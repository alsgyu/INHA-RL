import numpy as np
import time
import yaml
import logging
import threading
import queue
import hashlib
import json
import os
import subprocess
import sys

from booster_robotics_sdk_python import (
    ChannelFactory,
    B1LocoClient,
    B1LowCmdPublisher,
    B1LowStateSubscriber,
    LowCmd,
    LowState,
    RobotMode,
)

from utils.command import create_prepare_cmd
from utils.remote_control_service import JoystickConfig, RemoteControlService
from utils.rotate import rotate_vector_inverse_rpy
from utils.timer import TimerConfig, Timer
from utils.policy_thomas import Policy


class StdinCommandInput:
    def __init__(self):
        self.events = queue.Queue()
        self.thread = None

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._read_loop)
        self.thread.daemon = True
        self.thread.start()
        print("[cmd-input] line mode: b, r, stop, q, or '<vx> <vy> <vyaw>' such as '0.7 0.2 0.2'")

    def _read_loop(self):
        while True:
            try:
                line = sys.stdin.readline()
            except Exception as exc:
                self.events.put(("error", str(exc)))
                return
            if line == "":
                return
            self.events.put(self._parse(line))

    @staticmethod
    def _parse(line):
        text = line.strip()
        if not text:
            return ("noop", None)
        lowered = text.lower()
        if lowered in ("help", "h", "?"):
            return ("help", None)
        if lowered in ("b", "custom", "prepare"):
            return ("custom", None)
        if lowered in ("r", "rl", "walk", "gait"):
            return ("rl", None)
        if lowered in ("stop", "space", "zero", "hold", "0"):
            return ("cmd", (0.0, 0.0, 0.0))
        if lowered in ("q", "quit", "exit"):
            return ("quit", None)
        parts = lowered.replace(",", " ").split()
        if len(parts) != 3:
            return ("error", f"expected '<vx> <vy> <vyaw>', got '{text}'")
        try:
            return ("cmd", tuple(float(part) for part in parts))
        except ValueError:
            return ("error", f"could not parse command '{text}'")


class Controller:
    def __init__(self, cfg_file, args=None) -> None:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Load config
        with open(cfg_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

        # Initialize components
        remote_cfg = self.cfg.get("remote_control", {})
        args = args or object()
        self.policy_metadata_path = None
        self.policy_metadata_task = None
        self.policy_metadata_synced = False
        self._apply_policy_path_override(args)
        self._sync_policy_metadata_from_policy_path()
        self.remoteControlService = RemoteControlService(
            JoystickConfig(
                max_vx=float(getattr(args, "cmd_max_vx", None) or remote_cfg.get("max_vx", 0.5)),
                max_vy=float(getattr(args, "cmd_max_vy", None) or remote_cfg.get("max_vy", 0.5)),
                max_vyaw=float(getattr(args, "cmd_max_vyaw", None) or remote_cfg.get("max_vyaw", 0.5)),
                control_threshold=float(remote_cfg.get("control_threshold", 0.1)),
                keyboard_step_vx=float(remote_cfg.get("keyboard_step_vx", 0.1)),
                keyboard_step_vy=float(remote_cfg.get("keyboard_step_vy", 0.1)),
                keyboard_step_vyaw=float(remote_cfg.get("keyboard_step_vyaw", 0.1)),
                joystick_enabled=not bool(getattr(args, "stdin_cmd", False)),
                keyboard_enabled=not bool(getattr(args, "stdin_cmd", False)),
            )
        )
        self.policy = Policy(cfg=self.cfg)

        self._init_timer()
        self._init_low_state_values()
        self._init_communication()
        self.publish_runner = None
        self.running = True
        self.rl_start_time = None
        self.rl_start_target = None
        self.rl_motion_start_time = None
        self.motion_start_alpha = 0.0
        self.motion_command_alpha = 0.0
        self.last_debug_print_time = 0.0
        self.control_stage = "idle"
        self.safety_abort_triggered = False
        self.stdin_cmd = StdinCommandInput() if bool(getattr(args, "stdin_cmd", False)) else None

        self.publish_lock = threading.Lock()

    def _git_commit(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        try:
            return subprocess.check_output(
                ["git", "-C", repo_root, "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            return "unknown"

    def _resolve_policy_path(self, policy_path):
        if not policy_path:
            return policy_path
        if os.path.isabs(policy_path) and os.path.exists(policy_path):
            return policy_path

        deploy_dir = os.path.abspath(os.path.dirname(__file__))
        repo_dir = os.path.abspath(os.path.join(deploy_dir, ".."))
        relative_path = policy_path[2:] if policy_path.startswith("./") else policy_path
        candidates = [
            policy_path,
            os.path.abspath(policy_path),
            os.path.join(deploy_dir, relative_path),
            os.path.join(repo_dir, relative_path),
            os.path.join(repo_dir, "deploy", relative_path),
        ]
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return os.path.abspath(candidate)
        return os.path.abspath(policy_path)

    def _apply_policy_path_override(self, args):
        policy_path = getattr(args, "policy_path", None)
        if policy_path is None:
            self.cfg["policy"]["policy_path"] = self._resolve_policy_path(self.cfg["policy"]["policy_path"])
            return False

        resolved_path = self._resolve_policy_path(policy_path)
        changed = self.cfg["policy"].get("policy_path") != resolved_path
        self.cfg["policy"]["policy_path"] = resolved_path
        return changed

    @staticmethod
    def _merge_changed(target, source):
        changed = False
        for key, value in source.items():
            if target.get(key) != value:
                target[key] = value
                changed = True
        return changed

    def _sync_policy_metadata_from_policy_path(self):
        policy_path = self._resolve_policy_path(self.cfg["policy"]["policy_path"])
        self.cfg["policy"]["policy_path"] = policy_path
        metadata_path = os.path.splitext(policy_path)[0] + ".metadata.json"
        self.policy_metadata_path = metadata_path if os.path.exists(metadata_path) else None
        self.policy_metadata_task = None
        if self.policy_metadata_path is None:
            self.policy_metadata_synced = False
            return False

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        self.policy_metadata_task = metadata.get("task")

        changed = False
        commands = metadata.get("commands", {})
        if "command_slew_rate" in commands and self.cfg["policy"].get("command_slew_rate") != commands["command_slew_rate"]:
            self.cfg["policy"]["command_slew_rate"] = commands["command_slew_rate"]
            changed = True
        if "command_change_threshold" in commands:
            if self.cfg["policy"].get("command_change_threshold") != commands["command_change_threshold"]:
                self.cfg["policy"]["command_change_threshold"] = commands["command_change_threshold"]
                changed = True
        adapter = commands.get("adapter")
        if isinstance(adapter, dict):
            changed = self._merge_changed(self.cfg["policy"].setdefault("command_adapter", {}), adapter) or changed
        adapter_override = self.cfg["policy"].get("deploy_command_adapter_override")
        if isinstance(adapter_override, dict):
            changed = self._merge_changed(self.cfg["policy"].setdefault("command_adapter", {}), adapter_override) or changed

        normalization = metadata.get("normalization")
        if isinstance(normalization, dict):
            changed = self._merge_changed(self.cfg["policy"].setdefault("normalization", {}), normalization) or changed

        self.policy_metadata_synced = True
        return changed

    def _policy_sha1(self):
        policy_path = self._resolve_policy_path(self.cfg["policy"]["policy_path"])
        try:
            digest = hashlib.sha1()
            with open(policy_path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        except Exception:
            return "unknown"

    def apply_prepare_overrides(self, args):
        reload_policy = False
        reload_policy = self._apply_policy_path_override(args) or reload_policy
        reload_policy = self._sync_policy_metadata_from_policy_path() or reload_policy

        deploy_profile = str(getattr(args, "deploy_profile", "safe")).lower()
        if deploy_profile == "training_exact":
            policy_cfg = self.cfg["policy"]
            adapter = policy_cfg.setdefault("command_adapter", {})
            policy_cfg["deploy_action_clip"] = 1.0
            policy_cfg["deploy_action_scale"] = 1.0
            policy_cfg["deploy_scale_actions_in_policy"] = True
            policy_cfg["deploy_action_rate_limit"] = 0.0
            policy_cfg.pop("deploy_action_clip_by_index", None)
            policy_cfg.pop("deploy_action_scale_by_index", None)
            policy_cfg.pop("deploy_action_rate_limit_by_index", None)
            policy_cfg.pop("deploy_action_lower_by_index", None)
            policy_cfg.pop("deploy_action_upper_by_index", None)
            policy_cfg["rl_publish_mode"] = "policy_step"
            policy_cfg["rl_target_filter_alpha"] = 1.0
            policy_cfg["motion_start_action_ramp_s"] = 0.0
            adapter["gait_frequency_min"] = 1.15
            adapter["body_pitch_offset"] = 0.0
            adapter["forward_pitch_vx_comp_enabled"] = False
            adapter["heading_correction_enabled"] = False
            reload_policy = True

        pitch_ids = {
            "hip": [10, 16],
            "knee": [13, 19],
            "ankle": [14, 20],
        }
        for name, ids in pitch_ids.items():
            value = getattr(args, f"prepare_{name}_pitch", None)
            if value is None:
                continue
            for i in ids:
                self.cfg["prepare"]["default_qpos"][i] = float(value)
            reload_policy = True

        prepare_hold_current = getattr(args, "prepare_hold_current", None)
        if prepare_hold_current is not None:
            self.cfg["prepare"]["hold_current_on_custom"] = bool(prepare_hold_current)

        ankle_kp = getattr(args, "prepare_ankle_kp", None)
        if ankle_kp is not None:
            for i in self.cfg.get("mech", {}).get("parallel_mech_indexes", []):
                self.cfg["prepare"]["stiffness"][i] = float(ankle_kp)

        ankle_kd = getattr(args, "prepare_ankle_kd", None)
        if ankle_kd is not None:
            for i in self.cfg.get("mech", {}).get("parallel_mech_indexes", []):
                self.cfg["prepare"]["damping"][i] = float(ankle_kd)

        deploy_action_clip = getattr(args, "deploy_action_clip", None)
        if deploy_action_clip is not None:
            self.cfg["policy"]["deploy_action_clip"] = float(deploy_action_clip)

        deploy_action_scale = getattr(args, "deploy_action_scale", None)
        if deploy_action_scale is not None:
            self.cfg["policy"]["deploy_action_scale"] = float(deploy_action_scale)

        deploy_action_rate_limit = getattr(args, "deploy_action_rate_limit", None)
        if deploy_action_rate_limit is not None:
            self.cfg["policy"]["deploy_action_rate_limit"] = float(deploy_action_rate_limit)

        deploy_target_default_blend = getattr(args, "deploy_target_default_blend", None)
        if deploy_target_default_blend is not None:
            self.cfg["policy"]["deploy_target_default_blend"] = float(deploy_target_default_blend)
            reload_policy = True

        deploy_default_qpos_source = getattr(args, "deploy_default_qpos_source", None)
        if deploy_default_qpos_source is not None:
            self.cfg["policy"]["deploy_default_qpos_source"] = str(deploy_default_qpos_source)
            reload_policy = True

        rl_target_filter_alpha = getattr(args, "rl_target_filter_alpha", None)
        if rl_target_filter_alpha is not None:
            self.cfg["policy"]["rl_target_filter_alpha"] = float(rl_target_filter_alpha)

        rl_publish_mode = getattr(args, "rl_publish_mode", None)
        if rl_publish_mode is not None:
            self.cfg["policy"]["rl_publish_mode"] = str(rl_publish_mode)

        motion_ramp = getattr(args, "motion_start_action_ramp_s", None)
        if motion_ramp is not None:
            self.cfg["policy"]["motion_start_action_ramp_s"] = float(motion_ramp)

        motion_action_delay = getattr(args, "motion_start_action_delay_s", None)
        if motion_action_delay is not None:
            self.cfg["policy"]["motion_start_action_delay_s"] = float(motion_action_delay)

        motion_command_ramp = getattr(args, "motion_start_command_ramp_s", None)
        if motion_command_ramp is not None:
            self.cfg["policy"]["motion_start_command_ramp_s"] = float(motion_command_ramp)

        motion_command_delay = getattr(args, "motion_start_command_delay_s", None)
        if motion_command_delay is not None:
            self.cfg["policy"]["motion_start_command_delay_s"] = float(motion_command_delay)

        motion_hold = getattr(args, "motion_start_hold_s", None)
        if motion_hold is not None:
            self.cfg["policy"]["motion_start_hold_s"] = float(motion_hold)

        adapter = self.cfg["policy"].setdefault("command_adapter", {})
        adapter_float_overrides = {
            "adapter_stand_command_threshold": "stand_command_threshold",
            "adapter_gait_frequency_min": "gait_frequency_min",
            "adapter_gait_frequency_max": "gait_frequency_max",
            "adapter_forward_pitch_vx_comp_deadband": "forward_pitch_vx_comp_deadband",
            "adapter_forward_pitch_vx_comp_gain": "forward_pitch_vx_comp_gain",
            "adapter_forward_pitch_vx_comp_max": "forward_pitch_vx_comp_max",
            "adapter_forward_pitch_vx_comp_min_vx": "forward_pitch_vx_comp_min_vx",
            "adapter_body_pitch_offset": "body_pitch_offset",
            "adapter_body_pitch_balance_gain": "body_pitch_balance_gain",
            "adapter_heading_correction_gain": "heading_correction_gain",
            "adapter_heading_correction_deadband": "heading_correction_deadband",
            "adapter_heading_correction_max_yaw_rate": "heading_correction_max_yaw_rate",
        }
        for arg_name, key in adapter_float_overrides.items():
            value = getattr(args, arg_name, None)
            if value is not None:
                adapter[key] = float(value)

        forward_pitch_enabled = getattr(args, "adapter_forward_pitch_vx_comp_enabled", None)
        if forward_pitch_enabled is not None:
            adapter["forward_pitch_vx_comp_enabled"] = bool(forward_pitch_enabled)

        heading_enabled = getattr(args, "adapter_heading_correction_enabled", None)
        if heading_enabled is not None:
            adapter["heading_correction_enabled"] = bool(heading_enabled)

        heading_to_yaw = getattr(args, "adapter_heading_correction_apply_to_yaw_command", None)
        if heading_to_yaw is not None:
            adapter["heading_correction_apply_to_yaw_command"] = bool(heading_to_yaw)

        if reload_policy:
            self.policy = Policy(cfg=self.cfg)

    def print_startup_diagnostics(self, cfg_file):
        mech_indexes = self.cfg.get("mech", {}).get("parallel_mech_indexes", [])
        adapter = self.cfg["policy"].get("command_adapter", {})
        prepare_kp = [float(self.cfg["prepare"]["stiffness"][i]) for i in mech_indexes]
        common_kp = [float(self.cfg["common"]["stiffness"][i]) for i in mech_indexes]
        prepare_kd = [float(self.cfg["prepare"]["damping"][i]) for i in mech_indexes]
        common_kd = [float(self.cfg["common"]["damping"][i]) for i in mech_indexes]
        print(
            "[deploy-startup] "
            f"commit={self._git_commit()} "
            f"cwd={os.getcwd()} "
            f"script={os.path.abspath(__file__)} "
            f"config={os.path.abspath(cfg_file)} "
            f"metadata={self.policy_metadata_path or 'none'} "
            f"metadata_task={self.policy_metadata_task or 'none'}"
        )
        print(
            "[deploy-startup] "
            f"policy={self.cfg['policy']['policy_path']} "
            f"policy_sha1={self._policy_sha1()} "
            f"command_source={self.cfg['policy'].get('command_source', 'remote')} "
            f"command_slew_rate={self.cfg['policy'].get('command_slew_rate', 'default')} "
            f"prepare_publish=continuous "
            f"zero_hold={self.cfg['policy'].get('zero_command_hold_prepare', False)} "
            f"default_source={getattr(self.policy, 'default_dof_pos_source', 'unknown')} "
            f"target_default_blend={getattr(self.policy, 'target_default_blend', 'default')} "
            f"deploy_action_clip={self.cfg['policy'].get('deploy_action_clip', 'default')} "
            f"deploy_action_scale={self.cfg['policy'].get('deploy_action_scale', 1.0)} "
            f"scale_actions_in_policy={self.cfg['policy'].get('deploy_scale_actions_in_policy', False)} "
            f"action_rate_limit={self.cfg['policy'].get('deploy_action_rate_limit', 'off')} "
            f"action_rate_limit_by_index={self.cfg['policy'].get('deploy_action_rate_limit_by_index', 'off')} "
            f"action_lower_by_index={self.cfg['policy'].get('deploy_action_lower_by_index', 'off')} "
            f"action_upper_by_index={self.cfg['policy'].get('deploy_action_upper_by_index', 'off')} "
            f"rl_publish_mode={self._rl_publish_mode()} "
            f"target_filter_alpha={self.cfg['policy'].get('rl_target_filter_alpha', 0.2)} "
            f"dof_vel_source={self.cfg['policy'].get('policy_dof_vel_source', 'raw')} "
            f"dof_vel_filter_alpha={self.cfg['policy'].get('policy_dof_vel_filter_alpha', 'off')} "
            f"obs_dof_vel_scale={self.cfg['policy']['normalization'].get('dof_vel', 'default')} "
            f"control_action_scale={self.cfg['policy']['control'].get('action_scale', 'default')} "
            f"control_decimation={self.cfg['policy']['control'].get('decimation', 'default')} "
            f"motion_hold={self.cfg['policy'].get('motion_start_hold_s', 'default')} "
            f"motion_action_delay={self.cfg['policy'].get('motion_start_action_delay_s', 0.0)} "
            f"motion_ramp={self.cfg['policy'].get('motion_start_action_ramp_s', 'default')} "
            f"motion_cmd_delay={self.cfg['policy'].get('motion_start_command_delay_s', 0.0)} "
            f"motion_cmd_ramp={self.cfg['policy'].get('motion_start_command_ramp_s', 'default')} "
            f"gait_min={adapter.get('gait_frequency_min', 'default')} "
            f"gait_max={adapter.get('gait_frequency_max', 'default')} "
            f"stop_hold={adapter.get('stop_gait_hold_enabled', False)}:"
            f"{adapter.get('stop_gait_hold_s', 'default')}s "
            f"decel_hold={adapter.get('decel_gait_hold_enabled', False)}:"
            f"{adapter.get('decel_gait_hold_s', 'default')}s "
            f"body_pitch_gain={adapter.get('body_pitch_gain', 'default')} "
            f"body_pitch_offset={adapter.get('body_pitch_offset', 0.0)} "
            f"body_pitch_balance_gain={adapter.get('body_pitch_balance_gain', 0.0)} "
            f"forward_pitch_vx_comp={adapter.get('forward_pitch_vx_comp_enabled', False)} "
            f"forward_pitch_gain={adapter.get('forward_pitch_vx_comp_gain', 'default')} "
            f"forward_pitch_max={adapter.get('forward_pitch_vx_comp_max', 'default')} "
            f"forward_pitch_min_vx={adapter.get('forward_pitch_vx_comp_min_vx', 'default')} "
            f"heading_correction={adapter.get('heading_correction_enabled', False)} "
            f"heading_gain={adapter.get('heading_correction_gain', 'default')} "
            f"heading_max_yaw={adapter.get('heading_correction_max_yaw_rate', 'default')} "
            f"heading_to_yaw={adapter.get('heading_correction_apply_to_yaw_command', False)} "
            f"debug={self.cfg.get('debug', {}).get('enabled', False)}"
        )
        print(
            "[deploy-startup] "
            f"cmd_limits=(vx={self.remoteControlService.config.max_vx:.2f},"
            f"vy={self.remoteControlService.config.max_vy:.2f},"
            f"vyaw={self.remoteControlService.config.max_vyaw:.2f}) "
            f"stdin_cmd={self.stdin_cmd is not None}"
        )
        print(
            "[deploy-startup] "
            f"action_dof_indexes={[int(i) for i in getattr(self.policy, 'action_dof_indexes', [])]} "
            f"policy_joint_names={getattr(self.policy, 'policy_joint_names', 'unknown')}"
        )
        print(
            "[deploy-startup] "
            f"ankle_ids={mech_indexes} "
            f"mode=(prepare:{self._parallel_mech_mode('prepare')},rl:{self._parallel_mech_mode('rl')}) "
            f"prepare_hold_current={self.cfg.get('prepare', {}).get('hold_current_on_custom', False)} "
            f"prepare_transition_s={self.cfg.get('prepare', {}).get('transition_s', 0.0)} "
            f"prepare_safety={self.cfg.get('prepare', {}).get('safety_abort_enabled', False)} "
            f"prepare_kp={prepare_kp} prepare_kd={prepare_kd} "
            f"common_kp={common_kp} common_kd={common_kd}"
        )
        print(
            "[deploy-startup] "
            f"prepare_leg_pose="
            f"L(hip={self.cfg['prepare']['default_qpos'][10]:+.3f},"
            f"knee={self.cfg['prepare']['default_qpos'][13]:+.3f},"
            f"ankle={self.cfg['prepare']['default_qpos'][14]:+.3f}) "
            f"R(hip={self.cfg['prepare']['default_qpos'][16]:+.3f},"
            f"knee={self.cfg['prepare']['default_qpos'][19]:+.3f},"
            f"ankle={self.cfg['prepare']['default_qpos'][20]:+.3f})"
        )
        print(
            "[deploy-startup] "
            f"common_leg_pose="
            f"L(hip={self.cfg['common']['default_qpos'][10]:+.3f},"
            f"knee={self.cfg['common']['default_qpos'][13]:+.3f},"
            f"ankle={self.cfg['common']['default_qpos'][14]:+.3f}) "
            f"R(hip={self.cfg['common']['default_qpos'][16]:+.3f},"
            f"knee={self.cfg['common']['default_qpos'][19]:+.3f},"
            f"ankle={self.cfg['common']['default_qpos'][20]:+.3f})"
        )

    def _send_prepare_target_locked(self, target):
        create_prepare_cmd(self.low_cmd, self.cfg)
        for i in range(self.cfg["common"]["joint_cnt"]):
            self.low_cmd.motor_cmd[i].q = float(target[i])
            self.dof_target[i] = float(target[i])
            self.filtered_dof_target[i] = float(target[i])
        self.control_stage = "prepare"
        if self._abort_if_prepare_unstable_locked("prepare"):
            return
        self._apply_parallel_mech_cmd("prepare")
        self._send_cmd(self.low_cmd)

    def _ramp_to_prepare_target(self, start_target, final_target):
        transition_s = float(self.cfg.get("prepare", {}).get("transition_s", 0.0))
        if transition_s <= 1.0e-6:
            with self.publish_lock:
                self._send_prepare_target_locked(final_target)
            return

        transition_dt_s = max(float(self.cfg.get("prepare", {}).get("transition_dt_s", 0.02)), 0.002)
        ramp_start = time.perf_counter()
        print(f"[deploy-prepare] ramping prepare target over {transition_s:.2f}s")
        while True:
            elapsed = time.perf_counter() - ramp_start
            alpha = min(max(elapsed / transition_s, 0.0), 1.0)
            smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            target = start_target + smooth_alpha * (final_target - start_target)
            with self.publish_lock:
                self._send_prepare_target_locked(target)
            if not self.running:
                break
            if alpha >= 1.0:
                break
            time.sleep(transition_dt_s)

    def _init_timer(self):
        self.timer = Timer(TimerConfig(time_step=self.cfg["common"]["dt"]))
        self.next_publish_time = self.timer.get_time()
        self.next_inference_time = self.timer.get_time()

    def _init_low_state_values(self):
        self.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.base_rpy = np.zeros(3, dtype=np.float32)
        self.projected_gravity = np.zeros(3, dtype=np.float32)
        self.dof_pos = np.zeros(self.cfg["common"]["joint_cnt"], dtype=np.float32)
        self.dof_vel = np.zeros(self.cfg["common"]["joint_cnt"], dtype=np.float32)

        self.dof_target = np.zeros(self.cfg["common"]["joint_cnt"], dtype=np.float32)
        self.filtered_dof_target = np.zeros(self.cfg["common"]["joint_cnt"], dtype=np.float32)
        self.dof_pos_latest = np.zeros(self.cfg["common"]["joint_cnt"], dtype=np.float32)
        self.policy_dof_pos_prev = np.zeros(self.cfg["common"]["joint_cnt"], dtype=np.float32)
        self.policy_dof_vel = np.zeros(self.cfg["common"]["joint_cnt"], dtype=np.float32)
        self.policy_dof_vel_initialized = False

    def _init_communication(self) -> None:
        try:
            self.low_cmd = LowCmd()
            self.low_state_subscriber = B1LowStateSubscriber(self._low_state_handler)
            self.low_cmd_publisher = B1LowCmdPublisher()
            self.client = B1LocoClient()

            self.low_state_subscriber.InitChannel()
            self.low_cmd_publisher.InitChannel()
            self.client.Init()
        except Exception as e:
            self.logger.error(f"Failed to initialize communication: {e}")
            raise

    def _low_state_handler(self, low_state_msg: LowState):
        if abs(low_state_msg.imu_state.rpy[0]) > 1.0 or abs(low_state_msg.imu_state.rpy[1]) > 1.0:
            self.logger.warning("IMU base rpy values are too large: {}".format(low_state_msg.imu_state.rpy))
            self.running = False
        self.timer.tick_timer_if_sim()
        time_now = self.timer.get_time()
        self.base_rpy[:] = low_state_msg.imu_state.rpy
        for i, motor in enumerate(low_state_msg.motor_state_serial):
            self.dof_pos_latest[i] = motor.q
        if time_now >= self.next_inference_time:
            self.projected_gravity[:] = rotate_vector_inverse_rpy(
                low_state_msg.imu_state.rpy[0],
                low_state_msg.imu_state.rpy[1],
                low_state_msg.imu_state.rpy[2],
                np.array([0.0, 0.0, -1.0]),
            )
            self.base_ang_vel[:] = low_state_msg.imu_state.gyro
            for i, motor in enumerate(low_state_msg.motor_state_serial):
                self.dof_pos[i] = motor.q
                self.dof_vel[i] = motor.dq

    def _send_cmd(self, cmd: LowCmd):
        self.low_cmd_publisher.Write(cmd)

    def _set_joint_gains(self, section):
        for i in range(self.cfg["common"]["joint_cnt"]):
            self.low_cmd.motor_cmd[i].kp = float(self.cfg[section]["stiffness"][i])
            self.low_cmd.motor_cmd[i].kd = float(self.cfg[section]["damping"][i])
            self.low_cmd.motor_cmd[i].tau = 0.0

    def _remote_command_norm(self):
        vx = float(self.remoteControlService.get_vx_cmd())
        vy = float(self.remoteControlService.get_vy_cmd())
        vyaw = float(self.remoteControlService.get_vyaw_cmd())
        return float(np.sqrt(vx * vx + vy * vy + vyaw * vyaw))

    def start_stdin_command_input(self):
        if self.stdin_cmd is not None:
            self.stdin_cmd.start()

    def _process_stdin_commands(self):
        if self.stdin_cmd is None:
            return
        while True:
            try:
                kind, payload = self.stdin_cmd.events.get_nowait()
            except queue.Empty:
                return

            if kind == "noop":
                continue
            if kind == "help":
                print("[cmd-input] use: b | r | stop | q | <vx> <vy> <vyaw>")
                continue
            if kind == "custom":
                self.remoteControlService.request_custom_mode()
                print("[cmd-input] requested custom mode")
                continue
            if kind == "rl":
                self.remoteControlService.request_rl_gait()
                print("[cmd-input] requested RL gait")
                continue
            if kind == "quit":
                self.remoteControlService.set_velocity_command(0.0, 0.0, 0.0)
                self.running = False
                print("[cmd-input] stop command sent; exiting")
                continue
            if kind == "error":
                print(f"[cmd-input] {payload}")
                continue
            if kind == "cmd":
                vx, vy, vyaw = payload
                self.remoteControlService.set_velocity_command(vx, vy, vyaw)
                actual = (
                    self.remoteControlService.get_vx_cmd(),
                    self.remoteControlService.get_vy_cmd(),
                    self.remoteControlService.get_vyaw_cmd(),
                )
                clipped = "" if np.allclose(actual, payload, atol=1.0e-6) else " clipped"
                print(
                    "[cmd-input] "
                    f"cmd=({actual[0]:+.2f},{actual[1]:+.2f},{actual[2]:+.2f}){clipped}"
                )

    def _zero_command_hold_active(self):
        if not bool(self.cfg["policy"].get("zero_command_hold_prepare", False)):
            return False
        threshold = float(self.cfg["policy"].get("zero_command_hold_threshold", 0.035))
        if self._remote_command_norm() > threshold:
            return False

        if self.control_stage == "rl":
            smoothed = getattr(self.policy, "smoothed_commands", np.zeros(3, dtype=np.float32))
            if float(np.linalg.norm(smoothed)) > threshold:
                return False
            adapter = self.cfg["policy"].get("command_adapter", {})
            recovery_window_s = 0.0
            if bool(adapter.get("stop_gait_hold_enabled", False)):
                recovery_window_s = max(recovery_window_s, float(adapter.get("stop_gait_hold_s", 0.0)))
            if bool(adapter.get("decel_gait_hold_enabled", False)):
                recovery_window_s = max(recovery_window_s, float(adapter.get("decel_gait_hold_s", 0.0)))
            if float(getattr(self.policy, "command_age", 1.0e6)) <= recovery_window_s:
                return False

        return True

    def _rl_publish_mode(self):
        return str(self.cfg["policy"].get("rl_publish_mode", "continuous")).lower()

    def _reset_policy_dof_vel(self):
        self.policy_dof_pos_prev[:] = self.dof_pos
        self.policy_dof_vel[:] = 0.0
        self.policy_dof_vel_initialized = True

    def _update_policy_dof_vel(self):
        source = str(self.cfg["policy"].get("policy_dof_vel_source", "raw")).lower()
        if source in ("raw", "motor", "motor_state", "low_state"):
            self.policy_dof_vel[:] = self.dof_vel
            return
        if source not in ("finite_difference", "finite_diff", "policy_step"):
            raise ValueError(f"Unsupported policy_dof_vel_source '{source}'")

        if not self.policy_dof_vel_initialized:
            self._reset_policy_dof_vel()
            return

        dt = max(float(self.policy.get_policy_interval()), 1.0e-6)
        measured = (self.dof_pos - self.policy_dof_pos_prev) / dt
        vel_clip = self.cfg["policy"].get("policy_dof_vel_clip")
        if vel_clip is not None and float(vel_clip) > 0.0:
            measured = np.clip(measured, -float(vel_clip), float(vel_clip))
        alpha = float(np.clip(self.cfg["policy"].get("policy_dof_vel_filter_alpha", 1.0), 0.0, 1.0))
        self.policy_dof_vel[:] = (1.0 - alpha) * self.policy_dof_vel + alpha * measured
        self.policy_dof_pos_prev[:] = self.dof_pos

    def _start_publish_thread(self):
        if self.publish_runner is not None and self.publish_runner.is_alive():
            return
        self.next_publish_time = self.timer.get_time()
        self.publish_runner = threading.Thread(target=self._publish_cmd)
        self.publish_runner.daemon = True
        self.publish_runner.start()

    def _parallel_mech_mode(self, stage):
        mech_cfg = self.cfg.get("mech", {})
        return str(
            mech_cfg.get(f"parallel_mech_{stage}_mode", mech_cfg.get("parallel_mech_mode", "position"))
        ).lower()

    def _apply_parallel_mech_cmd(self, stage):
        mech_cfg = self.cfg.get("mech", {})
        indexes = mech_cfg.get("parallel_mech_indexes", [])
        if not indexes:
            return
        mode = self._parallel_mech_mode(stage)
        if mode in ("off", "none", "disabled"):
            return

        gain_section = "prepare" if stage == "prepare" else "common"
        stiffness = self.cfg[gain_section]["stiffness"]
        damping = self.cfg[gain_section]["damping"]
        torque_limit = self.cfg["common"].get("torque_limit", [0.0] * self.cfg["common"]["joint_cnt"])
        for i in indexes:
            target = float(self.filtered_dof_target[i])
            if mode in ("position", "pd", "servo"):
                self.low_cmd.motor_cmd[i].q = target
                self.low_cmd.motor_cmd[i].tau = 0.0
                self.low_cmd.motor_cmd[i].kp = float(stiffness[i])
                self.low_cmd.motor_cmd[i].kd = float(damping[i])
            elif mode == "torque":
                current = float(self.dof_pos_latest[i])
                limit = float(torque_limit[i])
                self.low_cmd.motor_cmd[i].q = current
                self.low_cmd.motor_cmd[i].tau = float(np.clip((target - current) * float(stiffness[i]), -limit, limit))
                self.low_cmd.motor_cmd[i].kp = 0.0
                self.low_cmd.motor_cmd[i].kd = float(damping[i])
            else:
                raise ValueError(f"Unsupported parallel_mech mode '{mode}' for stage '{stage}'")

    def _write_filtered_target_to_low_cmd(self, stage):
        for i in range(self.cfg["common"]["joint_cnt"]):
            self.low_cmd.motor_cmd[i].q = self.filtered_dof_target[i]
        self._apply_parallel_mech_cmd("prepare" if stage in ("prepare", "rl_hold") else "rl")

    def _abort_if_prepare_unstable_locked(self, stage):
        if stage not in ("prepare", "rl_hold") or self.safety_abort_triggered:
            return False
        prepare_cfg = self.cfg.get("prepare", {})
        if not bool(prepare_cfg.get("safety_abort_enabled", False)):
            return False

        action_dof_indexes = getattr(self.policy, "action_dof_indexes", np.arange(10, 22, dtype=np.int64))
        target = self.filtered_dof_target[action_dof_indexes]
        actual = self.dof_pos_latest[action_dof_indexes]
        leg_error_abs_max = float(np.max(np.abs(target - actual)))
        raw_leg_vel_abs_max = float(np.max(np.abs(self.dof_vel[action_dof_indexes])))
        pitch_abs = abs(float(self.base_rpy[1]))
        roll_abs = abs(float(self.base_rpy[0]))

        pitch_limit = float(prepare_cfg.get("safety_abort_pitch_abs", 10.0))
        roll_limit = float(prepare_cfg.get("safety_abort_roll_abs", 10.0))
        leg_error_limit = float(prepare_cfg.get("safety_abort_leg_error_abs", 10.0))
        raw_leg_vel_limit = float(prepare_cfg.get("safety_abort_raw_leg_vel_abs", 1000.0))

        reasons = []
        if pitch_abs > pitch_limit:
            reasons.append(f"pitch={self.base_rpy[1]:+.3f}>{pitch_limit:.3f}")
        if roll_abs > roll_limit:
            reasons.append(f"roll={self.base_rpy[0]:+.3f}>{roll_limit:.3f}")
        if leg_error_abs_max > leg_error_limit:
            reasons.append(f"leg_err={leg_error_abs_max:.3f}>{leg_error_limit:.3f}")
        if raw_leg_vel_abs_max > raw_leg_vel_limit:
            reasons.append(f"raw_leg_vel={raw_leg_vel_abs_max:.3f}>{raw_leg_vel_limit:.3f}")

        if not reasons:
            return False

        self.safety_abort_triggered = True
        self.running = False
        print(
            "[deploy-safety] prepare instability detected; switching to damping: "
            + ", ".join(reasons)
        )
        try:
            self.client.ChangeMode(RobotMode.kDamping)
        except Exception as exc:
            print(f"[deploy-safety] failed to switch to damping: {exc}")
        return True

    def _publish_policy_step_target(self):
        if self._rl_publish_mode() not in ("policy", "policy_step", "policy_dt"):
            return
        with self.publish_lock:
            if self.control_stage != "rl":
                return
            if self._abort_if_prepare_unstable_locked(self.control_stage):
                return
            alpha = float(np.clip(self.cfg["policy"].get("rl_target_filter_alpha", 1.0), 0.0, 1.0))
            self.filtered_dof_target[:] = self.filtered_dof_target * (1.0 - alpha) + self.dof_target * alpha
            self._write_filtered_target_to_low_cmd("rl")
            self._send_cmd(self.low_cmd)

    def _print_debug_state(self, time_now):
        debug_cfg = self.cfg.get("debug", {})
        if not bool(debug_cfg.get("enabled", False)):
            return
        interval = max(float(debug_cfg.get("print_interval_s", 1.0)), 1.0e-3)
        if time_now - self.last_debug_print_time < interval:
            return
        self.last_debug_print_time = time_now
        leg_ids = [10, 13, 14, 15, 16, 19, 20, 21]
        actual = [float(self.dof_pos_latest[i]) for i in leg_ids]
        target = [float(self.filtered_dof_target[i]) for i in leg_ids]
        error = [target[j] - actual[j] for j in range(len(leg_ids))]
        kp = [float(self.low_cmd.motor_cmd[i].kp) for i in leg_ids]
        kd = [float(self.low_cmd.motor_cmd[i].kd) for i in leg_ids]
        tau = [float(self.low_cmd.motor_cmd[i].tau) for i in leg_ids]
        action_dof_indexes = self.policy.action_dof_indexes
        leg_target = self.filtered_dof_target[action_dof_indexes]
        leg_actual = self.dof_pos_latest[action_dof_indexes]
        leg_desired = self.dof_target[action_dof_indexes]
        policy_default = getattr(self.policy, "default_dof_pos", self.cfg["common"]["default_qpos"])[action_dof_indexes]
        target_default = getattr(self.policy, "target_default_dof_pos", policy_default)[action_dof_indexes]
        leg_error_abs_max = float(np.max(np.abs(leg_target - leg_actual)))
        default_error_abs_max = float(np.max(np.abs(policy_default - leg_actual)))
        target_default_error_abs_max = float(np.max(np.abs(target_default - leg_actual)))
        target_lag_abs_max = float(np.max(np.abs(leg_desired - leg_target)))
        leg_vel_abs_max = float(np.max(np.abs(self.policy_dof_vel[action_dof_indexes])))
        raw_leg_vel_abs_max = float(np.max(np.abs(self.dof_vel[action_dof_indexes])))
        action_abs_max = float(np.max(np.abs(self.policy.actions))) if self.policy.actions.size else 0.0
        raw_action_abs_max = float(np.max(np.abs(getattr(self.policy, "raw_actions", self.policy.actions))))
        lateral_action_ids = [1, 2, 5, 7, 8, 11]
        lateral_action_abs_max = (
            float(np.max(np.abs(self.policy.actions[lateral_action_ids])))
            if self.policy.actions.size > max(lateral_action_ids)
            else 0.0
        )
        raw_actions = getattr(self.policy, "raw_actions", self.policy.actions)
        raw_lateral_action_abs_max = (
            float(np.max(np.abs(raw_actions[lateral_action_ids])))
            if raw_actions.size > max(lateral_action_ids)
            else 0.0
        )
        action_sample = [float(x) for x in self.policy.actions]
        command_block = [float(x) for x in getattr(self.policy, "command_block", [])]
        print(
            "[deploy-debug] "
            f"cmd=({self.remoteControlService.get_vx_cmd():+.2f},"
            f"{self.remoteControlService.get_vy_cmd():+.2f},"
            f"{self.remoteControlService.get_vyaw_cmd():+.2f}) "
            f"policy_cmd=({self.policy.policy_commands[0]:+.2f},"
            f"{self.policy.policy_commands[1]:+.2f},"
            f"{self.policy.policy_commands[2]:+.2f}) "
            f"cmd10={[round(x, 3) for x in command_block]} "
            f"gait={self.policy.gait_frequency:.2f} "
            f"recovery=(stop:{getattr(self.policy, 'stop_recovery', False)},"
            f"decel:{getattr(self.policy, 'decel_recovery', False)}) "
            f"cmd_age={getattr(self.policy, 'command_age', 0.0):.2f} "
            f"phase={getattr(self.policy, 'gait_process', 0.0):.2f} "
            f"vx_corr={getattr(self.policy, 'balance_vx_correction', 0.0):+.2f} "
            f"yaw_corr={getattr(self.policy, 'heading_correction_yaw', 0.0):+.2f} "
            f"heading_err={getattr(self.policy, 'heading_error_yaw', 0.0):+.2f} "
            f"alpha={self.motion_start_alpha:.2f} "
            f"cmd_alpha={self.motion_command_alpha:.2f} "
            f"stage={self.control_stage} "
            f"rpy=({self.base_rpy[0]:+.3f},{self.base_rpy[1]:+.3f},{self.base_rpy[2]:+.3f}) "
            f"grav=({self.projected_gravity[0]:+.3f},{self.projected_gravity[1]:+.3f},{self.projected_gravity[2]:+.3f}) "
            f"gyro=({self.base_ang_vel[0]:+.3f},{self.base_ang_vel[1]:+.3f},{self.base_ang_vel[2]:+.3f}) "
            f"mech=(prepare:{self._parallel_mech_mode('prepare')},rl:{self._parallel_mech_mode('rl')}) "
            f"actual[{leg_ids}]={[round(x, 3) for x in actual]} "
            f"target[{leg_ids}]={[round(x, 3) for x in target]} "
            f"err={[round(x, 3) for x in error]} "
            f"leg_err_max={leg_error_abs_max:.3f} "
            f"default_err_max={default_error_abs_max:.3f} "
            f"target_default_err_max={target_default_error_abs_max:.3f} "
            f"target_lag_max={target_lag_abs_max:.3f} "
            f"leg_vel_max={leg_vel_abs_max:.3f} "
            f"raw_leg_vel_max={raw_leg_vel_abs_max:.3f} "
            f"act_max={action_abs_max:.3f} "
            f"raw_act_max={raw_action_abs_max:.3f} "
            f"lat_act_max={lateral_action_abs_max:.3f} "
            f"raw_lat_act_max={raw_lateral_action_abs_max:.3f} "
            f"act={[round(x, 3) for x in action_sample]} "
            f"kp={[round(x, 1) for x in kp]} "
            f"kd={[round(x, 1) for x in kd]} "
            f"tau={[round(x, 2) for x in tau]}"
        )

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.remoteControlService.close()
        if hasattr(self, "low_cmd_publisher"):
            self.low_cmd_publisher.CloseChannel()
        if hasattr(self, "low_state_subscriber"):
            self.low_state_subscriber.CloseChannel()
        if hasattr(self, "publish_runner") and getattr(self, "publish_runner") != None:
            self.publish_runner.join(timeout=1.0)

    @staticmethod
    def _smoothstep(alpha):
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return alpha * alpha * (3.0 - 2.0 * alpha)

    def start_custom_mode_conditionally(self):
        print(f"{self.remoteControlService.get_custom_mode_operation_hint()}")
        while self.running:
            self._process_stdin_commands()
            if self.remoteControlService.start_custom_mode():
                break
            time.sleep(0.1)
        if not self.running:
            return
        start_time = time.perf_counter()
        prepare_target = np.array(self.cfg["prepare"]["default_qpos"], dtype=np.float32)
        start_target = np.copy(self.dof_pos_latest)
        if not np.any(np.abs(start_target) > 1.0e-6):
            start_target = np.copy(prepare_target)
        hold_current = bool(self.cfg.get("prepare", {}).get("hold_current_on_custom", False))
        if hold_current:
            prepare_target = np.copy(start_target)
        with self.publish_lock:
            self._send_prepare_target_locked(start_target)
        send_time = time.perf_counter()
        self.logger.debug(f"Send cmd took {(send_time - start_time)*1000:.4f} ms")
        self.client.ChangeMode(RobotMode.kCustom)
        self._start_publish_thread()
        if hold_current:
            print("[deploy-prepare] holding current joint target")
        else:
            self._ramp_to_prepare_target(start_target, prepare_target)
        if not self.running:
            return
        print("[deploy-prepare] custom mode active; prepare command sent")
        end_time = time.perf_counter()
        self.logger.debug(f"Change mode took {(end_time - send_time)*1000:.4f} ms")

    def start_rl_gait_conditionally(self):
        print(f"{self.remoteControlService.get_rl_gait_operation_hint()}")
        while self.running:
            self._process_stdin_commands()
            if self.remoteControlService.start_rl_gait():
                break
            time.sleep(0.1)
        if not self.running:
            return
        with self.publish_lock:
            self.policy.reset_runtime_state()
            self._reset_policy_dof_vel()
            current_target = np.copy(self.filtered_dof_target)
            self.rl_start_target = current_target
            self._set_joint_gains("prepare")
            for i in range(self.cfg["common"]["joint_cnt"]):
                self.low_cmd.motor_cmd[i].q = current_target[i]
                self.dof_target[i] = current_target[i]
                self.filtered_dof_target[i] = current_target[i]
            self._apply_parallel_mech_cmd("prepare")
            self._send_cmd(self.low_cmd)
            self.rl_start_time = self.timer.get_time()
            self.next_inference_time = self.rl_start_time
            self.rl_motion_start_time = None
            self.motion_command_alpha = 0.0
            self.control_stage = "rl_hold"
        self._start_publish_thread()
        print("[deploy-rl] RL gait armed; holding stable prepare target until movement command")
        print(f"{self.remoteControlService.get_operation_hint()}")

    def run(self):
        self._process_stdin_commands()
        if not self.running:
            return
        time_now = self.timer.get_time()
        if time_now < self.next_inference_time:
            time.sleep(0.001)
            return
        self.logger.debug("-----------------------------------------------------")
        self.next_inference_time += self.policy.get_policy_interval()
        self.logger.debug(f"Next start time: {self.next_inference_time}")
        start_time = time.perf_counter()

        if self._zero_command_hold_active():
            with self.publish_lock:
                hold_gain_section = "common" if self.control_stage == "rl" else "prepare"
                if self.control_stage == "rl":
                    self.rl_start_target = np.copy(self.filtered_dof_target)
                elif self.rl_start_target is None:
                    self.rl_start_target = np.copy(self.filtered_dof_target)
                self._set_joint_gains(hold_gain_section)
                self.control_stage = "rl_hold"
                self.rl_motion_start_time = None
                self.motion_start_alpha = 0.0
                self.motion_command_alpha = 0.0
                self.policy.reset_runtime_state()
                self._reset_policy_dof_vel()
                self.dof_target[:] = self.rl_start_target
            time.sleep(0.001)
            return

        if self.control_stage != "rl":
            with self.publish_lock:
                self.control_stage = "rl"
                self.rl_motion_start_time = time_now
                self.motion_start_alpha = 0.0
                self.motion_command_alpha = 0.0
                self.rl_start_target = np.copy(self.filtered_dof_target)
                self.dof_target[:] = self.rl_start_target
                self._set_joint_gains("common")
                for i in range(self.cfg["common"]["joint_cnt"]):
                    self.low_cmd.motor_cmd[i].q = self.filtered_dof_target[i]
                self._apply_parallel_mech_cmd("rl")
                self._send_cmd(self.low_cmd)
                self.policy.reset_runtime_state()
                self._reset_policy_dof_vel()

        hold_s = float(self.cfg["policy"].get("motion_start_hold_s", self.cfg["policy"].get("startup_hold_s", 0.0)))
        ramp_s = float(
            self.cfg["policy"].get("motion_start_action_ramp_s", self.cfg["policy"].get("startup_action_ramp_s", 0.0))
        )
        elapsed = 0.0 if self.rl_motion_start_time is None else max(0.0, time_now - self.rl_motion_start_time)
        if elapsed < hold_s:
            if self.rl_start_target is None:
                self.rl_start_target = np.copy(self.filtered_dof_target)
            self.dof_target[:] = self.rl_start_target
            self.motion_start_alpha = 0.0
            self.motion_command_alpha = 0.0
            self.policy.reset_runtime_state()
            time.sleep(0.001)
            return
        action_delay_s = float(
            self.cfg["policy"].get(
                "motion_start_action_delay_s",
                self.cfg["policy"].get("motion_start_command_delay_s", 0.0),
            )
        )
        if ramp_s > 1.0e-6:
            action_scale_multiplier = self._smoothstep((elapsed - hold_s - action_delay_s) / ramp_s)
        else:
            action_scale_multiplier = 0.0 if elapsed < hold_s + action_delay_s else 1.0
        self.motion_start_alpha = action_scale_multiplier
        command_delay_s = float(self.cfg["policy"].get("motion_start_command_delay_s", 0.0))
        command_ramp_s = float(self.cfg["policy"].get("motion_start_command_ramp_s", ramp_s))
        if command_ramp_s > 1.0e-6:
            command_scale_multiplier = self._smoothstep((elapsed - hold_s - command_delay_s) / command_ramp_s)
        else:
            command_scale_multiplier = 0.0 if elapsed < hold_s + command_delay_s else 1.0
        self.motion_command_alpha = command_scale_multiplier
        self._update_policy_dof_vel()

        policy_target = self.policy.inference(
            time_now=time_now,
            dof_pos=self.dof_pos,
            dof_vel=self.policy_dof_vel,
            base_ang_vel=self.base_ang_vel,
            projected_gravity=self.projected_gravity,
            vx=self.remoteControlService.get_vx_cmd() * command_scale_multiplier,
            vy=self.remoteControlService.get_vy_cmd() * command_scale_multiplier,
            vyaw=self.remoteControlService.get_vyaw_cmd() * command_scale_multiplier,
            action_scale_multiplier=action_scale_multiplier,
        )
        if action_scale_multiplier < 1.0 and self.rl_start_target is not None:
            self.dof_target[:] = (
                (1.0 - action_scale_multiplier) * self.rl_start_target
                + action_scale_multiplier * policy_target
            )
        else:
            self.dof_target[:] = policy_target

        inference_time = time.perf_counter()
        self.logger.debug(f"Inference took {(inference_time - start_time)*1000:.4f} ms")
        self._publish_policy_step_target()
        self._print_debug_state(time_now)
        time.sleep(0.001)

    def _publish_cmd(self):
        while self.running:
            time_now = self.timer.get_time()
            if time_now < self.next_publish_time:
                time.sleep(0.001)
                continue
            self.next_publish_time += self.cfg["common"]["dt"]
            self.logger.debug(f"Next publish time: {self.next_publish_time}")

            with self.publish_lock:
                stage = self.control_stage
                if stage == "rl" and self._rl_publish_mode() in ("policy", "policy_step", "policy_dt"):
                    continue
                if stage == "rl":
                    alpha = float(np.clip(self.cfg["policy"].get("rl_target_filter_alpha", 0.2), 0.0, 1.0))
                    self.filtered_dof_target = self.filtered_dof_target * (1.0 - alpha) + self.dof_target * alpha
                elif stage in ("prepare", "rl_hold"):
                    self.filtered_dof_target[:] = self.dof_target

                if self._abort_if_prepare_unstable_locked(stage):
                    continue

                self._write_filtered_target_to_low_cmd(stage)

                start_time = time.perf_counter()
                self._send_cmd(self.low_cmd)
            publish_time = time.perf_counter()
            self.logger.debug(f"Publish took {(publish_time - start_time)*1000:.4f} ms")
            self._print_debug_state(time_now)
            time.sleep(0.001)

    def __enter__(self) -> "Controller":
        return self

    def __exit__(self, *args) -> None:
        self.cleanup()


if __name__ == "__main__":
    import argparse
    import signal
    import sys
    def signal_handler(sig, frame):
        print("\nShutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Name of the configuration file.")
    parser.add_argument("--net", type=str, default="127.0.0.1", help="Network interface for SDK communication.")
    parser.add_argument("--policy_path", type=str, default=None, help="Override TorchScript .pt policy path.")
    parser.add_argument(
        "--deploy_profile",
        choices=["safe", "training_exact"],
        default="safe",
        help="Use safe real-robot limits or the training-like action path for diagnosis.",
    )
    parser.add_argument("--prepare_hip_pitch", type=float, default=None)
    parser.add_argument("--prepare_knee_pitch", type=float, default=None)
    parser.add_argument("--prepare_ankle_pitch", type=float, default=None)
    parser.add_argument("--prepare_hold_current", dest="prepare_hold_current", action="store_true", default=None)
    parser.add_argument("--no_prepare_hold_current", dest="prepare_hold_current", action="store_false")
    parser.add_argument("--prepare_ankle_kp", type=float, default=None)
    parser.add_argument("--prepare_ankle_kd", type=float, default=None)
    parser.add_argument("--deploy_action_clip", type=float, default=None)
    parser.add_argument("--deploy_action_scale", type=float, default=None)
    parser.add_argument("--deploy_action_rate_limit", type=float, default=None)
    parser.add_argument("--deploy_target_default_blend", type=float, default=None)
    parser.add_argument("--deploy_default_qpos_source", choices=["common", "prepare"], default=None)
    parser.add_argument("--rl_target_filter_alpha", type=float, default=None)
    parser.add_argument(
        "--rl_publish_mode",
        choices=["continuous", "policy", "policy_step", "policy_dt"],
        default=None,
        help="Publish RL targets continuously or once per policy step like Booster Deploy.",
    )
    parser.add_argument("--motion_start_action_delay_s", type=float, default=None)
    parser.add_argument("--motion_start_hold_s", type=float, default=None)
    parser.add_argument("--motion_start_action_ramp_s", type=float, default=None)
    parser.add_argument("--motion_start_command_delay_s", type=float, default=None)
    parser.add_argument("--motion_start_command_ramp_s", type=float, default=None)
    parser.add_argument("--adapter_stand_command_threshold", type=float, default=None)
    parser.add_argument("--adapter_gait_frequency_min", type=float, default=None)
    parser.add_argument("--adapter_gait_frequency_max", type=float, default=None)
    parser.add_argument("--adapter_forward_pitch_vx_comp_deadband", type=float, default=None)
    parser.add_argument("--adapter_forward_pitch_vx_comp_gain", type=float, default=None)
    parser.add_argument("--adapter_forward_pitch_vx_comp_max", type=float, default=None)
    parser.add_argument("--adapter_forward_pitch_vx_comp_min_vx", type=float, default=None)
    parser.add_argument("--adapter_forward_pitch_vx_comp_enabled", dest="adapter_forward_pitch_vx_comp_enabled", action="store_true", default=None)
    parser.add_argument("--adapter_no_forward_pitch_vx_comp", dest="adapter_forward_pitch_vx_comp_enabled", action="store_false")
    parser.add_argument("--adapter_body_pitch_offset", type=float, default=None)
    parser.add_argument("--adapter_body_pitch_balance_gain", type=float, default=None)
    parser.add_argument("--adapter_heading_correction_gain", type=float, default=None)
    parser.add_argument("--adapter_heading_correction_deadband", type=float, default=None)
    parser.add_argument("--adapter_heading_correction_max_yaw_rate", type=float, default=None)
    parser.add_argument("--adapter_heading_correction_enabled", dest="adapter_heading_correction_enabled", action="store_true", default=None)
    parser.add_argument("--adapter_no_heading_correction", dest="adapter_heading_correction_enabled", action="store_false")
    parser.add_argument(
        "--adapter_heading_correction_apply_to_yaw_command",
        dest="adapter_heading_correction_apply_to_yaw_command",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--adapter_no_heading_correction_apply_to_yaw_command",
        dest="adapter_heading_correction_apply_to_yaw_command",
        action="store_false",
    )
    parser.add_argument(
        "--stdin_cmd",
        action="store_true",
        help="Read line-based velocity commands from stdin: '<vx> <vy> <vyaw>', stop, b, r, q.",
    )
    parser.add_argument("--cmd_max_vx", type=float, default=None, help="Override stdin/keyboard vx command limit.")
    parser.add_argument("--cmd_max_vy", type=float, default=None, help="Override stdin/keyboard vy command limit.")
    parser.add_argument("--cmd_max_vyaw", type=float, default=None, help="Override stdin/keyboard vyaw command limit.")
    args = parser.parse_args()
    cfg_candidates = [
        args.config,
        os.path.join("configs", args.config),
        os.path.join(os.path.dirname(__file__), "configs", args.config),
    ]
    cfg_file = next((path for path in cfg_candidates if os.path.exists(path)), cfg_candidates[-1])

    print(f"Starting custom controller, connecting to {args.net} ...")
    ChannelFactory.Instance().Init(0, args.net)

    with Controller(cfg_file, args=args) as controller:
        controller.apply_prepare_overrides(args)
        controller.print_startup_diagnostics(cfg_file)
        controller.start_stdin_command_input()
        time.sleep(2)  # Wait for channels to initialize
        print("Initialization complete.")
        controller.start_custom_mode_conditionally()
        if controller.running:
            controller.start_rl_gait_conditionally()

        try:
            while controller.running:
                controller.run()
            controller.client.ChangeMode(RobotMode.kDamping)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Cleaning up...")
            controller.cleanup()
