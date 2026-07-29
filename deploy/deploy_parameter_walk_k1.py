import numpy as np
import time
import yaml
import logging
import threading
import os
import subprocess

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


class Controller:
    def __init__(self, cfg_file) -> None:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Load config
        with open(cfg_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

        # Initialize components
        remote_cfg = self.cfg.get("remote_control", {})
        self.remoteControlService = RemoteControlService(
            JoystickConfig(
                max_vx=float(remote_cfg.get("max_vx", 0.5)),
                max_vy=float(remote_cfg.get("max_vy", 0.5)),
                max_vyaw=float(remote_cfg.get("max_vyaw", 0.5)),
                control_threshold=float(remote_cfg.get("control_threshold", 0.1)),
                keyboard_step_vx=float(remote_cfg.get("keyboard_step_vx", 0.1)),
                keyboard_step_vy=float(remote_cfg.get("keyboard_step_vy", 0.1)),
                keyboard_step_vyaw=float(remote_cfg.get("keyboard_step_vyaw", 0.1)),
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
        self.last_debug_print_time = 0.0
        self.control_stage = "idle"

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

    def apply_prepare_overrides(self, args):
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

        motion_ramp = getattr(args, "motion_start_action_ramp_s", None)
        if motion_ramp is not None:
            self.cfg["policy"]["motion_start_action_ramp_s"] = float(motion_ramp)

    def print_startup_diagnostics(self, cfg_file):
        mech_indexes = self.cfg.get("mech", {}).get("parallel_mech_indexes", [])
        prepare_kp = [float(self.cfg["prepare"]["stiffness"][i]) for i in mech_indexes]
        common_kp = [float(self.cfg["common"]["stiffness"][i]) for i in mech_indexes]
        prepare_kd = [float(self.cfg["prepare"]["damping"][i]) for i in mech_indexes]
        common_kd = [float(self.cfg["common"]["damping"][i]) for i in mech_indexes]
        print(
            "[deploy-startup] "
            f"commit={self._git_commit()} "
            f"cwd={os.getcwd()} "
            f"script={os.path.abspath(__file__)} "
            f"config={os.path.abspath(cfg_file)}"
        )
        print(
            "[deploy-startup] "
            f"policy={self.cfg['policy']['policy_path']} "
            f"command_source={self.cfg['policy'].get('command_source', 'remote')} "
            f"prepare_publish=continuous "
            f"zero_hold={self.cfg['policy'].get('zero_command_hold_prepare', False)} "
            f"default_source={getattr(self.policy, 'default_dof_pos_source', 'unknown')} "
            f"deploy_action_clip={self.cfg['policy'].get('deploy_action_clip', 'default')} "
            f"deploy_action_scale={self.cfg['policy'].get('deploy_action_scale', 1.0)} "
            f"motion_ramp={self.cfg['policy'].get('motion_start_action_ramp_s', 'default')} "
            f"debug={self.cfg.get('debug', {}).get('enabled', False)}"
        )
        print(
            "[deploy-startup] "
            f"ankle_ids={mech_indexes} "
            f"mode=(prepare:{self._parallel_mech_mode('prepare')},rl:{self._parallel_mech_mode('rl')}) "
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

    def _zero_command_hold_active(self):
        if not bool(self.cfg["policy"].get("zero_command_hold_prepare", False)):
            return False
        threshold = float(self.cfg["policy"].get("zero_command_hold_threshold", 0.035))
        return self._remote_command_norm() <= threshold

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
        action_abs_max = float(np.max(np.abs(self.policy.actions))) if self.policy.actions.size else 0.0
        action_sample = [float(x) for x in self.policy.actions]
        print(
            "[deploy-debug] "
            f"cmd=({self.remoteControlService.get_vx_cmd():+.2f},"
            f"{self.remoteControlService.get_vy_cmd():+.2f},"
            f"{self.remoteControlService.get_vyaw_cmd():+.2f}) "
            f"policy_cmd=({self.policy.smoothed_commands[0]:+.2f},"
            f"{self.policy.smoothed_commands[1]:+.2f},"
            f"{self.policy.smoothed_commands[2]:+.2f}) "
            f"gait={self.policy.gait_frequency:.2f} "
            f"stage={self.control_stage} "
            f"rpy=({self.base_rpy[0]:+.3f},{self.base_rpy[1]:+.3f},{self.base_rpy[2]:+.3f}) "
            f"mech=(prepare:{self._parallel_mech_mode('prepare')},rl:{self._parallel_mech_mode('rl')}) "
            f"actual[{leg_ids}]={[round(x, 3) for x in actual]} "
            f"target[{leg_ids}]={[round(x, 3) for x in target]} "
            f"err={[round(x, 3) for x in error]} "
            f"act_max={action_abs_max:.3f} "
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

    def start_custom_mode_conditionally(self):
        print(f"{self.remoteControlService.get_custom_mode_operation_hint()}")
        while True:
            if self.remoteControlService.start_custom_mode():
                break
            time.sleep(0.1)
        start_time = time.perf_counter()
        with self.publish_lock:
            create_prepare_cmd(self.low_cmd, self.cfg)
            for i in range(self.cfg["common"]["joint_cnt"]):
                self.dof_target[i] = self.low_cmd.motor_cmd[i].q
                self.filtered_dof_target[i] = self.low_cmd.motor_cmd[i].q
            self.control_stage = "prepare"
            self._apply_parallel_mech_cmd("prepare")
            self._send_cmd(self.low_cmd)
        send_time = time.perf_counter()
        self.logger.debug(f"Send cmd took {(send_time - start_time)*1000:.4f} ms")
        self.client.ChangeMode(RobotMode.kCustom)
        with self.publish_lock:
            self._send_cmd(self.low_cmd)
        print("[deploy-prepare] custom mode active; prepare command sent")
        end_time = time.perf_counter()
        self.logger.debug(f"Change mode took {(end_time - send_time)*1000:.4f} ms")

    def start_rl_gait_conditionally(self):
        print(f"{self.remoteControlService.get_rl_gait_operation_hint()}")
        while True:
            if self.remoteControlService.start_rl_gait():
                break
            time.sleep(0.1)
        with self.publish_lock:
            self.policy.reset_runtime_state()
            current_target = np.copy(self.filtered_dof_target)
            self.rl_start_target = np.array(self.cfg["common"]["default_qpos"], dtype=np.float32)
            self._set_joint_gains("common")
            for i in range(self.cfg["common"]["joint_cnt"]):
                self.low_cmd.motor_cmd[i].q = current_target[i]
                self.dof_target[i] = current_target[i]
                self.filtered_dof_target[i] = current_target[i]
            self._apply_parallel_mech_cmd("rl")
            self._send_cmd(self.low_cmd)
            self.rl_start_time = self.timer.get_time()
            self.next_inference_time = self.rl_start_time
            self.rl_motion_start_time = None
            self.control_stage = "rl_hold"
        transition_s = max(float(self.cfg["policy"].get("rl_stance_transition_s", 1.5)), 0.0)
        transition_start = time.perf_counter()
        while transition_s > 1.0e-6:
            alpha = min((time.perf_counter() - transition_start) / transition_s, 1.0)
            target = (1.0 - alpha) * current_target + alpha * self.rl_start_target
            with self.publish_lock:
                self.dof_target[:] = target
                self.filtered_dof_target[:] = target
                for i in range(self.cfg["common"]["joint_cnt"]):
                    self.low_cmd.motor_cmd[i].q = self.filtered_dof_target[i]
                self._apply_parallel_mech_cmd("rl")
                self._send_cmd(self.low_cmd)
            if alpha >= 1.0:
                break
            time.sleep(self.cfg["common"]["dt"])
        with self.publish_lock:
            self.dof_target[:] = self.rl_start_target
            self.filtered_dof_target[:] = self.rl_start_target
            for i in range(self.cfg["common"]["joint_cnt"]):
                self.low_cmd.motor_cmd[i].q = self.filtered_dof_target[i]
            self._apply_parallel_mech_cmd("rl")
            self._send_cmd(self.low_cmd)
        self._start_publish_thread()
        print("[deploy-rl] RL gait armed; holding policy default target until movement command")
        print(f"{self.remoteControlService.get_operation_hint()}")

    def run(self):
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
                if self.control_stage == "rl":
                    self.rl_start_target = np.copy(self.filtered_dof_target)
                elif self.rl_start_target is None:
                    self.rl_start_target = np.copy(self.filtered_dof_target)
                self._set_joint_gains("common")
                self.control_stage = "rl_hold"
                self.rl_motion_start_time = None
                self.policy.reset_runtime_state()
                self.dof_target[:] = self.rl_start_target
            time.sleep(0.001)
            return

        if self.control_stage != "rl":
            with self.publish_lock:
                self.control_stage = "rl"
                self.rl_motion_start_time = time_now
                self.rl_start_target = np.copy(self.filtered_dof_target)
                self.dof_target[:] = self.rl_start_target
                self._set_joint_gains("common")
                for i in range(self.cfg["common"]["joint_cnt"]):
                    self.low_cmd.motor_cmd[i].q = self.filtered_dof_target[i]
                self._apply_parallel_mech_cmd("rl")
                self._send_cmd(self.low_cmd)
                self.policy.reset_runtime_state()

        hold_s = float(self.cfg["policy"].get("motion_start_hold_s", self.cfg["policy"].get("startup_hold_s", 0.0)))
        ramp_s = float(
            self.cfg["policy"].get("motion_start_action_ramp_s", self.cfg["policy"].get("startup_action_ramp_s", 0.0))
        )
        elapsed = 0.0 if self.rl_motion_start_time is None else max(0.0, time_now - self.rl_motion_start_time)
        if elapsed < hold_s:
            if self.rl_start_target is None:
                self.rl_start_target = np.copy(self.filtered_dof_target)
            self.dof_target[:] = self.rl_start_target
            self.policy.reset_runtime_state()
            time.sleep(0.001)
            return
        if ramp_s > 1.0e-6:
            action_scale_multiplier = min(max((elapsed - hold_s) / ramp_s, 0.0), 1.0)
        else:
            action_scale_multiplier = 1.0

        policy_target = self.policy.inference(
            time_now=time_now,
            dof_pos=self.dof_pos,
            dof_vel=self.dof_vel,
            base_ang_vel=self.base_ang_vel,
            projected_gravity=self.projected_gravity,
            vx=self.remoteControlService.get_vx_cmd(),
            vy=self.remoteControlService.get_vy_cmd(),
            vyaw=self.remoteControlService.get_vyaw_cmd(),
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
                if stage == "rl":
                    self.filtered_dof_target = self.filtered_dof_target * 0.8 + self.dof_target * 0.2
                elif stage in ("prepare", "rl_hold"):
                    self.filtered_dof_target[:] = self.dof_target

                for i in range(self.cfg["common"]["joint_cnt"]):
                    self.low_cmd.motor_cmd[i].q = self.filtered_dof_target[i]
                self._apply_parallel_mech_cmd("prepare" if stage == "prepare" else "rl")

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
    parser.add_argument("--prepare_hip_pitch", type=float, default=None)
    parser.add_argument("--prepare_knee_pitch", type=float, default=None)
    parser.add_argument("--prepare_ankle_pitch", type=float, default=None)
    parser.add_argument("--prepare_ankle_kp", type=float, default=None)
    parser.add_argument("--prepare_ankle_kd", type=float, default=None)
    parser.add_argument("--deploy_action_clip", type=float, default=None)
    parser.add_argument("--deploy_action_scale", type=float, default=None)
    parser.add_argument("--motion_start_action_ramp_s", type=float, default=None)
    args = parser.parse_args()
    cfg_file = os.path.join("configs", args.config)

    print(f"Starting custom controller, connecting to {args.net} ...")
    ChannelFactory.Instance().Init(0, args.net)

    with Controller(cfg_file) as controller:
        controller.apply_prepare_overrides(args)
        controller.print_startup_diagnostics(cfg_file)
        time.sleep(2)  # Wait for channels to initialize
        print("Initialization complete.")
        controller.start_custom_mode_conditionally()
        controller.start_rl_gait_conditionally()

        try:
            while controller.running:
                controller.run()
            controller.client.ChangeMode(RobotMode.kDamping)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Cleaning up...")
            controller.cleanup()
