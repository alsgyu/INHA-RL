import numpy as np
import torch
from utils.observation_controller import get_controller


class Policy:
    def __init__(self, cfg):
        try:
            self.cfg = cfg
            self.policy = torch.jit.load(self.cfg["policy"]["policy_path"])
            self.policy.eval()
        except Exception as e:
            print(f"Failed to load policy: {e}")
            raise
        self._init_inference_variables()
        # Initialize observation controller for live control
        self.obs_controller = get_controller()

    def get_policy_interval(self):
        return self.policy_interval

    def _init_inference_variables(self):
        default_source = str(self.cfg["policy"].get("deploy_default_qpos_source", "common")).lower()
        if default_source == "prepare":
            self.default_dof_pos = np.array(self.cfg["prepare"]["default_qpos"], dtype=np.float32)
        elif default_source == "common":
            self.default_dof_pos = np.array(self.cfg["common"]["default_qpos"], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported deploy_default_qpos_source '{default_source}'")
        self.default_dof_pos_source = default_source
        target_default_blend = self.cfg["policy"].get("deploy_target_default_blend")
        if target_default_blend is None:
            self.target_default_dof_pos = np.copy(self.default_dof_pos)
        else:
            blend = float(np.clip(float(target_default_blend), 0.0, 1.0))
            prepare_default = np.array(self.cfg["prepare"]["default_qpos"], dtype=np.float32)
            common_default = np.array(self.cfg["common"]["default_qpos"], dtype=np.float32)
            self.target_default_dof_pos = prepare_default + blend * (common_default - prepare_default)
        self.target_default_blend = target_default_blend
        self.stiffness = np.array(self.cfg["common"]["stiffness"], dtype=np.float32)
        self.damping = np.array(self.cfg["common"]["damping"], dtype=np.float32)

        self.commands = np.zeros(3, dtype=np.float32)
        self.smoothed_commands = np.zeros(3, dtype=np.float32)
        self.command_block = np.zeros(10, dtype=np.float32)

        self.command_adapter = self.cfg["policy"].get("command_adapter")
        self.command_source = str(self.cfg["policy"].get("command_source", "remote")).lower()
        self.gait_frequency = float(self.cfg["policy"]["gait_frequency"])
        self.gait_process = 0.0
        self.estimated_yaw = 0.0
        self.desired_yaw = 0.0
        self.heading_initialized = False
        self.heading_correction_yaw = 0.0
        self.dof_targets = np.copy(self.target_default_dof_pos)
        self.obs = np.zeros(self.cfg["policy"]["num_observations"], dtype=np.float32)
        self.raw_actions = np.zeros(self.cfg["policy"]["num_actions"], dtype=np.float32)
        self.actions = np.zeros(self.cfg["policy"]["num_actions"], dtype=np.float32)
        self.policy_interval = self.cfg["common"]["dt"] * self.cfg["policy"]["control"]["decimation"]
        self.action_dof_indexes = self._resolve_action_dof_indexes()
        self.leg_start_index = int(self.action_dof_indexes[0])
        self.leg_end_index = int(self.action_dof_indexes[-1]) + 1
        self._validate_action_vector("deploy_action_clip_by_index")
        self._validate_action_vector("deploy_action_scale_by_index")
        self._validate_action_vector("deploy_action_rate_limit_by_index")

    def _resolve_action_dof_indexes(self):
        num_actions = self.cfg["policy"]["num_actions"]
        joint_names = self.cfg["common"].get("joint_names")
        policy_joint_names = self.cfg["policy"].get("policy_joint_names")
        if policy_joint_names is None:
            leg_start_index = int(self.cfg["policy"].get("leg_start_index", len(self.default_dof_pos) - num_actions))
            indexes = np.arange(leg_start_index, leg_start_index + num_actions, dtype=np.int64)
            self.policy_joint_names = [f"dof_{int(i)}" for i in indexes]
            return indexes

        if len(policy_joint_names) != num_actions:
            raise ValueError(f"policy_joint_names must contain {num_actions} values, got {len(policy_joint_names)}")
        if joint_names is None or len(joint_names) != self.cfg["common"]["joint_cnt"]:
            raise ValueError("common.joint_names must be present and match common.joint_cnt when policy_joint_names is set")

        name_to_index = {name: index for index, name in enumerate(joint_names)}
        missing = [name for name in policy_joint_names if name not in name_to_index]
        if missing:
            raise ValueError(f"policy_joint_names contains joints not found in common.joint_names: {missing}")
        self.policy_joint_names = list(policy_joint_names)
        return np.asarray([name_to_index[name] for name in policy_joint_names], dtype=np.int64)

    def _validate_action_vector(self, key):
        values = self.cfg["policy"].get(key)
        if values is not None and len(values) != self.cfg["policy"]["num_actions"]:
            raise ValueError(
                f"{key} must contain {self.cfg['policy']['num_actions']} values, got {len(values)}"
            )

    def _use_observation_controller_commands(self):
        return self.command_source in ("observation_controller", "obs_controller", "live")

    def reset_runtime_state(self):
        self.commands[:] = 0.0
        self.smoothed_commands[:] = 0.0
        self.command_block[:] = 0.0
        self.gait_frequency = 0.0
        self.gait_process = 0.0
        self.estimated_yaw = 0.0
        self.desired_yaw = 0.0
        self.heading_initialized = False
        self.heading_correction_yaw = 0.0
        self.raw_actions[:] = 0.0
        self.actions[:] = 0.0
        self.dof_targets[:] = self.target_default_dof_pos

    def _adapter_value(self, key, default):
        if self.command_adapter is None:
            return default
        return self.command_adapter.get(key, default)

    @staticmethod
    def _wrap_to_pi(angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def _heading_correction(self, moving):
        adapter = self.command_adapter
        if adapter is None or not bool(adapter.get("heading_correction_enabled", False)):
            return 0.0

        if not moving:
            self.desired_yaw = self.estimated_yaw
            self.heading_initialized = False
            self.heading_correction_yaw = 0.0
            return 0.0

        if not self.heading_initialized:
            self.desired_yaw = self.estimated_yaw
            self.heading_initialized = True
        else:
            self.desired_yaw = self._wrap_to_pi(self.desired_yaw + float(self.smoothed_commands[2]) * self.policy_interval)

        min_vx = float(adapter.get("heading_correction_min_abs_vx", 0.08))
        max_abs_vy = float(adapter.get("heading_correction_max_abs_vy_command", 0.04))
        max_abs_yaw = float(adapter.get("heading_correction_max_abs_yaw_command", 0.08))
        straight = (
            abs(float(self.smoothed_commands[0])) > min_vx
            and abs(float(self.smoothed_commands[1])) < max_abs_vy
            and abs(float(self.smoothed_commands[2])) < max_abs_yaw
        )
        if not straight:
            self.heading_correction_yaw = 0.0
            return 0.0

        yaw_error = self._wrap_to_pi(self.estimated_yaw - self.desired_yaw)
        deadband = float(adapter.get("heading_correction_deadband", 0.015))
        yaw_error = np.sign(yaw_error) * max(abs(yaw_error) - deadband, 0.0)
        correction = -float(adapter.get("heading_correction_gain", 1.0)) * yaw_error
        max_correction = float(adapter.get("heading_correction_max_yaw_rate", 0.35))
        self.heading_correction_yaw = float(np.clip(correction, -max_correction, max_correction))
        return self.heading_correction_yaw

    def _resolve_command_block(self):
        adapter = self.command_adapter
        if adapter is None:
            moving = np.linalg.norm(self.smoothed_commands) > 1.0e-5
            self.gait_frequency = float(self.obs_controller.get_value(9)) if moving else 0.0
            return np.array(
                [
                    self.smoothed_commands[0],
                    self.smoothed_commands[1],
                    self.smoothed_commands[2],
                    self.gait_frequency,
                    self.obs_controller.get_value(10),
                    self.obs_controller.get_value(11),
                    self.obs_controller.get_value(12),
                    self.obs_controller.get_value(13),
                    self.obs_controller.get_value(14),
                    self.obs_controller.get_value(15),
                ],
                dtype=np.float32,
            )

        vx, vy, yaw = self.smoothed_commands
        stand_threshold = float(adapter.get("stand_command_threshold", 0.04))
        moving = np.sqrt(vx * vx + vy * vy + yaw * yaw) > stand_threshold
        internal_yaw = yaw + self._heading_correction(moving)

        linear_speed = np.sqrt(vx * vx + vy * vy)
        speed_min = float(adapter.get("gait_frequency_speed_min", 0.08))
        speed_max = max(float(adapter.get("gait_frequency_speed_max", 1.0)), speed_min + 1.0e-6)
        max_yaw = max(float(adapter.get("max_yaw_speed_for_drive", 1.0)), 1.0e-6)
        linear_drive = np.clip((linear_speed - speed_min) / (speed_max - speed_min), 0.0, 1.0)
        yaw_drive = np.clip(abs(internal_yaw) / max_yaw, 0.0, 1.0)
        drive = max(linear_drive, yaw_drive)

        if moving:
            gait_min = float(adapter.get("gait_frequency_min", 1.15))
            gait_max = float(adapter.get("gait_frequency_max", 1.95))
            self.gait_frequency = gait_min + drive * (gait_max - gait_min)
        else:
            self.gait_frequency = 0.0

        foot_clip = adapter.get("foot_yaw_target_clip", [-0.25, 0.25])
        foot_yaw = np.clip(
            internal_yaw * float(adapter.get("foot_yaw_from_yaw_gain", 0.12)),
            float(foot_clip[0]),
            float(foot_clip[1]),
        )
        pitch_clip = adapter.get("body_pitch_target_clip", [-0.04, 0.12])
        body_pitch = np.clip(
            vx * float(adapter.get("body_pitch_gain", 0.08)),
            float(pitch_clip[0]),
            float(pitch_clip[1]),
        )
        roll_clip = adapter.get("body_roll_target_clip", [-0.08, 0.08])
        body_roll = np.clip(
            vy * float(adapter.get("body_roll_gain", -0.08)),
            float(roll_clip[0]),
            float(roll_clip[1]),
        )
        if not moving:
            body_pitch = 0.0
            body_roll = 0.0
            foot_yaw = 0.0

        return np.array(
            [
                vx,
                vy,
                yaw,
                self.gait_frequency,
                foot_yaw,
                foot_yaw,
                body_pitch,
                body_roll,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

    def inference(
        self,
        time_now,
        dof_pos,
        dof_vel,
        base_ang_vel,
        projected_gravity,
        vx,
        vy,
        vyaw,
        action_scale_multiplier=1.0,
    ):
        self.estimated_yaw = self._wrap_to_pi(self.estimated_yaw + float(base_ang_vel[2]) * self.policy_interval)
        if self._use_observation_controller_commands():
            self.commands[0] = self.obs_controller.get_vx_cmd()
            self.commands[1] = self.obs_controller.get_vy_cmd()
            self.commands[2] = self.obs_controller.get_vyaw_cmd()
        else:
            self.commands[0] = vx
            self.commands[1] = vy
            self.commands[2] = vyaw
            
        command_slew_rate = float(self.cfg["policy"].get("command_slew_rate", 1.0))
        clip_delta = self.policy_interval * command_slew_rate
        clip_range = (-clip_delta, clip_delta)
        self.smoothed_commands += np.clip(self.commands - self.smoothed_commands, *clip_range)
        command_block = self._resolve_command_block()
        self.command_block[:] = command_block
        moving = self.gait_frequency > 1.0e-8
        self.gait_process = np.fmod(self.gait_process + self.policy_interval * self.gait_frequency, 1.0) if moving else 0.0

        norm = self.cfg["policy"]["normalization"]
        command_scale = np.array(
            [
                norm["lin_vel"],
                norm["lin_vel"],
                norm["ang_vel"],
                norm["gait_frequency"],
                norm["foot_yaw"],
                norm["foot_yaw"],
                norm["body_pitch_target"],
                norm["body_roll_target"],
                norm["feet_offset_x_target"],
                norm["feet_offset_y_target"],
            ],
            dtype=np.float32,
        )

        self.obs[0:3] = projected_gravity * norm["gravity"]
        self.obs[3:6] = base_ang_vel * norm["ang_vel"]
        self.obs[6:16] = command_block * command_scale
        self.obs[16] = np.cos(2 * np.pi * self.gait_process) * moving
        self.obs[17] = np.sin(2 * np.pi * self.gait_process) * moving
        self.obs[18:30] = (
            dof_pos[self.action_dof_indexes]
            - self.default_dof_pos[self.action_dof_indexes]
        ) * norm["dof_pos"]
        self.obs[30:42] = dof_vel[self.action_dof_indexes] * norm["dof_vel"]
        self.obs[42:54] = self.actions

        with torch.no_grad():
            output = self.policy(torch.from_numpy(self.obs).unsqueeze(0)).detach().numpy()[0]
        self.raw_actions[:] = output[: self.cfg["policy"]["num_actions"]]
        deploy_clip = float(self.cfg["policy"].get("deploy_action_clip", norm["clip_actions"]))
        deploy_clip = np.full(self.cfg["policy"]["num_actions"], deploy_clip, dtype=np.float32)
        deploy_clip_by_index = self.cfg["policy"].get("deploy_action_clip_by_index")
        if deploy_clip_by_index is not None:
            deploy_clip = np.minimum(
                deploy_clip,
                np.asarray(deploy_clip_by_index, dtype=np.float32),
            )
        desired_actions = np.clip(
            self.raw_actions,
            -deploy_clip,
            deploy_clip,
        )
        desired_actions *= float(self.cfg["policy"].get("deploy_action_scale", 1.0))
        deploy_scale_by_index = self.cfg["policy"].get("deploy_action_scale_by_index")
        if deploy_scale_by_index is not None:
            desired_actions *= np.asarray(deploy_scale_by_index, dtype=np.float32)
        if bool(self.cfg["policy"].get("deploy_scale_actions_in_policy", False)):
            desired_actions *= float(np.clip(action_scale_multiplier, 0.0, 1.0))
        action_rate_limit = self.cfg["policy"].get("deploy_action_rate_limit")
        action_rate_limit_by_index = self.cfg["policy"].get("deploy_action_rate_limit_by_index")
        if action_rate_limit_by_index is not None:
            max_delta = np.asarray(action_rate_limit_by_index, dtype=np.float32) * self.policy_interval
            self.actions[:] += np.clip(desired_actions - self.actions, -max_delta, max_delta)
        elif action_rate_limit is None or float(action_rate_limit) <= 0.0:
            self.actions[:] = desired_actions
        else:
            max_delta = float(action_rate_limit) * self.policy_interval
            self.actions[:] += np.clip(desired_actions - self.actions, -max_delta, max_delta)
        self.dof_targets[:] = self.target_default_dof_pos
        self.dof_targets[self.action_dof_indexes] += self.cfg["policy"]["control"]["action_scale"] * self.actions

        return self.dof_targets
