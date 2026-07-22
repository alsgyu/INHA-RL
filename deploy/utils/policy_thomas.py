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
        self.default_dof_pos = np.array(self.cfg["common"]["default_qpos"], dtype=np.float32)
        self.stiffness = np.array(self.cfg["common"]["stiffness"], dtype=np.float32)
        self.damping = np.array(self.cfg["common"]["damping"], dtype=np.float32)

        self.commands = np.zeros(3, dtype=np.float32)
        self.smoothed_commands = np.zeros(3, dtype=np.float32)

        self.command_adapter = self.cfg["policy"].get("command_adapter")
        self.gait_frequency = float(self.cfg["policy"]["gait_frequency"])
        self.gait_process = 0.0
        self.dof_targets = np.copy(self.default_dof_pos)
        self.obs = np.zeros(self.cfg["policy"]["num_observations"], dtype=np.float32)
        self.actions = np.zeros(self.cfg["policy"]["num_actions"], dtype=np.float32)
        self.policy_interval = self.cfg["common"]["dt"] * self.cfg["policy"]["control"]["decimation"]
        self.leg_start_index = int(
            self.cfg["policy"].get("leg_start_index", len(self.default_dof_pos) - self.cfg["policy"]["num_actions"])
        )
        self.leg_end_index = self.leg_start_index + self.cfg["policy"]["num_actions"]

    def _adapter_value(self, key, default):
        if self.command_adapter is None:
            return default
        return self.command_adapter.get(key, default)

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

        linear_speed = np.sqrt(vx * vx + vy * vy)
        speed_min = float(adapter.get("gait_frequency_speed_min", 0.08))
        speed_max = max(float(adapter.get("gait_frequency_speed_max", 1.0)), speed_min + 1.0e-6)
        max_yaw = max(float(adapter.get("max_yaw_speed_for_drive", 1.0)), 1.0e-6)
        linear_drive = np.clip((linear_speed - speed_min) / (speed_max - speed_min), 0.0, 1.0)
        yaw_drive = np.clip(abs(yaw) / max_yaw, 0.0, 1.0)
        drive = max(linear_drive, yaw_drive)

        if moving:
            gait_min = float(adapter.get("gait_frequency_min", 1.15))
            gait_max = float(adapter.get("gait_frequency_max", 1.95))
            self.gait_frequency = gait_min + drive * (gait_max - gait_min)
        else:
            self.gait_frequency = 0.0

        foot_clip = adapter.get("foot_yaw_target_clip", [-0.25, 0.25])
        foot_yaw = np.clip(
            yaw * float(adapter.get("foot_yaw_from_yaw_gain", 0.12)),
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

    def inference(self, time_now, dof_pos, dof_vel, base_ang_vel, projected_gravity, vx, vy, vyaw):
        try:
            self.commands[0] = self.obs_controller.get_vx_cmd()
            self.commands[1] = self.obs_controller.get_vy_cmd()
            self.commands[2] = self.obs_controller.get_vyaw_cmd()
        except Exception:
            self.commands[0] = vx
            self.commands[1] = vy
            self.commands[2] = vyaw
            
        clip_range = (-self.policy_interval, self.policy_interval)
        self.smoothed_commands += np.clip(self.commands - self.smoothed_commands, *clip_range)
        command_block = self._resolve_command_block()
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
            dof_pos[self.leg_start_index : self.leg_end_index]
            - self.default_dof_pos[self.leg_start_index : self.leg_end_index]
        ) * norm["dof_pos"]
        self.obs[30:42] = dof_vel[self.leg_start_index : self.leg_end_index] * norm["dof_vel"]
        self.obs[42:54] = self.actions

        with torch.no_grad():
            output = self.policy(torch.from_numpy(self.obs).unsqueeze(0)).detach().numpy()[0]
        self.actions[:] = output[: self.cfg["policy"]["num_actions"]]
        self.actions[:] = np.clip(
            self.actions,
            -norm["clip_actions"],
            norm["clip_actions"],
        )
        self.dof_targets[:] = self.default_dof_pos
        self.dof_targets[self.leg_start_index : self.leg_end_index] += self.cfg["policy"]["control"]["action_scale"] * self.actions

        return self.dof_targets
