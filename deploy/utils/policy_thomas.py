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
        moving = np.linalg.norm(self.smoothed_commands) > 1.0e-5
        self.gait_frequency = float(self.obs_controller.get_value(9)) if moving else 0.0
        self.gait_process = np.fmod(self.gait_process + self.policy_interval * self.gait_frequency, 1.0)

        norm = self.cfg["policy"]["normalization"]
        command_block = np.array(
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
