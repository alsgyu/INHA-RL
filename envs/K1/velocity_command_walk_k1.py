import torch

from isaacgym.torch_utils import torch_rand_float

from envs.K1.parameter_walk_k1 import ParameterWalkK1
from utils.utils import apply_randomization


class VelocityCommandWalkK1(ParameterWalkK1):
    """
    K1 velocity walking task with three public commands:
    lin_vel_x, lin_vel_y, and ang_vel_yaw.

    The policy observation remains compatible with the 54-dim ParameterWalkK1
    actor by resolving the three public commands into the existing 10-command
    internal block. Zero command is treated as an explicit stand command.
    """

    PUBLIC_COMMAND_KEYS = ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")

    def _init_buffers(self):
        super()._init_buffers()
        self.current_public_command_ranges = {
            key: list(self.cfg["commands"][key])
            for key in self.PUBLIC_COMMAND_KEYS
        }
        self.current_adapter_config = dict(self.cfg["commands"].get("adapter", {}))
        self.current_still_proportion = float(self.cfg["commands"].get("still_proportion", 0.0))
        self.current_straight_proportion = float(self.cfg["commands"].get("straight_proportion", 0.0))

    def update_training_curriculum(self, iteration):
        command_cfg = self.cfg["commands"]
        self.current_public_command_ranges = {
            key: list(command_cfg[key])
            for key in self.PUBLIC_COMMAND_KEYS
        }
        self.current_adapter_config = dict(command_cfg.get("adapter", {}))
        self.current_still_proportion = float(command_cfg.get("still_proportion", 0.0))
        self.current_straight_proportion = float(command_cfg.get("straight_proportion", 0.0))
        self.current_resampling_time = list(command_cfg.get("resampling_time_s", [3.0, 8.0]))
        self.current_disturbance_scale = 1.0
        self.training_phase_index = 0
        self.training_phase_progress = 1.0

        if command_cfg.get("training_mode", "sampled") == "fixed":
            return

        phases = command_cfg.get("training_phases", [])
        if not phases:
            return

        active_idx = 0
        for i, phase in enumerate(phases):
            if iteration >= int(phase["start_iteration"]):
                active_idx = i
        phase = phases[active_idx]

        for key in self.PUBLIC_COMMAND_KEYS:
            if key in phase:
                self.current_public_command_ranges[key] = list(phase[key])
        if "adapter" in phase:
            self.current_adapter_config.update(phase["adapter"])
        self.current_still_proportion = float(phase.get("still_proportion", self.current_still_proportion))
        self.current_straight_proportion = float(phase.get("straight_proportion", self.current_straight_proportion))
        self.current_resampling_time = list(phase.get("resampling_time_s", self.current_resampling_time))
        self.current_disturbance_scale = float(phase.get("disturbance_scale", self.current_disturbance_scale))
        self.training_phase_index = active_idx
        if active_idx < len(phases) - 1:
            next_iter = int(phases[active_idx + 1]["start_iteration"])
            span = max(next_iter - int(phase["start_iteration"]), 1)
            self.training_phase_progress = float(
                max(0.0, min(1.0, (iteration - int(phase["start_iteration"])) / span))
            )

    def _public_command_range(self, key):
        return self.current_public_command_ranges.get(key, self.cfg["commands"][key])

    def _adapter_value(self, key, default):
        return float(self.current_adapter_config.get(key, default))

    def _stand_mask(self):
        threshold = self._adapter_value("stand_command_threshold", 0.04)
        command_norm = torch.sqrt(torch.sum(torch.square(self.commands[:, 0:2]), dim=-1) + torch.square(self.commands[:, 2]))
        return (command_norm <= threshold).float()

    def _moving_mask(self):
        return 1.0 - self._stand_mask()

    def _resolve_internal_commands(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return

        adapter = self.current_adapter_config
        vx = self.commands[env_ids, 0]
        vy = self.commands[env_ids, 1]
        yaw = self.commands[env_ids, 2]
        linear_speed = torch.sqrt(torch.square(vx) + torch.square(vy))
        yaw_speed = torch.abs(yaw)

        speed_min = float(adapter.get("gait_frequency_speed_min", 0.08))
        speed_max = max(float(adapter.get("gait_frequency_speed_max", 1.0)), speed_min + 1.0e-6)
        max_yaw = max(float(adapter.get("max_yaw_speed_for_drive", self.cfg["commands"]["ang_vel_yaw"][1])), 1.0e-6)
        linear_drive = torch.clamp((linear_speed - speed_min) / (speed_max - speed_min), min=0.0, max=1.0)
        yaw_drive = torch.clamp(yaw_speed / max_yaw, min=0.0, max=1.0)
        drive = torch.maximum(linear_drive, yaw_drive)

        stand_threshold = float(adapter.get("stand_command_threshold", 0.04))
        command_norm = torch.sqrt(torch.square(vx) + torch.square(vy) + torch.square(yaw))
        moving = command_norm > stand_threshold

        gait_min = float(adapter.get("gait_frequency_min", 1.15))
        gait_max = float(adapter.get("gait_frequency_max", 1.95))
        gait_frequency = gait_min + drive * (gait_max - gait_min)
        gait_frequency = torch.where(moving, gait_frequency, torch.zeros_like(gait_frequency))

        foot_yaw_gain = float(adapter.get("foot_yaw_from_yaw_gain", 0.12))
        foot_yaw_clip = adapter.get("foot_yaw_target_clip", [-0.25, 0.25])
        foot_yaw = torch.clamp(
            yaw * foot_yaw_gain,
            min=float(foot_yaw_clip[0]),
            max=float(foot_yaw_clip[1]),
        )

        pitch_gain = float(adapter.get("body_pitch_gain", 0.08))
        pitch_clip = adapter.get("body_pitch_target_clip", [-0.04, 0.12])
        body_pitch = torch.clamp(
            vx * pitch_gain,
            min=float(pitch_clip[0]),
            max=float(pitch_clip[1]),
        )

        roll_gain = float(adapter.get("body_roll_gain", -0.08))
        roll_clip = adapter.get("body_roll_target_clip", [-0.08, 0.08])
        body_roll = torch.clamp(
            vy * roll_gain,
            min=float(roll_clip[0]),
            max=float(roll_clip[1]),
        )

        self.commands[env_ids, 3] = gait_frequency
        self.commands[env_ids, 4] = foot_yaw
        self.commands[env_ids, 5] = foot_yaw
        self.commands[env_ids, 6] = torch.where(moving, body_pitch, torch.zeros_like(body_pitch))
        self.commands[env_ids, 7] = torch.where(moving, body_roll, torch.zeros_like(body_roll))
        self.commands[env_ids, 8] = 0.0
        self.commands[env_ids, 9] = 0.0
        self.gait_frequency[env_ids] = gait_frequency
        self.gait_process[env_ids] = torch.where(
            moving,
            self.gait_process[env_ids],
            torch.zeros_like(self.gait_process[env_ids]),
        )

    def _apply_fixed_command(self, command_cfg, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return

        defaults = {"lin_vel_x": 0.0, "lin_vel_y": 0.0, "ang_vel_yaw": 0.0}
        for command_idx, key in enumerate(self.PUBLIC_COMMAND_KEYS):
            self.commands[env_ids, command_idx] = float(command_cfg.get(key, defaults[key]))
        self.commands[env_ids, 3:10] = 0.0
        self._resolve_internal_commands(env_ids)

    def _sample_public_commands(self, env_ids):
        for command_idx, key in enumerate(self.PUBLIC_COMMAND_KEYS):
            low, high = self._public_command_range(key)
            self.commands[env_ids, command_idx] = torch_rand_float(
                float(low),
                float(high),
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

    def _apply_straight_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        straight_cfg = self.cfg["commands"].get("straight", {})
        for command_idx, key in enumerate(self.PUBLIC_COMMAND_KEYS):
            value = straight_cfg.get(key, self.cfg["commands"].get("fixed", {}).get(key, 0.0))
            if isinstance(value, (list, tuple)) and len(value) == 2:
                self.commands[env_ids, command_idx] = torch_rand_float(
                    float(value[0]),
                    float(value[1]),
                    (len(env_ids), 1),
                    device=self.device,
                ).squeeze(1)
            else:
                self.commands[env_ids, command_idx] = float(value)

    def _resample_commands(self):
        if getattr(self, "is_play", False):
            self._apply_fixed_command(self.cfg["commands"].get("play", {}))
            return
        if getattr(self, "manual_control", False):
            self._resolve_internal_commands()
            return

        env_ids = (self.episode_length_buf == self.cmd_resample_time).nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return

        if self.cfg["commands"].get("training_mode", "sampled") == "fixed":
            self._apply_fixed_command(self.cfg["commands"].get("fixed", {}), env_ids)
            self.cmd_resample_time[env_ids] += self._sample_resample_steps(len(env_ids))
            return

        self.commands[env_ids, :10] = 0.0
        self._sample_public_commands(env_ids)

        perm = torch.randperm(len(env_ids), device=self.device)
        still_count = int(self.current_still_proportion * len(env_ids))
        still_envs = env_ids[perm[:still_count]]
        self.commands[still_envs, :10] = 0.0
        remaining_envs = env_ids[perm[still_count:]]
        straight_count = int(self.current_straight_proportion * len(remaining_envs))
        straight_envs = remaining_envs[:straight_count]
        self._apply_straight_commands(straight_envs)

        self._resolve_internal_commands(env_ids)
        self.cmd_resample_time[env_ids] += self._sample_resample_steps(len(env_ids))

    def _compute_observations(self):
        self._resolve_internal_commands()
        commands_scale = torch.tensor(
            [
                self.cfg["normalization"]["lin_vel"],
                self.cfg["normalization"]["lin_vel"],
                self.cfg["normalization"]["ang_vel"],
                self.cfg["normalization"]["gait_frequency"],
                self.cfg["normalization"]["foot_yaw"],
                self.cfg["normalization"]["foot_yaw"],
                self.cfg["normalization"]["body_pitch_target"],
                self.cfg["normalization"]["body_roll_target"],
                self.cfg["normalization"]["feet_offset_x_target"],
                self.cfg["normalization"]["feet_offset_y_target"],
            ],
            device=self.device,
        )
        moving = (self.gait_frequency > 1.0e-8).float()
        self.obs_buf = torch.cat(
            (
                apply_randomization(self.projected_gravity, self.cfg["noise"].get("gravity")) * self.cfg["normalization"]["gravity"],
                apply_randomization(self.base_ang_vel, self.cfg["noise"].get("ang_vel")) * self.cfg["normalization"]["ang_vel"],
                self.commands[:, :10] * commands_scale,
                (torch.cos(2 * torch.pi * self.gait_process) * moving).unsqueeze(-1),
                (torch.sin(2 * torch.pi * self.gait_process) * moving).unsqueeze(-1),
                apply_randomization(self.dof_pos - self.default_dof_pos, self.cfg["noise"].get("dof_pos")) * self.cfg["normalization"]["dof_pos"],
                apply_randomization(self.dof_vel, self.cfg["noise"].get("dof_vel")) * self.cfg["normalization"]["dof_vel"],
                self.actions,
            ),
            dim=-1,
        )

        if hasattr(self, "obs_csv_writer"):
            obs_env0 = self.obs_buf[0].cpu().numpy()
            self.obs_csv_writer.writerow(obs_env0)
            self.obs_csv_file.flush()

        if self.cfg.get("basic", {}).get("debug_tensors", False):
            print(self.obs_buf)
        self.privileged_obs_buf = torch.cat(
            (
                self.base_mass_scaled,
                apply_randomization(self.base_lin_vel, self.cfg["noise"].get("lin_vel")) * self.cfg["normalization"]["lin_vel"],
                apply_randomization(self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos), self.cfg["noise"].get("height")).unsqueeze(-1),
                self.pushing_forces[:, 0, :] * self.cfg["normalization"]["push_force"],
                self.pushing_torques[:, 0, :] * self.cfg["normalization"]["push_torque"],
            ),
            dim=-1,
        )
        self.extras["privileged_obs"] = self.privileged_obs_buf
        self.extras["sirl"] = self._compute_sirl_info()
        self.extras["sirl_commands"] = self.commands[:, :3].detach()
        self.extras["sirl_internal_commands"] = self.commands[:, :10].detach()

    def _swing_masks(self):
        left_swing = (torch.abs(self.gait_process - 0.25) < 0.5 * self.cfg["rewards"]["swing_period"]) & (
            self.gait_frequency > 1.0e-8
        )
        right_swing = (torch.abs(self.gait_process - 0.75) < 0.5 * self.cfg["rewards"]["swing_period"]) & (
            self.gait_frequency > 1.0e-8
        )
        return left_swing, right_swing

    def _foot_clearance(self):
        flat_feet_pos = self.feet_pos.reshape(-1, 3)
        ground_height = self.terrain.terrain_heights(flat_feet_pos).reshape(self.num_envs, len(self.feet_indices))
        return self.feet_pos[:, :, 2] - ground_height

    def _compute_sirl_info(self):
        velocity_error = (
            torch.abs(self.commands[:, 0] - self.filtered_lin_vel[:, 0])
            + 0.75 * torch.abs(self.commands[:, 1] - self.filtered_lin_vel[:, 1])
            + 0.50 * torch.abs(self.commands[:, 2] - self.filtered_ang_vel[:, 2])
        )
        feet_vel_xy = torch.norm((self.last_feet_pos[:, :, :2] - self.feet_pos[:, :, :2]) / self.dt, dim=-1)
        stance_count = torch.clamp(self.feet_contact.float().sum(dim=-1), min=1.0)
        feet_slip = torch.sum(feet_vel_xy * self.feet_contact.float(), dim=-1) / stance_count
        feet_slip = feet_slip * (self.episode_length_buf > 1).float()

        left_swing, right_swing = self._swing_masks()
        swing_mask = torch.stack((left_swing, right_swing), dim=-1).float()
        swing_count = torch.clamp(swing_mask.sum(dim=-1), min=1.0)
        clearance = self._foot_clearance()
        clearance_target = float(self.cfg["rewards"].get("swing_clearance_target", 0.10))
        swing_clearance_deficit = torch.sum(torch.clamp(clearance_target - clearance, min=0.0) * swing_mask, dim=-1) / swing_count

        base_tilt = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=-1)
        base_height = self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos)
        low_height_target = float(self.cfg["rewards"].get("base_height_target", 0.52)) - 0.04
        low_height = torch.clamp(low_height_target - base_height, min=0.0)
        stand_mask = self._stand_mask()
        stand_drift = (
            torch.sum(torch.square(self.filtered_lin_vel[:, :2]), dim=-1)
            + 0.5 * torch.square(self.filtered_ang_vel[:, 2])
        ) * stand_mask
        return {
            "velocity_error": velocity_error,
            "feet_slip": feet_slip,
            "swing_clearance_deficit": swing_clearance_deficit,
            "base_tilt": base_tilt,
            "low_height": low_height,
            "stand_drift": stand_drift,
        }

    def _reward_lin_vel_y_error(self):
        moving = (torch.abs(self.commands[:, 1]) > 0.03).float()
        return torch.square(self.commands[:, 1] - self.filtered_lin_vel[:, 1]) * moving

    def _reward_ang_vel_yaw_error(self):
        moving = (torch.abs(self.commands[:, 2]) > 0.05).float()
        return torch.square(self.commands[:, 2] - self.filtered_ang_vel[:, 2]) * moving

    def _reward_swing_clearance(self):
        left_swing, right_swing = self._swing_masks()
        swing_mask = torch.stack((left_swing, right_swing), dim=-1).float()
        swing_count = torch.clamp(swing_mask.sum(dim=-1), min=1.0)
        clearance = self._foot_clearance()
        target = float(self.cfg["rewards"].get("swing_clearance_target", 0.10))
        sigma = max(float(self.cfg["rewards"].get("swing_clearance_sigma", 0.015)), 1.0e-6)
        clearance_reward = torch.exp(-torch.square(clearance - target) / sigma)
        return torch.sum(clearance_reward * swing_mask, dim=-1) / swing_count

    def _reward_scuff_clearance(self):
        left_swing, right_swing = self._swing_masks()
        swing_mask = torch.stack((left_swing, right_swing), dim=-1).float()
        swing_count = torch.clamp(swing_mask.sum(dim=-1), min=1.0)
        clearance = self._foot_clearance()
        threshold = float(self.cfg["rewards"].get("scuff_clearance_threshold", 0.045))
        return torch.sum(torch.clamp(threshold - clearance, min=0.0) * swing_mask, dim=-1) / swing_count

    def _reward_command_stand_still(self):
        stand_mask = self._stand_mask()
        lin_xy = torch.sum(torch.square(self.filtered_lin_vel[:, :2]), dim=-1)
        ang_z = torch.square(self.filtered_ang_vel[:, 2])
        return (lin_xy + 0.5 * ang_z) * stand_mask

    def _reward_command_stand_action(self):
        return torch.sum(torch.square(self.actions), dim=-1) * self._stand_mask()

    def _reward_command_stand_default_pose(self):
        return torch.mean(torch.square(self.dof_pos - self.default_dof_pos), dim=-1) * self._stand_mask()

    def _reward_command_stand_feet_slip(self):
        feet_vel = (self.last_feet_pos - self.feet_pos) / self.dt
        feet_xy_vel = torch.sum(torch.square(feet_vel[:, :, :2]), dim=-1)
        return torch.sum(feet_xy_vel * self.feet_contact.float(), dim=-1) * self._stand_mask()
