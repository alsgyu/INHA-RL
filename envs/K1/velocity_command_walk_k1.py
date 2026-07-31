import torch

from isaacgym.torch_utils import get_euler_xyz, torch_rand_float

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
        self.public_command_targets = torch.zeros(
            self.num_envs,
            len(self.PUBLIC_COMMAND_KEYS),
            dtype=torch.float,
            device=self.device,
        )
        self.current_public_command_ranges = {
            key: list(self.cfg["commands"][key])
            for key in self.PUBLIC_COMMAND_KEYS
        }
        self.current_adapter_config = dict(self.cfg["commands"].get("adapter", {}))
        self.current_straight_config = dict(self.cfg["commands"].get("straight", {}))
        self.current_high_speed_proportion = float(self.cfg["commands"].get("high_speed_proportion", 0.0))
        self.current_high_speed_config = dict(self.cfg["commands"].get("high_speed", {}))
        self.current_still_proportion = float(self.cfg["commands"].get("still_proportion", 0.0))
        self.current_straight_proportion = float(self.cfg["commands"].get("straight_proportion", 0.0))
        self.current_command_slew_rate = float(self.cfg["commands"].get("command_slew_rate", 0.0))
        self.public_command_age = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    def update_training_curriculum(self, iteration):
        command_cfg = self.cfg["commands"]
        self.current_public_command_ranges = {
            key: list(command_cfg[key])
            for key in self.PUBLIC_COMMAND_KEYS
        }
        self.current_adapter_config = dict(command_cfg.get("adapter", {}))
        self.current_straight_config = dict(command_cfg.get("straight", {}))
        self.current_high_speed_proportion = float(command_cfg.get("high_speed_proportion", 0.0))
        self.current_high_speed_config = dict(command_cfg.get("high_speed", {}))
        self.current_still_proportion = float(command_cfg.get("still_proportion", 0.0))
        self.current_straight_proportion = float(command_cfg.get("straight_proportion", 0.0))
        self.current_command_slew_rate = float(command_cfg.get("command_slew_rate", self.current_command_slew_rate))
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
                self.current_straight_config[key] = phase[key]
                if key in self.current_high_speed_config:
                    self.current_high_speed_config[key] = phase[key]
        if "adapter" in phase:
            self.current_adapter_config.update(phase["adapter"])
        if "straight" in phase:
            self.current_straight_config.update(phase["straight"])
        if "high_speed_proportion" in phase:
            self.current_high_speed_proportion = float(phase["high_speed_proportion"])
        if "high_speed" in phase:
            self.current_high_speed_config.update(phase["high_speed"])
        self.current_command_slew_rate = float(phase.get("command_slew_rate", self.current_command_slew_rate))
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

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        self.public_command_targets[env_ids, :] = 0.0
        self.public_command_age[env_ids] = 0.0
        self.commands[env_ids, :10] = 0.0
        self.gait_frequency[env_ids] = 0.0
        self.gait_process[env_ids] = 0.0

    def _set_public_command_targets(self, env_ids, values):
        if values.dim() == 1:
            values = values.unsqueeze(0)
        if env_ids is None:
            if values.shape[0] == 1:
                values = values.expand(self.num_envs, -1)
            changed = torch.norm(values - self.public_command_targets, dim=-1) > float(
                self.cfg["commands"].get("command_change_threshold", 1.0e-4)
            )
            self.public_command_targets[:, :] = values
            self.public_command_age[changed] = 0.0
        else:
            if values.shape[0] == 1:
                values = values.expand(len(env_ids), -1)
            changed = torch.norm(values - self.public_command_targets[env_ids], dim=-1) > float(
                self.cfg["commands"].get("command_change_threshold", 1.0e-4)
            )
            self.public_command_targets[env_ids, :] = values
            self.public_command_age[env_ids[changed]] = 0.0

    def _apply_public_command_slew(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return

        targets = self.public_command_targets[env_ids]
        current = self.commands[env_ids, : len(self.PUBLIC_COMMAND_KEYS)]
        slew_rate = float(self.current_command_slew_rate)
        if slew_rate <= 0.0:
            next_command = targets
        else:
            max_delta = max(slew_rate * self.dt, 1.0e-8)
            next_command = current + torch.clamp(targets - current, min=-max_delta, max=max_delta)
        self.commands[env_ids, : len(self.PUBLIC_COMMAND_KEYS)] = next_command

    def _stand_mask(self):
        threshold = self._adapter_value("stand_command_threshold", 0.04)
        command_norm = torch.sqrt(torch.sum(torch.square(self.commands[:, 0:2]), dim=-1) + torch.square(self.commands[:, 2]))
        return (command_norm <= threshold).float()

    def _moving_mask(self):
        return 1.0 - self._stand_mask()

    def _public_command_norm(self, values):
        return torch.sqrt(torch.sum(torch.square(values[:, 0:2]), dim=-1) + torch.square(values[:, 2]))

    def _stop_transition_mask(self):
        threshold = self._adapter_value("stand_command_threshold", 0.04)
        target_norm = self._public_command_norm(self.public_command_targets)
        command_norm = self._public_command_norm(self.commands[:, : len(self.PUBLIC_COMMAND_KEYS)])
        window_s = float(self.cfg["rewards"].get("stop_transition_window_s", 1.4))
        min_command_norm = float(self.cfg["rewards"].get("stop_transition_min_command_norm", threshold))
        return (
            (target_norm <= threshold)
            & (command_norm > min_command_norm)
            & (self.public_command_age <= window_s)
        ).float()

    def _move_start_transition_mask(self):
        threshold = self._adapter_value("stand_command_threshold", 0.04)
        target_norm = self._public_command_norm(self.public_command_targets)
        command_norm = self._public_command_norm(self.commands[:, : len(self.PUBLIC_COMMAND_KEYS)])
        window_s = float(self.cfg["rewards"].get("move_start_window_s", 1.6))
        min_target_norm = float(self.cfg["rewards"].get("move_start_min_target_norm", threshold))
        return (
            (target_norm > min_target_norm)
            & (command_norm > threshold)
            & (self.public_command_age <= window_s)
        ).float()

    def _straight_walk_mask(self):
        min_speed = float(self.cfg["rewards"].get("straight_min_speed", 0.035))
        max_lateral_command = float(self.cfg["rewards"].get("straight_max_abs_y_command", 0.035))
        max_yaw_command = float(self.cfg["rewards"].get("straight_max_abs_yaw_command", 0.06))
        return (
            (torch.abs(self.commands[:, 0]) > min_speed)
            & (torch.abs(self.commands[:, 1]) < max_lateral_command)
            & (torch.abs(self.commands[:, 2]) < max_yaw_command)
        ).float()

    def _straight_sustain_mask(self):
        min_vx = float(self.cfg["rewards"].get("sustain_min_vx", 0.85))
        max_vx = float(self.cfg["rewards"].get("sustain_max_vx", 1.15))
        min_age_s = float(self.cfg["rewards"].get("sustain_min_command_age_s", 2.0))
        max_target_vy = float(self.cfg["rewards"].get("sustain_max_abs_target_vy", 0.035))
        max_target_yaw = float(self.cfg["rewards"].get("sustain_max_abs_target_yaw", 0.06))
        target = self.public_command_targets
        target_mask = (
            (target[:, 0] >= min_vx)
            & (target[:, 0] <= max_vx)
            & (torch.abs(target[:, 1]) < max_target_vy)
            & (torch.abs(target[:, 2]) < max_target_yaw)
        )
        return self._straight_walk_mask() * target_mask.float() * (self.public_command_age >= min_age_s).float()

    def _low_speed_straight_mask(self):
        min_vx = float(self.cfg["rewards"].get("low_speed_guard_min_vx", 0.04))
        max_vx = float(self.cfg["rewards"].get("low_speed_guard_max_vx", 0.36))
        in_range = (self.commands[:, 0] >= min_vx) & (self.commands[:, 0] <= max_vx)
        return self._straight_walk_mask() * in_range.float()

    def _resolve_internal_commands(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return

        adapter = self.current_adapter_config
        vx = self.commands[env_ids, 0]
        vy = self.commands[env_ids, 1]
        yaw = self.commands[env_ids, 2]
        if bool(adapter.get("forward_pitch_vx_comp_enabled", False)):
            forward_pitch = torch.clamp(
                self.projected_gravity[env_ids, 0] - float(adapter.get("forward_pitch_vx_comp_deadband", 0.04)),
                min=0.0,
            )
            correction = torch.minimum(
                forward_pitch * float(adapter.get("forward_pitch_vx_comp_gain", 0.8)),
                torch.full_like(forward_pitch, float(adapter.get("forward_pitch_vx_comp_max", 0.05))),
            )
            min_vx = float(adapter.get("forward_pitch_vx_comp_min_vx", 0.04))
            compensated_vx = torch.maximum(torch.full_like(vx, min_vx), vx - correction)
            vx = torch.where(vx > 0.0, torch.minimum(vx, compensated_vx), vx)
            self.commands[env_ids, 0] = vx
        internal_yaw = self._heading_corrected_yaw_command(env_ids, vx, vy, yaw, adapter)
        linear_speed = torch.sqrt(torch.square(vx) + torch.square(vy))
        yaw_speed = torch.abs(internal_yaw)

        speed_min = float(adapter.get("gait_frequency_speed_min", 0.08))
        speed_max = max(float(adapter.get("gait_frequency_speed_max", 1.0)), speed_min + 1.0e-6)
        max_yaw = max(float(adapter.get("max_yaw_speed_for_drive", self.cfg["commands"]["ang_vel_yaw"][1])), 1.0e-6)
        linear_drive = torch.clamp((linear_speed - speed_min) / (speed_max - speed_min), min=0.0, max=1.0)
        yaw_drive = torch.clamp(yaw_speed / max_yaw, min=0.0, max=1.0)
        drive = torch.maximum(linear_drive, yaw_drive)

        stand_threshold = float(adapter.get("stand_command_threshold", 0.04))
        command_norm = torch.sqrt(torch.square(vx) + torch.square(vy) + torch.square(yaw))
        moving = command_norm > stand_threshold
        was_standing = self.gait_frequency[env_ids] <= 1.0e-8

        gait_min = float(adapter.get("gait_frequency_min", 1.15))
        gait_max = float(adapter.get("gait_frequency_max", 1.95))
        gait_frequency = gait_min + drive * (gait_max - gait_min)
        gait_frequency = torch.where(moving, gait_frequency, torch.zeros_like(gait_frequency))

        foot_yaw_gain = float(adapter.get("foot_yaw_from_yaw_gain", 0.12))
        foot_yaw_clip = adapter.get("foot_yaw_target_clip", [-0.25, 0.25])
        foot_yaw = torch.clamp(
            internal_yaw * foot_yaw_gain,
            min=float(foot_yaw_clip[0]),
            max=float(foot_yaw_clip[1]),
        )

        pitch_gain = float(adapter.get("body_pitch_gain", 0.08))
        pitch_offset = float(adapter.get("body_pitch_offset", 0.0))
        pitch_clip = adapter.get("body_pitch_target_clip", [-0.04, 0.12])
        body_pitch = torch.clamp(
            vx * pitch_gain + pitch_offset,
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
        next_gait_process = self.gait_process[env_ids]
        if bool(self.cfg["commands"].get("randomize_gait_phase_on_start", False)):
            start_moving = moving & was_standing
            random_phase = torch.rand_like(next_gait_process)
            next_gait_process = torch.where(start_moving, random_phase, next_gait_process)
        self.gait_process[env_ids] = torch.where(
            moving,
            next_gait_process,
            torch.zeros_like(self.gait_process[env_ids]),
        )

    def _heading_corrected_yaw_command(self, env_ids, vx, vy, yaw, adapter):
        if not bool(adapter.get("heading_correction_enabled", False)):
            return yaw

        min_vx = float(adapter.get("heading_correction_min_abs_vx", 0.08))
        max_abs_vy = float(adapter.get("heading_correction_max_abs_vy_command", 0.04))
        max_abs_yaw = float(adapter.get("heading_correction_max_abs_yaw_command", 0.08))
        straight = (
            (torch.abs(vx) > min_vx)
            & (torch.abs(vy) < max_abs_vy)
            & (torch.abs(yaw) < max_abs_yaw)
        ).float()
        if float(straight.sum().item()) <= 0.0:
            return yaw

        base_yaw = self._get_base_yaw()[env_ids]
        yaw_error = self._wrap_to_pi(base_yaw - self.desired_yaw[env_ids])
        deadband = float(adapter.get("heading_correction_deadband", 0.015))
        yaw_error = torch.sign(yaw_error) * torch.clamp(torch.abs(yaw_error) - deadband, min=0.0)
        gain = float(adapter.get("heading_correction_gain", 1.0))
        max_correction = float(adapter.get("heading_correction_max_yaw_rate", 0.35))
        correction = torch.clamp(-gain * yaw_error, min=-max_correction, max=max_correction)
        return yaw + correction * straight

    def _apply_fixed_command(self, command_cfg, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return

        defaults = {"lin_vel_x": 0.0, "lin_vel_y": 0.0, "ang_vel_yaw": 0.0}
        values = torch.zeros(len(env_ids), len(self.PUBLIC_COMMAND_KEYS), dtype=torch.float, device=self.device)
        for command_idx, key in enumerate(self.PUBLIC_COMMAND_KEYS):
            values[:, command_idx] = float(command_cfg.get(key, defaults[key]))
        self._set_public_command_targets(env_ids, values)

    def _sample_public_commands(self, env_ids):
        for command_idx, key in enumerate(self.PUBLIC_COMMAND_KEYS):
            low, high = self._public_command_range(key)
            self.public_command_targets[env_ids, command_idx] = torch_rand_float(
                float(low),
                float(high),
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

    def _apply_straight_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        straight_cfg = self.current_straight_config
        for command_idx, key in enumerate(self.PUBLIC_COMMAND_KEYS):
            value = straight_cfg.get(
                key,
                self.current_public_command_ranges.get(
                    key,
                    self.cfg["commands"].get("fixed", {}).get(key, 0.0),
                ),
            )
            if isinstance(value, (list, tuple)) and len(value) == 2:
                self.public_command_targets[env_ids, command_idx] = torch_rand_float(
                    float(value[0]),
                    float(value[1]),
                    (len(env_ids), 1),
                    device=self.device,
                ).squeeze(1)
            else:
                self.public_command_targets[env_ids, command_idx] = float(value)

    def _apply_high_speed_commands(self, env_ids):
        if len(env_ids) == 0 or self.current_high_speed_proportion <= 0.0:
            return
        count = int(self.current_high_speed_proportion * len(env_ids))
        if count <= 0:
            return
        selected = env_ids[torch.randperm(len(env_ids), device=self.device)[:count]]
        high_speed_cfg = self.current_high_speed_config
        vx_range = high_speed_cfg.get("lin_vel_x", self.current_straight_config.get("lin_vel_x"))
        if not isinstance(vx_range, (list, tuple)) or len(vx_range) != 2:
            return
        self.public_command_targets[selected, 0] = torch_rand_float(
            float(vx_range[0]),
            float(vx_range[1]),
            (len(selected), 1),
            device=self.device,
        ).squeeze(1)
        self.public_command_targets[selected, 1] = float(high_speed_cfg.get("lin_vel_y", 0.0))
        self.public_command_targets[selected, 2] = float(high_speed_cfg.get("ang_vel_yaw", 0.0))

    def _resample_commands(self):
        if getattr(self, "is_play", False):
            self._apply_fixed_command(self.cfg["commands"].get("play", {}))
            return
        if getattr(self, "manual_control", False):
            self.public_command_targets[:, :] = self.commands[:, : len(self.PUBLIC_COMMAND_KEYS)]
            self._apply_public_command_slew()
            self._resolve_internal_commands()
            return

        env_ids = (self.episode_length_buf == self.cmd_resample_time).nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return

        if self.cfg["commands"].get("training_mode", "sampled") == "fixed":
            self._apply_fixed_command(self.cfg["commands"].get("fixed", {}), env_ids)
            self.cmd_resample_time[env_ids] += self._sample_resample_steps(len(env_ids))
            return

        previous_targets = self.public_command_targets[env_ids].clone()
        self.public_command_targets[env_ids, :] = 0.0
        self._sample_public_commands(env_ids)

        perm = torch.randperm(len(env_ids), device=self.device)
        still_count = int(self.current_still_proportion * len(env_ids))
        still_envs = env_ids[perm[:still_count]]
        self.public_command_targets[still_envs, :] = 0.0
        remaining_envs = env_ids[perm[still_count:]]
        straight_count = int(self.current_straight_proportion * len(remaining_envs))
        straight_envs = remaining_envs[:straight_count]
        self._apply_straight_commands(straight_envs)
        self._apply_high_speed_commands(straight_envs)
        changed = torch.norm(self.public_command_targets[env_ids] - previous_targets, dim=-1) > float(
            self.cfg["commands"].get("command_change_threshold", 1.0e-4)
        )
        self.public_command_age[env_ids[changed]] = 0.0

        self.cmd_resample_time[env_ids] += self._sample_resample_steps(len(env_ids))

    def _compute_observations(self):
        self._apply_public_command_slew()
        self._resolve_internal_commands()
        self.public_command_age += self.dt
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
        self.extras["sirl_command_targets"] = self.public_command_targets.detach()
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

    def _foot_front_edge_offset(self):
        edge_pos = self.cfg.get("asset", {}).get("feet_edge_pos", [])
        if not edge_pos:
            return 0.10
        return float(max(edge[0] for edge in edge_pos))

    def _feet_local_x(self):
        _, _, base_yaw = get_euler_xyz(self.base_quat)
        dx = self.feet_pos[:, :, 0] - self.base_pos[:, 0].unsqueeze(-1)
        dy = self.feet_pos[:, :, 1] - self.base_pos[:, 1].unsqueeze(-1)
        return torch.cos(base_yaw).unsqueeze(-1) * dx + torch.sin(base_yaw).unsqueeze(-1) * dy

    def _support_front_x(self):
        foot_front_x = self._feet_local_x() + self._foot_front_edge_offset()
        contact = self.feet_contact.float()
        contact_front_x = torch.where(
            contact > 0.0,
            foot_front_x,
            torch.full_like(foot_front_x, -1.0e6),
        )
        support_front_x = torch.max(contact_front_x, dim=-1).values
        fallback_front_x = torch.max(foot_front_x, dim=-1).values
        has_contact = torch.sum(contact, dim=-1) > 0.0
        return torch.where(has_contact, support_front_x, fallback_front_x)

    def _support_front_lag_value(self, mask, min_x_key="support_front_min_x"):
        min_vx = float(self.cfg["rewards"].get("support_front_min_abs_vx", 0.04))
        active = (torch.abs(self.commands[:, 0]) >= min_vx).float()
        min_x = float(self.cfg["rewards"].get(min_x_key, self.cfg["rewards"].get("support_front_min_x", 0.0)))
        lag = torch.clamp(min_x - self._support_front_x(), min=0.0)
        return torch.square(lag) * mask * active

    def _support_front_pitch_lag_value(
        self,
        mask,
        min_x_key="support_front_min_x",
        pitch_threshold_key="support_front_pitch_threshold",
    ):
        min_vx = float(self.cfg["rewards"].get("support_front_min_abs_vx", 0.04))
        active = (torch.abs(self.commands[:, 0]) >= min_vx).float()
        min_x = float(self.cfg["rewards"].get(min_x_key, self.cfg["rewards"].get("support_front_min_x", 0.0)))
        threshold = float(
            self.cfg["rewards"].get(
                pitch_threshold_key,
                self.cfg["rewards"].get("support_front_pitch_threshold", self.cfg["rewards"].get("straight_forward_pitch_threshold", 0.04)),
            )
        )
        lag = torch.clamp(min_x - self._support_front_x(), min=0.0)
        forward_pitch = torch.clamp(self.projected_gravity[:, 0] - threshold, min=0.0)
        return forward_pitch * lag * mask * active

    def _swing_foot_forward_lag_value(self, mask, min_x_key="swing_foot_forward_min_x"):
        left_swing, right_swing = self._swing_masks()
        swing_mask = torch.stack((left_swing, right_swing), dim=-1).float()
        side_weights = torch.as_tensor(
            self.cfg["rewards"].get(
                "swing_foot_forward_side_weights",
                self.cfg["rewards"].get("swing_clearance_side_weights", [1.0, 1.0]),
            ),
            dtype=torch.float,
            device=self.device,
        ).view(1, -1)
        min_x = float(self.cfg["rewards"].get(min_x_key, self.cfg["rewards"].get("swing_foot_forward_min_x", -0.03)))
        lag = torch.clamp(min_x - self._feet_local_x(), min=0.0)
        weighted = torch.square(lag) * swing_mask * side_weights
        denom = torch.clamp(torch.sum(swing_mask * side_weights, dim=-1), min=1.0)
        return torch.sum(weighted, dim=-1) / denom * mask

    def _compute_sirl_info(self):
        straight_mask = self._straight_walk_mask()
        lateral_weight = 0.75 + 1.25 * straight_mask
        yaw_weight = 0.50 + 1.00 * straight_mask
        velocity_error = (
            torch.abs(self.commands[:, 0] - self.filtered_lin_vel[:, 0])
            + lateral_weight * torch.abs(self.commands[:, 1] - self.filtered_lin_vel[:, 1])
            + yaw_weight * torch.abs(self.commands[:, 2] - self.filtered_ang_vel[:, 2])
        )
        feet_vel_xy = torch.norm((self.last_feet_pos[:, :, :2] - self.feet_pos[:, :, :2]) / self.dt, dim=-1)
        stance_count = torch.clamp(self.feet_contact.float().sum(dim=-1), min=1.0)
        feet_slip = torch.sum(feet_vel_xy * self.feet_contact.float(), dim=-1) / stance_count
        feet_slip = feet_slip * (self.episode_length_buf > 1).float()

        left_swing, right_swing = self._swing_masks()
        swing_mask = torch.stack((left_swing, right_swing), dim=-1).float()
        side_weights = torch.as_tensor(
            self.cfg["rewards"].get("swing_clearance_side_weights", [1.0, 1.0]),
            dtype=torch.float,
            device=self.device,
        ).view(1, -1)
        swing_weight = swing_mask * side_weights
        swing_count = torch.clamp(swing_mask.sum(dim=-1), min=1.0)
        clearance = self._foot_clearance()
        clearance_target = float(self.cfg["rewards"].get("swing_clearance_target", 0.10))
        swing_clearance_deficit = torch.sum(torch.clamp(clearance_target - clearance, min=0.0) * swing_weight, dim=-1) / torch.clamp(
            swing_weight.sum(dim=-1), min=1.0
        )
        swing_clearance_excess = self._straight_swing_clearance_excess_value()
        no_foot_contact = self._straight_no_foot_contact_value()
        support_front_lag = self._straight_support_front_lag_value()
        support_front_pitch_lag = self._straight_support_front_pitch_lag_value()
        swing_foot_forward_lag = self._straight_swing_foot_forward_lag_value()
        swing_edge_contact = self._straight_swing_edge_contact_value()
        swing_contact = self._straight_swing_contact_value()
        swing_pitch_asymmetry = self._straight_swing_pitch_asymmetry_value()
        contact_duty_asymmetry = self._straight_contact_duty_asymmetry_value()
        right_contact_duty_excess = self._straight_right_contact_duty_excess_value()
        forward_push_asymmetry = self._straight_forward_push_asymmetry_value()
        right_forward_push_excess = self._straight_right_forward_push_excess_value()
        forward_pitch_push = self._straight_forward_pitch_push_value()
        right_forward_pitch_push = self._straight_right_forward_pitch_push_value()
        forward_pitch_excess = self._straight_forward_pitch_excess_value()
        move_start_forward_pitch = self._move_start_forward_pitch_value()
        move_start_ankle_tracking = self._move_start_ankle_pitch_tracking_value()
        move_start_yaw_drift = self._move_start_yaw_drift_value()
        move_start_support_front_lag = self._move_start_support_front_lag_value()
        sustain_forward_pitch = self._sustain_forward_pitch_value()
        sustain_forward_pitch_rate = self._sustain_forward_pitch_rate_value()
        sustain_right_forward_push = self._sustain_right_forward_push_excess_value()
        sustain_right_forward_pitch_push = self._sustain_right_forward_pitch_push_value()
        sustain_ankle_tracking = self._sustain_ankle_pitch_tracking_value()
        sustain_yaw_drift = self._sustain_yaw_drift_value()
        sustain_swing_clearance_excess = self._sustain_swing_clearance_excess_value()
        sustain_no_foot_contact = self._sustain_no_foot_contact_value()
        sustain_support_front_lag = self._sustain_support_front_lag_value()
        sustain_support_front_pitch_lag = self._sustain_support_front_pitch_lag_value()
        sustain_swing_foot_forward_lag = self._sustain_swing_foot_forward_lag_value()

        base_tilt = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=-1)
        base_height = self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos)
        low_height_target = float(self.cfg["rewards"].get("base_height_target", 0.52)) - 0.04
        low_height = torch.clamp(low_height_target - base_height, min=0.0)
        stand_mask = self._stand_mask()
        stand_drift = (
            torch.sum(torch.square(self.filtered_lin_vel[:, :2]), dim=-1)
            + 0.5 * torch.square(self.filtered_ang_vel[:, 2])
        ) * stand_mask
        stop_transition_drift = self._stop_transition_drift_value()
        low_speed_mask = self._low_speed_straight_mask()
        low_speed_margin = float(
            self.cfg["rewards"].get(
                "low_speed_overspeed_margin",
                self.cfg["rewards"].get("lin_vel_x_overspeed_margin", 0.06),
            )
        )
        low_speed_overspeed = torch.square(
            torch.clamp(self.filtered_lin_vel[:, 0] - self.commands[:, 0] - low_speed_margin, min=0.0)
        )
        lateral_clip = float(self.cfg["rewards"].get("low_speed_lateral_vel_clip", 0.8))
        yaw_clip = float(self.cfg["rewards"].get("low_speed_yaw_vel_clip", 2.5))
        lateral_vel = torch.clamp(self.filtered_lin_vel[:, 1], min=-lateral_clip, max=lateral_clip)
        yaw_vel = torch.clamp(self.filtered_ang_vel[:, 2], min=-yaw_clip, max=yaw_clip)
        low_speed_lateral_instability = torch.square(lateral_vel) + 0.35 * torch.square(yaw_vel) + 0.60 * torch.square(
            self.projected_gravity[:, 1]
        )
        return {
            "velocity_error": velocity_error,
            "feet_slip": feet_slip,
            "swing_clearance_deficit": swing_clearance_deficit,
            "swing_clearance_excess": swing_clearance_excess,
            "no_foot_contact": no_foot_contact,
            "support_front_lag": support_front_lag,
            "support_front_pitch_lag": support_front_pitch_lag,
            "swing_foot_forward_lag": swing_foot_forward_lag,
            "low_speed_overspeed": low_speed_overspeed * low_speed_mask,
            "low_speed_lateral_instability": low_speed_lateral_instability * low_speed_mask,
            "swing_edge_contact": swing_edge_contact,
            "swing_contact": swing_contact,
            "swing_pitch_asymmetry": swing_pitch_asymmetry,
            "contact_duty_asymmetry": contact_duty_asymmetry,
            "right_contact_duty_excess": right_contact_duty_excess,
            "forward_push_asymmetry": forward_push_asymmetry,
            "right_forward_push_excess": right_forward_push_excess,
            "forward_pitch_push": forward_pitch_push,
            "right_forward_pitch_push": right_forward_pitch_push,
            "forward_pitch_excess": forward_pitch_excess,
            "move_start_forward_pitch": move_start_forward_pitch,
            "move_start_ankle_tracking": move_start_ankle_tracking,
            "move_start_yaw_drift": move_start_yaw_drift,
            "move_start_support_front_lag": move_start_support_front_lag,
            "sustain_forward_pitch": sustain_forward_pitch,
            "sustain_forward_pitch_rate": sustain_forward_pitch_rate,
            "sustain_right_forward_push": sustain_right_forward_push,
            "sustain_right_forward_pitch_push": sustain_right_forward_pitch_push,
            "sustain_ankle_tracking": sustain_ankle_tracking,
            "sustain_yaw_drift": sustain_yaw_drift,
            "sustain_swing_clearance_excess": sustain_swing_clearance_excess,
            "sustain_no_foot_contact": sustain_no_foot_contact,
            "sustain_support_front_lag": sustain_support_front_lag,
            "sustain_support_front_pitch_lag": sustain_support_front_pitch_lag,
            "sustain_swing_foot_forward_lag": sustain_swing_foot_forward_lag,
            "base_tilt": base_tilt,
            "low_height": low_height,
            "stand_drift": stand_drift,
            "stop_transition_drift": stop_transition_drift,
        }

    def _reward_lin_vel_y_error(self):
        active = ((torch.abs(self.commands[:, 1]) > 0.03) | (self._straight_walk_mask() > 0.0)).float()
        return torch.square(self.commands[:, 1] - self.filtered_lin_vel[:, 1]) * active

    def _reward_ang_vel_yaw_error(self):
        active = ((torch.abs(self.commands[:, 2]) > 0.05) | (self._straight_walk_mask() > 0.0)).float()
        return torch.square(self.commands[:, 2] - self.filtered_ang_vel[:, 2]) * active

    def _reward_lin_vel_x_overspeed(self):
        moving_forward = (self.commands[:, 0] > 0.05).float()
        margin = float(self.cfg["rewards"].get("lin_vel_x_overspeed_margin", 0.08))
        overspeed = torch.clamp(self.filtered_lin_vel[:, 0] - self.commands[:, 0] - margin, min=0.0)
        return torch.square(overspeed) * moving_forward

    def _reward_straight_low_speed_overspeed(self):
        margin = float(
            self.cfg["rewards"].get(
                "low_speed_overspeed_margin",
                self.cfg["rewards"].get("lin_vel_x_overspeed_margin", 0.06),
            )
        )
        overspeed = torch.clamp(self.filtered_lin_vel[:, 0] - self.commands[:, 0] - margin, min=0.0)
        return torch.square(overspeed) * self._low_speed_straight_mask()

    def _reward_straight_low_speed_lateral_instability(self):
        lateral_clip = float(self.cfg["rewards"].get("low_speed_lateral_vel_clip", 0.8))
        yaw_clip = float(self.cfg["rewards"].get("low_speed_yaw_vel_clip", 2.5))
        yaw_weight = float(self.cfg["rewards"].get("low_speed_lateral_yaw_weight", 0.35))
        roll_weight = float(self.cfg["rewards"].get("low_speed_lateral_roll_weight", 0.60))
        lateral_vel = torch.clamp(self.filtered_lin_vel[:, 1], min=-lateral_clip, max=lateral_clip)
        yaw_vel = torch.clamp(self.filtered_ang_vel[:, 2], min=-yaw_clip, max=yaw_clip)
        return (
            torch.square(lateral_vel)
            + yaw_weight * torch.square(yaw_vel)
            + roll_weight * torch.square(self.projected_gravity[:, 1])
        ) * self._low_speed_straight_mask()

    def _command_response_weight(self):
        window_s = float(self.cfg["rewards"].get("command_response_window_s", 1.0))
        tau_s = max(float(self.cfg["rewards"].get("command_response_tau_s", 0.35)), 1.0e-6)
        min_target = float(self.cfg["rewards"].get("command_response_min_abs_vx", 0.10))
        active = (self.public_command_age <= window_s).float() * (torch.abs(self.public_command_targets[:, 0]) > min_target).float()
        return active * torch.exp(-self.public_command_age / tau_s)

    def _reward_reactive_lin_vel_x_error(self):
        weight = self._command_response_weight()
        target_vx = self.public_command_targets[:, 0]
        return torch.square(target_vx - self.filtered_lin_vel[:, 0]) * weight

    def _reward_reactive_lin_vel_x_underspeed(self):
        weight = self._command_response_weight()
        target_vx = self.public_command_targets[:, 0]
        direction = torch.sign(target_vx)
        aligned_speed = self.filtered_lin_vel[:, 0] * direction
        lag = torch.clamp(torch.abs(target_vx) - aligned_speed, min=0.0)
        return torch.square(lag) * weight

    def _stop_transition_drift_value(self):
        mask = self._stop_transition_mask()
        lin_clip = float(self.cfg["rewards"].get("stop_transition_lin_vel_clip", 1.8))
        yaw_clip = float(self.cfg["rewards"].get("stop_transition_yaw_vel_clip", 3.0))
        yaw_weight = float(self.cfg["rewards"].get("stop_transition_yaw_weight", 0.35))
        tilt_weight = float(self.cfg["rewards"].get("stop_transition_tilt_weight", 1.2))
        lin_xy = torch.sum(torch.square(torch.clamp(self.filtered_lin_vel[:, :2], min=-lin_clip, max=lin_clip)), dim=-1)
        yaw = torch.square(torch.clamp(self.filtered_ang_vel[:, 2], min=-yaw_clip, max=yaw_clip))
        tilt = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=-1)
        return (lin_xy + yaw_weight * yaw + tilt_weight * tilt) * mask

    def _reward_stop_transition_drift(self):
        return self._stop_transition_drift_value()

    def _reward_stop_transition_action(self):
        return torch.sum(torch.square(self.actions), dim=-1) * self._stop_transition_mask()

    def _ankle_pitch_tracking_error_value(self):
        if self.num_actions < 11:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        ankle_pitch_idx = torch.as_tensor([4, 10], dtype=torch.long, device=self.device)
        err = self.last_dof_targets[:, ankle_pitch_idx] - self.dof_pos[:, ankle_pitch_idx]
        clip = float(self.cfg["rewards"].get("ankle_pitch_tracking_error_clip", 0.22))
        err = torch.clamp(err, min=-clip, max=clip)
        side_weights = torch.as_tensor(
            self.cfg["rewards"].get("ankle_pitch_tracking_side_weights", [1.0, 1.15]),
            dtype=torch.float,
            device=self.device,
        ).view(1, -1)
        return torch.sum(torch.square(err) * side_weights, dim=-1) / torch.clamp(side_weights.sum(), min=1.0e-6)

    def _move_start_forward_pitch_value(self):
        threshold = float(self.cfg["rewards"].get("move_start_forward_pitch_threshold", 0.035))
        forward_pitch = torch.clamp(self.projected_gravity[:, 0] - threshold, min=0.0)
        return torch.square(forward_pitch) * self._move_start_transition_mask()

    def _move_start_ankle_pitch_tracking_value(self):
        return self._ankle_pitch_tracking_error_value() * self._move_start_transition_mask()

    def _move_start_yaw_drift_value(self):
        yaw_clip = float(self.cfg["rewards"].get("move_start_yaw_vel_clip", 2.5))
        yaw_vel = torch.clamp(self.filtered_ang_vel[:, 2], min=-yaw_clip, max=yaw_clip)
        return torch.square(yaw_vel) * self._move_start_transition_mask()

    def _reward_move_start_forward_pitch(self):
        return self._move_start_forward_pitch_value()

    def _reward_move_start_ankle_pitch_tracking(self):
        return self._move_start_ankle_pitch_tracking_value()

    def _reward_move_start_yaw_drift(self):
        return self._move_start_yaw_drift_value()

    def _reward_straight_ankle_pitch_tracking(self):
        return self._ankle_pitch_tracking_error_value() * self._straight_walk_mask()

    def _swing_clearance_excess_value(self, mask):
        left_swing, right_swing = self._swing_masks()
        swing_mask = torch.stack((left_swing, right_swing), dim=-1).float()
        side_weights = torch.as_tensor(
            self.cfg["rewards"].get("swing_clearance_side_weights", [1.0, 1.0]),
            dtype=torch.float,
            device=self.device,
        ).view(1, -1)
        max_clearance = float(self.cfg["rewards"].get("swing_clearance_max", 0.115))
        excess = torch.clamp(self._foot_clearance() - max_clearance, min=0.0)
        weighted = torch.square(excess) * swing_mask * side_weights
        denom = torch.clamp(torch.sum(swing_mask * side_weights, dim=-1), min=1.0)
        return torch.sum(weighted, dim=-1) / denom * mask

    def _no_foot_contact_value(self, mask):
        min_contacts = float(self.cfg["rewards"].get("min_stance_foot_contacts", 1.0))
        contact_count = torch.sum(self.feet_contact.float(), dim=-1)
        return (contact_count < min_contacts).float() * mask

    def _straight_swing_clearance_excess_value(self):
        return self._swing_clearance_excess_value(self._straight_walk_mask())

    def _straight_no_foot_contact_value(self):
        return self._no_foot_contact_value(self._straight_walk_mask())

    def _straight_support_front_lag_value(self):
        return self._support_front_lag_value(self._straight_walk_mask())

    def _straight_support_front_pitch_lag_value(self):
        return self._support_front_pitch_lag_value(self._straight_walk_mask())

    def _straight_swing_foot_forward_lag_value(self):
        return self._swing_foot_forward_lag_value(self._straight_walk_mask())

    def _reward_straight_swing_clearance_excess(self):
        return self._straight_swing_clearance_excess_value()

    def _reward_straight_no_foot_contact(self):
        return self._straight_no_foot_contact_value()

    def _reward_straight_support_front_lag(self):
        return self._straight_support_front_lag_value()

    def _reward_straight_support_front_pitch_lag(self):
        return self._straight_support_front_pitch_lag_value()

    def _reward_straight_swing_foot_forward_lag(self):
        return self._straight_swing_foot_forward_lag_value()

    def _sustain_forward_pitch_value(self):
        threshold = float(self.cfg["rewards"].get("sustain_forward_pitch_threshold", 0.040))
        forward_pitch = torch.clamp(self.projected_gravity[:, 0] - threshold, min=0.0)
        return torch.square(forward_pitch) * self._straight_sustain_mask()

    def _sustain_forward_pitch_rate_value(self):
        clip = float(self.cfg["rewards"].get("sustain_forward_pitch_rate_clip", 2.5))
        pitch_rate = torch.clamp(self.filtered_ang_vel[:, 1], min=-clip, max=clip)
        return torch.square(pitch_rate) * self._straight_sustain_mask()

    def _sustain_right_forward_push_excess_value(self):
        deadband = float(
            self.cfg["rewards"].get(
                "sustain_right_forward_push_deadband",
                self.cfg["rewards"].get("right_forward_push_deadband", 0.05),
            )
        )
        push = self._straight_forward_push()
        right_excess = torch.clamp(push[:, 1] - push[:, 0] - deadband, min=0.0)
        return torch.square(right_excess) * self._straight_sustain_mask()

    def _sustain_right_forward_pitch_push_value(self):
        threshold = float(self.cfg["rewards"].get("sustain_forward_pitch_push_threshold", 0.030))
        forward_pitch = torch.clamp(self.projected_gravity[:, 0] - threshold, min=0.0)
        right_excess = torch.sqrt(torch.clamp(self._sustain_right_forward_push_excess_value(), min=0.0))
        return forward_pitch * right_excess * self._straight_sustain_mask()

    def _sustain_ankle_pitch_tracking_value(self):
        return self._ankle_pitch_tracking_error_value() * self._straight_sustain_mask()

    def _sustain_yaw_drift_value(self):
        yaw_clip = float(self.cfg["rewards"].get("sustain_yaw_vel_clip", 2.5))
        yaw_vel = torch.clamp(self.filtered_ang_vel[:, 2], min=-yaw_clip, max=yaw_clip)
        return torch.square(yaw_vel) * self._straight_sustain_mask()

    def _sustain_swing_clearance_excess_value(self):
        return self._swing_clearance_excess_value(self._straight_sustain_mask())

    def _sustain_no_foot_contact_value(self):
        return self._no_foot_contact_value(self._straight_sustain_mask())

    def _sustain_support_front_lag_value(self):
        return self._support_front_lag_value(self._straight_sustain_mask(), "sustain_support_front_min_x")

    def _sustain_support_front_pitch_lag_value(self):
        return self._support_front_pitch_lag_value(
            self._straight_sustain_mask(),
            "sustain_support_front_min_x",
            "sustain_support_front_pitch_threshold",
        )

    def _sustain_swing_foot_forward_lag_value(self):
        return self._swing_foot_forward_lag_value(self._straight_sustain_mask(), "sustain_swing_foot_forward_min_x")

    def _move_start_support_front_lag_value(self):
        return self._support_front_lag_value(self._move_start_transition_mask(), "move_start_support_front_min_x")

    def _reward_sustain_forward_pitch(self):
        return self._sustain_forward_pitch_value()

    def _reward_sustain_forward_pitch_rate(self):
        return self._sustain_forward_pitch_rate_value()

    def _reward_sustain_right_forward_push_excess(self):
        return self._sustain_right_forward_push_excess_value()

    def _reward_sustain_right_forward_pitch_push(self):
        return self._sustain_right_forward_pitch_push_value()

    def _reward_sustain_ankle_pitch_tracking(self):
        return self._sustain_ankle_pitch_tracking_value()

    def _reward_sustain_yaw_drift(self):
        return self._sustain_yaw_drift_value()

    def _reward_sustain_swing_clearance_excess(self):
        return self._sustain_swing_clearance_excess_value()

    def _reward_sustain_no_foot_contact(self):
        return self._sustain_no_foot_contact_value()

    def _reward_sustain_support_front_lag(self):
        return self._sustain_support_front_lag_value()

    def _reward_sustain_support_front_pitch_lag(self):
        return self._sustain_support_front_pitch_lag_value()

    def _reward_sustain_swing_foot_forward_lag(self):
        return self._sustain_swing_foot_forward_lag_value()

    def _reward_move_start_support_front_lag(self):
        return self._move_start_support_front_lag_value()

    def _straight_swing_mask(self):
        left_swing, right_swing = self._swing_masks()
        return torch.stack((left_swing, right_swing), dim=-1).float() * self._straight_walk_mask().unsqueeze(-1)

    def _reward_straight_phase_contact(self):
        left_swing, right_swing = self._swing_masks()
        straight = self._straight_walk_mask()
        left_contact = self.feet_contact[:, 0].float()
        right_contact = self.feet_contact[:, 1].float()
        left_phase = left_swing.float() * ((1.0 - left_contact) + right_contact)
        right_phase = right_swing.float() * ((1.0 - right_contact) + left_contact)
        phase_count = torch.clamp(left_swing.float() + right_swing.float(), min=1.0)
        return (left_phase + right_phase) / phase_count * straight

    def _reward_straight_roll_tilt(self):
        return torch.square(self.projected_gravity[:, 1]) * self._straight_walk_mask()

    def _straight_forward_pitch_excess_value(self):
        threshold = float(self.cfg["rewards"].get("straight_forward_pitch_threshold", 0.10))
        excess = torch.clamp(self.projected_gravity[:, 0] - threshold, min=0.0)
        return torch.square(excess) * self._straight_walk_mask()

    def _reward_straight_forward_pitch_excess(self):
        return self._straight_forward_pitch_excess_value()

    def _reward_straight_forward_pitch_rate(self):
        clip = float(self.cfg["rewards"].get("straight_forward_pitch_rate_clip", 3.0))
        pitch_rate = torch.clamp(self.filtered_ang_vel[:, 1], min=-clip, max=clip)
        return torch.square(pitch_rate) * self._straight_walk_mask()

    def _reward_straight_swing_flat_contact(self):
        if not hasattr(self, "feet_edge_contact") or self.feet_edge_contact.shape[-1] < 4:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        swing_mask = self._straight_swing_mask()
        front_contact = torch.any(self.feet_edge_contact[:, :, 0:2], dim=-1).float()
        rear_contact = torch.any(self.feet_edge_contact[:, :, 2:4], dim=-1).float()
        flat_contact = front_contact * rear_contact
        side_weights = torch.as_tensor(
            self.cfg["rewards"].get("swing_flat_contact_side_weights", [1.0, 1.0]),
            dtype=torch.float,
            device=self.device,
        ).view(1, -1)
        weighted = flat_contact * swing_mask * side_weights
        return torch.sum(weighted, dim=-1) / torch.clamp(torch.sum(swing_mask * side_weights, dim=-1), min=1.0)

    def _reward_straight_swing_pitch_balance(self):
        return self._straight_swing_pitch_asymmetry_value()

    def _straight_swing_contact_value(self):
        swing_mask = self._straight_swing_mask()
        side_weights = torch.as_tensor(
            self.cfg["rewards"].get("swing_contact_side_weights", [1.0, 1.0]),
            dtype=torch.float,
            device=self.device,
        ).view(1, -1)
        contact = self.feet_contact.float() * swing_mask * side_weights
        return torch.sum(contact, dim=-1) / torch.clamp(torch.sum(swing_mask * side_weights, dim=-1), min=1.0)

    def _reward_straight_swing_contact(self):
        return self._straight_swing_contact_value()

    def _straight_swing_pitch_asymmetry_value(self):
        swing_mask = self._straight_swing_mask()
        active = (torch.sum(swing_mask, dim=-1) > 0.0).float()
        pitch_mag = torch.abs(self.feet_pitch)
        return torch.square(pitch_mag[:, 0] - pitch_mag[:, 1]) * active

    def _contact_duty_diff(self):
        if not hasattr(self, "feet_contact_duty_ema") or self.feet_contact_duty_ema.shape[-1] < 2:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        return self.feet_contact_duty_ema[:, 1] - self.feet_contact_duty_ema[:, 0]

    def _straight_contact_duty_asymmetry_value(self):
        deadband = float(self.cfg["rewards"].get("contact_duty_deadband", 0.04))
        diff = torch.clamp(torch.abs(self._contact_duty_diff()) - deadband, min=0.0)
        return torch.square(diff) * self._straight_walk_mask()

    def _straight_right_contact_duty_excess_value(self):
        deadband = float(
            self.cfg["rewards"].get(
                "right_contact_duty_deadband",
                self.cfg["rewards"].get("contact_duty_deadband", 0.04),
            )
        )
        excess = torch.clamp(self._contact_duty_diff() - deadband, min=0.0)
        return torch.square(excess) * self._straight_walk_mask()

    def _reward_straight_contact_duty_balance(self):
        return self._straight_contact_duty_asymmetry_value()

    def _reward_straight_right_contact_duty_excess(self):
        return self._straight_right_contact_duty_excess_value()

    def _straight_forward_push(self):
        foot_force = self.contact_forces[:, self.feet_indices, :]
        _, _, base_yaw = get_euler_xyz(self.base_quat)
        forward_force = (
            torch.cos(base_yaw).unsqueeze(-1) * foot_force[:, :, 0]
            + torch.sin(base_yaw).unsqueeze(-1) * foot_force[:, :, 1]
        )
        force_clip = float(self.cfg["rewards"].get("forward_push_force_clip", 180.0))
        force_norm = max(float(self.cfg["rewards"].get("forward_push_force_norm", 120.0)), 1.0e-6)
        min_vertical = float(self.cfg["rewards"].get("forward_push_min_vertical_force", 5.0))
        contact = (torch.clamp(foot_force[:, :, 2], min=0.0) > min_vertical).float()
        contact = torch.maximum(contact, self.feet_contact.float())
        forward_force = torch.clamp(forward_force, min=-force_clip, max=force_clip)
        return torch.clamp(forward_force, min=0.0) * contact / force_norm

    def _straight_forward_push_asymmetry_value(self):
        deadband = float(self.cfg["rewards"].get("forward_push_asymmetry_deadband", 0.08))
        push = self._straight_forward_push()
        diff = torch.clamp(torch.abs(push[:, 1] - push[:, 0]) - deadband, min=0.0)
        return torch.square(diff) * self._straight_walk_mask()

    def _straight_right_forward_push_excess_value(self):
        deadband = float(
            self.cfg["rewards"].get(
                "right_forward_push_deadband",
                self.cfg["rewards"].get("forward_push_asymmetry_deadband", 0.08),
            )
        )
        push = self._straight_forward_push()
        excess = torch.clamp(push[:, 1] - push[:, 0] - deadband, min=0.0)
        return torch.square(excess) * self._straight_walk_mask()

    def _straight_forward_pitch_push_value(self):
        threshold = float(self.cfg["rewards"].get("forward_pitch_push_threshold", 0.05))
        forward_pitch = torch.clamp(self.projected_gravity[:, 0] - threshold, min=0.0)
        push = torch.mean(self._straight_forward_push(), dim=-1)
        return forward_pitch * push * self._straight_walk_mask()

    def _straight_right_forward_pitch_push_value(self):
        threshold = float(self.cfg["rewards"].get("forward_pitch_push_threshold", 0.05))
        deadband = float(
            self.cfg["rewards"].get(
                "right_forward_pitch_push_deadband",
                self.cfg["rewards"].get("right_forward_push_deadband", 0.06),
            )
        )
        forward_pitch = torch.clamp(self.projected_gravity[:, 0] - threshold, min=0.0)
        push = self._straight_forward_push()
        right_excess = torch.clamp(push[:, 1] - push[:, 0] - deadband, min=0.0)
        return forward_pitch * right_excess * self._straight_walk_mask()

    def _reward_straight_forward_push_balance(self):
        return self._straight_forward_push_asymmetry_value()

    def _reward_straight_right_forward_push_excess(self):
        return self._straight_right_forward_push_excess_value()

    def _reward_straight_forward_pitch_push(self):
        return self._straight_forward_pitch_push_value()

    def _reward_straight_right_forward_pitch_push(self):
        return self._straight_right_forward_pitch_push_value()

    def _straight_swing_edge_contact_value(self):
        if not hasattr(self, "feet_edge_contact") or self.feet_edge_contact.shape[-1] < 4:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        swing_mask = self._straight_swing_mask()
        front_contact = torch.any(self.feet_edge_contact[:, :, 0:2], dim=-1).float()
        rear_contact = torch.any(self.feet_edge_contact[:, :, 2:4], dim=-1).float()
        front_weight = float(self.cfg["rewards"].get("swing_front_edge_contact_weight", 1.4))
        rear_weight = float(self.cfg["rewards"].get("swing_rear_edge_contact_weight", 1.0))
        side_weights = torch.as_tensor(
            self.cfg["rewards"].get("swing_edge_contact_side_weights", [1.0, 1.0]),
            dtype=torch.float,
            device=self.device,
        ).view(1, -1)
        edge_contact = (front_weight * front_contact + rear_weight * rear_contact) * swing_mask * side_weights
        return torch.sum(edge_contact, dim=-1) / torch.clamp(torch.sum(swing_mask * side_weights, dim=-1), min=1.0)

    def _reward_straight_swing_edge_contact(self):
        return self._straight_swing_edge_contact_value()

    def _reward_straight_swing_foot_pitch(self):
        swing_mask = self._straight_swing_mask()
        side_weights = torch.as_tensor(
            self.cfg["rewards"].get("swing_pitch_side_weights", [1.0, 1.0]),
            dtype=torch.float,
            device=self.device,
        ).view(1, -1)
        pitch_sq = torch.square(self.feet_pitch) * swing_mask * side_weights
        return torch.sum(pitch_sq, dim=-1) / torch.clamp(torch.sum(swing_mask * side_weights, dim=-1), min=1.0)

    def _reward_straight_swing_foot_yaw(self):
        swing_mask = self._straight_swing_mask()
        swing_count = torch.clamp(swing_mask.sum(dim=-1), min=1.0)
        yaw_error = torch.square(self.feet_yaw_rel)
        return torch.sum(yaw_error * swing_mask, dim=-1) / swing_count

    def _reward_straight_swing_lateral_vel(self):
        swing_mask = self._straight_swing_mask()
        _, _, base_yaw = get_euler_xyz(self.base_quat)
        feet_vel = (self.feet_pos - self.last_feet_pos) / self.dt
        lateral_vel = -torch.sin(base_yaw).unsqueeze(-1) * feet_vel[:, :, 0] + torch.cos(base_yaw).unsqueeze(-1) * feet_vel[:, :, 1]
        clip = float(self.cfg["rewards"].get("straight_swing_lateral_vel_clip", 1.0))
        lateral_vel_sq = torch.clamp(torch.square(lateral_vel), max=clip * clip)
        return torch.sum(lateral_vel_sq * swing_mask, dim=-1) * (self.episode_length_buf > 1).float()

    def _reward_straight_swing_roll_yaw_action(self):
        swing_mask = self._straight_swing_mask()
        left_lateral = torch.sum(torch.square(self.actions[:, [1, 2, 5]]), dim=-1)
        right_lateral = torch.sum(torch.square(self.actions[:, [7, 8, 11]]), dim=-1)
        return left_lateral * swing_mask[:, 0] + right_lateral * swing_mask[:, 1]

    def _reward_swing_clearance(self):
        left_swing, right_swing = self._swing_masks()
        swing_mask = torch.stack((left_swing, right_swing), dim=-1).float()
        side_weights = torch.as_tensor(
            self.cfg["rewards"].get("swing_clearance_side_weights", [1.0, 1.0]),
            dtype=torch.float,
            device=self.device,
        ).view(1, -1)
        swing_weight = swing_mask * side_weights
        clearance = self._foot_clearance()
        target = float(self.cfg["rewards"].get("swing_clearance_target", 0.10))
        sigma = max(float(self.cfg["rewards"].get("swing_clearance_sigma", 0.015)), 1.0e-6)
        clearance_reward = torch.exp(-torch.square(clearance - target) / sigma)
        return torch.sum(clearance_reward * swing_weight, dim=-1) / torch.clamp(swing_weight.sum(dim=-1), min=1.0)

    def _reward_scuff_clearance(self):
        left_swing, right_swing = self._swing_masks()
        swing_mask = torch.stack((left_swing, right_swing), dim=-1).float()
        side_weights = torch.as_tensor(
            self.cfg["rewards"].get("scuff_clearance_side_weights", self.cfg["rewards"].get("swing_clearance_side_weights", [1.0, 1.0])),
            dtype=torch.float,
            device=self.device,
        ).view(1, -1)
        swing_weight = swing_mask * side_weights
        clearance = self._foot_clearance()
        threshold = float(self.cfg["rewards"].get("scuff_clearance_threshold", 0.045))
        return torch.sum(torch.clamp(threshold - clearance, min=0.0) * swing_weight, dim=-1) / torch.clamp(swing_weight.sum(dim=-1), min=1.0)

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
