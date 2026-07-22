import torch

from isaacgym.torch_utils import torch_rand_float

from envs.K1.parameter_walk_k1 import ParameterWalkK1
from utils.utils import apply_randomization


class TargetPoseWalkK1(ParameterWalkK1):
    """
    K1 walking task with six public commands:
    robot x, robot y, robot theta, target x, target y, target theta.

    The policy observation stays compatible with ParameterWalkK1 checkpoints by
    converting the target pose into an internal walk intent block.
    """

    TARGET_KEYS = ("target_local_x", "target_local_y", "target_heading_offset")

    def _init_buffers(self):
        super()._init_buffers()
        self.policy_commands = torch.zeros(self.num_envs, 10, dtype=torch.float, device=self.device)
        self.target_local_pos = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.target_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.prev_target_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.target_bearing_error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.target_heading_error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.target_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.target_hold_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.target_start_pos_xy = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.target_start_yaw = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.current_target_ranges = {key: list(self.cfg["commands"][key]) for key in self.TARGET_KEYS}
        self.current_navigation_config = dict(self.cfg["commands"].get("navigation", {}))
        self.current_straight_target_proportion = float(self.cfg["commands"].get("straight_target_proportion", 0.0))
        self.policy_command_defaults = torch.tensor(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                float(self.cfg["commands"].get("policy_defaults", {}).get("foot_yaw_L", 0.0)),
                float(self.cfg["commands"].get("policy_defaults", {}).get("foot_yaw_R", 0.0)),
                float(self.cfg["commands"].get("policy_defaults", {}).get("body_pitch_target", 0.0)),
                float(self.cfg["commands"].get("policy_defaults", {}).get("body_roll_target", 0.0)),
                float(self.cfg["commands"].get("policy_defaults", {}).get("feet_offset_x_target", 0.0)),
                float(self.cfg["commands"].get("policy_defaults", {}).get("feet_offset_y_target", 0.0)),
            ],
            dtype=torch.float,
            device=self.device,
        )
        self.hip_yaw_indices = self._find_dof_indices("Hip_Yaw")

    def update_training_curriculum(self, iteration):
        command_cfg = self.cfg["commands"]
        self.current_target_ranges = {key: list(command_cfg[key]) for key in self.TARGET_KEYS}
        self.current_resampling_time = list(command_cfg.get("resampling_time_s", [4.0, 8.0]))
        self.current_disturbance_scale = 1.0
        self.current_navigation_config = dict(command_cfg.get("navigation", {}))
        self.current_straight_target_proportion = float(command_cfg.get("straight_target_proportion", 0.0))
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

        for key in self.TARGET_KEYS:
            if key in phase:
                self.current_target_ranges[key] = list(phase[key])
        if "navigation" in phase:
            self.current_navigation_config.update(phase["navigation"])
        self.current_straight_target_proportion = float(
            phase.get("straight_target_proportion", self.current_straight_target_proportion)
        )
        self.current_resampling_time = list(phase.get("resampling_time_s", self.current_resampling_time))
        self.current_disturbance_scale = float(phase.get("disturbance_scale", self.current_disturbance_scale))
        self.training_phase_index = active_idx
        if active_idx < len(phases) - 1:
            next_iter = int(phases[active_idx + 1]["start_iteration"])
            span = max(next_iter - int(phase["start_iteration"]), 1)
            self.training_phase_progress = float(
                max(0.0, min(1.0, (iteration - int(phase["start_iteration"])) / span))
            )

    def _update_curriculum(self, env_ids):
        return

    def _find_dof_indices(self, pattern):
        indices = [i for i, name in enumerate(self.dof_names) if pattern in name]
        if not indices:
            return None
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def _target_range(self, key):
        return self.current_target_ranges.get(key, self.cfg["commands"][key])

    def _nav_value(self, key, default):
        return float(self.current_navigation_config.get(key, default))

    def _sync_current_pose_commands(self, env_ids=None):
        base_yaw = self._get_base_yaw()
        if env_ids is None:
            self.commands[:, 0:2] = self.root_states[:, 0:2]
            self.commands[:, 2] = base_yaw
        else:
            self.commands[env_ids, 0:2] = self.root_states[env_ids, 0:2]
            self.commands[env_ids, 2] = base_yaw[env_ids]
        return base_yaw

    def _reset_command_targets(self, env_ids):
        if len(env_ids) == 0:
            return
        base_yaw = self._sync_current_pose_commands(env_ids)
        self.commands[env_ids, 3:5] = self.root_states[env_ids, 0:2]
        self.commands[env_ids, 5] = base_yaw[env_ids]
        self.desired_yaw[env_ids] = base_yaw[env_ids]
        self.desired_pos_xy[env_ids] = self.root_states[env_ids, 0:2]
        self.target_start_pos_xy[env_ids] = self.root_states[env_ids, 0:2]
        self.target_start_yaw[env_ids] = base_yaw[env_ids]
        self.prev_target_distance[env_ids] = 0.0
        self.target_distance[env_ids] = 0.0
        self.target_reached[env_ids] = False
        self.target_hold_steps[env_ids] = 0
        self.policy_commands[env_ids] = 0.0

    def _sample_range_value(self, key, count):
        low, high = self._target_range(key)
        return torch_rand_float(float(low), float(high), (count, 1), device=self.device).squeeze(1)

    def _sample_target_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        base_yaw = self._sync_current_pose_commands(env_ids)
        local_x = self._sample_range_value("target_local_x", len(env_ids))
        local_y = self._sample_range_value("target_local_y", len(env_ids))
        heading_offset = self._sample_range_value("target_heading_offset", len(env_ids))
        straight_count = int(self.current_straight_target_proportion * len(env_ids))
        if straight_count > 0:
            perm = torch.randperm(len(env_ids), device=self.device)
            straight_slots = perm[:straight_count]
            local_y[straight_slots] = 0.0
            heading_offset[straight_slots] = 0.0
        cos_yaw = torch.cos(base_yaw[env_ids])
        sin_yaw = torch.sin(base_yaw[env_ids])
        self.target_start_pos_xy[env_ids] = self.root_states[env_ids, 0:2]
        self.target_start_yaw[env_ids] = base_yaw[env_ids]
        self.commands[env_ids, 3] = self.root_states[env_ids, 0] + cos_yaw * local_x - sin_yaw * local_y
        self.commands[env_ids, 4] = self.root_states[env_ids, 1] + sin_yaw * local_x + cos_yaw * local_y
        self.commands[env_ids, 5] = self._wrap_to_pi(base_yaw[env_ids] + heading_offset)
        self.target_hold_steps[env_ids] = 0
        self._update_target_state(env_ids, store_previous=False)
        self._compute_navigation_commands(env_ids)

    def _apply_fixed_target_command(self, command_cfg, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return
        base_yaw = self._sync_current_pose_commands(env_ids)
        self.target_start_pos_xy[env_ids] = self.root_states[env_ids, 0:2]
        self.target_start_yaw[env_ids] = base_yaw[env_ids]

        if "target_x" in command_cfg and "target_y" in command_cfg:
            self.commands[env_ids, 3] = float(command_cfg["target_x"])
            self.commands[env_ids, 4] = float(command_cfg["target_y"])
            target_theta = command_cfg.get("target_theta")
            if target_theta is None:
                self.commands[env_ids, 5] = base_yaw[env_ids]
            else:
                self.commands[env_ids, 5] = self._wrap_to_pi(
                    torch.full((len(env_ids),), float(target_theta), dtype=torch.float, device=self.device)
                )
        else:
            local_x = float(command_cfg.get("target_local_x", self.cfg["commands"]["play"].get("target_local_x", 1.5)))
            local_y = float(command_cfg.get("target_local_y", self.cfg["commands"]["play"].get("target_local_y", 0.0)))
            heading_offset = float(
                command_cfg.get("target_heading_offset", self.cfg["commands"]["play"].get("target_heading_offset", 0.0))
            )
            cos_yaw = torch.cos(base_yaw[env_ids])
            sin_yaw = torch.sin(base_yaw[env_ids])
            self.commands[env_ids, 3] = self.root_states[env_ids, 0] + cos_yaw * local_x - sin_yaw * local_y
            self.commands[env_ids, 4] = self.root_states[env_ids, 1] + sin_yaw * local_x + cos_yaw * local_y
            self.commands[env_ids, 5] = self._wrap_to_pi(base_yaw[env_ids] + heading_offset)

        self._update_target_state(env_ids, store_previous=False)
        self.target_hold_steps[env_ids] = 0
        self._compute_navigation_commands(env_ids)

    def _resample_commands(self):
        env_ids = (self.episode_length_buf == self.cmd_resample_time).nonzero(as_tuple=False).flatten()
        if getattr(self, "is_play", False):
            if len(env_ids) > 0:
                self._apply_fixed_target_command(self.cfg["commands"].get("play", {}), env_ids)
                self.cmd_resample_time[env_ids] += self._sample_resample_steps(len(env_ids))
            self._sync_navigation_state(store_previous=False)
            return
        if getattr(self, "manual_control", False):
            self._sync_navigation_state(store_previous=False)
            return

        if len(env_ids) == 0:
            self._sync_navigation_state(store_previous=False)
            return

        if self.cfg["commands"].get("training_mode", "sampled") == "fixed":
            self._apply_fixed_target_command(self.cfg["commands"].get("fixed", {}), env_ids)
        else:
            self._sample_target_commands(env_ids)

        still_count = int(self.cfg["commands"].get("still_proportion", 0.0) * len(env_ids))
        if still_count > 0:
            perm = torch.randperm(len(env_ids), device=self.device)
            still_envs = env_ids[perm[:still_count]]
            self.commands[still_envs, 3:5] = self.commands[still_envs, 0:2]
            self.commands[still_envs, 5] = self.commands[still_envs, 2]
            self.policy_commands[still_envs] = 0.0
            self.gait_frequency[still_envs] = 0.0
            self.target_hold_steps[still_envs] = 0
            self.target_start_pos_xy[still_envs] = self.commands[still_envs, 0:2]
            self.target_start_yaw[still_envs] = self.commands[still_envs, 2]
            self._update_target_state(still_envs, store_previous=False)

        self.cmd_resample_time[env_ids] += self._sample_resample_steps(len(env_ids))
        self._sync_navigation_state(store_previous=False)

    def _update_command_targets(self):
        self._sync_navigation_state(store_previous=True)
        self.desired_pos_xy[:, :] = self.commands[:, 3:5]
        self.desired_yaw[:] = self.commands[:, 5]
        self.target_hold_steps[:] = torch.where(
            self.target_reached,
            self.target_hold_steps + 1,
            torch.zeros_like(self.target_hold_steps),
        )

    def _sync_navigation_state(self, store_previous):
        self._sync_current_pose_commands()
        self._update_target_state(store_previous=store_previous)
        self._compute_navigation_commands()

    def _update_target_state(self, env_ids=None, store_previous=True):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return
        if store_previous:
            self.prev_target_distance[env_ids] = self.target_distance[env_ids]

        base_yaw = self.commands[env_ids, 2]
        dx = self.commands[env_ids, 3] - self.commands[env_ids, 0]
        dy = self.commands[env_ids, 4] - self.commands[env_ids, 1]
        cos_yaw = torch.cos(base_yaw)
        sin_yaw = torch.sin(base_yaw)
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy
        self.target_local_pos[env_ids, 0] = local_x
        self.target_local_pos[env_ids, 1] = local_y
        distance = torch.sqrt(torch.square(local_x) + torch.square(local_y) + 1.0e-8)
        self.target_distance[env_ids] = distance
        self.target_bearing_error[env_ids] = self._wrap_to_pi(torch.atan2(local_y, local_x))
        self.target_heading_error[env_ids] = self._wrap_to_pi(self.commands[env_ids, 5] - base_yaw)
        arrival_distance = float(self.cfg["rewards"].get("target_arrival_distance", 0.18))
        arrival_heading = float(self.cfg["rewards"].get("target_arrival_heading", 0.20))
        self.target_reached[env_ids] = (distance < arrival_distance) & (torch.abs(self.target_heading_error[env_ids]) < arrival_heading)
        if not store_previous:
            self.prev_target_distance[env_ids] = distance

    def _compute_navigation_commands(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return

        local_x = self.target_local_pos[env_ids, 0]
        local_y = self.target_local_pos[env_ids, 1]
        distance = self.target_distance[env_ids]
        bearing = self.target_bearing_error[env_ids]
        heading_error = self.target_heading_error[env_ids]

        stop_distance = self._nav_value("stop_distance", 0.18)
        heading_stop_error = self._nav_value("heading_stop_error", 0.20)
        final_heading_distance = self._nav_value("final_heading_distance", 0.45)
        max_forward_speed = self._nav_value("max_forward_speed", 1.25)
        max_lateral_speed = self._nav_value("max_lateral_speed", 0.20)
        max_yaw_speed = self._nav_value("max_yaw_speed", 1.0)
        target_speed_gain = self._nav_value("target_speed_gain", 0.75)
        lateral_speed_gain = self._nav_value("lateral_speed_gain", 0.55)
        bearing_yaw_gain = self._nav_value("bearing_yaw_gain", 1.35)
        final_heading_gain = self._nav_value("final_heading_gain", 1.15)

        speed_distance = torch.clamp(distance - stop_distance, min=0.0)
        forward_alignment = torch.clamp(torch.cos(bearing), min=0.0, max=1.0)
        vx = torch.clamp(target_speed_gain * speed_distance, min=0.0, max=max_forward_speed) * forward_alignment
        vy = torch.clamp(lateral_speed_gain * local_y, min=-max_lateral_speed, max=max_lateral_speed)
        far_yaw = bearing_yaw_gain * bearing
        near_yaw = final_heading_gain * heading_error
        near_weight = torch.exp(-torch.square(distance) / max(final_heading_distance * final_heading_distance, 1.0e-6))
        yaw_cmd = torch.clamp((1.0 - near_weight) * far_yaw + near_weight * near_yaw, min=-max_yaw_speed, max=max_yaw_speed)

        stopped = distance < stop_distance
        arrived = stopped & (torch.abs(heading_error) < heading_stop_error)
        vx = torch.where(stopped, torch.zeros_like(vx), vx)
        vy = torch.where(stopped, torch.zeros_like(vy), vy)
        yaw_cmd = torch.where(stopped, torch.clamp(near_yaw, min=-max_yaw_speed, max=max_yaw_speed), yaw_cmd)
        yaw_cmd = torch.where(arrived, torch.zeros_like(yaw_cmd), yaw_cmd)

        gait_min = self._nav_value("gait_frequency_min", 1.20)
        gait_max = self._nav_value("gait_frequency_max", 1.95)
        gait_speed_min = self._nav_value("gait_frequency_speed_min", 0.15)
        gait_speed_max = self._nav_value("gait_frequency_speed_max", max_forward_speed)
        gait_drive = torch.sqrt(torch.square(vx) + torch.square(vy)) + 0.15 * torch.abs(yaw_cmd)
        drive = torch.clamp((gait_drive - gait_speed_min) / max(gait_speed_max - gait_speed_min, 1.0e-6), min=0.0, max=1.0)
        gait_frequency = gait_min + drive * (gait_max - gait_min)
        moving = gait_drive > self._nav_value("stand_command_threshold", 0.03)
        gait_frequency = torch.where(moving, gait_frequency, torch.zeros_like(gait_frequency))

        self.policy_commands[env_ids] = self.policy_command_defaults.unsqueeze(0)
        self.policy_commands[env_ids, 0] = vx
        self.policy_commands[env_ids, 1] = vy
        self.policy_commands[env_ids, 2] = yaw_cmd
        self.policy_commands[env_ids, 3] = gait_frequency
        self.gait_frequency[env_ids] = gait_frequency

    def _compute_observations(self):
        self._sync_navigation_state(store_previous=False)
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
                self.policy_commands * commands_scale,
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

    def _swing_masks(self):
        swing_half_period = 0.5 * self.cfg["rewards"]["swing_period"]
        gait_active = self.gait_frequency > 1.0e-8
        left_swing = (torch.abs(self.gait_process - 0.25) < swing_half_period) & gait_active
        right_swing = (torch.abs(self.gait_process - 0.75) < swing_half_period) & gait_active
        return left_swing, right_swing

    def _foot_clearance(self):
        flat_feet_pos = self.feet_pos.reshape(-1, 3)
        ground_height = self.terrain.terrain_heights(flat_feet_pos).reshape(self.num_envs, len(self.feet_indices))
        return self.feet_pos[:, :, 2] - ground_height

    def _compute_sirl_info(self):
        velocity_error = (
            torch.abs(self.policy_commands[:, 0] - self.filtered_lin_vel[:, 0])
            + 0.75 * torch.abs(self.policy_commands[:, 1] - self.filtered_lin_vel[:, 1])
            + 0.50 * torch.abs(self.policy_commands[:, 2] - self.filtered_ang_vel[:, 2])
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
        return {
            "velocity_error": velocity_error,
            "feet_slip": feet_slip,
            "swing_clearance_deficit": swing_clearance_deficit,
            "base_tilt": base_tilt,
            "low_height": low_height,
            "target_distance": self.target_distance,
        }

    def _check_termination(self):
        super()._check_termination()
        hold_steps = int(float(self.cfg["rewards"].get("target_hold_time_s", 2.0)) / self.dt)
        arrival_done = self.target_reached & (self.target_hold_steps >= hold_steps)
        self.reset_buf |= arrival_done

    def _straight_walk_mask(self):
        return (
            (torch.abs(self.policy_commands[:, 0]) > 0.05)
            & (torch.abs(self.policy_commands[:, 1]) < 0.05)
            & (torch.abs(self.policy_commands[:, 2]) < 0.05)
        ).float()

    def _reward_tracking_lin_vel_x(self):
        return torch.exp(-torch.square(self.policy_commands[:, 0] - self.filtered_lin_vel[:, 0]) / self.cfg["rewards"]["tracking_sigma"])

    def _reward_tracking_lin_vel_y(self):
        return torch.exp(-torch.square(self.policy_commands[:, 1] - self.filtered_lin_vel[:, 1]) / self.cfg["rewards"]["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        return torch.exp(-torch.square(self.policy_commands[:, 2] - self.filtered_ang_vel[:, 2]) / self.cfg["rewards"]["tracking_sigma"])

    def _reward_lin_vel_x_error(self):
        moving = (torch.abs(self.policy_commands[:, 0]) > 0.05).float()
        return torch.square(self.policy_commands[:, 0] - self.filtered_lin_vel[:, 0]) * moving

    def _reward_lin_vel_y_error(self):
        moving = (torch.abs(self.policy_commands[:, 1]) > 0.03).float()
        return torch.square(self.policy_commands[:, 1] - self.filtered_lin_vel[:, 1]) * moving

    def _reward_ang_vel_yaw_error(self):
        moving = (torch.abs(self.policy_commands[:, 2]) > 0.05).float()
        return torch.square(self.policy_commands[:, 2] - self.filtered_ang_vel[:, 2]) * moving

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=-1)

    def _reward_fast_stance_feet_xy_vel(self):
        feet_vel = (self.last_feet_pos - self.feet_pos) / self.dt
        speed = torch.sqrt(torch.square(self.policy_commands[:, 0]) + torch.square(self.policy_commands[:, 1]))
        speed_weight = torch.clamp((speed - 0.70) / 0.50, min=0.0, max=1.0)
        return (
            torch.sum(torch.sum(torch.square(feet_vel[:, :, :2]), dim=-1) * self.feet_contact.float(), dim=-1)
            * speed_weight
            * (self.episode_length_buf > 1).float()
        )

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

    def _reward_target_progress(self):
        progress_clip = float(self.cfg["rewards"].get("target_progress_clip", 0.08))
        return torch.clamp(self.prev_target_distance - self.target_distance, min=-progress_clip, max=progress_clip)

    def _reward_target_position(self):
        sigma = float(self.cfg["rewards"].get("target_position_sigma", 0.25))
        return torch.exp(-torch.square(self.target_distance) / sigma)

    def _reward_target_heading(self):
        sigma = float(self.cfg["rewards"].get("target_heading_sigma", 0.35))
        final_heading_distance = self._nav_value("final_heading_distance", 0.45)
        near_weight = torch.exp(-torch.square(self.target_distance) / max(final_heading_distance * final_heading_distance, 1.0e-6))
        return torch.exp(-torch.square(self.target_heading_error) / sigma) * near_weight

    def _reward_target_arrival(self):
        return self.target_reached.float()

    def _reward_target_hold(self):
        hold_steps = max(1, int(float(self.cfg["rewards"].get("target_hold_time_s", 2.0)) / self.dt))
        return torch.clamp(self.target_hold_steps.float() / hold_steps, max=1.0) * self.target_reached.float()

    def _reward_target_path_lateral(self):
        path_vec = self.commands[:, 3:5] - self.target_start_pos_xy
        current_vec = self.commands[:, 0:2] - self.target_start_pos_xy
        path_len = torch.sqrt(torch.sum(torch.square(path_vec), dim=-1) + 1.0e-8)
        cross = path_vec[:, 0] * current_vec[:, 1] - path_vec[:, 1] * current_vec[:, 0]
        lateral_error = cross / torch.clamp(path_len, min=0.20)
        error_clip = float(self.cfg["rewards"].get("target_path_lateral_error_clip", 0.45))
        moving_mask = (self.target_distance > float(self.cfg["rewards"].get("target_arrival_distance", 0.18))).float()
        valid_path = (path_len > 0.25).float()
        return torch.clamp(torch.square(lateral_error), max=error_clip * error_clip) * moving_mask * valid_path

    def _reward_target_bearing(self):
        sigma = float(self.cfg["rewards"].get("target_bearing_sigma", 0.30))
        final_heading_distance = self._nav_value("final_heading_distance", 0.45)
        far_weight = 1.0 - torch.exp(-torch.square(self.target_distance) / max(final_heading_distance * final_heading_distance, 1.0e-6))
        return torch.exp(-torch.square(self.target_bearing_error) / sigma) * far_weight

    def _reward_target_stand_still(self):
        stand_mask = self._stand_mask()
        lin_xy = torch.sum(torch.square(self.filtered_lin_vel[:, :2]), dim=-1)
        ang_z = torch.square(self.filtered_ang_vel[:, 2])
        return (lin_xy + 0.5 * ang_z) * stand_mask

    def _reward_target_stand_upright(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=-1) * self._stand_mask()

    def _reward_target_stand_action(self):
        return torch.sum(torch.square(self.actions), dim=-1) * self._stand_mask()

    def _reward_target_stand_default_pose(self):
        return torch.mean(torch.square(self.dof_pos - self.default_dof_pos), dim=-1) * self._stand_mask()

    def _reward_nav_lateral_vel(self):
        return torch.square(self.filtered_lin_vel[:, 1] - self.policy_commands[:, 1])

    def _reward_nav_yaw_oscillation(self):
        return torch.square(self.filtered_ang_vel[:, 2] - self.policy_commands[:, 2])

    def _reward_feet_yaw_zero(self):
        return torch.sum(torch.square(self.feet_yaw_rel), dim=-1)

    def _reward_feet_yaw_asymmetry(self):
        if self.feet_yaw_rel.shape[1] < 2:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        return torch.square(self.feet_yaw_rel[:, 0] - self.feet_yaw_rel[:, 1])

    def _reward_feet_width(self):
        _, feet_y_offset = self.get_feet_offset()
        width_clip = float(self.cfg["rewards"].get("feet_width_error_clip", 0.12))
        return torch.clamp(torch.square(feet_y_offset), max=width_clip * width_clip)

    def _reward_hip_yaw_zero(self):
        if self.hip_yaw_indices is None:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        return torch.sum(torch.square(self.dof_pos[:, self.hip_yaw_indices]), dim=-1)

    def _stand_mask(self):
        stand_distance = float(self.cfg["rewards"].get("target_stand_distance", 0.28))
        stand_heading = float(self.cfg["rewards"].get("target_stand_heading", self.cfg["rewards"].get("target_arrival_heading", 0.20)))
        return ((self.target_distance < stand_distance) & (torch.abs(self.target_heading_error) < stand_heading)).float()
