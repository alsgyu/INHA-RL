import torch

from isaacgym.torch_utils import get_euler_xyz

from envs.K1.velocity_command_walk_k1 import VelocityCommandWalkK1


class VelocityCommandWalkPPOK1(VelocityCommandWalkK1):
    """Clean PPO velocity walking task kept separate from SIRL fine-tune configs."""

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        if hasattr(self, "heading_drift_ref_yaw"):
            self.heading_drift_ref_yaw[env_ids] = self._get_base_yaw()[env_ids]

    def _heading_drift_reference(self):
        if not hasattr(self, "heading_drift_ref_yaw"):
            self.heading_drift_ref_yaw = self._get_base_yaw().detach().clone()
        return self.heading_drift_ref_yaw

    def _command_vector(self):
        return self.commands[:, :3]

    def _command_vector_norm(self):
        command = self._command_vector()
        return torch.sqrt(torch.sum(torch.square(command[:, :2]), dim=-1) + torch.square(command[:, 2]))

    def _moving_command_mask(self):
        threshold = float(self.cfg["rewards"].get("command_threshold", 0.05))
        return (self._command_vector_norm() > threshold).float()

    def _pure_axis_masks(self):
        command_deadband = float(self.cfg["rewards"].get("pure_axis_command_deadband", 0.05))
        zero_deadband = float(self.cfg["rewards"].get("pure_axis_zero_deadband", 0.05))
        command = self._command_vector()
        pure_x = (
            (torch.abs(command[:, 0]) > command_deadband)
            & (torch.abs(command[:, 1]) < zero_deadband)
            & (torch.abs(command[:, 2]) < zero_deadband)
        ).float()
        pure_y = (
            (torch.abs(command[:, 1]) > command_deadband)
            & (torch.abs(command[:, 0]) < zero_deadband)
            & (torch.abs(command[:, 2]) < zero_deadband)
        ).float()
        zero_yaw = (torch.abs(command[:, 2]) < zero_deadband).float()
        return pure_x, pure_y, zero_yaw

    def _reward_velocity_alignment(self):
        command = self._command_vector()
        actual = torch.stack(
            (
                self.filtered_lin_vel[:, 0],
                self.filtered_lin_vel[:, 1],
                self.filtered_ang_vel[:, 2],
            ),
            dim=-1,
        )
        scales = torch.as_tensor(
            self.cfg["rewards"].get("velocity_alignment_scales", [1.0, 1.0, 1.0]),
            dtype=torch.float,
            device=self.device,
        ).view(1, 3)
        command_n = command / torch.clamp(scales, min=1.0e-6)
        actual_n = actual / torch.clamp(scales, min=1.0e-6)
        command_norm = torch.norm(command_n, dim=-1)
        actual_norm = torch.norm(actual_n, dim=-1)
        cosine = torch.sum(command_n * actual_n, dim=-1) / torch.clamp(command_norm * actual_norm, min=1.0e-6)
        return torch.clamp(cosine, min=0.0, max=1.0) * self._moving_command_mask()

    def _reward_heading_drift(self):
        ref_yaw = self._heading_drift_reference()
        yaw = self._get_base_yaw()
        deadband = float(self.cfg["rewards"].get("heading_drift_yaw_command_deadband", 0.10))
        turning = torch.abs(self.commands[:, 2]) >= deadband
        reset_ref = (self.episode_length_buf <= 1) | turning
        ref_yaw[reset_ref] = yaw[reset_ref].detach()
        yaw_error = self._wrap_to_pi(yaw - ref_yaw)
        return torch.square(yaw_error) * (~turning).float()

    def _reward_pure_x_lateral_drift(self):
        pure_x, _, _ = self._pure_axis_masks()
        return torch.square(self.filtered_lin_vel[:, 1]) * pure_x

    def _reward_pure_x_yaw_drift(self):
        pure_x, _, _ = self._pure_axis_masks()
        return torch.square(self.filtered_ang_vel[:, 2]) * pure_x

    def _reward_pure_y_forward_drift(self):
        _, pure_y, _ = self._pure_axis_masks()
        return torch.square(self.filtered_lin_vel[:, 0]) * pure_y

    def _reward_pure_y_yaw_drift(self):
        _, pure_y, _ = self._pure_axis_masks()
        return torch.square(self.filtered_ang_vel[:, 2]) * pure_y

    def _reward_zero_yaw_command_ang_vel(self):
        _, _, zero_yaw = self._pure_axis_masks()
        return torch.square(self.filtered_ang_vel[:, 2]) * zero_yaw

    def _reward_xy_perpendicular_velocity(self):
        command_xy = self.commands[:, :2]
        velocity_xy = self.filtered_lin_vel[:, :2]
        command_norm = torch.norm(command_xy, dim=-1, keepdim=True)
        command_dir = command_xy / torch.clamp(command_norm, min=1.0e-6)
        velocity_parallel = torch.sum(velocity_xy * command_dir, dim=-1, keepdim=True) * command_dir
        velocity_perp = velocity_xy - velocity_parallel
        active = (command_norm.squeeze(-1) > float(self.cfg["rewards"].get("xy_perpendicular_command_deadband", 0.05))).float()
        return torch.sum(torch.square(velocity_perp), dim=-1) * active

    def _reward_root_feet_lateral_diff(self):
        return self._root_feet_lateral_diff_value(torch.ones(self.num_envs, dtype=torch.float, device=self.device))

    def _joint_index(self, name):
        if name in self.dof_names:
            return self.dof_names.index(name)
        matches = [idx for idx, dof_name in enumerate(self.dof_names) if name in dof_name]
        return matches[0] if matches else None

    def _roll_yaw_symmetry_pairs(self):
        if hasattr(self, "_cached_roll_yaw_symmetry_pairs"):
            return self._cached_roll_yaw_symmetry_pairs
        pairs = []
        for left_name, right_name in self.cfg["rewards"].get(
            "joint_roll_yaw_symmetry_pairs",
            [
                ["Left_Hip_Roll", "Right_Hip_Roll"],
                ["Left_Hip_Yaw", "Right_Hip_Yaw"],
                ["Left_Ankle_Roll", "Right_Ankle_Roll"],
            ],
        ):
            left_idx = self._joint_index(left_name)
            right_idx = self._joint_index(right_name)
            if left_idx is not None and right_idx is not None:
                pairs.append((left_idx, right_idx))
        self._cached_roll_yaw_symmetry_pairs = pairs
        return pairs

    def _reward_joint_roll_yaw_symmetry(self):
        value = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for left_idx, right_idx in self._roll_yaw_symmetry_pairs():
            value += torch.square(self.dof_pos[:, left_idx] + self.dof_pos[:, right_idx])
        return value

    def _reward_feet_distance(self):
        separation = torch.norm(self.feet_pos[:, 0, :2] - self.feet_pos[:, 1, :2], dim=-1)
        ref = float(self.cfg["rewards"].get("feet_distance_ref", 0.18))
        return torch.clamp(ref - separation, min=0.0)

    def _von_mises_foot_height_profiles(self, maximum_height, kappa):
        centers = torch.as_tensor([0.25, 0.75], dtype=torch.float, device=self.device).view(1, 2)
        phase = self.gait_process.view(-1, 1)
        profile = torch.exp(kappa * torch.cos(2.0 * torch.pi * (phase - centers))) / torch.exp(
            torch.as_tensor(kappa, dtype=torch.float, device=self.device)
        )
        return maximum_height * profile

    def _reward_footswing_trajectory(self):
        command = self._command_vector()
        moving = self._moving_command_mask() * (self.gait_frequency > 1.0e-8).float()

        _, _, base_yaw = get_euler_xyz(self.base_quat)
        cos_yaw = torch.cos(base_yaw)
        sin_yaw = torch.sin(base_yaw)
        desired_velocity_w = torch.stack(
            (
                cos_yaw * command[:, 0] - sin_yaw * command[:, 1],
                sin_yaw * command[:, 0] + cos_yaw * command[:, 1],
            ),
            dim=-1,
        )
        current_velocity_w = self.root_states[:, 7:9]
        stance_fraction = float(self.cfg["rewards"].get("footswing_stance_fraction", 0.5))
        frequency = torch.clamp(self.gait_frequency, min=1.0e-6)
        stance_time = stance_fraction / frequency
        velocity_gain = float(self.cfg["rewards"].get("footswing_velocity_error_gain", 0.2))
        placement_offset = (
            0.5 * stance_time.unsqueeze(-1) * current_velocity_w
            + velocity_gain * (current_velocity_w - desired_velocity_w)
        )

        half_width = 0.5 * float(self.cfg["rewards"].get("feet_distance_ref", 0.18))
        hip_local = torch.as_tensor(
            self.cfg["rewards"].get("footswing_hip_local_pos", [[0.0, half_width], [0.0, -half_width]]),
            dtype=torch.float,
            device=self.device,
        ).view(1, 2, 2)
        hip_w_x = self.base_pos[:, 0].view(-1, 1) + cos_yaw.view(-1, 1) * hip_local[:, :, 0] - sin_yaw.view(-1, 1) * hip_local[:, :, 1]
        hip_w_y = self.base_pos[:, 1].view(-1, 1) + sin_yaw.view(-1, 1) * hip_local[:, :, 0] + cos_yaw.view(-1, 1) * hip_local[:, :, 1]
        target_xy = torch.stack((hip_w_x, hip_w_y), dim=-1) + placement_offset.unsqueeze(1)
        xy_error = torch.sum(torch.square(self.feet_pos[:, :, :2] - target_xy), dim=(1, 2))

        target_height = self._von_mises_foot_height_profiles(
            float(self.cfg["rewards"].get("footswing_maximum_height", 0.06)),
            float(self.cfg["rewards"].get("footswing_height_kappa", 4.0)),
        )
        actual_height = self._foot_clearance()
        height_scale = float(self.cfg["rewards"].get("footswing_height_error_scale", 5.0))
        height_error = torch.sum(torch.square(actual_height - target_height), dim=-1)
        return (xy_error + height_scale * height_error) * moving

    def _reward_straight_feet_overtake(self):
        left_swing, right_swing = self._swing_masks()
        feet_x = self._feet_local_x()
        margin = float(self.cfg["rewards"].get("straight_feet_overtake_margin", 0.015))
        side_weights = torch.as_tensor(
            self.cfg["rewards"].get("straight_feet_overtake_side_weights", [1.0, 1.25]),
            dtype=torch.float,
            device=self.device,
        )
        left_lag = torch.clamp(feet_x[:, 1] + margin - feet_x[:, 0], min=0.0)
        right_lag = torch.clamp(feet_x[:, 0] + margin - feet_x[:, 1], min=0.0)
        value = side_weights[0] * torch.square(left_lag) * left_swing.float()
        value += side_weights[1] * torch.square(right_lag) * right_swing.float()
        return value * self._straight_walk_mask()

    def _reward_straight_right_foot_body_lag(self):
        _, right_swing = self._swing_masks()
        min_x = float(self.cfg["rewards"].get("right_foot_body_min_x", 0.015))
        right_lag = torch.clamp(min_x - self._feet_local_x()[:, 1], min=0.0)
        return torch.square(right_lag) * right_swing.float() * self._straight_walk_mask()
