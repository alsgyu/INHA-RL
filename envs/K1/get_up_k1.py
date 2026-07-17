import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_from_euler_xyz, torch_rand_float

import numpy as np

from envs.K1.base_walk_k1 import BaseWalkK1
from utils.utils import apply_randomization


class GetUpK1(BaseWalkK1):
    """Whole-body get-up task for K1.

    This task reuses the K1 locomotion simulation pipeline but samples fallen
    initial trunk poses and trains a separate 22-DoF recovery policy.
    """

    def _init_buffers(self):
        super()._init_buffers()
        self.play_reset_count = 0

    def _sample_cfg_range(self, value, count):
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return torch_rand_float(float(value[0]), float(value[1]), (count, 1), device=self.device).squeeze(1)
        return torch.full((count,), float(value), dtype=torch.float, device=self.device)

    def _apply_named_dof_pose(self, env_ids, pose_cfg):
        for i, dof_name in enumerate(self.dof_names):
            for key, value in pose_cfg.items():
                if key in dof_name:
                    self.dof_pos[env_ids, i] = float(value)
                    break

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = self.default_dof_pos
        self._apply_named_dof_pose(env_ids, self.cfg.get("getup", {}).get("fallen_joint_angles", {}))
        self.dof_pos[env_ids] = apply_randomization(self.dof_pos[env_ids], self.cfg["randomization"].get("init_dof_pos"))
        self.dof_pos[env_ids] = torch.max(torch.min(self.dof_pos[env_ids], self.dof_pos_limits[:, 1]), self.dof_pos_limits[:, 0])
        self.dof_vel[env_ids] = 0.0
        self.prev_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.custom_dof_vel[env_ids] = 0.0
        self.filtered_custom_dof_vel[env_ids] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)
        )

    def _sample_pose_ids(self, env_ids):
        poses = self.cfg["getup"]["fallen_poses"]
        play_pose = self.cfg["getup"].get("play_pose", "random")
        if getattr(self, "is_play", False) and play_pose != "random":
            pose_names = [pose["name"] for pose in poses]
            if play_pose == "cycle":
                pose_ids = (torch.arange(len(env_ids), device=self.device) + self.play_reset_count) % len(poses)
                self.play_reset_count += len(env_ids)
                return pose_ids.long()
            if play_pose in pose_names:
                return torch.full((len(env_ids),), pose_names.index(play_pose), dtype=torch.long, device=self.device)

        weights = torch.tensor([float(pose.get("probability", 1.0)) for pose in poses], dtype=torch.float, device=self.device)
        weights = torch.clamp(weights, min=0.0)
        if float(weights.sum().item()) <= 0.0:
            weights[:] = 1.0
        return torch.multinomial(weights / weights.sum(), len(env_ids), replacement=True)

    def _reset_root_states(self, env_ids):
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :2] += self.env_origins[env_ids, :2]
        self.root_states[env_ids, :2] = apply_randomization(self.root_states[env_ids, :2], self.cfg["randomization"].get("init_base_pos_xy"))

        pose_ids = self._sample_pose_ids(env_ids)
        rolls = torch.zeros(len(env_ids), dtype=torch.float, device=self.device)
        pitches = torch.zeros(len(env_ids), dtype=torch.float, device=self.device)
        heights = torch.zeros(len(env_ids), dtype=torch.float, device=self.device)
        poses = self.cfg["getup"]["fallen_poses"]
        for pose_idx, pose in enumerate(poses):
            local_ids = (pose_ids == pose_idx).nonzero(as_tuple=False).flatten()
            if len(local_ids) == 0:
                continue
            rolls[local_ids] = self._sample_cfg_range(pose.get("roll", 0.0), len(local_ids))
            pitches[local_ids] = self._sample_cfg_range(pose.get("pitch", 0.0), len(local_ids))
            heights[local_ids] = self._sample_cfg_range(pose.get("height", 0.25), len(local_ids))

        yaws = torch.rand(len(env_ids), device=self.device) * (2 * torch.pi)
        self.root_states[env_ids, 2] = self.terrain.terrain_heights(self.root_states[env_ids, :2]) + heights
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(rolls, pitches, yaws)
        self.root_states[env_ids, 7:13] = 0.0
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _resample_commands(self):
        self.commands.zero_()
        self.gait_frequency.zero_()
        self.gait_process.zero_()
        max_episode_steps = int(np.ceil(self.cfg["rewards"]["episode_length_s"] / self.dt))
        self.cmd_resample_time[:] = max_episode_steps + 1

    def _kick_robots(self):
        if not bool(self.cfg["randomization"].get("enable_disturbances", False)):
            return
        super()._kick_robots()

    def _push_robots(self):
        if not bool(self.cfg["randomization"].get("enable_disturbances", False)):
            return
        super()._push_robots()

    def _check_termination(self):
        self.reset_buf = self.root_states[:, 7:13].square().sum(dim=-1) > self.cfg["rewards"]["terminate_vel"]
        self.time_out_buf = self.episode_length_buf > np.ceil(self.cfg["rewards"]["episode_length_s"] / self.dt)
        self.reset_buf |= self.time_out_buf

    def _base_height(self):
        return self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos)

    def _upright_error(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=-1)

    def _standing_mask(self):
        return (
            (self._base_height() > float(self.cfg["rewards"]["success_height"]))
            & (self._upright_error() < float(self.cfg["rewards"]["success_upright_error"]))
            & (torch.norm(self.root_states[:, 7:13], dim=-1) < float(self.cfg["rewards"]["success_vel"]))
        )

    def _reward_upright(self):
        return torch.exp(-self._upright_error() / float(self.cfg["rewards"]["upright_sigma"]))

    def _reward_stand_height(self):
        height_error = self._base_height() - float(self.cfg["rewards"]["stand_height_target"])
        return torch.exp(-torch.square(height_error) / float(self.cfg["rewards"]["stand_height_sigma"]))

    def _reward_stand_still(self):
        vel_error = torch.sum(torch.square(self.root_states[:, 7:13]), dim=-1)
        return torch.exp(-vel_error / float(self.cfg["rewards"]["stable_vel_sigma"]))

    def _reward_getup_success(self):
        return self._standing_mask().float()

    def _reward_dof_stand(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=-1)
