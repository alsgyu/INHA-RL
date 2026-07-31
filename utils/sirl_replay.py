from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


SIRL_METRIC_KEYS = (
    "velocity_error",
    "feet_slip",
    "swing_clearance_deficit",
    "low_speed_overspeed",
    "low_speed_lateral_instability",
    "swing_edge_contact",
    "swing_contact",
    "swing_pitch_asymmetry",
    "contact_duty_asymmetry",
    "right_contact_duty_excess",
    "forward_push_asymmetry",
    "right_forward_push_excess",
    "forward_pitch_push",
    "right_forward_pitch_push",
    "forward_pitch_excess",
    "move_start_forward_pitch",
    "move_start_ankle_tracking",
    "move_start_yaw_drift",
    "base_tilt",
    "low_height",
    "stand_drift",
    "stop_transition_drift",
)


def percentile(values, percentile_value):
    if values.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float, device=values.device)
    return torch.quantile(values.float().reshape(-1), float(percentile_value) / 100.0)


def normalize_by_percentiles(values, low_percentile=50.0, high_percentile=90.0, eps=1.0e-6):
    low = percentile(values, low_percentile)
    high = percentile(values, high_percentile)
    return torch.clamp((values.float() - low) / (high - low + eps), min=0.0, max=1.0)


@dataclass
class TrajectorySegment:
    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    next_obs: Optional[torch.Tensor]
    command: Optional[torch.Tensor]
    metrics: Dict[str, torch.Tensor]
    return_value: float
    length: int
    terminal: bool
    fall: bool
    score: float
    bc_weight: float
    model_error: Optional[float] = None
    model_done_prob: Optional[float] = None


class TrajectorySIRLReplayBuffer:
    """CPU trajectory replay used for trajectory-return SIRL imitation."""

    def __init__(self, obs_dim, action_dim, capacity_transitions=131072):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.capacity_transitions = max(1, int(capacity_transitions))
        self.trajectories: List[TrajectorySegment] = []
        self.transition_count = 0
        self._flat_cache = None

    @property
    def trajectory_count(self):
        return len(self.trajectories)

    def __len__(self):
        return self.transition_count

    def clear(self):
        self.trajectories = []
        self.transition_count = 0
        self._flat_cache = None

    def _append(self, trajectory):
        self.trajectories.append(trajectory)
        self.transition_count += trajectory.length
        self._flat_cache = None
        while self.transition_count > self.capacity_transitions and len(self.trajectories) > 1:
            removed = self.trajectories.pop(0)
            self.transition_count -= removed.length

    def add_segments(
        self,
        obses,
        actions,
        rewards,
        dones,
        time_outs=None,
        next_obses=None,
        commands=None,
        metrics=None,
        env_ids=None,
        returns=None,
        scores=None,
        bc_weights=None,
        model_errors=None,
        model_done_probs=None,
    ):
        if env_ids is None:
            env_ids = torch.arange(obses.shape[1], device=obses.device)
        if env_ids.numel() == 0:
            return 0

        env_ids = env_ids.detach().long()
        obses_cpu = obses[:, env_ids, :].detach().float().cpu()
        actions_cpu = actions[:, env_ids, :].detach().float().cpu()
        rewards_cpu = rewards[:, env_ids].detach().float().cpu()
        dones_cpu = dones[:, env_ids].detach().bool().cpu()
        time_outs_cpu = None
        if time_outs is not None:
            time_outs_cpu = time_outs[:, env_ids].detach().bool().cpu()
        next_obses_cpu = None
        if next_obses is not None:
            next_obses_cpu = next_obses[:, env_ids, :].detach().float().cpu()
        commands_cpu = None
        if commands is not None:
            commands_cpu = commands[:, env_ids, :].detach().float().cpu()

        metrics_cpu = {}
        if metrics:
            for key, value in metrics.items():
                metrics_cpu[key] = value[:, env_ids].detach().float().cpu()

        selected_count = int(env_ids.numel())
        returns_cpu = (
            returns[env_ids].detach().float().cpu()
            if returns is not None
            else rewards[:, env_ids].detach().float().sum(dim=0).cpu()
        )
        scores_cpu = (
            scores[env_ids].detach().float().cpu()
            if scores is not None
            else torch.zeros(selected_count, dtype=torch.float)
        )
        weights_cpu = (
            bc_weights[env_ids].detach().float().cpu()
            if bc_weights is not None
            else torch.ones(selected_count, dtype=torch.float)
        )
        model_errors_cpu = model_errors[env_ids].detach().float().cpu() if model_errors is not None else None
        model_done_cpu = model_done_probs[env_ids].detach().float().cpu() if model_done_probs is not None else None

        for i in range(selected_count):
            done_i = dones_cpu[:, i]
            timeout_i = torch.zeros_like(done_i) if time_outs_cpu is None else time_outs_cpu[:, i]
            metric_i = {key: value[:, i] for key, value in metrics_cpu.items()}
            length = int(obses_cpu.shape[0])
            trajectory = TrajectorySegment(
                obs=obses_cpu[:, i, :],
                action=actions_cpu[:, i, :],
                reward=rewards_cpu[:, i],
                done=done_i,
                next_obs=None if next_obses_cpu is None else next_obses_cpu[:, i, :],
                command=None if commands_cpu is None else commands_cpu[:, i, :],
                metrics=metric_i,
                return_value=float(returns_cpu[i].item()),
                length=length,
                terminal=bool(done_i.any().item()),
                fall=bool((done_i & ~timeout_i).any().item()),
                score=float(scores_cpu[i].item()),
                bc_weight=float(weights_cpu[i].item()),
                model_error=None if model_errors_cpu is None else float(model_errors_cpu[i].item()),
                model_done_prob=None if model_done_cpu is None else float(model_done_cpu[i].item()),
            )
            self._append(trajectory)
        return selected_count

    def _ensure_flat_cache(self):
        if self._flat_cache is not None:
            return self._flat_cache
        if not self.trajectories:
            self._flat_cache = {}
            return self._flat_cache

        obs = torch.cat([trajectory.obs for trajectory in self.trajectories], dim=0)
        actions = torch.cat([trajectory.action for trajectory in self.trajectories], dim=0)
        weights = torch.cat(
            [
                torch.full((trajectory.length,), float(trajectory.bc_weight), dtype=torch.float)
                for trajectory in self.trajectories
            ],
            dim=0,
        )
        returns = torch.cat(
            [
                torch.full((trajectory.length,), float(trajectory.return_value), dtype=torch.float)
                for trajectory in self.trajectories
            ],
            dim=0,
        )
        cache = {
            "obs": obs,
            "action": actions,
            "bc_weight": weights,
            "return": returns,
        }
        if all(trajectory.next_obs is not None for trajectory in self.trajectories):
            cache["next_obs"] = torch.cat([trajectory.next_obs for trajectory in self.trajectories], dim=0)
        if all(trajectory.command is not None for trajectory in self.trajectories):
            cache["command"] = torch.cat([trajectory.command for trajectory in self.trajectories], dim=0)
        self._flat_cache = cache
        return cache

    def sample_transitions(self, batch_size, device):
        if self.transition_count <= 0:
            raise ValueError("Cannot sample from an empty SIRL replay buffer")
        cache = self._ensure_flat_cache()
        batch_size = min(int(batch_size), self.transition_count)
        indices = torch.randint(0, self.transition_count, (batch_size,), device="cpu")
        batch = {}
        for key, value in cache.items():
            batch[key] = value[indices].to(device)
        return batch

    def stats(self):
        stats = {
            "sirl/buffer_trajectories": float(self.trajectory_count),
            "sirl/buffer_transitions": float(self.transition_count),
            "sirl/buffer_count": float(self.transition_count),
        }
        if not self.trajectories:
            stats.update(
                {
                    "sirl/buffer_return_mean": 0.0,
                    "sirl/buffer_return_p50": 0.0,
                    "sirl/buffer_return_p90": 0.0,
                }
            )
            return stats
        returns = torch.tensor([trajectory.return_value for trajectory in self.trajectories], dtype=torch.float)
        stats.update(
            {
                "sirl/buffer_return_mean": float(returns.mean().item()),
                "sirl/buffer_return_p50": float(percentile(returns, 50).item()),
                "sirl/buffer_return_p90": float(percentile(returns, 90).item()),
            }
        )
        return stats


class WorldModelReplayBuffer:
    """CPU transition replay for auxiliary world-model training."""

    def __init__(self, obs_dim, action_dim, capacity_transitions=1000000, command_dim=0):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.command_dim = int(command_dim)
        self.capacity = max(1, int(capacity_transitions))
        self.obs = torch.zeros(self.capacity, self.obs_dim, dtype=torch.float)
        self.action = torch.zeros(self.capacity, self.action_dim, dtype=torch.float)
        self.next_obs = torch.zeros(self.capacity, self.obs_dim, dtype=torch.float)
        self.reward = torch.zeros(self.capacity, dtype=torch.float)
        self.done = torch.zeros(self.capacity, dtype=torch.float)
        self.command = None
        if self.command_dim > 0:
            self.command = torch.zeros(self.capacity, self.command_dim, dtype=torch.float)
        self.count = 0
        self.write_idx = 0

    def __len__(self):
        return self.count

    def _write_range(self, storage, values):
        size = int(values.shape[0])
        first = min(size, self.capacity - self.write_idx)
        storage[self.write_idx : self.write_idx + first] = values[:first]
        remaining = size - first
        if remaining > 0:
            storage[:remaining] = values[first:]

    def add(self, obses, actions, rewards, dones, next_obses, commands=None):
        obs = obses.reshape(-1, self.obs_dim).detach().float().cpu()
        action = actions.reshape(-1, self.action_dim).detach().float().cpu()
        reward = rewards.reshape(-1).detach().float().cpu()
        done = dones.reshape(-1).detach().float().cpu()
        next_obs = next_obses.reshape(-1, self.obs_dim).detach().float().cpu()
        command = None
        if self.command is not None and commands is not None:
            command = commands.reshape(-1, self.command_dim).detach().float().cpu()

        size = int(obs.shape[0])
        if size == 0:
            return 0
        if size >= self.capacity:
            self.obs[:] = obs[-self.capacity :]
            self.action[:] = action[-self.capacity :]
            self.reward[:] = reward[-self.capacity :]
            self.done[:] = done[-self.capacity :]
            self.next_obs[:] = next_obs[-self.capacity :]
            if self.command is not None and command is not None:
                self.command[:] = command[-self.capacity :]
            self.count = self.capacity
            self.write_idx = 0
            return size

        self._write_range(self.obs, obs)
        self._write_range(self.action, action)
        self._write_range(self.reward, reward)
        self._write_range(self.done, done)
        self._write_range(self.next_obs, next_obs)
        if self.command is not None and command is not None:
            self._write_range(self.command, command)
        self.write_idx = (self.write_idx + size) % self.capacity
        self.count = min(self.capacity, self.count + size)
        return size

    def sample(self, batch_size, device):
        if self.count <= 0:
            raise ValueError("Cannot sample from an empty world model replay buffer")
        batch_size = min(int(batch_size), self.count)
        indices = torch.randint(0, self.count, (batch_size,), device="cpu")
        batch = {
            "obs": self.obs[indices].to(device),
            "action": self.action[indices].to(device),
            "reward": self.reward[indices].to(device),
            "done": self.done[indices].to(device),
            "next_obs": self.next_obs[indices].to(device),
        }
        if self.command is not None:
            batch["command"] = self.command[indices].to(device)
        return batch


class OffPolicyReplayBuffer:
    """CPU transition replay for off-policy actor-critic and model rollouts."""

    def __init__(self, obs_dim, action_dim, capacity, privileged_obs_dim=0, command_dim=0):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.privileged_obs_dim = int(privileged_obs_dim)
        self.command_dim = int(command_dim)
        self.capacity = max(1, int(capacity))
        self.obs = torch.zeros(self.capacity, self.obs_dim, dtype=torch.float)
        self.action = torch.zeros(self.capacity, self.action_dim, dtype=torch.float)
        self.reward = torch.zeros(self.capacity, dtype=torch.float)
        self.done = torch.zeros(self.capacity, dtype=torch.float)
        self.next_obs = torch.zeros(self.capacity, self.obs_dim, dtype=torch.float)
        self.ret = torch.zeros(self.capacity, dtype=torch.float)
        self.sirl_weight = torch.zeros(self.capacity, dtype=torch.float)
        self.is_model = torch.zeros(self.capacity, dtype=torch.float)
        self.sample_weight = torch.ones(self.capacity, dtype=torch.float)
        self.privileged_obs = None
        self.next_privileged_obs = None
        self.command = None
        if self.privileged_obs_dim > 0:
            self.privileged_obs = torch.zeros(self.capacity, self.privileged_obs_dim, dtype=torch.float)
            self.next_privileged_obs = torch.zeros(self.capacity, self.privileged_obs_dim, dtype=torch.float)
        if self.command_dim > 0:
            self.command = torch.zeros(self.capacity, self.command_dim, dtype=torch.float)
        self.count = 0
        self.write_idx = 0

    def __len__(self):
        return self.count

    def _write_range(self, storage, values):
        size = int(values.shape[0])
        first = min(size, self.capacity - self.write_idx)
        storage[self.write_idx : self.write_idx + first] = values[:first]
        remaining = size - first
        if remaining > 0:
            storage[:remaining] = values[first:]

    def add(
        self,
        obs,
        action,
        reward,
        done,
        next_obs,
        privileged_obs=None,
        next_privileged_obs=None,
        command=None,
        ret=None,
        sirl_weight=None,
        is_model=0.0,
        sample_weight=None,
    ):
        obs = obs.reshape(-1, self.obs_dim).detach().float().cpu()
        action = action.reshape(-1, self.action_dim).detach().float().cpu()
        reward = reward.reshape(-1).detach().float().cpu()
        done = done.reshape(-1).detach().float().cpu()
        next_obs = next_obs.reshape(-1, self.obs_dim).detach().float().cpu()
        size = int(obs.shape[0])
        if size == 0:
            return 0
        if ret is None:
            ret = torch.zeros(size, dtype=torch.float)
        else:
            ret = ret.reshape(-1).detach().float().cpu()
        if sirl_weight is None:
            sirl_weight = torch.zeros(size, dtype=torch.float)
        else:
            sirl_weight = sirl_weight.reshape(-1).detach().float().cpu()
        if torch.is_tensor(is_model):
            is_model = is_model.reshape(-1).detach().float().cpu()
            if is_model.numel() == 1 and size > 1:
                is_model = is_model.expand(size).clone()
        else:
            is_model = torch.full((size,), float(is_model), dtype=torch.float)
        if sample_weight is None:
            sample_weight = torch.ones(size, dtype=torch.float)
        elif torch.is_tensor(sample_weight):
            sample_weight = sample_weight.reshape(-1).detach().float().cpu()
            if sample_weight.numel() == 1 and size > 1:
                sample_weight = sample_weight.expand(size).clone()
        else:
            sample_weight = torch.full((size,), float(sample_weight), dtype=torch.float)

        privileged_obs_cpu = None
        next_privileged_obs_cpu = None
        command_cpu = None
        if self.privileged_obs is not None:
            if privileged_obs is None:
                privileged_obs_cpu = torch.zeros(size, self.privileged_obs_dim, dtype=torch.float)
            else:
                privileged_obs_cpu = privileged_obs.reshape(-1, self.privileged_obs_dim).detach().float().cpu()
            if next_privileged_obs is None:
                next_privileged_obs_cpu = torch.zeros(size, self.privileged_obs_dim, dtype=torch.float)
            else:
                next_privileged_obs_cpu = next_privileged_obs.reshape(-1, self.privileged_obs_dim).detach().float().cpu()
        if self.command is not None:
            if command is None:
                command_cpu = torch.zeros(size, self.command_dim, dtype=torch.float)
            else:
                command_cpu = command.reshape(size, -1).detach().float().cpu()
                if command_cpu.shape[-1] < self.command_dim:
                    padded_command = torch.zeros(size, self.command_dim, dtype=torch.float)
                    padded_command[:, : command_cpu.shape[-1]] = command_cpu
                    command_cpu = padded_command
                elif command_cpu.shape[-1] > self.command_dim:
                    command_cpu = command_cpu[:, : self.command_dim]

        if size >= self.capacity:
            obs = obs[-self.capacity :]
            action = action[-self.capacity :]
            reward = reward[-self.capacity :]
            done = done[-self.capacity :]
            next_obs = next_obs[-self.capacity :]
            ret = ret[-self.capacity :]
            sirl_weight = sirl_weight[-self.capacity :]
            is_model = is_model[-self.capacity :]
            sample_weight = sample_weight[-self.capacity :]
            self.obs[:] = obs
            self.action[:] = action
            self.reward[:] = reward
            self.done[:] = done
            self.next_obs[:] = next_obs
            self.ret[:] = ret
            self.sirl_weight[:] = sirl_weight
            self.is_model[:] = is_model
            self.sample_weight[:] = sample_weight
            if self.privileged_obs is not None:
                self.privileged_obs[:] = privileged_obs_cpu[-self.capacity :]
                self.next_privileged_obs[:] = next_privileged_obs_cpu[-self.capacity :]
            if self.command is not None:
                self.command[:] = command_cpu[-self.capacity :]
            self.count = self.capacity
            self.write_idx = 0
            return size

        self._write_range(self.obs, obs)
        self._write_range(self.action, action)
        self._write_range(self.reward, reward)
        self._write_range(self.done, done)
        self._write_range(self.next_obs, next_obs)
        self._write_range(self.ret, ret)
        self._write_range(self.sirl_weight, sirl_weight)
        self._write_range(self.is_model, is_model)
        self._write_range(self.sample_weight, sample_weight)
        if self.privileged_obs is not None:
            self._write_range(self.privileged_obs, privileged_obs_cpu)
            self._write_range(self.next_privileged_obs, next_privileged_obs_cpu)
        if self.command is not None:
            self._write_range(self.command, command_cpu)
        self.write_idx = (self.write_idx + size) % self.capacity
        self.count = min(self.capacity, self.count + size)
        return size

    def sample(self, batch_size, device):
        if self.count <= 0:
            raise ValueError("Cannot sample from an empty off-policy replay buffer")
        batch_size = min(int(batch_size), self.count)
        indices = torch.randint(0, self.count, (batch_size,), device="cpu")
        batch = {
            "obs": self.obs[indices].to(device),
            "action": self.action[indices].to(device),
            "reward": self.reward[indices].to(device),
            "done": self.done[indices].to(device),
            "next_obs": self.next_obs[indices].to(device),
            "return": self.ret[indices].to(device),
            "sirl_weight": self.sirl_weight[indices].to(device),
            "is_model": self.is_model[indices].to(device),
            "sample_weight": self.sample_weight[indices].to(device),
        }
        if self.privileged_obs is not None:
            batch["privileged_obs"] = self.privileged_obs[indices].to(device)
            batch["next_privileged_obs"] = self.next_privileged_obs[indices].to(device)
        if self.command is not None:
            batch["command"] = self.command[indices].to(device)
        return batch
