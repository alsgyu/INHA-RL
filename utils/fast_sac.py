from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

torch.set_float32_matmul_precision("high")


class EmpiricalNormalizer(nn.Module):
    """Running mean/std normalizer used by the FastSAC actor and critic."""

    def __init__(self, shape, device, eps=1.0e-2, until=None):
        super().__init__()
        self.eps = float(eps)
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape, device=device).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape, device=device).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape, device=device).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long, device=device))

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    @torch.no_grad()
    def forward(self, x, update=True, center=True):
        if self.training and update:
            self.update(x)
        if center:
            return (x - self._mean) / (self._std + self.eps)
        return x / (self._std + self.eps)

    @torch.no_grad()
    def update(self, x):
        if self.until is not None and self.count >= self.until:
            return
        batch_count = x.shape[0]
        if batch_count <= 0:
            return
        batch_mean = torch.mean(x, dim=0, keepdim=True)
        batch_var = torch.var(x, dim=0, keepdim=True, unbiased=False)
        new_count = self.count + batch_count

        delta = batch_mean - self._mean
        self._mean.copy_(self._mean + delta * (batch_count / new_count))
        delta2 = batch_mean - self._mean
        m_a = self._var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta2.pow(2) * (self.count * batch_count / new_count)
        self._var.copy_(m2 / new_count)
        self._std.copy_(self._var.sqrt())
        self.count.copy_(new_count)


class FastSACActor(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_dim=512,
        log_std_min=-5.0,
        log_std_max=0.0,
        use_tanh=True,
        use_layer_norm=True,
        action_scale=None,
        device=None,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.use_tanh = bool(use_tanh)

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim, device=device),
            nn.LayerNorm(hidden_dim, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.LayerNorm(hidden_dim // 2, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.LayerNorm(hidden_dim // 4, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim // 4, action_dim, device=device)
        self.fc_logstd = nn.Linear(hidden_dim // 4, action_dim, device=device)
        nn.init.constant_(self.fc_mu.weight, 0.0)
        nn.init.constant_(self.fc_mu.bias, 0.0)
        nn.init.constant_(self.fc_logstd.weight, 0.0)
        nn.init.constant_(self.fc_logstd.bias, 0.0)

        if action_scale is None:
            action_scale = torch.ones(action_dim, dtype=torch.float, device=device)
        self.register_buffer("action_scale", torch.as_tensor(action_scale, dtype=torch.float, device=device))

    def distribution_params(self, obs):
        x = self.net(obs)
        mean = self.fc_mu(x)
        log_std = torch.tanh(self.fc_logstd(x))
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1.0)
        return mean, log_std

    def forward(self, obs):
        mean, _ = self.distribution_params(obs)
        if not self.use_tanh:
            return mean
        return torch.tanh(mean) * self.action_scale

    def sample(self, obs):
        mean, log_std = self.distribution_params(obs)
        dist = torch.distributions.Normal(mean, log_std.exp())
        raw_action = dist.rsample()
        log_prob = dist.log_prob(raw_action)
        if self.use_tanh:
            action = torch.tanh(raw_action)
            log_prob = log_prob - torch.log(1.0 - action.pow(2) + 1.0e-6)
            log_prob = log_prob - torch.log(self.action_scale + 1.0e-6)
            action = action * self.action_scale
        else:
            action = raw_action
        return action, log_prob.sum(dim=-1)

    @torch.no_grad()
    def explore(self, obs, deterministic=False):
        if deterministic:
            return self(obs)
        action, _ = self.sample(obs)
        return action


class DistributionalQNetwork(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        num_atoms=101,
        v_min=-20.0,
        v_max=20.0,
        hidden_dim=768,
        use_layer_norm=True,
        device=None,
    ):
        super().__init__()
        self.num_atoms = int(num_atoms)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim, device=device),
            nn.LayerNorm(hidden_dim, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.LayerNorm(hidden_dim // 2, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.LayerNorm(hidden_dim // 4, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, num_atoms, device=device),
        )

    def forward(self, obs, action):
        return self.net(torch.cat((obs, action), dim=-1))

    def projection(self, obs, action, reward, bootstrap, discount, q_support):
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = reward.shape[0]
        target_z = reward.unsqueeze(1) + bootstrap.unsqueeze(1) * discount.unsqueeze(1) * q_support
        target_z = target_z.clamp(self.v_min, self.v_max)
        b = (target_z - self.v_min) / delta_z
        lower = torch.floor(b).long()
        upper = torch.ceil(b).long()

        same_bin = upper == lower
        lower_mask = (lower > 0) & same_bin
        upper_mask = (lower == 0) & same_bin
        lower = torch.where(lower_mask, lower - 1, lower)
        upper = torch.where(upper_mask, upper + 1, upper)
        lower = torch.clamp(lower, 0, self.num_atoms - 1)
        upper = torch.clamp(upper, 0, self.num_atoms - 1)

        next_dist = F.softmax(self(obs, action), dim=-1)
        proj_dist = torch.zeros_like(next_dist)
        offset = (
            torch.arange(batch_size, device=obs.device).unsqueeze(1) * self.num_atoms
        ).expand(batch_size, self.num_atoms)
        proj_dist.view(-1).index_add_(0, (lower + offset).reshape(-1), (next_dist * (upper.float() - b)).reshape(-1))
        proj_dist.view(-1).index_add_(0, (upper + offset).reshape(-1), (next_dist * (b - lower.float())).reshape(-1))
        return proj_dist


class FastSACCritic(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        num_atoms=101,
        v_min=-20.0,
        v_max=20.0,
        hidden_dim=768,
        use_layer_norm=True,
        num_q_networks=2,
        device=None,
    ):
        super().__init__()
        if num_q_networks < 1:
            raise ValueError("num_q_networks must be at least 1")
        self.num_q_networks = int(num_q_networks)
        self.qnets = nn.ModuleList(
            [
                DistributionalQNetwork(
                    obs_dim,
                    action_dim,
                    num_atoms=num_atoms,
                    v_min=v_min,
                    v_max=v_max,
                    hidden_dim=hidden_dim,
                    use_layer_norm=use_layer_norm,
                    device=device,
                )
                for _ in range(self.num_q_networks)
            ]
        )
        self.register_buffer("q_support", torch.linspace(v_min, v_max, num_atoms, device=device))

    def forward(self, obs, action):
        return torch.stack([qnet(obs, action) for qnet in self.qnets], dim=0)

    def projection(self, obs, action, reward, bootstrap, discount):
        return torch.stack(
            [
                qnet.projection(obs, action, reward, bootstrap, discount, self.q_support)
                for qnet in self.qnets
            ],
            dim=0,
        )

    def get_value(self, probs):
        return torch.sum(probs * self.q_support, dim=-1)


class FastSACReplayBuffer:
    """GPU circular replay with one ring per environment, matching Holosoma's layout."""

    def __init__(self, num_envs, buffer_size, obs_dim, action_dim, critic_obs_dim, device):
        self.num_envs = int(num_envs)
        self.buffer_size = int(buffer_size)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.critic_obs_dim = int(critic_obs_dim)
        self.device = device
        self.observations = torch.zeros(self.num_envs, self.buffer_size, self.obs_dim, device=device)
        self.actions = torch.zeros(self.num_envs, self.buffer_size, self.action_dim, device=device)
        self.rewards = torch.zeros(self.num_envs, self.buffer_size, device=device)
        self.dones = torch.zeros(self.num_envs, self.buffer_size, dtype=torch.bool, device=device)
        self.truncations = torch.zeros(self.num_envs, self.buffer_size, dtype=torch.bool, device=device)
        self.next_observations = torch.zeros(self.num_envs, self.buffer_size, self.obs_dim, device=device)
        self.critic_observations = torch.zeros(self.num_envs, self.buffer_size, self.critic_obs_dim, device=device)
        self.next_critic_observations = torch.zeros(self.num_envs, self.buffer_size, self.critic_obs_dim, device=device)
        self.ptr = 0

    def __len__(self):
        return min(self.ptr, self.buffer_size) * self.num_envs

    @property
    def valid_steps(self):
        return min(self.ptr, self.buffer_size)

    def extend(self, obs, action, reward, done, truncation, next_obs, critic_obs, next_critic_obs):
        slot = self.ptr % self.buffer_size
        self.observations[:, slot] = obs.detach()
        self.actions[:, slot] = action.detach()
        self.rewards[:, slot] = reward.detach()
        self.dones[:, slot] = done.detach().bool()
        self.truncations[:, slot] = truncation.detach().bool()
        self.next_observations[:, slot] = next_obs.detach()
        self.critic_observations[:, slot] = critic_obs.detach()
        self.next_critic_observations[:, slot] = next_critic_obs.detach()
        self.ptr += 1

    @torch.no_grad()
    def sample(self, per_env_batch_size):
        valid_steps = self.valid_steps
        if valid_steps <= 0:
            raise ValueError("Cannot sample from an empty FastSAC replay buffer")
        indices = torch.randint(0, valid_steps, (self.num_envs, int(per_env_batch_size)), device=self.device)
        obs_idx = indices.unsqueeze(-1).expand(-1, -1, self.obs_dim)
        act_idx = indices.unsqueeze(-1).expand(-1, -1, self.action_dim)
        critic_idx = indices.unsqueeze(-1).expand(-1, -1, self.critic_obs_dim)
        return {
            "observations": torch.gather(self.observations, 1, obs_idx).reshape(-1, self.obs_dim),
            "actions": torch.gather(self.actions, 1, act_idx).reshape(-1, self.action_dim),
            "rewards": torch.gather(self.rewards, 1, indices).reshape(-1),
            "dones": torch.gather(self.dones, 1, indices).reshape(-1),
            "truncations": torch.gather(self.truncations, 1, indices).reshape(-1),
            "next_observations": torch.gather(self.next_observations, 1, obs_idx).reshape(-1, self.obs_dim),
            "critic_observations": torch.gather(self.critic_observations, 1, critic_idx).reshape(-1, self.critic_obs_dim),
            "next_critic_observations": torch.gather(
                self.next_critic_observations,
                1,
                critic_idx,
            ).reshape(-1, self.critic_obs_dim),
            "effective_n_steps": torch.ones(self.num_envs * int(per_env_batch_size), device=self.device),
        }

    def state_dict(self):
        return {
            "ptr": self.ptr,
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "truncations": self.truncations,
            "next_observations": self.next_observations,
            "critic_observations": self.critic_observations,
            "next_critic_observations": self.next_critic_observations,
        }

    def load_state_dict(self, state):
        self.ptr = int(state.get("ptr", 0))
        for key in (
            "observations",
            "actions",
            "rewards",
            "dones",
            "truncations",
            "next_observations",
            "critic_observations",
            "next_critic_observations",
        ):
            if key in state:
                getattr(self, key).copy_(state[key].to(self.device))

    def export_npz(self, path, max_transitions=None):
        valid_steps = self.valid_steps
        if valid_steps <= 0:
            return 0
        tensors = {
            "obs": self.observations[:, :valid_steps].reshape(-1, self.obs_dim),
            "action": self.actions[:, :valid_steps].reshape(-1, self.action_dim),
            "reward": self.rewards[:, :valid_steps].reshape(-1),
            "done": self.dones[:, :valid_steps].reshape(-1).float(),
            "truncation": self.truncations[:, :valid_steps].reshape(-1).float(),
            "next_obs": self.next_observations[:, :valid_steps].reshape(-1, self.obs_dim),
            "critic_obs": self.critic_observations[:, :valid_steps].reshape(-1, self.critic_obs_dim),
            "next_critic_obs": self.next_critic_observations[:, :valid_steps].reshape(-1, self.critic_obs_dim),
        }
        total = tensors["reward"].shape[0]
        if max_transitions is not None and total > int(max_transitions):
            keep = torch.randperm(total, device=self.device)[: int(max_transitions)]
            tensors = {key: value[keep] for key, value in tensors.items()}
            total = int(max_transitions)
        arrays = {key: value.detach().cpu().numpy() for key, value in tensors.items()}
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(path, **arrays)
        return total

    def load_npz(self, path, max_transitions=None):
        data = np.load(path, allow_pickle=False)
        obs = torch.as_tensor(data["obs"], dtype=torch.float, device=self.device)
        action_key = "action" if "action" in data.files else "actions"
        action = torch.as_tensor(data[action_key], dtype=torch.float, device=self.device)
        reward = torch.as_tensor(data["reward"], dtype=torch.float, device=self.device).reshape(-1)
        done = torch.as_tensor(data["done"], dtype=torch.float, device=self.device).reshape(-1) > 0.5
        next_obs = torch.as_tensor(data["next_obs"], dtype=torch.float, device=self.device)
        if "truncation" in data.files:
            truncation = torch.as_tensor(data["truncation"], dtype=torch.float, device=self.device).reshape(-1) > 0.5
        else:
            truncation = torch.zeros_like(done)
        critic_obs = (
            torch.as_tensor(data["critic_obs"], dtype=torch.float, device=self.device)
            if "critic_obs" in data.files
            else obs
        )
        next_critic_obs = (
            torch.as_tensor(data["next_critic_obs"], dtype=torch.float, device=self.device)
            if "next_critic_obs" in data.files
            else next_obs
        )
        if critic_obs.shape[-1] < self.critic_obs_dim:
            padded = torch.zeros(critic_obs.shape[0], self.critic_obs_dim, dtype=torch.float, device=self.device)
            padded[:, : critic_obs.shape[-1]] = critic_obs
            critic_obs = padded
        elif critic_obs.shape[-1] > self.critic_obs_dim:
            critic_obs = critic_obs[:, : self.critic_obs_dim]
        if next_critic_obs.shape[-1] < self.critic_obs_dim:
            padded = torch.zeros(next_critic_obs.shape[0], self.critic_obs_dim, dtype=torch.float, device=self.device)
            padded[:, : next_critic_obs.shape[-1]] = next_critic_obs
            next_critic_obs = padded
        elif next_critic_obs.shape[-1] > self.critic_obs_dim:
            next_critic_obs = next_critic_obs[:, : self.critic_obs_dim]
        count = min(obs.shape[0], action.shape[0], reward.shape[0], done.shape[0], next_obs.shape[0])
        if max_transitions is not None:
            count = min(count, int(max_transitions))
        obs = obs[:count]
        action = action[:count]
        reward = reward[:count]
        done = done[:count]
        truncation = truncation[:count]
        next_obs = next_obs[:count]
        critic_obs = critic_obs[:count]
        next_critic_obs = next_critic_obs[:count]

        added = 0
        for start in range(0, count, self.num_envs):
            end = min(start + self.num_envs, count)
            size = end - start
            if size < self.num_envs:
                break
            self.extend(
                obs[start:end],
                action[start:end],
                reward[start:end],
                done[start:end],
                truncation[start:end],
                next_obs[start:end],
                critic_obs[start:end],
                next_critic_obs[start:end],
            )
            added += size
        return added


class FastSACPolicyWrapper(nn.Module):
    """Deployment wrapper with the same obs -> action interface as existing policies."""

    def __init__(self, actor, obs_normalizer: Optional[EmpiricalNormalizer] = None):
        super().__init__()
        self.actor = actor
        self.obs_normalizer = obs_normalizer

    def forward(self, obs):
        if self.obs_normalizer is not None:
            obs = self.obs_normalizer(obs, update=False)
        return self.actor(obs)
