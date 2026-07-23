import os
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.models.WorldModel import QNetwork, WorldModel
from utils.sirl_replay import (
    OffPolicyReplayBuffer,
    SIRL_METRIC_KEYS,
    TrajectorySIRLReplayBuffer,
    WorldModelReplayBuffer,
    normalize_by_percentiles,
)


def main():
    torch.manual_seed(7)
    horizon = 8
    num_envs = 16
    obs_dim = 54
    action_dim = 12
    command_dim = 3

    obs = torch.randn(horizon, num_envs, obs_dim)
    action = torch.randn(horizon, num_envs, action_dim)
    next_obs = obs + 0.05 * torch.randn(horizon, num_envs, obs_dim)
    rewards = torch.randn(horizon, num_envs)
    dones = torch.zeros(horizon, num_envs, dtype=torch.bool)
    dones[-1, 0] = True
    time_outs = torch.zeros_like(dones)
    commands = torch.randn(horizon, num_envs, command_dim)
    metrics = {key: torch.rand(horizon, num_envs) for key in SIRL_METRIC_KEYS}

    returns = rewards.sum(dim=0)
    selected = torch.topk(returns, 4).indices
    bc_weights = normalize_by_percentiles(returns, 50, 90)

    trajectory_replay = TrajectorySIRLReplayBuffer(obs_dim, action_dim, capacity_transitions=128)
    added = trajectory_replay.add_segments(
        obs,
        action,
        rewards,
        dones,
        time_outs=time_outs,
        next_obses=next_obs,
        commands=commands,
        metrics=metrics,
        env_ids=selected,
        returns=returns,
        scores=returns,
        bc_weights=bc_weights,
    )
    assert added == 4
    assert trajectory_replay.trajectory_count == 4
    batch = trajectory_replay.sample_transitions(32, "cpu")
    assert batch["obs"].shape == (32, obs_dim)
    assert batch["action"].shape == (32, action_dim)
    assert batch["bc_weight"].shape == (32,)

    wm_replay = WorldModelReplayBuffer(obs_dim, action_dim, capacity_transitions=256, command_dim=command_dim)
    wm_replay.add(obs, action, rewards, dones, next_obs, commands=commands)
    wm_batch = wm_replay.sample(32, "cpu")
    world_model = WorldModel(obs_dim, action_dim, hidden_dims=[64, 64], command_dim=command_dim)
    pred = world_model(wm_batch["obs"], wm_batch["action"], wm_batch["command"])
    loss = (
        F.mse_loss(pred["delta_obs"], wm_batch["next_obs"] - wm_batch["obs"])
        + F.mse_loss(pred["reward"], wm_batch["reward"])
        + F.binary_cross_entropy_with_logits(pred["done_logit"], wm_batch["done"])
    )
    loss.backward()
    assert torch.isfinite(loss).item()

    offpolicy = OffPolicyReplayBuffer(obs_dim, action_dim, capacity=256, privileged_obs_dim=14, command_dim=command_dim)
    privileged = torch.randn(horizon, num_envs, 14)
    next_privileged = torch.randn(horizon, num_envs, 14)
    offpolicy.add(
        obs,
        action,
        rewards,
        dones,
        next_obs,
        privileged_obs=privileged,
        next_privileged_obs=next_privileged,
        command=commands,
        ret=rewards.flip(0).cumsum(0).flip(0),
        sirl_weight=torch.ones(horizon, num_envs),
    )
    off_batch = offpolicy.sample(32, "cpu")
    q_net = QNetwork(obs_dim, action_dim, privileged_obs_dim=14, hidden_dims=[64, 64])
    q_value = q_net(off_batch["obs"], off_batch["action"], off_batch["privileged_obs"])
    q_loss = F.mse_loss(q_value, off_batch["return"])
    q_loss.backward()
    assert q_value.shape == (32,)
    assert torch.isfinite(q_loss).item()
    print("sirl replay/world model smoke test passed")


if __name__ == "__main__":
    main()
