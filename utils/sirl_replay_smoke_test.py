import os
import sys
import types

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


class TinyActorCritic(torch.nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, action_dim),
        )
        self.logstd = torch.nn.Parameter(torch.full((1, action_dim), -3.0))


def build_fake_sirl_worldmodel_runner(obs_dim, action_dim, privileged_dim, command_dim):
    sys.modules.setdefault("envs", types.ModuleType("envs"))
    sys.modules.setdefault("imageio", types.ModuleType("imageio"))
    recorder_module = types.ModuleType("utils.recorder")
    recorder_module.Recorder = type("Recorder", (), {})
    sys.modules.setdefault("utils.recorder", recorder_module)
    from utils.sirl_worldmodel_runner import SIRLWorldModelRunner

    runner = SIRLWorldModelRunner.__new__(SIRLWorldModelRunner)
    runner.device = "cpu"
    runner.cfg = {
        "algorithm": {"gamma": 0.99},
        "commands": {"adapter": {"stand_command_threshold": 0.04}},
        "normalization": {"lin_vel": 1.0, "ang_vel": 1.0},
    }
    runner.wm_cfg = {
        "world_model_learning_starts": 1,
        "world_model_batch_size": 32,
        "world_model_gradient_steps": 1,
        "world_model_obs_delta_coef": 1.0,
        "world_model_reward_coef": 0.5,
        "world_model_done_coef": 0.2,
        "world_model_velocity_coef": 0.25,
        "world_model_max_grad_norm": 1.0,
        "world_model_use_command": True,
        "world_model_command_dim": command_dim,
        "world_model_velocity_head_enabled": True,
        "world_model_velocity_dim": 3,
        "world_model_velocity_privileged_start": 4,
        "model_rollout_starts": 1,
        "model_rollout_min_wm_updates": 1,
        "model_rollout_max_wm_loss": 1.0e9,
        "model_rollout_batch_size": 32,
        "model_rollout_horizon": 1,
        "model_done_threshold": 1.1,
        "model_done_filter_threshold": 1.1,
        "model_delta_norm_max": 100.0,
        "model_delta_clip": 1.0,
        "model_reward_clip": 5.0,
        "model_uncertainty_filter_enabled": True,
        "model_uncertainty_quantile": 1.0,
        "model_uncertainty_weight_enabled": False,
        "model_low_speed_guard_enabled": False,
        "model_rollout_ensemble_samples": 2,
        "model_loss_weight": 0.35,
        "model_batch_ratio_start": 0.0,
        "model_batch_ratio_final": 0.5,
        "model_batch_ratio_warmup_iterations": 1,
        "model_batch_ratio_max": 0.5,
        "model_batch_ratio_min_accept_rate": 0.0,
        "action_clip": 1.0,
        "actor_mean_clip": 5.0,
        "log_std_min": -5.0,
        "log_std_max": -1.0,
        "parameter_abs_clip": 20.0,
    }
    runner.env = types.SimpleNamespace(
        num_obs=obs_dim,
        num_actions=action_dim,
        num_privileged_obs=privileged_dim,
    )
    runner.gamma = 0.99
    runner.tau = 0.005
    runner.action_clip = 1.0
    runner.actor_bound_limit = 1.0
    runner.actor_mean_clip = 5.0
    runner.reward_clip = None
    runner.target_q_clip = 100.0
    runner.q_value_clip = 100.0
    runner.parameter_abs_clip = 20.0
    runner.log_std_min = -5.0
    runner.log_std_max = -1.0
    runner.current_iteration = 10
    runner.use_privileged_q = False
    runner.q_privileged_dim = 0
    runner.replay_privileged_dim = privileged_dim
    runner.replay_command_dim = command_dim
    runner.world_model_command_dim = command_dim
    runner.world_model_velocity_dim = 3
    runner.world_model_velocity_privileged_start = 4
    runner._obs_public_command_scales = torch.ones(3)
    runner.last_model_rollout_accept_rate = 0.0
    runner.last_model_ratio_used = 0.0
    runner.last_update_model_count = 0
    runner.last_update_real_count = 0
    runner.world_model_updates = 0
    runner.last_world_model_loss = None

    runner.model = TinyActorCritic(obs_dim, action_dim)
    runner.world_models = torch.nn.ModuleList(
        [
            WorldModel(obs_dim, action_dim, hidden_dims=[32, 32], command_dim=command_dim, velocity_dim=3)
            for _ in range(2)
        ]
    )
    runner.world_model_optimizers = [
        torch.optim.Adam(model.parameters(), lr=1.0e-3)
        for model in runner.world_models
    ]
    runner.real_replay = OffPolicyReplayBuffer(
        obs_dim,
        action_dim,
        capacity=256,
        privileged_obs_dim=privileged_dim,
        command_dim=command_dim,
    )
    runner.model_replay = OffPolicyReplayBuffer(
        obs_dim,
        action_dim,
        capacity=256,
        privileged_obs_dim=privileged_dim,
        command_dim=command_dim,
    )
    runner.sirl_replay = OffPolicyReplayBuffer(obs_dim, action_dim, capacity=64, command_dim=command_dim)
    return runner


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

    fake_runner = build_fake_sirl_worldmodel_runner(obs_dim, action_dim, 14, command_dim)
    fake_runner.real_replay.add(
        obs,
        action,
        rewards,
        dones.float(),
        next_obs,
        privileged_obs=privileged,
        next_privileged_obs=next_privileged,
        command=commands,
        sample_weight=1.0,
    )
    wm_stats = fake_runner._train_world_model()
    assert wm_stats["world_model/loss"] >= 0.0
    assert fake_runner.world_model_updates == 1
    rollout_stats = fake_runner._rollout_world_model()
    assert rollout_stats["model_rollout/attempted"] > 0.0
    assert rollout_stats["model_rollout/added"] > 0.0
    update_batch = fake_runner._sample_update_batch()
    assert update_batch["obs"].shape[1] == obs_dim
    assert update_batch["action"].shape[1] == action_dim
    assert update_batch["sample_weight"].shape == update_batch["is_model"].shape
    assert update_batch["sample_weight"].ndim == 1
    assert bool((update_batch["is_model"] > 0.5).any().item())
    assert bool((update_batch["sample_weight"][update_batch["is_model"] > 0.5] <= 0.35 + 1.0e-6).all().item())
    print("sirl replay/world model smoke test passed")


if __name__ == "__main__":
    main()
