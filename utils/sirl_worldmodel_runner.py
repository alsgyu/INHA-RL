import argparse
import copy
import glob
import os
import random
import signal
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F

from envs import *
from utils.models.WorldModel import QNetwork, WorldModel
from utils.recorder import Recorder
from utils.runner import get_model_class, get_task_class, load_config
from utils.sirl_replay import OffPolicyReplayBuffer, SIRL_METRIC_KEYS, normalize_by_percentiles, percentile


class SIRLWorldModelRunner:
    """Off-policy SIRL learner with SAC-style updates and short world-model rollouts."""

    def __init__(self, test=False):
        self.test = test
        self._get_args()
        self._update_cfg_from_args()
        self._set_seed()

        task_name = self.cfg["basic"]["task"]
        if "/" in task_name:
            task_name = task_name.split("/")[-1]
        task_class = get_task_class(task_name)
        if task_class is None:
            raise ValueError(f"Unknown task: {task_name}. Could not find a matching env class.")
        self.env = task_class(self.cfg)
        self.env.is_play = test

        self.device = self.cfg["basic"]["rl_device"]
        self.wm_cfg = self.cfg.get("algorithm", {}).get("sirl_worldmodel", {})
        self.collect_steps = int(self.wm_cfg.get("collect_steps_per_iter", 16))
        self.gamma = float(self.wm_cfg.get("gamma", self.cfg["algorithm"].get("gamma", 0.99)))
        self.tau = float(self.wm_cfg.get("target_tau", 0.005))
        self.action_clip = self.wm_cfg.get("action_clip", None)
        self.log_std_min = float(self.wm_cfg.get("log_std_min", -5.0))
        self.log_std_max = float(self.wm_cfg.get("log_std_max", -1.0))
        self.use_privileged_q = bool(self.wm_cfg.get("use_privileged_q", False))
        self.q_privileged_dim = self.env.num_privileged_obs if self.use_privileged_q else 0

        model_name = self.cfg["basic"].get("model", "BaseActorCritic")
        model_class = get_model_class(model_name)
        self.model = model_class(self.env.num_actions, self.env.num_obs, self.env.num_privileged_obs).to(self.device)
        q_hidden_dims = self.wm_cfg.get("q_hidden_dims", [512, 512])
        self.q1 = QNetwork(self.env.num_obs, self.env.num_actions, self.q_privileged_dim, q_hidden_dims).to(self.device)
        self.q2 = QNetwork(self.env.num_obs, self.env.num_actions, self.q_privileged_dim, q_hidden_dims).to(self.device)
        self.q1_target = copy.deepcopy(self.q1).to(self.device)
        self.q2_target = copy.deepcopy(self.q2).to(self.device)
        for target in (self.q1_target, self.q2_target):
            target.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(
            list(self.model.actor.parameters()) + [self.model.logstd],
            lr=float(self.wm_cfg.get("actor_lr", 3.0e-4)),
        )
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=float(self.wm_cfg.get("critic_lr", 3.0e-4)),
        )
        self.auto_alpha = bool(self.wm_cfg.get("auto_alpha", True))
        alpha_init = float(self.wm_cfg.get("alpha", 0.05))
        self.log_alpha = torch.tensor(np.log(alpha_init), dtype=torch.float, device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=float(self.wm_cfg.get("alpha_lr", 3.0e-4)))
        self.target_entropy = float(self.wm_cfg.get("target_entropy", -self.env.num_actions))

        self.world_models = torch.nn.ModuleList()
        self.world_model_optimizers = []
        self.world_model_command_dim = int(self.wm_cfg.get("world_model_command_dim", 3)) if bool(self.wm_cfg.get("world_model_use_command", False)) else 0
        ensemble_size = max(1, int(self.wm_cfg.get("world_model_ensemble_size", 5)))
        for _ in range(ensemble_size):
            model = WorldModel(
                self.env.num_obs,
                self.env.num_actions,
                hidden_dims=self.wm_cfg.get("world_model_hidden_dims", [512, 512]),
                command_dim=self.world_model_command_dim,
            ).to(self.device)
            self.world_models.append(model)
            self.world_model_optimizers.append(
                torch.optim.Adam(model.parameters(), lr=float(self.wm_cfg.get("world_model_lr", 1.0e-4)))
            )

        real_capacity = int(self.wm_cfg.get("real_buffer_size", 1000000))
        model_capacity = int(self.wm_cfg.get("model_buffer_size", 250000))
        sirl_capacity = int(self.wm_cfg.get("sirl_buffer_size", 250000))
        self.replay_command_dim = max(3, self.world_model_command_dim)
        self.real_replay = OffPolicyReplayBuffer(
            self.env.num_obs,
            self.env.num_actions,
            real_capacity,
            privileged_obs_dim=self.q_privileged_dim,
            command_dim=self.replay_command_dim,
        )
        self.model_replay = OffPolicyReplayBuffer(
            self.env.num_obs,
            self.env.num_actions,
            model_capacity,
            privileged_obs_dim=self.q_privileged_dim,
            command_dim=self.replay_command_dim,
        )
        self.sirl_replay = OffPolicyReplayBuffer(
            self.env.num_obs,
            self.env.num_actions,
            sirl_capacity,
            privileged_obs_dim=self.q_privileged_dim,
            command_dim=self.replay_command_dim,
        )
        self._load()

    def _get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", required=True, type=str, help="Name of the task to run.")
        parser.add_argument("--checkpoint", type=str, help="Path of the checkpoint to load. Use -1 for latest.")
        parser.add_argument("--num_envs", type=int, help="Number of environments.")
        parser.add_argument("--headless", type=bool, help="Run headless.")
        parser.add_argument("--sim_device", type=str, help="IsaacGym sim device.")
        parser.add_argument("--rl_device", type=str, help="Learner device.")
        parser.add_argument("--seed", type=int, help="Random seed.")
        parser.add_argument("--max_iterations", type=int, help="Training iterations.")
        parser.add_argument("--model", type=str, help="Actor model class name.")
        parser.add_argument("--disable_record_video", action="store_true", help="Disable video recording in training.")
        self.args = parser.parse_args()

    def _update_cfg_from_args(self):
        cfg_file = os.path.join("envs", "{}.yaml".format(self.args.task))
        self.cfg = load_config(cfg_file)
        if "model" not in self.cfg.get("basic", {}):
            self.cfg.setdefault("basic", {})["model"] = "BaseActorCritic"
        self.cfg.setdefault("algorithm", {}).setdefault("sirl_worldmodel", {})
        for arg in vars(self.args):
            value = getattr(self.args, arg)
            if value is None:
                continue
            if arg == "num_envs":
                self.cfg["env"][arg] = value
            elif arg == "task":
                continue
            elif arg == "disable_record_video":
                if value:
                    self.cfg["viewer"]["record_video"] = False
            else:
                self.cfg["basic"][arg] = value
        if not self.test:
            self.cfg["viewer"]["record_video"] = False

    def _set_seed(self):
        if self.cfg["basic"]["seed"] == -1:
            self.cfg["basic"]["seed"] = np.random.randint(0, 10000)
        print("Setting seed: {}".format(self.cfg["basic"]["seed"]))
        random.seed(self.cfg["basic"]["seed"])
        np.random.seed(self.cfg["basic"]["seed"])
        torch.manual_seed(self.cfg["basic"]["seed"])
        os.environ["PYTHONHASHSEED"] = str(self.cfg["basic"]["seed"])
        torch.cuda.manual_seed(self.cfg["basic"]["seed"])
        torch.cuda.manual_seed_all(self.cfg["basic"]["seed"])

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    def _policy_dist(self, obs):
        action_mean = self.model.actor(obs)
        log_std = torch.clamp(self.model.logstd, min=self.log_std_min, max=self.log_std_max).expand_as(action_mean)
        return torch.distributions.Normal(action_mean, torch.exp(log_std))

    def _policy_action(self, obs, deterministic=False):
        dist = self._policy_dist(obs)
        action = dist.loc if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        if self.action_clip is not None:
            action = torch.clamp(action, -float(self.action_clip), float(self.action_clip))
        return action, log_prob, dist

    def _random_action(self, shape):
        scale = float(self.wm_cfg.get("random_action_scale", 1.0))
        return torch.empty(shape, device=self.device).uniform_(-scale, scale)

    def _q_input_privileged(self, batch, next_obs=False):
        if not self.use_privileged_q:
            return None
        key = "next_privileged_obs" if next_obs else "privileged_obs"
        return batch.get(key)

    def _soft_update_targets(self):
        with torch.no_grad():
            for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)
            for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)

    def _compute_segment_returns(self, rewards, dones):
        returns = torch.zeros_like(rewards)
        running = torch.zeros(rewards.shape[1], dtype=torch.float, device=rewards.device)
        for t in reversed(range(rewards.shape[0])):
            running = rewards[t] + self.gamma * running * (1.0 - dones[t].float())
            returns[t] = running
        return returns

    def _collect_command(self):
        if hasattr(self.env, "commands"):
            command = self.env.commands[:, : self.replay_command_dim]
        else:
            command = torch.zeros(self.env.num_envs, self.replay_command_dim, dtype=torch.float, device=self.device)
        return command.to(self.device).float()

    def _collect_rollout(self, obs, privileged_obs, iteration):
        obs_list = []
        action_list = []
        reward_list = []
        done_list = []
        next_obs_list = []
        privileged_list = []
        next_privileged_list = []
        command_list = []
        time_out_list = []
        metric_buffers = {key: [] for key in SIRL_METRIC_KEYS}
        reward_sum = 0.0
        done_count = 0
        start_random_steps = int(self.wm_cfg.get("start_random_steps", 0))

        for step in range(self.collect_steps):
            obs_list.append(obs.detach())
            privileged_list.append(privileged_obs.detach())
            command_list.append(self._collect_command())
            with torch.no_grad():
                if len(self.real_replay) < start_random_steps:
                    action = self._random_action((self.env.num_envs, self.env.num_actions))
                else:
                    action, _, _ = self._policy_action(obs, deterministic=False)
            next_obs, reward, done, infos = self.env.step(action)
            next_obs = next_obs.to(self.device)
            reward = reward.to(self.device)
            done = done.to(self.device)
            next_privileged_obs = infos["privileged_obs"].to(self.device)
            action_list.append(action.detach())
            reward_list.append(reward.detach())
            done_list.append(done.detach())
            next_obs_list.append(next_obs.detach())
            next_privileged_list.append(next_privileged_obs.detach())
            time_out_list.append(infos["time_outs"].to(self.device).detach())
            for key in SIRL_METRIC_KEYS:
                value = infos.get("sirl", {}).get(key)
                if value is None:
                    value = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
                metric_buffers[key].append(value.to(self.device).float().detach())
            reward_sum += float(reward.mean().item())
            done_count += int(done.sum().item())
            obs = next_obs
            privileged_obs = next_privileged_obs

        rollout = {
            "obs": torch.stack(obs_list, dim=0),
            "action": torch.stack(action_list, dim=0),
            "reward": torch.stack(reward_list, dim=0),
            "done": torch.stack(done_list, dim=0),
            "next_obs": torch.stack(next_obs_list, dim=0),
            "privileged_obs": torch.stack(privileged_list, dim=0),
            "next_privileged_obs": torch.stack(next_privileged_list, dim=0),
            "command": torch.stack(command_list, dim=0),
            "time_out": torch.stack(time_out_list, dim=0),
            "metrics": {key: torch.stack(values, dim=0) for key, values in metric_buffers.items()},
        }
        stats = {
            "rollout/reward_mean": reward_sum / max(self.collect_steps, 1),
            "rollout/done_rate": done_count / max(self.collect_steps * self.env.num_envs, 1),
        }
        return obs, privileged_obs, rollout, stats

    def _score_and_store_rollout(self, rollout):
        returns = self._compute_segment_returns(rollout["reward"], rollout["done"])
        segment_return = rollout["reward"].sum(dim=0)
        normalized_return = normalize_by_percentiles(segment_return, 50, 90)
        score = normalized_return.clone()
        penalties = self.wm_cfg.get("sirl_metric_penalties", {})
        for key, penalty in penalties.items():
            metric = rollout["metrics"].get(key)
            if metric is None:
                continue
            score = score - float(penalty) * normalize_by_percentiles(metric.mean(dim=0), 50, 90)
        fall = (rollout["done"].bool() & ~rollout["time_out"].bool()).any(dim=0).float()
        score = score - float(self.wm_cfg.get("sirl_fall_penalty", 2.0)) * fall

        min_percentile = float(self.wm_cfg.get("sirl_min_return_percentile", 60))
        eligible = segment_return >= percentile(segment_return, min_percentile)
        if int(eligible.sum().item()) == 0:
            elite_env_ids = torch.empty(0, dtype=torch.long, device=self.device)
        else:
            top_fraction = float(self.wm_cfg.get("sirl_top_fraction", 0.15))
            top_k = max(1, int(np.ceil(top_fraction * int(eligible.sum().item()))))
            masked_score = torch.where(eligible, score, torch.full_like(score, -torch.inf))
            _, elite_env_ids = torch.topk(masked_score, top_k)

        self.real_replay.add(
            rollout["obs"],
            rollout["action"],
            rollout["reward"],
            rollout["done"],
            rollout["next_obs"],
            privileged_obs=rollout["privileged_obs"] if self.use_privileged_q else None,
            next_privileged_obs=rollout["next_privileged_obs"] if self.use_privileged_q else None,
            command=rollout["command"],
            ret=returns,
            sirl_weight=torch.zeros_like(returns),
            is_model=0.0,
        )

        stats = {
            "sirl/trajectory_return_mean": float(segment_return.mean().item()),
            "sirl/trajectory_return_p50": float(percentile(segment_return, 50).item()),
            "sirl/trajectory_return_p90": float(percentile(segment_return, 90).item()),
            "sirl/selected_trajectories": float(elite_env_ids.numel()),
            "replay/real_count": float(len(self.real_replay)),
            "replay/sirl_count": float(len(self.sirl_replay)),
        }
        if elite_env_ids.numel() == 0:
            return stats

        return_weight = normalize_by_percentiles(segment_return, 50, 90).pow(float(self.wm_cfg.get("sirl_return_alpha", 1.0)))
        return_weight = torch.clamp(return_weight, min=0.0, max=float(self.wm_cfg.get("sirl_max_weight", 1.0)))
        elite_weight = return_weight[elite_env_ids].unsqueeze(0).expand(self.collect_steps, -1)
        self.sirl_replay.add(
            rollout["obs"][:, elite_env_ids, :],
            rollout["action"][:, elite_env_ids, :],
            rollout["reward"][:, elite_env_ids],
            rollout["done"][:, elite_env_ids],
            rollout["next_obs"][:, elite_env_ids, :],
            privileged_obs=rollout["privileged_obs"][:, elite_env_ids, :] if self.use_privileged_q else None,
            next_privileged_obs=rollout["next_privileged_obs"][:, elite_env_ids, :] if self.use_privileged_q else None,
            command=rollout["command"][:, elite_env_ids, :],
            ret=returns[:, elite_env_ids],
            sirl_weight=elite_weight,
            is_model=0.0,
        )
        stats.update(
            {
                "sirl/bc_weight_mean": float(elite_weight.mean().item()),
                "sirl/bc_weight_max": float(elite_weight.max().item()),
                "replay/sirl_count": float(len(self.sirl_replay)),
            }
        )
        return stats

    def _sample_update_batch(self):
        batch_size = int(self.wm_cfg.get("batch_size", 1024))
        model_ratio = float(self.wm_cfg.get("model_batch_ratio", 0.25))
        if len(self.model_replay) <= 0:
            return self.real_replay.sample(batch_size, self.device)
        model_batch_size = min(int(batch_size * model_ratio), len(self.model_replay))
        real_batch_size = max(1, batch_size - model_batch_size)
        real_batch = self.real_replay.sample(real_batch_size, self.device)
        model_batch = self.model_replay.sample(model_batch_size, self.device)
        batch = {}
        for key, value in real_batch.items():
            if key in model_batch:
                batch[key] = torch.cat((value, model_batch[key]), dim=0)
            else:
                batch[key] = value
        return batch

    def _sac_update(self):
        if len(self.real_replay) < int(self.wm_cfg.get("learning_starts", 8192)):
            return {}
        batch = self._sample_update_batch()
        alpha = self.alpha
        with torch.no_grad():
            next_action, next_log_prob, _ = self._policy_action(batch["next_obs"])
            next_priv = self._q_input_privileged(batch, next_obs=True)
            next_q = torch.min(
                self.q1_target(batch["next_obs"], next_action, next_priv),
                self.q2_target(batch["next_obs"], next_action, next_priv),
            )
            target_q = batch["reward"] + self.gamma * (1.0 - batch["done"]) * (next_q - alpha * next_log_prob)

        priv = self._q_input_privileged(batch)
        q1 = self.q1(batch["obs"], batch["action"], priv)
        q2 = self.q2(batch["obs"], batch["action"], priv)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        sirl_adv1 = torch.clamp(batch["return"] - q1, min=0.0)
        sirl_adv2 = torch.clamp(batch["return"] - q2, min=0.0)
        lower_bound_loss = 0.5 * torch.mean(batch["sirl_weight"] * (sirl_adv1.square() + sirl_adv2.square()))
        q_loss = critic_loss + float(self.wm_cfg.get("sirl_q_coef", 0.25)) * lower_bound_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), float(self.wm_cfg.get("max_grad_norm", 10.0)))
        self.q_optimizer.step()

        policy_action, log_prob, dist = self._policy_action(batch["obs"])
        q_pi = torch.min(self.q1(batch["obs"], policy_action, priv), self.q2(batch["obs"], policy_action, priv))
        actor_loss = (alpha * log_prob - q_pi).mean()
        bound_loss = torch.clip(dist.loc - 1.0, min=0.0).square().mean() + torch.clip(dist.loc + 1.0, max=0.0).square().mean()
        actor_loss = actor_loss + float(self.wm_cfg.get("actor_bound_coef", 1.0)) * bound_loss
        actor_loss = actor_loss + float(self.wm_cfg.get("actor_mean_l2_coef", 0.0)) * dist.loc.square().mean()
        actor_loss = actor_loss + float(self.wm_cfg.get("actor_action_l2_coef", 0.0)) * policy_action.square().mean()
        actor_loss = actor_loss + float(self.wm_cfg.get("logstd_l2_coef", 0.0)) * self.model.logstd.square().mean()

        sirl_actor_loss = torch.tensor(0.0, dtype=torch.float, device=self.device)
        sirl_batch_size = min(int(self.wm_cfg.get("sirl_batch_size", 512)), len(self.sirl_replay))
        if sirl_batch_size > 0 and float(self.wm_cfg.get("sirl_actor_coef", 0.0)) > 0.0:
            sirl_batch = self.sirl_replay.sample(sirl_batch_size, self.device)
            sirl_priv = self._q_input_privileged(sirl_batch)
            with torch.no_grad():
                q_sirl = torch.min(
                    self.q1(sirl_batch["obs"], sirl_batch["action"], sirl_priv),
                    self.q2(sirl_batch["obs"], sirl_batch["action"], sirl_priv),
                )
                positive_adv = torch.clamp(sirl_batch["return"] - q_sirl, min=0.0)
                adv_scale = positive_adv / (positive_adv.mean() + 1.0e-6)
                sample_weight = sirl_batch["sirl_weight"] * torch.clamp(adv_scale, max=float(self.wm_cfg.get("sirl_adv_weight_clip", 5.0)))
            sirl_dist = self.model.act(sirl_batch["obs"])
            if str(self.wm_cfg.get("sirl_loss", "nll")).lower() == "mse":
                per_sample_loss = torch.mean(torch.square(sirl_dist.loc - sirl_batch["action"]), dim=-1)
            else:
                per_sample_loss = -sirl_dist.log_prob(sirl_batch["action"]).sum(dim=-1)
            sirl_actor_loss = torch.mean(sample_weight * per_sample_loss)
            actor_loss = actor_loss + float(self.wm_cfg.get("sirl_actor_coef", 0.1)) * sirl_actor_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.model.actor.parameters()) + [self.model.logstd], float(self.wm_cfg.get("max_grad_norm", 10.0)))
        self.actor_optimizer.step()
        with torch.no_grad():
            self.model.logstd.clamp_(min=self.log_std_min, max=self.log_std_max)

        alpha_loss = torch.tensor(0.0, dtype=torch.float, device=self.device)
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self._soft_update_targets()
        return {
            "sac/q_loss": float(q_loss.detach().item()),
            "sac/critic_loss": float(critic_loss.detach().item()),
            "sac/lower_bound_loss": float(lower_bound_loss.detach().item()),
            "sac/actor_loss": float(actor_loss.detach().item()),
            "sac/sirl_actor_loss": float(sirl_actor_loss.detach().item()),
            "sac/alpha": float(self.alpha.item()),
            "sac/alpha_loss": float(alpha_loss.detach().item()),
            "sac/q_mean": float(0.5 * (q1.detach().mean().item() + q2.detach().mean().item())),
        }

    def _train_world_model(self):
        if len(self.real_replay) < int(self.wm_cfg.get("world_model_learning_starts", 8192)):
            return {}
        batch_size = int(self.wm_cfg.get("world_model_batch_size", 1024))
        steps = int(self.wm_cfg.get("world_model_gradient_steps", 1))
        obs_coef = float(self.wm_cfg.get("world_model_obs_delta_coef", 1.0))
        reward_coef = float(self.wm_cfg.get("world_model_reward_coef", 0.5))
        done_coef = float(self.wm_cfg.get("world_model_done_coef", 0.2))
        totals = {"loss": 0.0, "obs": 0.0, "reward": 0.0, "done": 0.0}
        for _ in range(steps):
            batch = self.real_replay.sample(batch_size, self.device)
            model_idx = random.randrange(len(self.world_models))
            model = self.world_models[model_idx]
            command = batch.get("command")
            if self.world_model_command_dim > 0 and command is not None:
                command = command[:, : self.world_model_command_dim]
            else:
                command = None
            pred = model(batch["obs"], batch["action"], command)
            target_delta = batch["next_obs"] - batch["obs"]
            obs_loss = F.mse_loss(pred["delta_obs"], target_delta)
            reward_loss = F.mse_loss(pred["reward"], batch["reward"])
            done_loss = F.binary_cross_entropy_with_logits(pred["done_logit"], batch["done"])
            loss = obs_coef * obs_loss + reward_coef * reward_loss + done_coef * done_loss
            self.world_model_optimizers[model_idx].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(self.wm_cfg.get("world_model_max_grad_norm", 10.0)))
            self.world_model_optimizers[model_idx].step()
            totals["loss"] += float(loss.detach().item())
            totals["obs"] += float(obs_loss.detach().item())
            totals["reward"] += float(reward_loss.detach().item())
            totals["done"] += float(done_loss.detach().item())
        return {
            "world_model/loss": totals["loss"] / steps,
            "world_model/obs_delta_loss": totals["obs"] / steps,
            "world_model/reward_loss": totals["reward"] / steps,
            "world_model/done_loss": totals["done"] / steps,
        }

    def _rollout_world_model(self):
        if len(self.real_replay) < int(self.wm_cfg.get("model_rollout_starts", 16384)):
            return {"model_rollout/added": 0.0}
        num_starts = int(self.wm_cfg.get("model_rollout_batch_size", 4096))
        horizon = int(self.wm_cfg.get("model_rollout_horizon", 1))
        done_threshold = float(self.wm_cfg.get("model_done_threshold", 0.8))
        batch = self.real_replay.sample(num_starts, self.device)
        obs = batch["obs"]
        command = batch.get("command")
        added = 0
        for _ in range(horizon):
            with torch.no_grad():
                action, _, _ = self._policy_action(obs)
                model_idx = random.randrange(len(self.world_models))
                model = self.world_models[model_idx]
                model_command = command[:, : self.world_model_command_dim] if self.world_model_command_dim > 0 and command is not None else None
                pred = model(obs, action, model_command)
                next_obs = obs + pred["delta_obs"]
                reward = pred["reward"]
                done_prob = torch.sigmoid(pred["done_logit"])
                done = (done_prob > done_threshold).float()
            self.model_replay.add(
                obs,
                action,
                reward,
                done,
                next_obs,
                command=command,
                is_model=1.0,
            )
            added += int(obs.shape[0])
            obs = next_obs.detach()
        return {
            "model_rollout/added": float(added),
            "replay/model_count": float(len(self.model_replay)),
        }

    def _checkpoint_state(self):
        return {
            "model": self.model.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "world_models": [model.state_dict() for model in self.world_models],
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "world_model_optimizers": [optimizer.state_dict() for optimizer in self.world_model_optimizers],
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "curriculum": getattr(self.env, "curriculum_prob", None),
            "ball_curriculum_level": getattr(self.env, "ball_curriculum_global_level", 0),
            "learner": "sirl_worldmodel_sac_mbpo",
        }

    def _load(self):
        checkpoint = self.cfg["basic"].get("checkpoint")
        if not checkpoint:
            return
        if checkpoint in ("-1", -1):
            task_name = self.cfg["basic"].get("log_task", self.cfg["basic"]["task"])
            patterns = [
                os.path.join("logs", task_name.split("/", 1)[0], task_name, "**/*.pth"),
                os.path.join("logs", "**/*.pth"),
            ]
            matches = []
            for pattern in patterns:
                matches = sorted(glob.glob(pattern, recursive=True), key=os.path.getmtime)
                if matches:
                    break
            if not matches:
                return
            checkpoint = matches[-1]
        print("Loading model from {}".format(checkpoint))
        model_dict = torch.load(checkpoint, map_location=self.device, weights_only=True)
        if "model" in model_dict:
            self.model.load_state_dict(model_dict["model"], strict=False)
        for name in ("q1", "q2", "q1_target", "q2_target"):
            if name in model_dict:
                getattr(self, name).load_state_dict(model_dict[name], strict=False)
        if "world_models" in model_dict:
            for model, state in zip(self.world_models, model_dict["world_models"]):
                model.load_state_dict(state, strict=False)
        if "log_alpha" in model_dict:
            self.log_alpha.data.copy_(model_dict["log_alpha"].to(self.device))
        try:
            if "actor_optimizer" in model_dict:
                self.actor_optimizer.load_state_dict(model_dict["actor_optimizer"])
            if "q_optimizer" in model_dict:
                self.q_optimizer.load_state_dict(model_dict["q_optimizer"])
            if "alpha_optimizer" in model_dict:
                self.alpha_optimizer.load_state_dict(model_dict["alpha_optimizer"])
            if "world_model_optimizers" in model_dict:
                for optimizer, state in zip(self.world_model_optimizers, model_dict["world_model_optimizers"]):
                    optimizer.load_state_dict(state)
        except Exception as e:
            print(f"Failed to load optimizer state: {e}")

    def train(self):
        self.recorder = Recorder(self.cfg)
        if hasattr(self.env, "update_training_curriculum"):
            self.env.update_training_curriculum(0)
        obs, infos = self.env.reset()
        obs = obs.to(self.device)
        privileged_obs = infos["privileged_obs"].to(self.device)
        max_iterations = int(self.cfg["basic"]["max_iterations"])
        save_interval = int(self.cfg["runner"]["save_interval"])
        progress_interval = max(1, int(self.cfg["runner"].get("progress_interval", 10)))
        updates_per_iter = int(self.wm_cfg.get("updates_per_iter", 64))
        world_model_train_every = max(1, int(self.wm_cfg.get("world_model_train_every", 1)))
        model_rollout_every = max(1, int(self.wm_cfg.get("model_rollout_every", 1)))
        train_start_time = time.time()

        print(f"SIRL world-model logs: {self.recorder.dir}")
        print(
            f"SIRL world-model progress: 0/{max_iterations} | "
            f"num_envs={self.env.num_envs} collect_steps={self.collect_steps}"
        )

        for iteration in range(max_iterations):
            if hasattr(self.env, "update_training_curriculum"):
                self.env.update_training_curriculum(iteration)
            obs, privileged_obs, rollout, rollout_stats = self._collect_rollout(obs, privileged_obs, iteration)
            sirl_stats = self._score_and_store_rollout(rollout)

            wm_stats = {}
            if iteration % world_model_train_every == 0:
                wm_stats = self._train_world_model()
            model_rollout_stats = {}
            if iteration % model_rollout_every == 0:
                model_rollout_stats = self._rollout_world_model()

            sac_totals = {}
            valid_updates = 0
            for _ in range(updates_per_iter):
                sac_stats = self._sac_update()
                if not sac_stats:
                    continue
                valid_updates += 1
                for key, value in sac_stats.items():
                    sac_totals[key] = sac_totals.get(key, 0.0) + value
            if valid_updates > 0:
                for key in sac_totals:
                    sac_totals[key] /= valid_updates

            stats = {}
            stats.update(rollout_stats)
            stats.update(sirl_stats)
            stats.update(wm_stats)
            stats.update(model_rollout_stats)
            stats.update(sac_totals)
            stats["replay/real_count"] = float(len(self.real_replay))
            stats["replay/model_count"] = float(len(self.model_replay))
            stats["replay/sirl_count"] = float(len(self.sirl_replay))
            stats["updates/valid_sac"] = float(valid_updates)
            self.recorder.record_statistics(stats, iteration)

            if (iteration + 1) % save_interval == 0:
                self.recorder.save(self._checkpoint_state(), iteration + 1)

            if (iteration + 1) == 1 or (iteration + 1) == max_iterations or (iteration + 1) % progress_interval == 0:
                elapsed = time.time() - train_start_time
                avg_iter_time = elapsed / (iteration + 1)
                eta = avg_iter_time * max(0, max_iterations - iteration - 1)
                print(
                    f"[sirl-wm] {iteration + 1}/{max_iterations} | "
                    f"elapsed {self._format_duration(elapsed)} | eta {self._format_duration(eta)} | "
                    f"reward {stats.get('rollout/reward_mean', 0.0):.4f} | "
                    f"done {100.0 * stats.get('rollout/done_rate', 0.0):.2f}% | "
                    f"real {len(self.real_replay)} | model {len(self.model_replay)} | sirl {len(self.sirl_replay)} | "
                    f"q {stats.get('sac/q_loss', 0.0):.4f} | actor {stats.get('sac/actor_loss', 0.0):.4f} | "
                    f"wm {stats.get('world_model/loss', 0.0):.4f}"
                )

    @staticmethod
    def _format_duration(seconds):
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
        return f"{minutes:d}m {seconds:02d}s"

    def play(self):
        obs, _ = self.env.reset()
        obs = obs.to(self.device)
        if self.cfg["viewer"]["record_video"]:
            os.makedirs("videos", exist_ok=True)
            name = time.strftime("%Y-%m-%d-%H-%M-%S.mp4", time.localtime())
            record_time = self.cfg["viewer"]["record_interval"]
        try:
            while True:
                with torch.no_grad():
                    action, _, _ = self._policy_action(obs, deterministic=True)
                    obs, _, done, _ = self.env.step(action)
                    obs = obs.to(self.device)
                if done[0]:
                    print("[play] env 0 reset")
                if self.cfg["viewer"]["record_video"]:
                    record_time -= self.env.dt
                    if record_time < 0:
                        record_time += self.cfg["viewer"]["record_interval"]
                        self.interrupt = False
                        signal.signal(signal.SIGINT, self.interrupt_handler)
                        with imageio.get_writer(os.path.join("videos", name), fps=int(1.0 / self.env.dt)) as self.writer:
                            for frame in self.env.camera_frames:
                                self.writer.append_data(frame)
                        if self.interrupt:
                            raise KeyboardInterrupt
                        signal.signal(signal.SIGINT, signal.default_int_handler)
        finally:
            pass

    def interrupt_handler(self, signal, frame):
        self.interrupt = True
