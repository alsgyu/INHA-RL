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
        self.actor_bound_limit = float(self.wm_cfg.get("actor_bound_limit", 1.0))
        self.actor_mean_clip = self.wm_cfg.get("actor_mean_clip", None)
        self.reward_clip = self.wm_cfg.get("reward_clip", None)
        self.target_q_clip = self.wm_cfg.get("target_q_clip", None)
        self.q_value_clip = self.wm_cfg.get("q_value_clip", None)
        self.parameter_abs_clip = self.wm_cfg.get("parameter_abs_clip", None)
        self.log_std_min = float(self.wm_cfg.get("log_std_min", -5.0))
        self.log_std_max = float(self.wm_cfg.get("log_std_max", -1.0))
        self.collect_deterministic = bool(self.wm_cfg.get("collect_deterministic", True))
        self.collect_noise_std = float(self.wm_cfg.get("collect_noise_std", 0.05))
        self.collect_action_clip = self.wm_cfg.get("collect_action_clip", self.action_clip)
        self.deploy_action_clip = self.wm_cfg.get("deploy_action_clip", self.collect_action_clip)
        self._mirror_action_indices = torch.tensor([6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5], dtype=torch.long, device=self.device)
        self._mirror_action_signs = torch.tensor([1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0], dtype=torch.float, device=self.device)
        self._obs_public_command_scales = torch.tensor(
            [
                float(self.cfg["normalization"].get("lin_vel", 1.0)),
                float(self.cfg["normalization"].get("lin_vel", 1.0)),
                float(self.cfg["normalization"].get("ang_vel", 1.0)),
            ],
            dtype=torch.float,
            device=self.device,
        )
        self.use_privileged_q = bool(self.wm_cfg.get("use_privileged_q", False))
        self.q_privileged_dim = self.env.num_privileged_obs if self.use_privileged_q else 0
        self.current_iteration = 0

        model_name = self.cfg["basic"].get("model", "BaseActorCritic")
        model_class = get_model_class(model_name)
        self.model_class = model_class
        self.model = model_class(self.env.num_actions, self.env.num_obs, self.env.num_privileged_obs).to(self.device)
        self._init_actor_for_safe_start()
        self.teacher_policy = None
        self.teacher_enabled = bool(self.wm_cfg.get("teacher_enabled", False))
        self._load_teacher_policy()
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
        self.min_log_alpha = float(np.log(float(self.wm_cfg.get("alpha_min", 1.0e-4))))
        self.max_log_alpha = float(np.log(float(self.wm_cfg.get("alpha_max", 1.0))))

        self.world_models = torch.nn.ModuleList()
        self.world_model_optimizers = []
        self.world_model_command_dim = int(self.wm_cfg.get("world_model_command_dim", 3)) if bool(self.wm_cfg.get("world_model_use_command", False)) else 0
        self.world_model_updates = 0
        self.last_world_model_loss = None
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

    def _finite_tensor(self, value, clip=None):
        if clip is None:
            return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        clip = float(clip)
        value = torch.nan_to_num(value, nan=0.0, posinf=clip, neginf=-clip)
        return torch.clamp(value, -clip, clip)

    def _sanitize_module_(self, module):
        if self.parameter_abs_clip is None:
            clip = None
        else:
            clip = float(self.parameter_abs_clip)
        with torch.no_grad():
            for param in module.parameters():
                self._sanitize_parameter_(param, clip)

    def _sanitize_parameter_(self, param, clip=None):
        if clip is None and self.parameter_abs_clip is not None:
            clip = float(self.parameter_abs_clip)
        param.data = self._finite_tensor(param.data, clip)

    def _resolve_path(self, path):
        if not path:
            return None
        if os.path.isabs(path):
            return path if os.path.exists(path) else None
        candidates = [
            path,
            os.path.join(os.getcwd(), path),
            os.path.join(os.path.dirname(__file__), "..", path),
            os.path.join(os.path.dirname(__file__), "..", "..", path),
        ]
        for candidate in candidates:
            candidate = os.path.normpath(candidate)
            if os.path.exists(candidate):
                return candidate
        return None

    def _resolve_checkpoint_arg(self, checkpoint):
        if not checkpoint:
            return None
        if checkpoint not in ("-1", -1):
            return self._resolve_path(str(checkpoint))
        task_name = self.cfg["basic"].get("log_task", self.cfg["basic"]["task"])
        patterns = [
            os.path.join("logs", task_name.split("/", 1)[0], task_name, "**/*.pth"),
            os.path.join("logs", "**/*.pth"),
        ]
        for pattern in patterns:
            matches = sorted(glob.glob(pattern, recursive=True), key=os.path.getmtime)
            if matches:
                return matches[-1]
        return None

    def _load_actor_checkpoint_policy(self, checkpoint_path):
        try:
            try:
                model_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            except TypeError:
                model_dict = torch.load(checkpoint_path, map_location=self.device)
            state_dict = model_dict["model"] if isinstance(model_dict, dict) and "model" in model_dict else model_dict
            teacher_model = self.model_class(self.env.num_actions, self.env.num_obs, self.env.num_privileged_obs).to(self.device)
            teacher_model.load_state_dict(state_dict, strict=False)
            teacher_model.actor.eval()
            for param in teacher_model.actor.parameters():
                param.requires_grad_(False)
            return teacher_model.actor
        except Exception as exc:
            print(f"[sirl-wm] failed to load teacher actor checkpoint '{checkpoint_path}': {exc}")
            return None

    def _load_teacher_policy(self):
        if not self.teacher_enabled:
            return
        teacher_spec = self.wm_cfg.get("teacher_policy_path", "deploy/models/parameter_walk_k1.pt")
        if str(teacher_spec).lower() in ("__checkpoint__", "$checkpoint", "checkpoint", "self"):
            teacher_path = self._resolve_checkpoint_arg(self.cfg["basic"].get("checkpoint"))
        else:
            teacher_path = self._resolve_path(teacher_spec)
        if teacher_path is None:
            print("[sirl-wm] teacher policy requested but path was not found; continuing without teacher.")
            self.teacher_enabled = False
            return
        try:
            self.teacher_policy = torch.jit.load(teacher_path, map_location=self.device)
            self.teacher_policy.eval()
            print(f"[sirl-wm] teacher policy loaded: {teacher_path}")
        except Exception as exc:
            checkpoint_policy = self._load_actor_checkpoint_policy(teacher_path)
            if checkpoint_policy is None:
                print(f"[sirl-wm] failed to load teacher policy '{teacher_path}': {exc}")
                self.teacher_policy = None
                self.teacher_enabled = False
                return
            self.teacher_policy = checkpoint_policy
            print(f"[sirl-wm] teacher actor checkpoint loaded: {teacher_path}")

    def _teacher_action(self, obs):
        if self.teacher_policy is None:
            return None
        with torch.no_grad():
            action = self.teacher_policy(obs)
            clip = self.wm_cfg.get("teacher_action_clip", self.action_clip)
            if clip is not None:
                action = torch.clamp(action, -float(clip), float(clip))
            return action.detach()

    def _teacher_collect_blend(self, iteration):
        if self.teacher_policy is None:
            return 0.0
        until = int(self.wm_cfg.get("teacher_collect_iterations", 0))
        if until <= 0:
            return 0.0
        if iteration >= until:
            return 0.0
        start = float(self.wm_cfg.get("teacher_collect_start_blend", 1.0))
        end = float(self.wm_cfg.get("teacher_collect_end_blend", 0.15))
        decay = max(1, int(self.wm_cfg.get("teacher_collect_decay_iterations", until)))
        mix = min(max(iteration / decay, 0.0), 1.0)
        return max(end, start + mix * (end - start))

    def _teacher_bc_coef(self):
        if self.teacher_policy is None:
            return 0.0
        coef = float(self.wm_cfg.get("teacher_bc_coef", 0.0))
        if coef <= 0.0:
            return 0.0
        decay = int(self.wm_cfg.get("teacher_bc_decay_iterations", 0))
        min_coef = float(self.wm_cfg.get("teacher_bc_min_coef", 0.0))
        if decay <= 0:
            return coef
        mix = min(max(self.current_iteration / max(decay, 1), 0.0), 1.0)
        return max(min_coef, coef * (1.0 - mix))

    def _scheduled_coef(self, coef_key, default, start_key=None, warmup_key=None):
        coef = float(self.wm_cfg.get(coef_key, default))
        start = int(self.wm_cfg.get(start_key, 0)) if start_key is not None else 0
        if self.current_iteration < start:
            return 0.0
        warmup = int(self.wm_cfg.get(warmup_key, 0)) if warmup_key is not None else 0
        if warmup <= 0:
            return coef
        mix = min(max((self.current_iteration - start) / max(warmup, 1), 0.0), 1.0)
        return coef * mix

    def _teacher_distill_update(self, rollout=None):
        if self.teacher_policy is None:
            return {}
        max_iteration = int(self.wm_cfg.get("teacher_distill_iterations", 0))
        if max_iteration <= 0 or self.current_iteration >= max_iteration:
            return {}
        steps = int(self.wm_cfg.get("teacher_distill_steps_per_iter", 0))
        if steps <= 0:
            return {}

        batch_size = int(self.wm_cfg.get("teacher_distill_batch_size", 2048))
        replay_fraction = float(self.wm_cfg.get("teacher_distill_replay_fraction", 0.5))
        rollout_obs = None
        if rollout is not None:
            rollout_obs = rollout["obs"].reshape(-1, self.env.num_obs).detach()
        if rollout_obs is None and len(self.real_replay) <= 0:
            return {}

        total_loss = 0.0
        valid_steps = 0
        for _ in range(steps):
            obs_parts = []
            replay_count = 0
            if len(self.real_replay) > 0 and replay_fraction > 0.0:
                replay_count = min(int(batch_size * replay_fraction), batch_size)
                if replay_count > 0:
                    obs_parts.append(self.real_replay.sample(replay_count, self.device)["obs"])
            rollout_count = max(0, batch_size - replay_count)
            if rollout_obs is not None and rollout_count > 0:
                indices = torch.randint(0, rollout_obs.shape[0], (rollout_count,), device=rollout_obs.device)
                obs_parts.append(rollout_obs[indices].to(self.device))
            if not obs_parts:
                continue

            obs = torch.cat(obs_parts, dim=0)
            teacher_action = self._teacher_action(obs)
            if teacher_action is None:
                return {}
            student_action = self._policy_dist(obs).loc
            distill_loss = F.mse_loss(student_action, teacher_action)
            if not torch.isfinite(distill_loss):
                continue

            self.actor_optimizer.zero_grad()
            distill_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.actor.parameters()) + [self.model.logstd],
                float(self.wm_cfg.get("max_grad_norm", 10.0)),
            )
            self.actor_optimizer.step()
            self._sanitize_module_(self.model.actor)
            with torch.no_grad():
                self._sanitize_parameter_(self.model.logstd, None)
                self.model.logstd.clamp_(min=self.log_std_min, max=self.log_std_max)
            total_loss += float(distill_loss.detach().item())
            valid_steps += 1

        if valid_steps == 0:
            return {"teacher/distill_skipped_nonfinite": 1.0}
        return {
            "teacher/distill_loss": total_loss / valid_steps,
            "teacher/distill_steps": float(valid_steps),
        }

    def _actor_state_snapshot(self):
        return {key: value.detach().clone() for key, value in self.model.actor.state_dict().items()}, self.model.logstd.detach().clone()

    def _restore_actor_state(self, snapshot):
        actor_state, logstd = snapshot
        self.model.actor.load_state_dict(actor_state)
        with torch.no_grad():
            self.model.logstd.data.copy_(logstd.to(self.device))

    def _mirror_action(self, action):
        if action.shape[-1] != 12:
            return action
        indices = self._mirror_action_indices.to(action.device)
        signs = self._mirror_action_signs.to(action.device, dtype=action.dtype)
        return action.index_select(action.dim() - 1, indices) * signs

    def _mirror_obs(self, obs):
        if obs.shape[-1] < 54 or self.env.num_actions != 12:
            return obs
        mirrored = obs.clone()
        mirrored[..., 0] = obs[..., 0]
        mirrored[..., 1] = -obs[..., 1]
        mirrored[..., 2] = obs[..., 2]
        mirrored[..., 3] = -obs[..., 3]
        mirrored[..., 4] = obs[..., 4]
        mirrored[..., 5] = -obs[..., 5]
        mirrored[..., 6] = obs[..., 6]
        mirrored[..., 7] = -obs[..., 7]
        mirrored[..., 8] = -obs[..., 8]
        mirrored[..., 9] = obs[..., 9]
        mirrored[..., 10] = -obs[..., 11]
        mirrored[..., 11] = -obs[..., 10]
        mirrored[..., 12] = obs[..., 12]
        mirrored[..., 13] = -obs[..., 13]
        mirrored[..., 14] = obs[..., 14]
        mirrored[..., 15] = -obs[..., 15]
        mirrored[..., 16] = -obs[..., 16]
        mirrored[..., 17] = -obs[..., 17]
        mirrored[..., 18:30] = self._mirror_action(obs[..., 18:30])
        mirrored[..., 30:42] = self._mirror_action(obs[..., 30:42])
        mirrored[..., 42:54] = self._mirror_action(obs[..., 42:54])
        return mirrored

    def _batch_public_command(self, batch):
        command = batch.get("command")
        if command is not None and command.shape[-1] >= 3:
            return command[:, :3]
        obs = batch.get("obs")
        if obs is None or obs.shape[-1] < 9:
            return None
        scales = self._obs_public_command_scales.to(obs.device, dtype=obs.dtype)
        return obs[:, 6:9] / torch.clamp(scales, min=1.0e-6)

    def _straight_symmetry_mask(self, batch):
        command = self._batch_public_command(batch)
        if command is None:
            return None
        min_vx = float(self.wm_cfg.get("actor_symmetry_min_abs_vx", 0.03))
        max_vy = float(self.wm_cfg.get("actor_symmetry_max_abs_vy", 0.04))
        max_yaw = float(self.wm_cfg.get("actor_symmetry_max_abs_yaw", 0.08))
        straight = (
            (torch.abs(command[:, 0]) >= min_vx)
            & (torch.abs(command[:, 1]) <= max_vy)
            & (torch.abs(command[:, 2]) <= max_yaw)
        )
        if bool(self.wm_cfg.get("actor_symmetry_include_stand", True)):
            stand_threshold = float(self.cfg["commands"].get("adapter", {}).get("stand_command_threshold", 0.04))
            command_norm = torch.sqrt(torch.sum(torch.square(command[:, 0:2]), dim=-1) + torch.square(command[:, 2]))
            straight = straight | (command_norm <= stand_threshold)
        return straight.float()

    def _init_actor_for_safe_start(self):
        if not bool(self.wm_cfg.get("zero_init_actor", True)):
            return
        final_layer = None
        for module in reversed(self.model.actor):
            if isinstance(module, torch.nn.Linear):
                final_layer = module
                break
        if final_layer is not None:
            torch.nn.init.zeros_(final_layer.weight)
            torch.nn.init.zeros_(final_layer.bias)
        with torch.no_grad():
            self.model.logstd.fill_(float(self.wm_cfg.get("initial_logstd", -4.0)))

    def _policy_dist(self, obs):
        action_mean = self.model.actor(obs)
        action_mean = self._finite_tensor(action_mean, self.actor_mean_clip)
        log_std_param = torch.nan_to_num(self.model.logstd, nan=self.log_std_min, posinf=self.log_std_max, neginf=self.log_std_min)
        log_std = torch.clamp(log_std_param, min=self.log_std_min, max=self.log_std_max).expand_as(action_mean)
        return torch.distributions.Normal(action_mean, torch.exp(log_std))

    def _policy_action(self, obs, deterministic=False):
        dist = self._policy_dist(obs)
        action = dist.loc if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        if self.action_clip is not None:
            action = torch.clamp(action, -float(self.action_clip), float(self.action_clip))
        return action, log_prob, dist

    def _collect_action(self, obs, iteration=0):
        if not self.collect_deterministic:
            action, _, _ = self._policy_action(obs, deterministic=False)
        else:
            dist = self._policy_dist(obs)
            action = dist.loc
            if self.collect_noise_std > 0.0:
                action = action + self.collect_noise_std * torch.randn_like(action)
        teacher_action = self._teacher_action(obs)
        teacher_blend = self._teacher_collect_blend(iteration)
        if teacher_action is not None and teacher_blend > 0.0:
            action = teacher_blend * teacher_action + (1.0 - teacher_blend) * action
        if self.collect_action_clip is not None:
            action = torch.clamp(action, -float(self.collect_action_clip), float(self.collect_action_clip))
        return action

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
                    action = self._collect_action(obs, iteration)
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
        if model_ratio <= 0.0 or len(self.model_replay) <= 0:
            return self.real_replay.sample(batch_size, self.device)
        model_batch_size = min(int(batch_size * model_ratio), len(self.model_replay))
        if model_batch_size <= 0:
            return self.real_replay.sample(batch_size, self.device)
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
        self._sanitize_module_(self.model.actor)
        self._sanitize_module_(self.q1)
        self._sanitize_module_(self.q2)
        self._sanitize_module_(self.q1_target)
        self._sanitize_module_(self.q2_target)
        with torch.no_grad():
            self._sanitize_parameter_(self.model.logstd, None)
            self.model.logstd.clamp_(min=self.log_std_min, max=self.log_std_max)
            self.log_alpha.data = torch.clamp(torch.nan_to_num(self.log_alpha.data, nan=np.log(0.05)), self.min_log_alpha, self.max_log_alpha)
        batch = self._sample_update_batch()
        alpha = self.alpha
        reward = batch["reward"]
        if self.reward_clip is not None:
            reward = torch.clamp(reward, -float(self.reward_clip), float(self.reward_clip))
        with torch.no_grad():
            next_action, next_log_prob, _ = self._policy_action(batch["next_obs"])
            next_priv = self._q_input_privileged(batch, next_obs=True)
            next_q = torch.min(
                self.q1_target(batch["next_obs"], next_action, next_priv),
                self.q2_target(batch["next_obs"], next_action, next_priv),
            )
            next_q = self._finite_tensor(next_q, self.q_value_clip)
            next_log_prob = self._finite_tensor(next_log_prob, self.q_value_clip)
            target_q = reward + self.gamma * (1.0 - batch["done"]) * (next_q - alpha * next_log_prob)
            target_q = self._finite_tensor(target_q, self.target_q_clip)

        priv = self._q_input_privileged(batch)
        q1 = self.q1(batch["obs"], batch["action"], priv)
        q2 = self.q2(batch["obs"], batch["action"], priv)
        q1_loss_value = self._finite_tensor(q1, self.q_value_clip)
        q2_loss_value = self._finite_tensor(q2, self.q_value_clip)
        if str(self.wm_cfg.get("critic_loss", "huber")).lower() == "mse":
            critic_loss = F.mse_loss(q1_loss_value, target_q) + F.mse_loss(q2_loss_value, target_q)
        else:
            critic_loss = F.smooth_l1_loss(q1_loss_value, target_q) + F.smooth_l1_loss(q2_loss_value, target_q)

        sirl_adv1 = torch.clamp(batch["return"] - q1_loss_value, min=0.0)
        sirl_adv2 = torch.clamp(batch["return"] - q2_loss_value, min=0.0)
        lower_bound_loss = 0.5 * torch.mean(batch["sirl_weight"] * (sirl_adv1.square() + sirl_adv2.square()))
        q_loss = critic_loss + float(self.wm_cfg.get("sirl_q_coef", 0.25)) * lower_bound_loss
        if not torch.isfinite(q_loss):
            return {"sac/skipped_nonfinite": 1.0}

        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), float(self.wm_cfg.get("max_grad_norm", 10.0)))
        self.q_optimizer.step()
        self._sanitize_module_(self.q1)
        self._sanitize_module_(self.q2)

        policy_action, log_prob, dist = self._policy_action(batch["obs"])
        q_pi = torch.min(self.q1(batch["obs"], policy_action, priv), self.q2(batch["obs"], policy_action, priv))
        q_pi = self._finite_tensor(q_pi, self.q_value_clip)
        log_prob = self._finite_tensor(log_prob, self.q_value_clip)
        actor_rl_loss = (alpha * log_prob - q_pi).mean()
        actor_rl_coef = self._scheduled_coef(
            "actor_rl_coef",
            1.0,
            "actor_rl_start_iteration",
            "actor_rl_warmup_iterations",
        )
        actor_loss = actor_rl_coef * actor_rl_loss
        raw_actor_mean = self._finite_tensor(
            self.model.actor(batch["obs"]),
            self.wm_cfg.get("actor_bound_loss_clip", 5.0),
        )
        bound_loss = (
            torch.clip(raw_actor_mean - self.actor_bound_limit, min=0.0).square().mean()
            + torch.clip(raw_actor_mean + self.actor_bound_limit, max=0.0).square().mean()
        )
        actor_loss = actor_loss + float(self.wm_cfg.get("actor_bound_coef", 1.0)) * bound_loss
        actor_loss = actor_loss + float(self.wm_cfg.get("actor_mean_l2_coef", 0.0)) * raw_actor_mean.square().mean()
        actor_loss = actor_loss + float(self.wm_cfg.get("actor_action_l2_coef", 0.0)) * policy_action.square().mean()
        actor_loss = actor_loss + float(self.wm_cfg.get("logstd_l2_coef", 0.0)) * self.model.logstd.square().mean()

        sirl_actor_loss = torch.tensor(0.0, dtype=torch.float, device=self.device)
        teacher_bc_loss = torch.tensor(0.0, dtype=torch.float, device=self.device)
        actor_symmetry_loss = torch.tensor(0.0, dtype=torch.float, device=self.device)
        actor_symmetry_mask_frac = 0.0
        sirl_batch_size = min(int(self.wm_cfg.get("sirl_batch_size", 512)), len(self.sirl_replay))
        sirl_actor_coef = self._scheduled_coef(
            "sirl_actor_coef",
            0.0,
            "sirl_actor_start_iteration",
            "sirl_actor_warmup_iterations",
        )
        if sirl_batch_size > 0 and sirl_actor_coef > 0.0:
            sirl_batch = self.sirl_replay.sample(sirl_batch_size, self.device)
            sirl_priv = self._q_input_privileged(sirl_batch)
            with torch.no_grad():
                q_sirl = torch.min(
                    self.q1(sirl_batch["obs"], sirl_batch["action"], sirl_priv),
                    self.q2(sirl_batch["obs"], sirl_batch["action"], sirl_priv),
                )
                q_sirl = self._finite_tensor(q_sirl, self.q_value_clip)
                positive_adv = torch.clamp(sirl_batch["return"] - q_sirl, min=0.0)
                adv_scale = positive_adv / (positive_adv.mean() + 1.0e-6)
                sample_weight = sirl_batch["sirl_weight"] * torch.clamp(adv_scale, max=float(self.wm_cfg.get("sirl_adv_weight_clip", 5.0)))
            sirl_dist = self._policy_dist(sirl_batch["obs"])
            if str(self.wm_cfg.get("sirl_loss", "nll")).lower() == "mse":
                per_sample_loss = torch.mean(torch.square(sirl_dist.loc - sirl_batch["action"]), dim=-1)
            else:
                per_sample_loss = -sirl_dist.log_prob(sirl_batch["action"]).sum(dim=-1)
            sirl_actor_loss = torch.mean(sample_weight * per_sample_loss)
            actor_loss = actor_loss + sirl_actor_coef * sirl_actor_loss
        teacher_coef = self._teacher_bc_coef()
        guard_enabled = bool(self.wm_cfg.get("teacher_guard_enabled", False)) and self.teacher_policy is not None
        guard_snapshot = None
        guard_teacher_action = None
        guard_before = torch.tensor(0.0, dtype=torch.float, device=self.device)
        guard_after = torch.tensor(0.0, dtype=torch.float, device=self.device)
        guard_rollback = 0.0
        if guard_enabled:
            guard_teacher_action = self._teacher_action(batch["obs"])
            if guard_teacher_action is not None:
                guard_before = F.mse_loss(self._policy_dist(batch["obs"]).loc, guard_teacher_action).detach()
                guard_snapshot = self._actor_state_snapshot()
        if teacher_coef > 0.0:
            teacher_action = guard_teacher_action if guard_teacher_action is not None else self._teacher_action(batch["obs"])
            if teacher_action is not None:
                student_action = self._policy_dist(batch["obs"]).loc
                teacher_bc_loss = F.mse_loss(student_action, teacher_action)
                actor_loss = actor_loss + teacher_coef * teacher_bc_loss
        symmetry_coef = self._scheduled_coef(
            "actor_symmetry_coef",
            0.0,
            "actor_symmetry_start_iteration",
            "actor_symmetry_warmup_iterations",
        )
        if symmetry_coef > 0.0 and self.env.num_actions == 12 and self.env.num_obs >= 54:
            symmetry_mask = self._straight_symmetry_mask(batch)
            if symmetry_mask is not None and float(symmetry_mask.sum().item()) > 0.0:
                student_mean = self._policy_dist(batch["obs"]).loc
                mirrored_mean = self._policy_dist(self._mirror_obs(batch["obs"])).loc
                mirrored_target = self._mirror_action(student_mean).detach()
                per_sample_symmetry = torch.mean(torch.square(mirrored_mean - mirrored_target), dim=-1)
                actor_symmetry_loss = torch.sum(per_sample_symmetry * symmetry_mask) / (torch.sum(symmetry_mask) + 1.0e-6)
                actor_loss = actor_loss + symmetry_coef * actor_symmetry_loss
                actor_symmetry_mask_frac = float(symmetry_mask.mean().detach().item())
        if not torch.isfinite(actor_loss):
            return {"sac/skipped_nonfinite": 1.0}

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.model.actor.parameters()) + [self.model.logstd], float(self.wm_cfg.get("max_grad_norm", 10.0)))
        self.actor_optimizer.step()
        self._sanitize_module_(self.model.actor)
        with torch.no_grad():
            self._sanitize_parameter_(self.model.logstd, None)
            self.model.logstd.clamp_(min=self.log_std_min, max=self.log_std_max)
        if guard_snapshot is not None and guard_teacher_action is not None:
            guard_after = F.mse_loss(self._policy_dist(batch["obs"]).loc, guard_teacher_action).detach()
            max_mse = float(self.wm_cfg.get("teacher_guard_max_mse", 0.02))
            max_increase = float(self.wm_cfg.get("teacher_guard_max_increase", 0.005))
            if float(guard_after.item()) > max_mse or float((guard_after - guard_before).item()) > max_increase:
                self._restore_actor_state(guard_snapshot)
                guard_after = guard_before
                guard_rollback = 1.0

        alpha_loss = torch.tensor(0.0, dtype=torch.float, device=self.device)
        if self.auto_alpha and actor_rl_coef > 0.0:
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            if torch.isfinite(alpha_loss):
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                with torch.no_grad():
                    self.log_alpha.clamp_(min=self.min_log_alpha, max=self.max_log_alpha)

        self._soft_update_targets()
        self._sanitize_module_(self.q1_target)
        self._sanitize_module_(self.q2_target)
        return {
            "sac/q_loss": float(q_loss.detach().item()),
            "sac/critic_loss": float(critic_loss.detach().item()),
            "sac/lower_bound_loss": float(lower_bound_loss.detach().item()),
            "sac/actor_loss": float(actor_loss.detach().item()),
            "sac/actor_rl_loss": float(actor_rl_loss.detach().item()),
            "sac/actor_rl_coef": float(actor_rl_coef),
            "sac/sirl_actor_loss": float(sirl_actor_loss.detach().item()),
            "sac/sirl_actor_coef": float(sirl_actor_coef),
            "sac/teacher_bc_loss": float(teacher_bc_loss.detach().item()),
            "sac/teacher_bc_coef": float(teacher_coef),
            "sac/teacher_guard_before": float(guard_before.detach().item()),
            "sac/teacher_guard_after": float(guard_after.detach().item()),
            "sac/teacher_guard_rollback": float(guard_rollback),
            "sac/actor_symmetry_loss": float(actor_symmetry_loss.detach().item()),
            "sac/actor_symmetry_coef": float(symmetry_coef),
            "sac/actor_symmetry_mask_frac": actor_symmetry_mask_frac,
            "sac/alpha": float(self.alpha.item()),
            "sac/alpha_loss": float(alpha_loss.detach().item()),
            "sac/q_mean": float(0.5 * (q1_loss_value.detach().mean().item() + q2_loss_value.detach().mean().item())),
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
        valid_steps = 0
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
            if not torch.isfinite(loss):
                continue
            self.world_model_optimizers[model_idx].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(self.wm_cfg.get("world_model_max_grad_norm", 10.0)))
            self.world_model_optimizers[model_idx].step()
            self._sanitize_module_(model)
            totals["loss"] += float(loss.detach().item())
            totals["obs"] += float(obs_loss.detach().item())
            totals["reward"] += float(reward_loss.detach().item())
            totals["done"] += float(done_loss.detach().item())
            self.world_model_updates += 1
            valid_steps += 1
        if valid_steps == 0:
            return {"world_model/skipped_nonfinite": 1.0}
        self.last_world_model_loss = totals["loss"] / valid_steps
        return {
            "world_model/loss": totals["loss"] / valid_steps,
            "world_model/obs_delta_loss": totals["obs"] / valid_steps,
            "world_model/reward_loss": totals["reward"] / valid_steps,
            "world_model/done_loss": totals["done"] / valid_steps,
        }

    def _rollout_world_model(self):
        if len(self.real_replay) < int(self.wm_cfg.get("model_rollout_starts", 16384)):
            return {"model_rollout/added": 0.0}
        min_updates = int(self.wm_cfg.get("model_rollout_min_wm_updates", 1000))
        if self.world_model_updates < min_updates:
            return {"model_rollout/added": 0.0, "model_rollout/skipped_not_ready": 1.0}
        max_wm_loss = self.wm_cfg.get("model_rollout_max_wm_loss", None)
        if max_wm_loss is not None and (self.last_world_model_loss is None or self.last_world_model_loss > float(max_wm_loss)):
            return {"model_rollout/added": 0.0, "model_rollout/skipped_wm_loss": 1.0}
        num_starts = int(self.wm_cfg.get("model_rollout_batch_size", 4096))
        horizon = int(self.wm_cfg.get("model_rollout_horizon", 1))
        done_threshold = float(self.wm_cfg.get("model_done_threshold", 0.8))
        delta_clip = self.wm_cfg.get("model_delta_clip", None)
        model_reward_clip = self.wm_cfg.get("model_reward_clip", self.reward_clip)
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
                delta_obs = self._finite_tensor(pred["delta_obs"], delta_clip)
                next_obs = self._finite_tensor(obs + delta_obs, None)
                reward = self._finite_tensor(pred["reward"], model_reward_clip)
                done_prob = torch.sigmoid(pred["done_logit"])
                done = (done_prob > done_threshold).float()
                finite_mask = (
                    torch.isfinite(obs).all(dim=-1)
                    & torch.isfinite(action).all(dim=-1)
                    & torch.isfinite(next_obs).all(dim=-1)
                    & torch.isfinite(reward)
                    & torch.isfinite(done)
                )
            if int(finite_mask.sum().item()) == 0:
                break
            self.model_replay.add(
                obs[finite_mask],
                action[finite_mask],
                reward[finite_mask],
                done[finite_mask],
                next_obs[finite_mask],
                command=command[finite_mask] if command is not None else None,
                is_model=1.0,
            )
            added += int(finite_mask.sum().item())
            obs = next_obs[finite_mask].detach()
            command = command[finite_mask] if command is not None else None
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
            "deploy_action_clip": None if self.deploy_action_clip is None else float(self.deploy_action_clip),
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
        load_critic_state = bool(self.wm_cfg.get("load_critic_state", True))
        load_world_model_state = bool(self.wm_cfg.get("load_world_model_state", True))
        load_alpha_state = bool(self.wm_cfg.get("load_alpha_state", True))
        load_optimizer_state = bool(self.wm_cfg.get("load_optimizer_state", True))
        if load_critic_state:
            for name in ("q1", "q2", "q1_target", "q2_target"):
                if name in model_dict:
                    getattr(self, name).load_state_dict(model_dict[name], strict=False)
        if load_world_model_state and "world_models" in model_dict:
            for model, state in zip(self.world_models, model_dict["world_models"]):
                model.load_state_dict(state, strict=False)
        if load_alpha_state and "log_alpha" in model_dict:
            self.log_alpha.data.copy_(model_dict["log_alpha"].to(self.device))
        if load_optimizer_state:
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
        else:
            print("[sirl-wm] optimizer state reset for this run.")
        self._sanitize_module_(self.model.actor)
        self._sanitize_module_(self.q1)
        self._sanitize_module_(self.q2)
        self._sanitize_module_(self.q1_target)
        self._sanitize_module_(self.q2_target)
        for model in self.world_models:
            self._sanitize_module_(model)
        with torch.no_grad():
            self._sanitize_parameter_(self.model.logstd, None)
            self.model.logstd.clamp_(min=self.log_std_min, max=self.log_std_max)
            self.log_alpha.data = torch.clamp(torch.nan_to_num(self.log_alpha.data, nan=np.log(0.05)), self.min_log_alpha, self.max_log_alpha)

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
        model_rollout_enabled = bool(
            self.wm_cfg.get("model_rollout_enabled", float(self.wm_cfg.get("model_batch_ratio", 0.0)) > 0.0)
        )
        train_start_time = time.time()

        print(f"SIRL world-model logs: {self.recorder.dir}")
        print(
            f"SIRL world-model progress: 0/{max_iterations} | "
            f"num_envs={self.env.num_envs} collect_steps={self.collect_steps}"
        )

        for iteration in range(max_iterations):
            self.current_iteration = iteration
            if hasattr(self.env, "update_training_curriculum"):
                self.env.update_training_curriculum(iteration)
            obs, privileged_obs, rollout, rollout_stats = self._collect_rollout(obs, privileged_obs, iteration)
            sirl_stats = self._score_and_store_rollout(rollout)
            teacher_stats = self._teacher_distill_update(rollout)

            wm_stats = {}
            if iteration % world_model_train_every == 0:
                wm_stats = self._train_world_model()
            model_rollout_stats = {}
            if model_rollout_enabled and iteration % model_rollout_every == 0:
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
            stats.update(teacher_stats)
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
                    f"bc {stats.get('teacher/distill_loss', stats.get('sac/teacher_bc_loss', 0.0)):.4f} | "
                    f"sym {stats.get('sac/actor_symmetry_loss', 0.0):.4f} | "
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
