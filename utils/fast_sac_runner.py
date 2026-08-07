from __future__ import annotations

import argparse
import copy
import glob
import os
import random
import time
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from envs import *
from utils.fast_sac import (
    EmpiricalNormalizer,
    FastSACActor,
    FastSACCritic,
    FastSACPolicyWrapper,
    FastSACReplayBuffer,
)
from utils.recorder import Recorder
from utils.runner import get_task_class, load_config


class FastSACK1Runner:
    """Holosoma-style FastSAC trainer adapted to the existing HTWK Gym env API."""

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
        self.sac_cfg = self.cfg["algorithm"]["fast_sac"]
        self.global_step = 0
        self.use_privileged_critic = bool(self.sac_cfg.get("use_privileged_critic", True))
        self.critic_obs_dim = self.env.num_obs + (self.env.num_privileged_obs if self.use_privileged_critic else 0)
        self.use_amp = bool(self.sac_cfg.get("amp", True)) and str(self.device).startswith("cuda")
        self.amp_dtype = torch.bfloat16 if str(self.sac_cfg.get("amp_dtype", "bf16")).lower() == "bf16" else torch.float16
        self.action_scale = self._action_scale_tensor()

        self.actor = FastSACActor(
            self.env.num_obs,
            self.env.num_actions,
            hidden_dim=int(self.sac_cfg.get("actor_hidden_dim", 512)),
            log_std_min=float(self.sac_cfg.get("log_std_min", -5.0)),
            log_std_max=float(self.sac_cfg.get("log_std_max", 0.0)),
            use_tanh=bool(self.sac_cfg.get("use_tanh", True)),
            use_layer_norm=bool(self.sac_cfg.get("use_layer_norm", True)),
            action_scale=self.action_scale,
            device=self.device,
        ).to(self.device)
        self.qnet = FastSACCritic(
            self.critic_obs_dim,
            self.env.num_actions,
            num_atoms=int(self.sac_cfg.get("num_atoms", 101)),
            v_min=float(self.sac_cfg.get("v_min", -20.0)),
            v_max=float(self.sac_cfg.get("v_max", 20.0)),
            hidden_dim=int(self.sac_cfg.get("critic_hidden_dim", 768)),
            use_layer_norm=bool(self.sac_cfg.get("use_layer_norm", True)),
            num_q_networks=int(self.sac_cfg.get("num_q_networks", 2)),
            device=self.device,
        ).to(self.device)
        self.qnet_target = copy.deepcopy(self.qnet).to(self.device)
        self.obs_normalizer = EmpiricalNormalizer(self.env.num_obs, self.device)
        self.critic_obs_normalizer = EmpiricalNormalizer(self.critic_obs_dim, self.device)
        if not bool(self.sac_cfg.get("obs_normalization", True)):
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()

        self.log_alpha = torch.tensor(
            np.log(float(self.sac_cfg.get("alpha_init", 0.001))),
            dtype=torch.float,
            device=self.device,
            requires_grad=True,
        )
        self.target_entropy = -self.env.num_actions * float(self.sac_cfg.get("target_entropy_ratio", 0.0))
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.actor_optimizer = self._make_optimizer(
            self.actor.parameters(),
            float(self.sac_cfg.get("actor_learning_rate", 3.0e-4)),
        )
        self.q_optimizer = self._make_optimizer(
            self.qnet.parameters(),
            float(self.sac_cfg.get("critic_learning_rate", 3.0e-4)),
        )
        self.alpha_optimizer = self._make_optimizer(
            [self.log_alpha],
            float(self.sac_cfg.get("alpha_learning_rate", 3.0e-4)),
            weight_decay=0.0,
        )
        self.replay = FastSACReplayBuffer(
            self.env.num_envs,
            int(self.sac_cfg.get("buffer_size", 1024)),
            self.env.num_obs,
            self.env.num_actions,
            self.critic_obs_dim,
            self.device,
        )
        self.teacher_policy = self._load_teacher_policy()
        self.base_reward_scales = dict(getattr(self.env, "reward_scales", {}))
        self.training_metrics = {}
        self.last_perf_time = time.time()
        self.last_perf_step = 0
        self._load()
        self._load_preload_replay()

    def _get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", required=True, type=str, help="Name of the task to run.")
        parser.add_argument("--checkpoint", type=str, help="Path of a FastSAC checkpoint to resume. Use -1 for latest.")
        parser.add_argument("--num_envs", type=int, help="Number of environments.")
        parser.add_argument("--headless", type=bool, help="Run headless.")
        parser.add_argument("--sim_device", type=str, help="IsaacGym sim device.")
        parser.add_argument("--rl_device", type=str, help="Learner device.")
        parser.add_argument("--seed", type=int, help="Random seed.")
        parser.add_argument("--max_iterations", type=int, help="Number of FastSAC environment steps.")
        parser.add_argument("--replay_path", type=str, help="Comma-separated replay .npz files to preload.")
        parser.add_argument("--teacher_policy_path", type=str, help="Optional TorchScript teacher policy for BC warm start.")
        parser.add_argument("--disable_record_video", action="store_true", help="Disable viewer video recording.")
        self.args = parser.parse_args()

    def _fast_sac_defaults(self):
        return {
            "num_learning_iterations": self.cfg["basic"].get("max_iterations", 100000),
            "critic_learning_rate": 3.0e-4,
            "actor_learning_rate": 3.0e-4,
            "alpha_learning_rate": 3.0e-4,
            "buffer_size": 1024,
            "gamma": 0.97,
            "tau": 0.125,
            "batch_size": 8192,
            "learning_starts": 10,
            "policy_frequency": 4,
            "num_updates": 8,
            "target_entropy_ratio": 0.0,
            "num_atoms": 101,
            "v_min": -20.0,
            "v_max": 20.0,
            "critic_hidden_dim": 768,
            "actor_hidden_dim": 512,
            "use_symmetry": True,
            "alpha_init": 0.001,
            "use_autotune": True,
            "use_tanh": True,
            "log_std_max": 0.0,
            "log_std_min": -5.0,
            "compile": True,
            "obs_normalization": True,
            "use_layer_norm": True,
            "use_privileged_critic": True,
            "num_q_networks": 2,
            "max_grad_norm": 0.0,
            "amp": True,
            "amp_dtype": "bf16",
            "weight_decay": 0.001,
            "save_interval": self.cfg.get("runner", {}).get("save_interval", 1000),
            "logging_interval": self.cfg.get("runner", {}).get("progress_interval", 100),
            "num_steps": 1,
            "save_replay_interval": 0,
            "save_final_replay": False,
            "max_replay_export_transitions": 1000000,
            "teacher_bc_coef": 0.0,
            "teacher_bc_start_step": 0,
            "teacher_bc_warmup_steps": 0,
            "teacher_bc_decay_steps": 0,
            "teacher_bc_min_coef": 0.0,
            "penalty_curriculum": {
                "enabled": True,
                "terms": ["action_rate"],
                "initial_scale": 0.5,
                "final_scale": 1.0,
                "warmup_steps": 30000,
            },
        }

    @staticmethod
    def _merge_dicts(base, override):
        merged = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = FastSACK1Runner._merge_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _update_cfg_from_args(self):
        cfg_file = os.path.join("envs", f"{self.args.task}.yaml")
        self.cfg = load_config(cfg_file)
        self.cfg.setdefault("algorithm", {})
        self.cfg["algorithm"]["fast_sac"] = self._merge_dicts(
            self._fast_sac_defaults(),
            self.cfg["algorithm"].get("fast_sac", {}),
        )
        for arg, value in vars(self.args).items():
            if value is None:
                continue
            if arg == "task":
                continue
            if arg == "num_envs":
                self.cfg["env"]["num_envs"] = value
            elif arg == "replay_path":
                paths = [path.strip() for path in str(value).split(",") if path.strip()]
                self.cfg["algorithm"]["fast_sac"]["preload_replay_paths"] = paths
            elif arg == "teacher_policy_path":
                self.cfg["algorithm"]["fast_sac"]["teacher_policy_path"] = str(value)
            elif arg == "disable_record_video":
                if value:
                    self.cfg["viewer"]["record_video"] = False
            elif arg in ("checkpoint", "headless", "sim_device", "rl_device", "seed", "max_iterations"):
                self.cfg["basic"][arg] = value
        if not self.test:
            self.cfg["viewer"]["record_video"] = False
        self.cfg["basic"]["max_iterations"] = int(
            self.cfg["algorithm"]["fast_sac"].get(
                "num_learning_iterations",
                self.cfg["basic"].get("max_iterations", 100000),
            )
            if self.args.max_iterations is None
            else self.cfg["basic"]["max_iterations"]
        )

    def _set_seed(self):
        if self.cfg["basic"]["seed"] == -1:
            self.cfg["basic"]["seed"] = np.random.randint(0, 10000)
        print(f"Setting seed: {self.cfg['basic']['seed']}")
        random.seed(self.cfg["basic"]["seed"])
        np.random.seed(self.cfg["basic"]["seed"])
        torch.manual_seed(self.cfg["basic"]["seed"])
        os.environ["PYTHONHASHSEED"] = str(self.cfg["basic"]["seed"])
        torch.cuda.manual_seed(self.cfg["basic"]["seed"])
        torch.cuda.manual_seed_all(self.cfg["basic"]["seed"])

    def _action_scale_tensor(self):
        configured = self.sac_cfg.get("action_scale")
        if configured is not None:
            if isinstance(configured, (list, tuple)):
                return torch.tensor(configured, dtype=torch.float, device=self.device)
            return torch.full((self.env.num_actions,), float(configured), dtype=torch.float, device=self.device)
        if getattr(self.env, "action_clip_by_index", None) is not None:
            return self.env.action_clip_by_index.squeeze(0).to(self.device).float()
        return torch.full(
            (self.env.num_actions,),
            float(self.cfg["normalization"].get("clip_actions", 1.0)),
            dtype=torch.float,
            device=self.device,
        )

    def _make_optimizer(self, parameters, lr, weight_decay=None):
        weight_decay = float(self.sac_cfg.get("weight_decay", 0.001) if weight_decay is None else weight_decay)
        kwargs = {"lr": lr, "weight_decay": weight_decay, "betas": (0.9, 0.95)}
        if str(self.device).startswith("cuda") and bool(self.sac_cfg.get("fused_optimizer", True)):
            try:
                return torch.optim.AdamW(parameters, fused=True, **kwargs)
            except TypeError:
                pass
        return torch.optim.AdamW(parameters, **kwargs)

    @contextmanager
    def _maybe_amp(self):
        if not self.use_amp:
            with nullcontext():
                yield
            return
        with torch.cuda.amp.autocast(dtype=self.amp_dtype, enabled=True):
            yield

    def _critic_obs(self, obs, privileged_obs):
        if not self.use_privileged_critic:
            return obs
        return torch.cat((obs, privileged_obs), dim=-1)

    def _normalize_obs(self, obs, update=True):
        if not bool(self.sac_cfg.get("obs_normalization", True)):
            return obs
        return self.obs_normalizer(obs, update=update)

    def _normalize_critic_obs(self, obs, update=True):
        if not bool(self.sac_cfg.get("obs_normalization", True)):
            return obs
        return self.critic_obs_normalizer(obs, update=update)

    def _policy(self, obs, dones=None):
        return self.actor.explore(obs, deterministic=False)

    @staticmethod
    def _identity_normalize(obs, update=True):
        return obs

    def _compiled_runtime(self):
        policy = self._policy
        update_critic_and_alpha = self._update_critic_and_alpha
        update_actor = self._update_actor
        if bool(self.sac_cfg.get("obs_normalization", True)):
            normalize_obs = self.obs_normalizer.forward
            normalize_critic_obs = self.critic_obs_normalizer.forward
        else:
            normalize_obs = self._identity_normalize
            normalize_critic_obs = self._identity_normalize
        if not bool(self.sac_cfg.get("compile", True)):
            return policy, normalize_obs, normalize_critic_obs, update_critic_and_alpha, update_actor
        if not hasattr(torch, "compile"):
            print("[fastsac] torch.compile unavailable; running without compilation.")
            return policy, normalize_obs, normalize_critic_obs, update_critic_and_alpha, update_actor
        try:
            return (
                torch.compile(policy),
                torch.compile(normalize_obs),
                torch.compile(normalize_critic_obs),
                torch.compile(update_critic_and_alpha),
                torch.compile(update_actor),
            )
        except Exception as exc:
            print(f"[fastsac] torch.compile setup failed; running uncompiled: {exc}")
            return policy, normalize_obs, normalize_critic_obs, update_critic_and_alpha, update_actor

    def _warn_if_not_fast_path(self):
        sim_device = str(self.cfg["basic"].get("sim_device", ""))
        rl_device = str(self.cfg["basic"].get("rl_device", self.device))
        if not sim_device.startswith("cuda") or not rl_device.startswith("cuda") or not torch.cuda.is_available():
            print(
                "[fastsac] WARNING: the 15-minute paper result assumes GPU physics and GPU learning "
                "on a single RTX 4090-class device. Current devices: "
                f"sim_device={sim_device}, rl_device={rl_device}, cuda_available={torch.cuda.is_available()}."
            )

    def _load_teacher_policy(self):
        path = self.sac_cfg.get("teacher_policy_path")
        if not path:
            return None
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        if not os.path.exists(path):
            print(f"[fastsac] teacher policy not found: {path}")
            return None
        try:
            policy = torch.jit.load(path, map_location=self.device)
            policy.eval()
            print(f"[fastsac] teacher policy loaded: {path}")
            return policy
        except Exception as exc:
            print(f"[fastsac] failed to load teacher policy '{path}': {exc}")
            return None

    def _resolve_checkpoint_arg(self, checkpoint):
        if not checkpoint:
            return None
        if checkpoint not in ("-1", -1):
            return str(checkpoint)
        task_name = self.cfg["basic"].get("log_task", self.cfg["basic"]["task"])
        robot_type = self._get_robot_type(task_name)
        patterns = [
            os.path.join("logs", robot_type, task_name, "**", "*.pth"),
            os.path.join("logs", robot_type, "**", "*.pth"),
            os.path.join("logs", "**", "*.pth"),
        ]
        for pattern in patterns:
            matches = sorted(glob.glob(pattern, recursive=True), key=os.path.getmtime)
            matches = [path for path in matches if self._looks_like_fastsac_checkpoint(path)]
            if matches:
                return matches[-1]
        return None

    def _looks_like_fastsac_checkpoint(self, path):
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
            return bool(state.get("fast_sac", False) or "actor_state_dict" in state)
        except Exception:
            return False

    def _load(self):
        checkpoint_path = self._resolve_checkpoint_arg(self.cfg["basic"].get("checkpoint"))
        if checkpoint_path is None:
            return
        print(f"[fastsac] loading checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(state["actor_state_dict"])
        self.qnet.load_state_dict(state["qnet_state_dict"])
        self.qnet_target.load_state_dict(state.get("qnet_target_state_dict", state["qnet_state_dict"]))
        if state.get("obs_normalizer_state") is not None:
            self.obs_normalizer.load_state_dict(state["obs_normalizer_state"])
        if state.get("critic_obs_normalizer_state") is not None:
            self.critic_obs_normalizer.load_state_dict(state["critic_obs_normalizer_state"])
        self.log_alpha.data.copy_(state.get("log_alpha", self.log_alpha.detach()).to(self.device))
        if bool(self.sac_cfg.get("load_optimizer", True)):
            for key, optimizer in (
                ("actor_optimizer_state_dict", self.actor_optimizer),
                ("q_optimizer_state_dict", self.q_optimizer),
                ("alpha_optimizer_state_dict", self.alpha_optimizer),
            ):
                if key in state:
                    try:
                        optimizer.load_state_dict(state[key])
                    except Exception as exc:
                        print(f"[fastsac] failed to load {key}: {exc}")
            if state.get("grad_scaler_state_dict") is not None:
                self.scaler.load_state_dict(state["grad_scaler_state_dict"])
        if bool(self.sac_cfg.get("load_replay_from_checkpoint", False)) and "replay" in state:
            self.replay.load_state_dict(state["replay"])
        self.global_step = int(state.get("global_step", 0))
        try:
            if "curriculum" in state:
                self.env.curriculum_prob = state["curriculum"].to(self.device)
        except Exception as exc:
            print(f"[fastsac] failed to restore curriculum: {exc}")

    def _load_preload_replay(self):
        paths = self.sac_cfg.get("preload_replay_paths", [])
        if isinstance(paths, str):
            paths = [path.strip() for path in paths.split(",") if path.strip()]
        max_transitions = self.sac_cfg.get("max_preload_replay_transitions")
        for path in paths or []:
            resolved = path if os.path.isabs(path) else os.path.abspath(path)
            if not os.path.exists(resolved):
                print(f"[fastsac] preload replay not found: {path}")
                continue
            try:
                added = self.replay.load_npz(resolved, max_transitions=max_transitions)
                print(f"[fastsac] preloaded {added} replay transitions from {resolved}")
            except Exception as exc:
                print(f"[fastsac] failed to preload replay '{path}': {exc}")

    def _checkpoint_state(self, include_replay=False):
        state = {
            "fast_sac": True,
            "actor_state_dict": self.actor.state_dict(),
            "qnet_state_dict": self.qnet.state_dict(),
            "qnet_target_state_dict": self.qnet_target.state_dict(),
            "obs_normalizer_state": self.obs_normalizer.state_dict(),
            "critic_obs_normalizer_state": self.critic_obs_normalizer.state_dict(),
            "log_alpha": self.log_alpha.detach(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "q_optimizer_state_dict": self.q_optimizer.state_dict(),
            "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict(),
            "grad_scaler_state_dict": self.scaler.state_dict(),
            "task_cfg": copy.deepcopy(self.cfg),
            "global_step": self.global_step,
            "curriculum": getattr(self.env, "curriculum_prob", None),
            "policy_wrapper": "utils.fast_sac.FastSACPolicyWrapper",
        }
        if include_replay:
            state["replay"] = self.replay.state_dict()
        return state

    def _save_checkpoint(self, recorder, step, include_replay=False):
        path = os.path.join(recorder.model_dir, f"model_{step}.pth")
        print(f"[fastsac] saving checkpoint to {path}")
        torch.save(self._checkpoint_state(include_replay=include_replay), path)
        return path

    def _save_replay_npz(self, recorder, step):
        replay_dir = os.path.join(recorder.dir, "replay")
        path = os.path.join(replay_dir, f"replay_{step}.npz")
        max_transitions = self.sac_cfg.get("max_replay_export_transitions")
        count = self.replay.export_npz(path, max_transitions=max_transitions)
        print(f"[fastsac] saved {count} replay transitions to {path}")
        return path

    def _penalty_curriculum_scale(self):
        cfg = self.sac_cfg.get("penalty_curriculum", {})
        if not bool(cfg.get("enabled", False)):
            return 1.0
        warmup_steps = max(1, int(cfg.get("warmup_steps", 1)))
        mix = min(max(self.global_step / warmup_steps, 0.0), 1.0)
        initial = float(cfg.get("initial_scale", 1.0))
        final = float(cfg.get("final_scale", 1.0))
        return initial + mix * (final - initial)

    def _apply_penalty_curriculum(self):
        cfg = self.sac_cfg.get("penalty_curriculum", {})
        if not bool(cfg.get("enabled", False)):
            return 1.0
        scale = self._penalty_curriculum_scale()
        for term in cfg.get("terms", []):
            if term in self.base_reward_scales:
                self.env.reward_scales[term] = self.base_reward_scales[term] * scale
        return scale

    def _mirror_action(self, action):
        if action.shape[-1] != 12:
            return action
        indices = torch.tensor([6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5], dtype=torch.long, device=action.device)
        signs = torch.tensor([1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0], dtype=action.dtype, device=action.device)
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

    def _mirror_privileged_obs(self, privileged_obs):
        if privileged_obs.shape[-1] < 14:
            return privileged_obs
        mirrored = privileged_obs.clone()
        mirrored[..., 1] = -privileged_obs[..., 1]
        mirrored[..., 5] = -privileged_obs[..., 5]
        mirrored[..., 9] = -privileged_obs[..., 9]
        mirrored[..., 11] = -privileged_obs[..., 11]
        mirrored[..., 13] = -privileged_obs[..., 13]
        return mirrored

    def _mirror_critic_obs(self, critic_obs):
        if not self.use_privileged_critic:
            return self._mirror_obs(critic_obs)
        public = self._mirror_obs(critic_obs[..., : self.env.num_obs])
        privileged = self._mirror_privileged_obs(critic_obs[..., self.env.num_obs :])
        return torch.cat((public, privileged), dim=-1)

    def _augment_batch(self, batch):
        if not bool(self.sac_cfg.get("use_symmetry", True)):
            return batch
        return {
            "observations": torch.cat((batch["observations"], self._mirror_obs(batch["observations"])), dim=0),
            "actions": torch.cat((batch["actions"], self._mirror_action(batch["actions"])), dim=0),
            "rewards": batch["rewards"].repeat(2),
            "dones": batch["dones"].repeat(2),
            "truncations": batch["truncations"].repeat(2),
            "next_observations": torch.cat((batch["next_observations"], self._mirror_obs(batch["next_observations"])), dim=0),
            "critic_observations": torch.cat((batch["critic_observations"], self._mirror_critic_obs(batch["critic_observations"])), dim=0),
            "next_critic_observations": torch.cat(
                (batch["next_critic_observations"], self._mirror_critic_obs(batch["next_critic_observations"])),
                dim=0,
            ),
            "effective_n_steps": batch["effective_n_steps"].repeat(2),
            "raw_observations": torch.cat((batch["observations"], self._mirror_obs(batch["observations"])), dim=0),
        }

    def _prepare_update_batches(self, normalize_obs=None, normalize_critic_obs=None):
        if normalize_obs is None:
            normalize_obs = self._normalize_obs
        if normalize_critic_obs is None:
            normalize_critic_obs = self._normalize_critic_obs
        batch_size = int(self.sac_cfg.get("batch_size", 8192))
        num_updates = int(self.sac_cfg.get("num_updates", 8))
        per_env_batch = max(batch_size // self.env.num_envs, 1)
        large_batch = self.replay.sample(per_env_batch * num_updates)
        if "raw_observations" not in large_batch:
            large_batch["raw_observations"] = large_batch["observations"]
        large_batch = self._augment_batch(large_batch)
        raw_obs = large_batch["raw_observations"]
        large_batch["observations"] = normalize_obs(large_batch["observations"], update=True)
        large_batch["next_observations"] = normalize_obs(large_batch["next_observations"], update=True)
        large_batch["critic_observations"] = normalize_critic_obs(large_batch["critic_observations"], update=True)
        large_batch["next_critic_observations"] = normalize_critic_obs(
            large_batch["next_critic_observations"],
            update=True,
        )
        large_batch["raw_observations"] = raw_obs

        samples_per_update = per_env_batch * self.env.num_envs
        if bool(self.sac_cfg.get("use_symmetry", True)):
            samples_per_update *= 2
        batches = []
        for i in range(num_updates):
            start = i * samples_per_update
            end = (i + 1) * samples_per_update
            batches.append({key: value[start:end] for key, value in large_batch.items()})
        return batches

    def _soft_update_targets(self):
        tau = float(self.sac_cfg.get("tau", 0.125))
        with torch.no_grad():
            src_ps = [p.data for p in self.qnet.parameters()]
            tgt_ps = [p.data for p in self.qnet_target.parameters()]
            try:
                torch._foreach_mul_(tgt_ps, 1.0 - tau)
                torch._foreach_add_(tgt_ps, src_ps, alpha=tau)
            except Exception:
                for src, target in zip(src_ps, tgt_ps):
                    target.mul_(1.0 - tau).add_(src, alpha=tau)

    def _teacher_bc_coef(self):
        if self.teacher_policy is None:
            return 0.0
        coef = float(self.sac_cfg.get("teacher_bc_coef", 0.0))
        start = int(self.sac_cfg.get("teacher_bc_start_step", 0))
        if coef <= 0.0 or self.global_step < start:
            return 0.0
        warmup = int(self.sac_cfg.get("teacher_bc_warmup_steps", 0))
        if warmup > 0:
            coef *= min(max((self.global_step - start + 1) / warmup, 0.0), 1.0)
        decay = int(self.sac_cfg.get("teacher_bc_decay_steps", 0))
        if decay > 0:
            mix = min(max((self.global_step - start) / decay, 0.0), 1.0)
            coef = max(float(self.sac_cfg.get("teacher_bc_min_coef", 0.0)), coef * (1.0 - mix))
        return coef

    def _update_critic_and_alpha(self, batch):
        gamma = float(self.sac_cfg.get("gamma", 0.97))
        with self._maybe_amp():
            with torch.no_grad():
                next_action, next_log_prob = self.actor.sample(batch["next_observations"])
                discount = gamma ** batch["effective_n_steps"]
                bootstrap = (batch["truncations"].bool() | ~batch["dones"].bool()).float()
                target_dist = self.qnet_target.projection(
                    batch["next_critic_observations"],
                    next_action,
                    batch["rewards"] - discount * bootstrap * self.log_alpha.exp() * next_log_prob,
                    bootstrap,
                    discount,
                )
                target_values = self.qnet_target.get_value(target_dist)
            q_outputs = self.qnet(batch["critic_observations"], batch["actions"])
            critic_log_probs = F.log_softmax(q_outputs, dim=-1)
            critic_losses = -torch.sum(target_dist * critic_log_probs, dim=-1)
            q_loss = critic_losses.mean(dim=1).sum(dim=0)

        self.q_optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(q_loss).backward()
        self.scaler.unscale_(self.q_optimizer)
        max_grad_norm = float(self.sac_cfg.get("max_grad_norm", 0.0))
        if max_grad_norm > 0.0:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), max_grad_norm)
        else:
            critic_grad_norm = torch.tensor(0.0, device=self.device)
        self.scaler.step(self.q_optimizer)
        self.scaler.update()

        alpha_loss = torch.tensor(0.0, dtype=torch.float, device=self.device)
        if bool(self.sac_cfg.get("use_autotune", True)):
            self.alpha_optimizer.zero_grad(set_to_none=True)
            with self._maybe_amp():
                alpha_loss = (-self.log_alpha.exp() * (next_log_prob.detach() + self.target_entropy)).mean()
            self.scaler.scale(alpha_loss).backward()
            self.scaler.unscale_(self.alpha_optimizer)
            self.scaler.step(self.alpha_optimizer)
            self.scaler.update()

        return {
            "q_loss": q_loss.detach(),
            "critic_grad_norm": critic_grad_norm.detach(),
            "alpha_loss": alpha_loss.detach(),
            "target_q_max": target_values.max().detach(),
            "target_q_min": target_values.min().detach(),
        }

    def _update_actor(self, batch):
        with self._maybe_amp():
            action, log_prob = self.actor.sample(batch["observations"])
            q_outputs = self.qnet(batch["critic_observations"], action)
            q_probs = F.softmax(q_outputs, dim=-1)
            q_values = self.qnet.get_value(q_probs)
            actor_loss = (self.log_alpha.exp().detach() * log_prob - q_values.mean(dim=0)).mean()
            teacher_coef = self._teacher_bc_coef()
            teacher_bc_loss = torch.tensor(0.0, dtype=torch.float, device=self.device)
            if teacher_coef > 0.0:
                with torch.no_grad():
                    teacher_action = self.teacher_policy(batch["raw_observations"]).to(self.device)
                student_action = self.actor(batch["observations"])
                teacher_bc_loss = F.mse_loss(student_action, teacher_action)
                actor_loss = actor_loss + teacher_coef * teacher_bc_loss
            with torch.no_grad():
                _, log_std = self.actor.distribution_params(batch["observations"])
                policy_entropy = -log_prob.mean()
                action_std = log_std.exp().mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(actor_loss).backward()
        self.scaler.unscale_(self.actor_optimizer)
        max_grad_norm = float(self.sac_cfg.get("max_grad_norm", 0.0))
        if max_grad_norm > 0.0:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
        else:
            actor_grad_norm = torch.tensor(0.0, device=self.device)
        self.scaler.step(self.actor_optimizer)
        self.scaler.update()
        return {
            "actor_loss": actor_loss.detach(),
            "actor_grad_norm": actor_grad_norm.detach(),
            "policy_entropy": policy_entropy.detach(),
            "action_std": action_std.detach(),
            "teacher_bc_loss": teacher_bc_loss.detach(),
            "teacher_bc_coef": torch.tensor(teacher_coef, dtype=torch.float, device=self.device),
        }

    def _metric_add(self, metrics):
        for key, value in metrics.items():
            if torch.is_tensor(value):
                value = float(value.detach().mean().item())
            else:
                value = float(value)
            total, count = self.training_metrics.get(key, (0.0, 0))
            self.training_metrics[key] = (total + value, count + 1)

    def _metric_mean_and_clear(self):
        out = {
            key: total / max(count, 1)
            for key, (total, count) in self.training_metrics.items()
        }
        self.training_metrics = {}
        return out

    @staticmethod
    def _format_duration(seconds):
        seconds = max(0, int(seconds))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
        return f"{minutes:d}m {seconds:02d}s"

    @staticmethod
    def _get_robot_type(task_name):
        if task_name.startswith("K1"):
            return "K1"
        if task_name.startswith("T1"):
            return "T1"
        return "Unknown"

    def train(self):
        recorder = Recorder(self.cfg)
        self._warn_if_not_fast_path()
        policy, normalize_obs, normalize_critic_obs, update_critic_and_alpha, update_actor = self._compiled_runtime()
        obs, infos = self.env.reset()
        obs = obs.to(self.device)
        privileged_obs = infos["privileged_obs"].to(self.device)
        max_iterations = int(self.cfg["basic"]["max_iterations"])
        save_interval = int(self.sac_cfg.get("save_interval", self.cfg["runner"].get("save_interval", 1000)))
        logging_interval = max(1, int(self.sac_cfg.get("logging_interval", self.cfg["runner"].get("progress_interval", 100))))
        replay_save_interval = int(self.sac_cfg.get("save_replay_interval", 0))
        learning_starts = int(self.sac_cfg.get("learning_starts", 10))
        policy_frequency = max(1, int(self.sac_cfg.get("policy_frequency", 4)))
        start_time = time.time()
        print(f"Training logs: {recorder.dir}")
        print(
            f"FastSAC progress: {self.global_step}/{max_iterations} | "
            f"num_envs={self.env.num_envs} buffer_size={self.replay.buffer_size} "
            f"batch={self.sac_cfg.get('batch_size', 8192)} updates={self.sac_cfg.get('num_updates', 8)}"
        )

        dones_for_policy = None
        while self.global_step < max_iterations:
            if hasattr(self.env, "update_training_curriculum"):
                self.env.update_training_curriculum(self.global_step)
            penalty_scale = self._apply_penalty_curriculum()

            critic_obs = self._critic_obs(obs, privileged_obs)
            with torch.no_grad(), self._maybe_amp():
                norm_obs = normalize_obs(obs, update=False)
                action = policy(norm_obs, dones_for_policy)
            next_obs, reward, done, infos = self.env.step(action.float())
            next_obs = next_obs.to(self.device)
            reward = reward.to(self.device)
            done = done.to(self.device)
            next_privileged_obs = infos["privileged_obs"].to(self.device)
            final_obs = infos.get("final_obs", next_obs).to(self.device)
            final_privileged_obs = infos.get("final_privileged_obs", next_privileged_obs).to(self.device)
            final_critic_obs = self._critic_obs(final_obs, final_privileged_obs)
            time_outs = infos["time_outs"].to(self.device)
            truncations = time_outs
            self.replay.extend(
                obs,
                action.float(),
                reward,
                done,
                truncations,
                final_obs,
                critic_obs,
                final_critic_obs,
            )

            ep_info = {"reward": reward}
            if self.cfg["runner"].get("log_reward_terms", False):
                ep_info.update(infos.get("rew_terms", {}))
            recorder.record_episode_statistics(done, ep_info, self.global_step, self.global_step % logging_interval == 0)

            obs = next_obs
            privileged_obs = next_privileged_obs
            dones_for_policy = done
            self.global_step += 1

            if self.global_step > learning_starts:
                update_start = time.time()
                prepared_batches = self._prepare_update_batches(normalize_obs, normalize_critic_obs)
                for update_idx, batch in enumerate(prepared_batches):
                    critic_metrics = update_critic_and_alpha(batch)
                    self._metric_add({f"sac/{key}": value for key, value in critic_metrics.items()})
                    should_update_actor = (
                        (len(prepared_batches) > 1 and update_idx % policy_frequency == 1)
                        or (len(prepared_batches) == 1 and self.global_step % policy_frequency == 0)
                    )
                    if should_update_actor:
                        actor_metrics = update_actor(batch)
                        self._metric_add({f"sac/{key}": value for key, value in actor_metrics.items()})
                    self._soft_update_targets()
                update_time = max(time.time() - update_start, 1.0e-9)
                self._metric_add({"perf/updates_per_sec": len(prepared_batches) / update_time})

            if save_interval > 0 and self.global_step % save_interval == 0:
                self._save_checkpoint(
                    recorder,
                    self.global_step,
                    include_replay=bool(self.sac_cfg.get("checkpoint_replay", False)),
                )
            if replay_save_interval > 0 and self.global_step % replay_save_interval == 0:
                self._save_replay_npz(recorder, self.global_step)

            if self.global_step == 1 or self.global_step % logging_interval == 0 or self.global_step >= max_iterations:
                now = time.time()
                interval = max(now - self.last_perf_time, 1.0e-9)
                step_delta = self.global_step - self.last_perf_step
                steps_per_sec = step_delta / interval
                env_steps_per_sec = steps_per_sec * self.env.num_envs
                elapsed = now - start_time
                eta = (max_iterations - self.global_step) / max(steps_per_sec, 1.0e-9)
                mean_metrics = self._metric_mean_and_clear()
                statistics = {
                    "sac/alpha": float(self.log_alpha.exp().detach().item()),
                    "replay/count": float(len(self.replay)),
                    "replay/valid_steps": float(self.replay.valid_steps),
                    "perf/steps_per_sec": float(steps_per_sec),
                    "perf/env_steps_per_sec": float(env_steps_per_sec),
                    "perf/replay_ratio": float(self.sac_cfg.get("num_updates", 8) * self.sac_cfg.get("batch_size", 8192) / max(self.env.num_envs, 1)),
                    "rollout/reward_mean": float(reward.mean().item()),
                    "rollout/done_rate": float(done.float().mean().item()),
                    "curriculum/penalty_scale": float(penalty_scale),
                    "training_phase/index": float(getattr(self.env, "training_phase_index", 0)),
                    "training_phase/progress": float(getattr(self.env, "training_phase_progress", 0.0)),
                }
                statistics.update(mean_metrics)
                recorder.record_statistics(statistics, self.global_step)
                self.last_perf_time = now
                self.last_perf_step = self.global_step
                percent = 100.0 * self.global_step / max(max_iterations, 1)
                print(
                    f"[fastsac] {self.global_step}/{max_iterations} ({percent:5.2f}%) | "
                    f"elapsed {self._format_duration(elapsed)} | eta {self._format_duration(eta)} | "
                    f"reward {statistics['rollout/reward_mean']:.4f} | done {statistics['rollout/done_rate'] * 100.0:.2f}% | "
                    f"env_steps/s {env_steps_per_sec:,.0f} | replay {len(self.replay):,} | "
                    f"q {statistics.get('sac/q_loss', 0.0):.4f} | actor {statistics.get('sac/actor_loss', 0.0):.4f} | "
                    f"alpha {statistics['sac/alpha']:.5f}"
                )

        self._save_checkpoint(recorder, self.global_step, include_replay=bool(self.sac_cfg.get("checkpoint_replay", False)))
        if bool(self.sac_cfg.get("save_final_replay", True)):
            self._save_replay_npz(recorder, self.global_step)

    def export_policy_wrapper(self):
        wrapper = FastSACPolicyWrapper(
            copy.deepcopy(self.actor).to("cpu"),
            copy.deepcopy(self.obs_normalizer).to("cpu") if bool(self.sac_cfg.get("obs_normalization", True)) else None,
        )
        wrapper.eval()
        return wrapper
