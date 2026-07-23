import os
import glob
import yaml
import argparse
import numpy as np
import random
import time
import signal
import imageio
import subprocess
import sys

# Import envs first to initialize isaacgym modules
from envs import *

# Import torch and utils after isaacgym modules are initialized
import torch
import torch.nn.functional as F
from utils.models.BaseAC import *
from utils.models.WorldModel import WorldModel
from utils.buffer import ExperienceBuffer
from utils.command_metrics import CommandVelocityMetrics
from utils.sirl_replay import (
    SIRL_METRIC_KEYS,
    TrajectorySIRLReplayBuffer,
    WorldModelReplayBuffer,
    normalize_by_percentiles,
    percentile,
)
from utils.utils import discount_values, surrogate_loss
from utils.recorder import Recorder

# Dynamic task class loading
import importlib
import inspect
import pkgutil


def merge_dicts(base, override):
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(cfg_file, visited=None):
    if visited is None:
        visited = set()

    cfg_file = os.path.normpath(cfg_file)
    if cfg_file in visited:
        raise ValueError(f"Recursive config inheritance detected for {cfg_file}")
    visited.add(cfg_file)

    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    parent = cfg.pop("extends", None)
    if not parent:
        return cfg

    if not parent.endswith(".yaml"):
        parent = os.path.join("envs", f"{parent}.yaml")
    elif not os.path.isabs(parent):
        parent = os.path.join(os.path.dirname(cfg_file), parent)

    parent_cfg = load_config(parent, visited)
    return merge_dicts(parent_cfg, cfg)

def get_task_class(task_name):
    """
    Dynamically load task class by name.
    Searches through all modules in the envs package for classes that match the task name.
    Handles different naming conventions (Base_Walk vs BaseWalk, etc.)
    """
    # Generate possible class name variations
    possible_names = [task_name]
    
    # Handle underscore to camelCase conversion (Base_Walk -> BaseWalk)
    if '_' in task_name:
        camel_case = ''.join(word.capitalize() for word in task_name.split('_'))
        possible_names.append(camel_case)
    
    # Handle camelCase to underscore conversion (BaseWalk -> Base_Walk)
    if not '_' in task_name and any(c.isupper() for c in task_name[1:]):
        import re
        snake_case = re.sub(r'(?<!^)(?=[A-Z])', '_', task_name).lower()
        snake_case = snake_case[0].upper() + snake_case[1:]  # Capitalize first letter
        possible_names.append(snake_case)
    
    # First try to get from the envs module (which imports all task classes)
    try:
        envs_module = importlib.import_module('envs')
        for name, obj in inspect.getmembers(envs_module):
            if inspect.isclass(obj) and name in possible_names:
                return obj
    except Exception as e:
        print(f"Error loading from envs module: {e}")
    
    # If not found, try to import from specific paths
    task_paths = [
        f"envs.T1.{task_name.lower()}",
        f"envs.K1.{task_name.lower()}",
        f"envs.{task_name}",
    ]
    
    for path in task_paths:
        try:
            module = importlib.import_module(path)
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and name in possible_names:
                    return obj
        except ImportError:
            continue
        except Exception as e:
            print(f"Error loading from {path}: {e}")
            continue
    
    return None


def get_model_class(model_name):
    """
    Resolve a model class by name. Supports names from config/CLI like
    "BaseActorCritic", "BaseAC", "OdometryActorCritic", or "OdometryAC".
    Falls back to BaseActorCritic if name is None/empty.
    """
    # Default
    if not model_name:
        return BaseActorCritic

    key = str(model_name)
    key_norm = key.replace("_", "").replace(" ", "").lower()

    # Quick direct matches for common defaults
    if key_norm in {"baseactorcritic", "baseac"}:
        return BaseActorCritic

    # Dynamically scan utils.models package for classes
    try:
        models_pkg = importlib.import_module('utils.models')
        discovered = []
        for finder, mod_name, is_pkg in pkgutil.walk_packages(models_pkg.__path__, models_pkg.__name__ + '.'):
            try:
                module = importlib.import_module(mod_name)
            except Exception:
                continue
            for attr_name, obj in inspect.getmembers(module, inspect.isclass):
                # Only consider classes that are defined in the module (avoid imported aliases)
                if getattr(obj, '__module__', '').startswith(mod_name):
                    try:
                        import torch
                        if issubclass(obj, torch.nn.Module):
                            discovered.append(obj)
                    except Exception:
                        continue
        # Try exact name match first
        for cls in discovered:
            if cls.__name__ == key:
                return cls
        # Try normalized name match (ignore underscores/spaces and case)
        for cls in discovered:
            if cls.__name__.replace("_", "").replace(" ", "").lower() == key_norm:
                return cls
        # As a convenience, prefer classes ending with 'ActorCritic' if multiple choices
        for cls in discovered:
            if cls.__name__.lower().endswith('actorcritic') and cls.__name__.lower() == key_norm:
                return cls
        available = ', '.join(sorted({c.__name__ for c in discovered}))
        raise ValueError(f"Unknown model class: {model_name}. Available: {available}")
    except Exception as e:
        raise ValueError(f"Unknown model class: {model_name} ({e})")


class Runner:

    def __init__(self, test=False):
        self.test = test
        # prepare the environment
        self._get_args()
        self._update_cfg_from_args()
        self._set_seed()
        task_name = self.cfg["basic"]["task"]
        # Extract task name from path (e.g., "T1/T1" -> "T1")
        if "/" in task_name:
            task_name = task_name.split("/")[-1]
        
        # Dynamically load the task class
        task_class = get_task_class(task_name)
        if task_class is None:
            raise ValueError(f"Unknown task: {task_name}. Could not find a class named '{task_name}' in the envs package.")
        
        self.env = task_class(self.cfg)
        self.env.is_play = test

        self.device = self.cfg["basic"]["rl_device"]
        self.learning_rate = self.cfg["algorithm"]["learning_rate"]
        self.init_learning_rate = self.learning_rate
        # Select model by config/CLI
        model_name = self.cfg["basic"].get("model", "BaseActorCritic")
        model_class = get_model_class(model_name)
        self.model = model_class(self.env.num_actions, self.env.num_obs, self.env.num_privileged_obs).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.buffer = ExperienceBuffer(self.cfg["runner"]["horizon_length"], self.env.num_envs, self.device)
        self.buffer.add_buffer("actions", (self.env.num_actions,))
        self.buffer.add_buffer("obses", (self.env.num_obs,))
        self.buffer.add_buffer("next_obses", (self.env.num_obs,))
        self.buffer.add_buffer("privileged_obses", (self.env.num_privileged_obs,))
        self.buffer.add_buffer("rewards", ())
        self.buffer.add_buffer("dones", (), dtype=bool)
        self.buffer.add_buffer("time_outs", (), dtype=bool)
        self._init_sirl()
        self._init_world_model()
        self._load()

    def _get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", required=True, type=str, help="Name of the task to run.")
        parser.add_argument("--checkpoint", type=str, help="Path of the model checkpoint to load. Overrides config file if provided.")
        parser.add_argument("--num_envs", type=int, help="Number of environments to create. Overrides config file if provided.")
        parser.add_argument("--headless", type=bool, help="Run headless without creating a viewer window. Overrides config file if provided.")
        parser.add_argument("--sim_device", type=str, help="Device for physics simulation. Overrides config file if provided.")
        parser.add_argument("--rl_device", type=str, help="Device for the RL algorithm. Overrides config file if provided.")
        parser.add_argument("--seed", type=int, help="Random seed. Overrides config file if provided.")
        parser.add_argument("--max_iterations", type=int, help="Maximum number of training iterations. Overrides config file if provided.")
        parser.add_argument("--model", type=str, help="Model class name to use (e.g., BaseActorCritic, OdometryActorCritic). Overrides config file if provided.")
        # Play-mode command overrides for parameterized walking tasks.
        parser.add_argument("--play_straight_eval", action="store_true", help="Use a clean straight-walk play command with fixed yaw and no disturbances.")
        parser.add_argument("--play_fixed_yaw", action="store_true", help="Start play episodes with yaw=0 instead of randomized yaw.")
        parser.add_argument("--play_free_yaw", action="store_true", help="Allow randomized initial yaw in play mode.")
        parser.add_argument("--play_no_disturbance", action="store_true", help="Disable random kick/push disturbances in play mode.")
        parser.add_argument("--play_with_disturbance", action="store_true", help="Enable random kick/push disturbances in play mode.")
        parser.add_argument("--play_lin_vel_x", type=float, help="Play command forward velocity [m/s].")
        parser.add_argument("--play_lin_vel_y", type=float, help="Play command lateral velocity [m/s].")
        parser.add_argument("--play_ang_vel_yaw", type=float, help="Play command yaw velocity [rad/s].")
        parser.add_argument("--play_gait_frequency", type=float, help="Play command gait frequency [Hz].")
        parser.add_argument("--play_foot_yaw_l", type=float, help="Play command left foot yaw [rad].")
        parser.add_argument("--play_foot_yaw_r", type=float, help="Play command right foot yaw [rad].")
        parser.add_argument("--play_body_pitch", type=float, help="Play command body pitch target [rad].")
        parser.add_argument("--play_body_roll", type=float, help="Play command body roll target [rad].")
        parser.add_argument("--play_feet_offset_x", type=float, help="Play command feet x-offset target [m].")
        parser.add_argument("--play_feet_offset_y", type=float, help="Play command feet y-offset target [m].")
        parser.add_argument("--play_target_x", type=float, help="Play absolute target x [m] for target-pose tasks.")
        parser.add_argument("--play_target_y", type=float, help="Play absolute target y [m] for target-pose tasks.")
        parser.add_argument("--play_target_theta", type=float, help="Play absolute target heading [rad] for target-pose tasks.")
        parser.add_argument("--play_target_local_x", type=float, help="Play local target x [m] for target-pose tasks.")
        parser.add_argument("--play_target_local_y", type=float, help="Play local target y [m] for target-pose tasks.")
        parser.add_argument("--play_target_heading_offset", type=float, help="Play target heading offset [rad] for target-pose tasks.")
        parser.add_argument("--play_default_hip_pitch", type=float, help="Override play default hip pitch [rad].")
        parser.add_argument("--play_default_knee_pitch", type=float, help="Override play default knee pitch [rad].")
        parser.add_argument("--play_default_ankle_pitch", type=float, help="Override play default ankle pitch [rad].")
        parser.add_argument("--play_base_height", type=float, help="Override play initial base height [m].")
        parser.add_argument("--play_velocity_metrics", action="store_true", help="Print commanded vs actual base velocity in play mode.")
        parser.add_argument("--play_metrics_interval_s", type=float, default=1.0, help="Seconds between play velocity metric prints.")
        parser.add_argument("--play_metrics_window_s", type=float, default=3.0, help="Rolling window for play velocity averages.")
        parser.add_argument("--play_metrics_warmup_s", type=float, default=1.0, help="Warmup before play velocity samples count.")
        parser.add_argument("--play_metrics_csv", type=str, help="Optional CSV path for play velocity metrics.")
        parser.add_argument("--play_metrics_csv_sample_s", type=float, default=0.05, help="Seconds between CSV samples.")
        # Video recording mode arguments (for separate process recording)
        parser.add_argument("--record_video_mode", action="store_true", help="Enable video recording mode (record and exit).")
        parser.add_argument("--disable_record_video", action="store_true", help="Disable video recording even if the config enables it.")
        parser.add_argument("--video_duration", type=float, help="Duration of video to record in seconds.")
        parser.add_argument("--video_iteration", type=int, help="Iteration number for wandb logging.")
        parser.add_argument("--video_output_path", type=str, help="Path where to save the video file.")
        parser.add_argument("--rewards_output_path", type=str, help="Path where to save the reward data JSON file.")
        self.args = parser.parse_args()

    # Override config file with args if needed
    def _update_cfg_from_args(self):
        cfg_file = os.path.join("envs", "{}.yaml".format(self.args.task))
        self.cfg = load_config(cfg_file)
        # Ensure default model if not present in config
        if "model" not in self.cfg.get("basic", {}):
            self.cfg.setdefault("basic", {})["model"] = "BaseActorCritic"
        play_command_arg_map = {
            "play_lin_vel_x": "lin_vel_x",
            "play_lin_vel_y": "lin_vel_y",
            "play_ang_vel_yaw": "ang_vel_yaw",
            "play_gait_frequency": "gait_frequency",
            "play_foot_yaw_l": "foot_yaw_L",
            "play_foot_yaw_r": "foot_yaw_R",
            "play_body_pitch": "body_pitch_target",
            "play_body_roll": "body_roll_target",
            "play_feet_offset_x": "feet_offset_x_target",
            "play_feet_offset_y": "feet_offset_y_target",
        }
        play_target_arg_map = {
            "play_target_x": "target_x",
            "play_target_y": "target_y",
            "play_target_theta": "target_theta",
            "play_target_local_x": "target_local_x",
            "play_target_local_y": "target_local_y",
            "play_target_heading_offset": "target_heading_offset",
        }
        play_arg_names = set(play_command_arg_map)
        play_arg_names.update(play_target_arg_map)
        play_arg_names.update(
            {
                "play_straight_eval",
                "play_fixed_yaw",
                "play_free_yaw",
                "play_no_disturbance",
                "play_with_disturbance",
                "play_velocity_metrics",
                "play_metrics_interval_s",
                "play_metrics_window_s",
                "play_metrics_warmup_s",
                "play_metrics_csv",
                "play_metrics_csv_sample_s",
                "play_default_hip_pitch",
                "play_default_knee_pitch",
                "play_default_ankle_pitch",
                "play_base_height",
            }
        )
        for arg in vars(self.args):
            if getattr(self.args, arg) is not None:
                if arg == "num_envs":
                    self.cfg["env"][arg] = getattr(self.args, arg)
                elif arg == "task" or arg in play_arg_names:
                    continue
                else:
                    self.cfg["basic"][arg] = getattr(self.args, arg)
        play_cfg = self.cfg.setdefault("commands", {}).setdefault("play", {})
        if self.args.play_straight_eval:
            play_cfg.update(
                {
                    "lin_vel_x": 0.2,
                    "lin_vel_y": 0.0,
                    "ang_vel_yaw": 0.0,
                    "gait_frequency": 1.5,
                    "foot_yaw_L": 0.0,
                    "foot_yaw_R": 0.0,
                    "body_pitch_target": 0.0,
                    "body_roll_target": 0.0,
                    "feet_offset_x_target": 0.0,
                    "feet_offset_y_target": 0.0,
                    "fixed_yaw": True,
                    "no_disturbance": True,
                }
            )
        for arg, cfg_key in play_command_arg_map.items():
            value = getattr(self.args, arg)
            if value is not None:
                play_cfg[cfg_key] = value
        for arg, cfg_key in play_target_arg_map.items():
            value = getattr(self.args, arg)
            if value is not None:
                play_cfg[cfg_key] = value
        if self.args.play_fixed_yaw:
            play_cfg["fixed_yaw"] = True
        if self.args.play_free_yaw:
            play_cfg["fixed_yaw"] = False
        if self.args.play_no_disturbance:
            play_cfg["no_disturbance"] = True
        if self.args.play_with_disturbance:
            play_cfg["no_disturbance"] = False
        default_joint_arg_map = {
            "play_default_hip_pitch": "Hip_Pitch",
            "play_default_knee_pitch": "Knee_Pitch",
            "play_default_ankle_pitch": "Ankle_Pitch",
        }
        default_joint_angles = self.cfg.setdefault("init_state", {}).setdefault("default_joint_angles", {})
        for arg, cfg_key in default_joint_arg_map.items():
            value = getattr(self.args, arg)
            if value is not None:
                default_joint_angles[cfg_key] = float(value)
        if self.args.play_base_height is not None:
            self.cfg.setdefault("init_state", {}).setdefault("pos", [0.0, 0.0, 0.58])[2] = float(self.args.play_base_height)
        if self.args.record_video_mode:
            self.cfg["viewer"]["record_video"] = True
        elif self.args.disable_record_video:
            self.cfg["viewer"]["record_video"] = False
        elif not self.test:
            # Disable video recording in training process - videos will be recorded in separate process
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

    def _init_sirl(self):
        self.sirl_cfg = self.cfg.get("algorithm", {}).get("sirl", {})
        self.sirl_enabled = bool(self.sirl_cfg.get("enabled", False))
        self.sirl_mode = str(self.sirl_cfg.get("mode", "sirl_lite")).lower()
        self.sirl_trajectory_mode = self.sirl_enabled and self.sirl_mode in {
            "trajectory",
            "trajectory_return",
            "return",
        }
        self.sirl_metric_buffers = {}
        self.sirl_last_stats = {}
        self.sirl_last_bc_stats = {}
        self.sirl_trajectory_replay = None
        self.sirl_replay_size = int(self.sirl_cfg.get("buffer_size", 131072))
        if not self.sirl_enabled:
            self.sirl_replay_count = 0
            self.sirl_replay_write_idx = 0
            return

        if self.sirl_trajectory_mode:
            self.sirl_trajectory_replay = TrajectorySIRLReplayBuffer(
                self.env.num_obs,
                self.env.num_actions,
                capacity_transitions=self.sirl_replay_size,
            )
            self.sirl_replay_count = 0
            self.sirl_replay_write_idx = 0
            return

        self.sirl_replay_obs = torch.zeros(self.sirl_replay_size, self.env.num_obs, dtype=torch.float, device=self.device)
        self.sirl_replay_actions = torch.zeros(
            self.sirl_replay_size,
            self.env.num_actions,
            dtype=torch.float,
            device=self.device,
        )
        self.sirl_replay_count = 0
        self.sirl_replay_write_idx = 0

    def _init_world_model(self):
        self.world_model_cfg = self.cfg.get("algorithm", {}).get("world_model", {})
        self.world_model_enabled = bool(self.world_model_cfg.get("enabled", False))
        self.world_model = None
        self.world_model_optimizer = None
        self.world_model_replay = None
        self.world_model_command_dim = 0
        self.world_model_use_amp = False
        self.world_model_scaler = None
        self.world_model_last_stats = {}
        if not self.world_model_enabled:
            return

        if bool(self.world_model_cfg.get("use_command", False)):
            self.world_model_command_dim = int(self.world_model_cfg.get("command_dim", 3))
        hidden_dims = self.world_model_cfg.get("hidden_dims", [512, 512])
        self.world_model = WorldModel(
            self.env.num_obs,
            self.env.num_actions,
            hidden_dims=hidden_dims,
            command_dim=self.world_model_command_dim,
        ).to(self.device)
        self.world_model_optimizer = torch.optim.Adam(
            self.world_model.parameters(),
            lr=float(self.world_model_cfg.get("lr", 1.0e-4)),
        )
        self.world_model_replay = WorldModelReplayBuffer(
            self.env.num_obs,
            self.env.num_actions,
            capacity_transitions=int(self.world_model_cfg.get("buffer_size", 1000000)),
            command_dim=self.world_model_command_dim,
        )
        self.world_model_use_amp = bool(self.world_model_cfg.get("use_amp", False)) and str(self.device).startswith("cuda")
        self.world_model_scaler = torch.cuda.amp.GradScaler(enabled=self.world_model_use_amp)

    def _rollout_command_dim(self):
        command_dims = []
        if self.sirl_trajectory_mode:
            command_dims.append(int(self.sirl_cfg.get("command_dim", 3)))
        if self.world_model_enabled and self.world_model_command_dim > 0:
            command_dims.append(self.world_model_command_dim)
        return max(command_dims) if command_dims else 0

    def _reset_sirl_rollout_metrics(self):
        if self.sirl_enabled:
            self.sirl_metric_buffers = {}
        command_dim = self._rollout_command_dim()
        if command_dim > 0:
            horizon = self.cfg["runner"]["horizon_length"]
            self.rollout_command_buffer = torch.zeros(
                horizon,
                self.env.num_envs,
                command_dim,
                dtype=torch.float,
                device=self.device,
            )
        else:
            self.rollout_command_buffer = None

    def _store_rollout_command(self, step_idx, obs=None):
        if getattr(self, "rollout_command_buffer", None) is None:
            return
        command_dim = self.rollout_command_buffer.shape[-1]
        command = None
        if hasattr(self.env, "commands"):
            command = self.env.commands[:, : min(command_dim, self.env.commands.shape[-1])]
        elif obs is not None and obs.shape[-1] >= 6 + command_dim:
            command = obs[:, 6 : 6 + command_dim]
        if command is None:
            return
        if command.shape[-1] < command_dim:
            padded_command = torch.zeros(self.env.num_envs, command_dim, dtype=torch.float, device=self.device)
            padded_command[:, : command.shape[-1]] = command.to(self.device).float()
            command = padded_command
        self.rollout_command_buffer[step_idx, :, :] = command.to(self.device).float().detach()

    def _store_sirl_infos(self, step_idx, infos):
        if not self.sirl_enabled:
            return
        sirl_infos = infos.get("sirl", {})
        if not sirl_infos:
            return
        horizon = self.cfg["runner"]["horizon_length"]
        for key, value in sirl_infos.items():
            if not torch.is_tensor(value):
                value = torch.as_tensor(value, dtype=torch.float, device=self.device)
            value = value.to(self.device).float()
            if value.ndim == 0:
                value = value.expand(self.env.num_envs)
            if value.numel() != self.env.num_envs:
                continue
            value = value.reshape(self.env.num_envs)
            if key not in self.sirl_metric_buffers:
                self.sirl_metric_buffers[key] = torch.zeros(horizon, self.env.num_envs, dtype=torch.float, device=self.device)
            self.sirl_metric_buffers[key][step_idx, :] = value.detach()

    def _add_sirl_samples(self, obses, actions):
        obses = obses.reshape(-1, self.env.num_obs).detach()
        actions = actions.reshape(-1, self.env.num_actions).detach()
        sample_count = obses.shape[0]
        if sample_count == 0:
            return
        if sample_count >= self.sirl_replay_size:
            obses = obses[-self.sirl_replay_size :]
            actions = actions[-self.sirl_replay_size :]
            sample_count = self.sirl_replay_size

        indices = (torch.arange(sample_count, device=self.device) + self.sirl_replay_write_idx) % self.sirl_replay_size
        self.sirl_replay_obs[indices] = obses
        self.sirl_replay_actions[indices] = actions
        self.sirl_replay_write_idx = (self.sirl_replay_write_idx + sample_count) % self.sirl_replay_size
        self.sirl_replay_count = min(self.sirl_replay_size, self.sirl_replay_count + sample_count)

    def _sirl_bc_coef(self, iteration):
        if not self.sirl_enabled:
            return 0.0
        start_iteration = int(self.sirl_cfg.get("start_iteration", 0))
        if iteration < start_iteration:
            return 0.0
        coef = float(self.sirl_cfg.get("bc_coef", 0.0))
        warmup_iterations = int(self.sirl_cfg.get("coef_warmup_iterations", 0))
        if warmup_iterations > 0:
            progress = min(1.0, max(0.0, (iteration - start_iteration + 1) / warmup_iterations))
            coef *= progress
        return coef

    def _update_sirl_replay(self, iteration):
        if self.sirl_trajectory_mode:
            return self._update_trajectory_sirl_replay(iteration)
        return self._update_sirl_lite_replay(iteration)

    def _update_sirl_lite_replay(self, iteration):
        stats = {
            "sirl/elite_score_mean": 0.0,
            "sirl/elite_score_min": 0.0,
            "sirl/selected_envs": 0.0,
            "sirl/added_samples": 0.0,
            "sirl/buffer_count": float(self.sirl_replay_count),
        }
        if not self.sirl_enabled or iteration < int(self.sirl_cfg.get("start_iteration", 0)):
            return stats

        rewards = self.buffer["rewards"].detach()
        score = rewards.sum(dim=0)
        reduction = str(self.sirl_cfg.get("score_metric_reduction", "mean"))
        for key, weight in self.sirl_cfg.get("score_weights", {}).items():
            metric = self.sirl_metric_buffers.get(key)
            if metric is None:
                continue
            if reduction == "sum":
                metric_score = metric.sum(dim=0)
            elif reduction == "max":
                metric_score = metric.max(dim=0).values
            else:
                metric_score = metric.mean(dim=0)
            score = score + float(weight) * metric_score

        done_count = self.buffer["dones"].float().sum(dim=0)
        max_dones = int(self.sirl_cfg.get("max_dones_per_horizon", 0))
        eligible = done_count <= max_dones
        if not bool(eligible.any().item()):
            return stats

        terminal_penalty = float(self.sirl_cfg.get("terminal_penalty", 0.0))
        score = score - terminal_penalty * done_count
        score = torch.where(eligible, score, torch.full_like(score, -torch.inf))

        top_fraction = float(self.sirl_cfg.get("top_fraction", 0.15))
        eligible_count = int(eligible.sum().item())
        top_k = max(1, int(np.ceil(top_fraction * self.env.num_envs)))
        top_k = min(top_k, eligible_count)
        elite_scores, elite_env_ids = torch.topk(score, top_k)
        if "min_score" in self.sirl_cfg:
            keep = elite_scores >= float(self.sirl_cfg["min_score"])
            elite_scores = elite_scores[keep]
            elite_env_ids = elite_env_ids[keep]
        if elite_env_ids.numel() == 0:
            return stats

        elite_obses = self.buffer["obses"][:, elite_env_ids, :]
        elite_actions = self.buffer["actions"][:, elite_env_ids, :]
        added_samples = elite_obses.numel() // self.env.num_obs
        self._add_sirl_samples(elite_obses, elite_actions)

        stats.update(
            {
                "sirl/elite_score_mean": float(elite_scores.mean().item()),
                "sirl/elite_score_min": float(elite_scores.min().item()),
                "sirl/selected_envs": float(elite_env_ids.numel()),
                "sirl/added_samples": float(added_samples),
                "sirl/buffer_count": float(self.sirl_replay_count),
            }
        )
        return stats

    def _empty_trajectory_sirl_stats(self):
        stats = {
            "sirl/elite_score_mean": 0.0,
            "sirl/elite_score_min": 0.0,
            "sirl/selected_envs": 0.0,
            "sirl/selected_trajectories": 0.0,
            "sirl/added_samples": 0.0,
            "sirl/trajectory_return_mean": 0.0,
            "sirl/trajectory_return_p50": 0.0,
            "sirl/trajectory_return_p90": 0.0,
            "sirl/bc_weight_mean": 0.0,
            "sirl/bc_weight_max": 0.0,
            "sirl/model_error_mean": 0.0,
            "sirl/model_done_prob_mean": 0.0,
            "sirl/model_error_high_return_mean": 0.0,
            "sirl/buffer_trajectories": 0.0,
            "sirl/buffer_transitions": 0.0,
            "sirl/buffer_count": 0.0,
        }
        if self.sirl_trajectory_replay is not None:
            stats.update(self.sirl_trajectory_replay.stats())
        return stats

    def _rollout_segment_returns(self):
        rewards = self.buffer["rewards"].detach()
        if bool(self.sirl_cfg.get("discounted_return", False)):
            gamma = float(self.sirl_cfg.get("return_gamma", self.cfg["algorithm"].get("gamma", 1.0)))
            steps = torch.arange(rewards.shape[0], dtype=torch.float, device=self.device)
            discounts = torch.pow(torch.full_like(steps, gamma), steps).unsqueeze(-1)
            return (rewards * discounts).sum(dim=0)
        return rewards.sum(dim=0)

    def _trajectory_metric_penalties(self):
        penalties = dict(self.sirl_cfg.get("metric_penalties", {}))
        if penalties:
            return penalties
        for key, weight in self.sirl_cfg.get("score_weights", {}).items():
            weight = float(weight)
            if weight < 0.0:
                penalties[key] = -weight
        return penalties

    def _score_rollout_with_world_model(self):
        if not self.world_model_enabled or self.world_model is None or self.world_model_replay is None:
            return None
        filter_cfg = self.sirl_cfg.get("world_model_filter", {})
        min_replay_size = int(filter_cfg.get("min_replay_size", self.world_model_cfg.get("batch_size", 1024)))
        if len(self.world_model_replay) < min_replay_size:
            return None

        obs = self.buffer["obses"].detach().reshape(-1, self.env.num_obs)
        action = self.buffer["actions"].detach().reshape(-1, self.env.num_actions)
        next_obs = self.buffer["next_obses"].detach().reshape(-1, self.env.num_obs)
        reward = self.buffer["rewards"].detach().reshape(-1)
        done = self.buffer["dones"].detach().reshape(-1)
        command = None
        if self.world_model_command_dim > 0 and getattr(self, "rollout_command_buffer", None) is not None:
            command = self.rollout_command_buffer[:, :, : self.world_model_command_dim].detach().reshape(
                -1,
                self.world_model_command_dim,
            )

        eval_batch_size = int(self.world_model_cfg.get("eval_batch_size", 16384))
        obs_errors = []
        done_probs = []
        self.world_model.eval()
        with torch.no_grad():
            for start in range(0, obs.shape[0], eval_batch_size):
                end = min(start + eval_batch_size, obs.shape[0])
                batch_command = None if command is None else command[start:end]
                pred = self.world_model(obs[start:end], action[start:end], batch_command)
                target_delta = next_obs[start:end] - obs[start:end]
                delta_error = torch.mean(torch.square(pred["delta_obs"] - target_delta), dim=-1)
                reward_error = torch.square(pred["reward"] - reward[start:end])
                obs_errors.append(delta_error + 0.1 * reward_error)
                done_probs.append(torch.sigmoid(pred["done_logit"]))
        self.world_model.train()

        horizon = self.cfg["runner"]["horizon_length"]
        obs_error = torch.cat(obs_errors, dim=0).reshape(horizon, self.env.num_envs)
        done_prob = torch.cat(done_probs, dim=0).reshape(horizon, self.env.num_envs)
        valid = (~done.bool()).float().reshape(horizon, self.env.num_envs)
        if bool(self.world_model_cfg.get("mask_done_obs_delta", True)):
            model_error = (obs_error * valid).sum(dim=0) / torch.clamp(valid.sum(dim=0), min=1.0)
        else:
            model_error = obs_error.mean(dim=0)
        return {
            "model_error": model_error,
            "done_prob": done_prob.mean(dim=0),
        }

    def _update_trajectory_sirl_replay(self, iteration):
        stats = self._empty_trajectory_sirl_stats()
        if not self.sirl_enabled or iteration < int(self.sirl_cfg.get("start_iteration", 0)):
            return stats

        returns = self._rollout_segment_returns()
        return_p50 = percentile(returns, 50)
        return_p90 = percentile(returns, 90)
        normalized_return = normalize_by_percentiles(returns, 50, 90)
        return_alpha = max(float(self.sirl_cfg.get("return_weight_alpha", 1.0)), 0.0)
        score = normalized_return.pow(return_alpha).clone()

        normalize_metrics = bool(self.sirl_cfg.get("normalize_metric_penalties", True))
        metric_penalties = self._trajectory_metric_penalties()
        for key in SIRL_METRIC_KEYS:
            if key not in metric_penalties:
                continue
            metric = self.sirl_metric_buffers.get(key)
            if metric is None:
                continue
            metric_mean = metric.mean(dim=0)
            metric_score = normalize_by_percentiles(metric_mean, 50, 90) if normalize_metrics else metric_mean
            score = score - float(metric_penalties[key]) * metric_score
        for key, weight in metric_penalties.items():
            if key in SIRL_METRIC_KEYS:
                continue
            metric = self.sirl_metric_buffers.get(key)
            if metric is None:
                continue
            metric_mean = metric.mean(dim=0)
            metric_score = normalize_by_percentiles(metric_mean, 50, 90) if normalize_metrics else metric_mean
            score = score - float(weight) * metric_score

        done_count = self.buffer["dones"].float().sum(dim=0)
        time_outs = self.buffer["time_outs"].bool()
        terminal = self.buffer["dones"].bool().any(dim=0).float()
        max_dones = int(self.sirl_cfg.get("max_dones_per_horizon", 0))
        eligible = done_count <= max_dones
        min_return_percentile = float(self.sirl_cfg.get("min_return_percentile", 0.0))
        if min_return_percentile > 0.0:
            eligible &= returns >= percentile(returns, min_return_percentile)
        fall_penalty = float(self.sirl_cfg.get("fall_penalty", self.sirl_cfg.get("terminal_penalty", 0.0)))
        score = score - fall_penalty * terminal

        world_scores = self._score_rollout_with_world_model()
        model_error = None
        model_done_prob = None
        normalized_model_error = None
        world_filter_cfg = self.sirl_cfg.get("world_model_filter", {})
        if world_scores is not None and bool(world_filter_cfg.get("enabled", True)):
            model_error = world_scores["model_error"]
            model_done_prob = world_scores["done_prob"]
            normalized_model_error = normalize_by_percentiles(model_error, 50, 90)
            score = score - float(world_filter_cfg.get("error_penalty", 0.0)) * normalized_model_error
            score = score - float(world_filter_cfg.get("done_prob_penalty", 0.0)) * model_done_prob
            if "max_done_prob" in world_filter_cfg:
                eligible &= model_done_prob <= float(world_filter_cfg["max_done_prob"])

        eligible_count = int(eligible.sum().item())
        stats.update(
            {
                "sirl/trajectory_return_mean": float(returns.mean().item()),
                "sirl/trajectory_return_p50": float(return_p50.item()),
                "sirl/trajectory_return_p90": float(return_p90.item()),
            }
        )
        if model_error is not None:
            high_return = returns >= return_p90
            high_error = model_error[high_return].mean() if bool(high_return.any().item()) else model_error.mean()
            stats.update(
                {
                    "sirl/model_error_mean": float(model_error.mean().item()),
                    "sirl/model_done_prob_mean": float(model_done_prob.mean().item()),
                    "sirl/model_error_high_return_mean": float(high_error.item()),
                }
            )
        if eligible_count <= 0:
            stats.update(self.sirl_trajectory_replay.stats())
            return stats

        score = torch.where(eligible, score, torch.full_like(score, -torch.inf))
        top_fraction = float(self.sirl_cfg.get("top_fraction", 0.15))
        top_k = max(1, int(np.ceil(top_fraction * eligible_count)))
        elite_scores, elite_env_ids = torch.topk(score, top_k)
        finite_keep = torch.isfinite(elite_scores)
        if "min_score" in self.sirl_cfg:
            finite_keep &= elite_scores >= float(self.sirl_cfg["min_score"])
        elite_scores = elite_scores[finite_keep]
        elite_env_ids = elite_env_ids[finite_keep]
        if elite_env_ids.numel() == 0:
            stats.update(self.sirl_trajectory_replay.stats())
            return stats

        max_bc_weight = float(self.sirl_cfg.get("max_bc_weight", 1.0))
        if bool(self.sirl_cfg.get("dynamic_bc_weight", True)):
            bc_weights = normalized_return.pow(return_alpha)
        else:
            bc_weights = torch.ones_like(normalized_return)
        bc_weights = torch.clamp(bc_weights * max_bc_weight, min=0.0, max=max_bc_weight)
        if model_error is not None and normalized_model_error is not None:
            error_decay = float(world_filter_cfg.get("bc_weight_error_decay", 0.0))
            done_decay = float(world_filter_cfg.get("bc_weight_done_decay", 0.0))
            model_scale = torch.clamp(1.0 - error_decay * normalized_model_error - done_decay * model_done_prob, min=0.0, max=1.0)
            bc_weights = bc_weights * model_scale

        selected_weights = bc_weights[elite_env_ids]
        selected_returns = returns[elite_env_ids]
        added_trajectories = self.sirl_trajectory_replay.add_segments(
            self.buffer["obses"],
            self.buffer["actions"],
            self.buffer["rewards"],
            self.buffer["dones"],
            time_outs=time_outs,
            next_obses=self.buffer["next_obses"],
            commands=getattr(self, "rollout_command_buffer", None),
            metrics=self.sirl_metric_buffers,
            env_ids=elite_env_ids,
            returns=returns,
            scores=score,
            bc_weights=bc_weights,
            model_errors=model_error,
            model_done_probs=model_done_prob,
        )
        added_samples = added_trajectories * self.cfg["runner"]["horizon_length"]
        self.sirl_replay_count = self.sirl_trajectory_replay.transition_count

        stats.update(
            {
                "sirl/elite_score_mean": float(elite_scores.mean().item()),
                "sirl/elite_score_min": float(elite_scores.min().item()),
                "sirl/selected_envs": float(elite_env_ids.numel()),
                "sirl/selected_trajectories": float(elite_env_ids.numel()),
                "sirl/added_samples": float(added_samples),
                "sirl/trajectory_return_mean": float(selected_returns.mean().item()),
                "sirl/bc_weight_mean": float(selected_weights.mean().item()),
                "sirl/bc_weight_max": float(selected_weights.max().item()),
            }
        )
        stats.update(self.sirl_trajectory_replay.stats())
        return stats

    def _compute_sirl_bc_loss(self, iteration):
        self.sirl_last_bc_stats = {}
        coef = self._sirl_bc_coef(iteration)
        min_replay_size = int(self.sirl_cfg.get("min_replay_size", 4096))
        if coef <= 0.0 or self.sirl_replay_count < min_replay_size:
            return None, 0.0

        if self.sirl_trajectory_mode:
            if self.sirl_trajectory_replay is None or len(self.sirl_trajectory_replay) < min_replay_size:
                return None, 0.0
            batch_size = min(int(self.sirl_cfg.get("bc_batch_size", 4096)), len(self.sirl_trajectory_replay))
            batch = self.sirl_trajectory_replay.sample_transitions(batch_size, self.device)
            bc_obs = batch["obs"]
            bc_actions = batch["action"]
            replay_weights = batch["bc_weight"].float()
            sample_weights = torch.clamp(coef * replay_weights, min=0.0, max=float(self.sirl_cfg.get("max_bc_weight", 1.0)) * coef)
            bc_dist = self.model.act(bc_obs)
            if str(self.sirl_cfg.get("loss", "mse")).lower() == "nll":
                per_sample_loss = -bc_dist.log_prob(bc_actions).sum(dim=-1)
            else:
                per_sample_loss = torch.mean(torch.square(bc_dist.loc - bc_actions), dim=-1)
            bc_loss = torch.mean(sample_weights * per_sample_loss)
            self.sirl_last_bc_stats = {
                "sirl/bc_weight_mean": float(sample_weights.detach().mean().item()),
                "sirl/bc_weight_max": float(sample_weights.detach().max().item()),
            }
            return bc_loss, coef

        batch_size = min(int(self.sirl_cfg.get("bc_batch_size", 4096)), self.sirl_replay_count)
        indices = torch.randint(0, self.sirl_replay_count, (batch_size,), device=self.device)
        bc_obs = self.sirl_replay_obs[indices]
        bc_actions = self.sirl_replay_actions[indices]
        bc_dist = self.model.act(bc_obs)
        if str(self.sirl_cfg.get("loss", "mse")).lower() == "nll":
            bc_loss = -bc_dist.log_prob(bc_actions).sum(dim=-1).mean()
        else:
            bc_loss = F.mse_loss(bc_dist.loc, bc_actions)
        self.sirl_last_bc_stats = {
            "sirl/bc_weight_mean": float(coef),
            "sirl/bc_weight_max": float(coef),
        }
        return bc_loss, coef

    def _update_world_model_replay(self):
        stats = {}
        if not self.world_model_enabled or self.world_model_replay is None:
            return stats
        command = None
        if self.world_model_command_dim > 0 and getattr(self, "rollout_command_buffer", None) is not None:
            command = self.rollout_command_buffer[:, :, : self.world_model_command_dim]
        self.world_model_replay.add(
            self.buffer["obses"],
            self.buffer["actions"],
            self.buffer["rewards"],
            self.buffer["dones"],
            self.buffer["next_obses"],
            commands=command,
        )
        stats["world_model/buffer_count"] = float(len(self.world_model_replay))
        return stats

    def _train_world_model(self, iteration):
        if not self.world_model_enabled or self.world_model is None or self.world_model_replay is None:
            return {}
        stats = {
            "world_model/loss": 0.0,
            "world_model/obs_delta_loss": 0.0,
            "world_model/reward_loss": 0.0,
            "world_model/done_loss": 0.0,
            "world_model/buffer_count": float(len(self.world_model_replay)),
        }
        train_every = max(1, int(self.world_model_cfg.get("train_every", 1)))
        if (iteration + 1) % train_every != 0:
            return stats
        batch_size = int(self.world_model_cfg.get("batch_size", 1024))
        min_train_size = int(self.world_model_cfg.get("min_train_size", batch_size))
        if len(self.world_model_replay) < min_train_size:
            return stats

        gradient_steps = max(1, int(self.world_model_cfg.get("gradient_steps", 1)))
        obs_coef = float(self.world_model_cfg.get("obs_delta_coef", 1.0))
        reward_coef = float(self.world_model_cfg.get("reward_coef", 0.5))
        done_coef = float(self.world_model_cfg.get("done_coef", 0.2))
        mask_done_obs_delta = bool(self.world_model_cfg.get("mask_done_obs_delta", True))
        max_grad_norm = float(self.world_model_cfg.get("max_grad_norm", 1.0))
        totals = {
            "loss": 0.0,
            "obs_delta_loss": 0.0,
            "reward_loss": 0.0,
            "done_loss": 0.0,
        }

        self.world_model.train()
        for _ in range(gradient_steps):
            batch = self.world_model_replay.sample(batch_size, self.device)
            command = batch.get("command")
            with torch.cuda.amp.autocast(enabled=self.world_model_use_amp):
                pred = self.world_model(batch["obs"], batch["action"], command)
                target_delta = batch["next_obs"] - batch["obs"]
                obs_delta_loss_per_sample = torch.mean(torch.square(pred["delta_obs"] - target_delta), dim=-1)
                if mask_done_obs_delta:
                    valid = 1.0 - batch["done"].float()
                    obs_delta_loss = (obs_delta_loss_per_sample * valid).sum() / torch.clamp(valid.sum(), min=1.0)
                else:
                    obs_delta_loss = obs_delta_loss_per_sample.mean()
                reward_loss = F.mse_loss(pred["reward"], batch["reward"])
                done_loss = F.binary_cross_entropy_with_logits(pred["done_logit"], batch["done"].float())
                loss = obs_coef * obs_delta_loss + reward_coef * reward_loss + done_coef * done_loss

            self.world_model_optimizer.zero_grad()
            if self.world_model_use_amp:
                self.world_model_scaler.scale(loss).backward()
                self.world_model_scaler.unscale_(self.world_model_optimizer)
                torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_grad_norm)
                self.world_model_scaler.step(self.world_model_optimizer)
                self.world_model_scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_grad_norm)
                self.world_model_optimizer.step()

            totals["loss"] += float(loss.detach().item())
            totals["obs_delta_loss"] += float(obs_delta_loss.detach().item())
            totals["reward_loss"] += float(reward_loss.detach().item())
            totals["done_loss"] += float(done_loss.detach().item())

        stats.update(
            {
                "world_model/loss": totals["loss"] / gradient_steps,
                "world_model/obs_delta_loss": totals["obs_delta_loss"] / gradient_steps,
                "world_model/reward_loss": totals["reward_loss"] / gradient_steps,
                "world_model/done_loss": totals["done_loss"] / gradient_steps,
                "world_model/buffer_count": float(len(self.world_model_replay)),
            }
        )
        return stats

    @staticmethod
    def _format_duration(seconds):
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
        return f"{minutes:d}m {seconds:02d}s"

    def _load(self):
        if not self.cfg["basic"]["checkpoint"]:
            return
        if (self.cfg["basic"]["checkpoint"] == "-1") or (self.cfg["basic"]["checkpoint"] == -1):
            # Look for models in hierarchical structure: logs/robot_type/task_name/**/*.pth
            task_name = self.cfg["basic"].get("log_task", self.cfg["basic"]["task"])
            robot_type = self._get_robot_type(task_name)

            # First try: exact log task in robot-specific folder. A config can
            # keep basic.task for class loading while storing experiments under
            # a separate basic.log_task name.
            search_task_names = [task_name]
            run_task_name = self.cfg["basic"]["task"]
            if run_task_name not in search_task_names:
                search_task_names.append(run_task_name)

            task_models = []
            for search_task_name in search_task_names:
                task_log_pattern = os.path.join("logs", robot_type, search_task_name, "**/*.pth")
                task_models = sorted(glob.glob(task_log_pattern, recursive=True), key=os.path.getmtime)
                if task_models:
                    break

            if task_models:
                self.cfg["basic"]["checkpoint"] = task_models[-1]
            else:
                # Second try: any task in robot-specific folder
                robot_log_pattern = os.path.join("logs", robot_type, "**/*.pth")
                robot_models = sorted(glob.glob(robot_log_pattern, recursive=True), key=os.path.getmtime)
                
                if robot_models:
                    self.cfg["basic"]["checkpoint"] = robot_models[-1]
                else:
                    # Fallback: all logs if no robot-specific models found
                    self.cfg["basic"]["checkpoint"] = sorted(glob.glob(os.path.join("logs", "**/*.pth"), recursive=True), key=os.path.getmtime)[-1]
        print("Loading model from {}".format(self.cfg["basic"]["checkpoint"]))
        model_dict = torch.load(self.cfg["basic"]["checkpoint"], map_location=self.device, weights_only=True)
        self.model.load_state_dict(model_dict["model"], strict=False)
        try:
            self.env.curriculum_prob = model_dict["curriculum"]
        except Exception as e:
            print(f"Failed to load curriculum: {e}")
        try:
            if hasattr(self.env, 'ball_curriculum_global_level') and "ball_curriculum_level" in model_dict:
                self.env.ball_curriculum_global_level = int(model_dict["ball_curriculum_level"])
                self.env.ball_curriculum_level[:] = self.env.ball_curriculum_global_level
                print(f"Restored ball curriculum level: {self.env.ball_curriculum_global_level}")
        except Exception as e:
            print(f"Failed to load ball curriculum level: {e}")
        try:
            self.optimizer.load_state_dict(model_dict["optimizer"])
        except Exception as e:
            print(f"Failed to load optimizer: {e}")
        if self.world_model_enabled and self.world_model is not None:
            try:
                if "world_model" in model_dict:
                    self.world_model.load_state_dict(model_dict["world_model"], strict=False)
                if "world_model_optimizer" in model_dict and self.world_model_optimizer is not None:
                    self.world_model_optimizer.load_state_dict(model_dict["world_model_optimizer"])
            except Exception as e:
                print(f"Failed to load world model: {e}")

    def _checkpoint_state(self):
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "curriculum": self.env.curriculum_prob,
            "ball_curriculum_level": getattr(self.env, 'ball_curriculum_global_level', 0),
        }
        if self.world_model_enabled and self.world_model is not None:
            state["world_model"] = self.world_model.state_dict()
            if self.world_model_optimizer is not None:
                state["world_model_optimizer"] = self.world_model_optimizer.state_dict()
        return state

    def train(self):
        self.recorder = Recorder(self.cfg)
        if hasattr(self.env, "update_training_curriculum"):
            self.env.update_training_curriculum(0)
        obs, infos = self.env.reset()
        obs = obs.to(self.device)
        privileged_obs = infos["privileged_obs"].to(self.device)
        
        # Get video logging configuration
        use_wandb = self.cfg["runner"].get("use_wandb", False)
        log_video_interval = self.cfg["runner"].get("log_video_interval", None)
        if log_video_interval is None:
            log_video_interval = self.cfg["runner"].get("save_interval", None)
        # Ensure log_video_interval is a positive integer
        if log_video_interval is not None and log_video_interval <= 0:
            log_video_interval = None
        log_video_duration = self.cfg["runner"].get("log_video_duration", 10.0)
        log_reward_terms = self.cfg["runner"].get("log_reward_terms", False)
        log_env_metrics = self.cfg["runner"].get("log_env_metrics", False)
        progress_interval = max(1, int(self.cfg["runner"].get("progress_interval", 10)))
        max_iterations = self.cfg["basic"]["max_iterations"]
        train_start_time = time.time()

        print(f"Training logs: {self.recorder.dir}")
        print(
            f"Training progress: 0/{max_iterations} (0.00%) | "
            f"num_envs={self.env.num_envs} horizon={self.cfg['runner']['horizon_length']}"
        )

        for it in range(max_iterations):
            rollout_reward_sum = 0.0
            rollout_done_count = 0
            self._reset_sirl_rollout_metrics()
            if hasattr(self.env, "update_training_curriculum"):
                self.env.update_training_curriculum(it)
            # Check if it's time to log a video
            should_log_video = (use_wandb and 
                               log_video_interval is not None and 
                               log_video_interval > 0 and
                               (it + 1) % log_video_interval == 0)
            
            # Save checkpoint if needed (for video recording or regular save interval)
            should_save = False
            checkpoint_path = None
            if (it + 1) % self.cfg["runner"]["save_interval"] == 0:
                should_save = True
                checkpoint_path = os.path.join(self.recorder.model_dir, f"model_{it + 1}.pth")
                self.recorder.save(self._checkpoint_state(), it + 1)
            
            if should_log_video:
                # If we didn't save yet, save checkpoint now for video recording
                if not should_save:
                    checkpoint_path = os.path.join(self.recorder.model_dir, f"model_{it + 1}.pth")
                    self.recorder.save(self._checkpoint_state(), it + 1)
                # Spawn separate process to record video (will wait for completion)
                # Note: Video will be uploaded at step it+1 (after training loop logs at step it)
                self._spawn_video_recording_process(checkpoint_path, it, log_video_duration)
            # within horizon_length, env.step() is called with same act
            for n in range(self.cfg["runner"]["horizon_length"]):
                self.buffer.update_data("obses", n, obs)
                self.buffer.update_data("privileged_obses", n, privileged_obs)
                self._store_rollout_command(n, obs)
                with torch.no_grad():
                    dist = self.model.act(obs)
                    act = dist.sample()
                obs, rew, done, infos = self.env.step(act)
                obs, rew, done = obs.to(self.device), rew.to(self.device), done.to(self.device)
                privileged_obs = infos["privileged_obs"].to(self.device)
                rollout_reward_sum += float(rew.mean().item())
                rollout_done_count += int(done.sum().item())
                self.buffer.update_data("actions", n, act)
                self.buffer.update_data("rewards", n, rew)
                self.buffer.update_data("dones", n, done)
                self.buffer.update_data("next_obses", n, obs)
                self.buffer.update_data("time_outs", n, infos["time_outs"].to(self.device))
                self._store_sirl_infos(n, infos)
                ep_info = {"reward": rew}
                if log_reward_terms:
                    ep_info.update(infos["rew_terms"])
                if log_env_metrics and "metrics" in infos and bool(done.any().item()):
                    ep_info.update(infos["metrics"])
                self.recorder.record_episode_statistics(done, ep_info, it, n == (self.cfg["runner"]["horizon_length"] - 1))

            sirl_rollout_stats = self._update_sirl_replay(it)
            world_model_replay_stats = self._update_world_model_replay()

            with torch.no_grad():
                old_dist = self.model.act(self.buffer["obses"])
                old_actions_log_prob = old_dist.log_prob(self.buffer["actions"]).sum(dim=-1)
                # Store old values for value loss clipping
                old_values = self.model.est_value(self.buffer["obses"], self.buffer["privileged_obses"])
                old_last_values = self.model.est_value(obs, privileged_obs)
                # Compute returns once using old values (they shouldn't change during mini epochs)
                self.buffer["rewards"][self.buffer["time_outs"]] = old_values[self.buffer["time_outs"]]
                advantages = discount_values(
                    self.buffer["rewards"],
                    self.buffer["dones"] | self.buffer["time_outs"],
                    old_values,
                    old_last_values,
                    self.cfg["algorithm"]["gamma"],
                    self.cfg["algorithm"]["lam"],
                )
                returns = old_values + advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Get value clip parameter (default to None for no clipping, for backwards compatibility)
            value_clip_param = self.cfg["algorithm"].get("value_clip_param", None)

            mean_value_loss = 0
            mean_actor_loss = 0
            mean_bound_loss = 0
            mean_entropy = 0
            mean_sirl_bc_loss = 0.0
            mean_sirl_bc_coef = 0.0
            mean_sirl_bc_weight_mean = 0.0
            mean_sirl_bc_weight_max = 0.0
            sirl_bc_update_count = 0
            for n in range(self.cfg["runner"]["mini_epochs"]):
                values = self.model.est_value(self.buffer["obses"], self.buffer["privileged_obses"])

                # Value loss with optional clipping
                if value_clip_param is not None:
                    # Clipped value prediction
                    values_clipped = old_values + torch.clamp(
                        values - old_values, -value_clip_param, value_clip_param
                    )
                    # Unclipped and clipped value losses
                    value_loss_unclipped = (values - returns).pow(2)
                    value_loss_clipped = (values_clipped - returns).pow(2)
                    # Take the maximum (more conservative)
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    value_loss = F.mse_loss(values, returns)

                dist = self.model.act(self.buffer["obses"])
                actions_log_prob = dist.log_prob(self.buffer["actions"]).sum(dim=-1)
                actor_loss = surrogate_loss(old_actions_log_prob, actions_log_prob, advantages)

                bound_loss = torch.clip(dist.loc - 1.0, min=0.0).square().mean() + torch.clip(dist.loc + 1.0, max=0.0).square().mean()

                entropy = dist.entropy().sum(dim=-1)

                if self.cfg["algorithm"]["min_entropy"] is not None and self.cfg["algorithm"]["max_entropy"] is not None:
                    min_entropy = self.cfg["algorithm"]["min_entropy"]
                    max_entropy = self.cfg["algorithm"]["max_entropy"]
                    loss_entropy = torch.mean((torch.clamp(entropy.mean(), min=min_entropy, max=max_entropy) - entropy.mean())**2)
                else:
                    loss_entropy = 0.0
                sirl_bc_loss, sirl_bc_coef = self._compute_sirl_bc_loss(it)
                loss = (
                    value_loss
                    + actor_loss
                    + self.cfg["algorithm"]["bound_coef"] * bound_loss
                    + self.cfg["algorithm"]["entropy_coef"] * entropy.mean()
                    + 0.01 * loss_entropy
                    #+ self.cfg["algorithm"]["symmetry_coef"] * sym_loss
                )
                if sirl_bc_loss is not None:
                    if self.sirl_trajectory_mode:
                        loss = loss + sirl_bc_loss
                    else:
                        loss = loss + sirl_bc_coef * sirl_bc_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_actor_loss += actor_loss.item()
                mean_bound_loss += bound_loss.item()
                mean_entropy += entropy.mean()
                if sirl_bc_loss is not None:
                    mean_sirl_bc_loss += sirl_bc_loss.item()
                    mean_sirl_bc_coef += sirl_bc_coef
                    mean_sirl_bc_weight_mean += float(self.sirl_last_bc_stats.get("sirl/bc_weight_mean", 0.0))
                    mean_sirl_bc_weight_max = max(
                        mean_sirl_bc_weight_max,
                        float(self.sirl_last_bc_stats.get("sirl/bc_weight_max", 0.0)),
                    )
                    sirl_bc_update_count += 1

            world_model_stats = self._train_world_model(it)

            # Calculate KL divergence after all mini epochs (between old and final policy)
            with torch.no_grad():
                final_dist = self.model.act(self.buffer["obses"])
                kl = torch.sum(
                    torch.log(final_dist.scale / old_dist.scale)
                    + 0.5 * (torch.square(old_dist.scale) + torch.square(final_dist.loc - old_dist.loc)) / torch.square(final_dist.scale)
                    - 0.5,
                    axis=-1,
                )
                kl_mean = torch.mean(kl)

                # Adapt learning rate based on KL divergence
                lr_min = float(self.cfg["algorithm"].get("adaptive_lr_min", 1e-5))
                lr_max = float(self.cfg["algorithm"].get("adaptive_lr_max", 1e-2))
                if kl_mean > self.cfg["algorithm"]["desired_kl"] * 2:
                    self.learning_rate = max(lr_min, self.learning_rate / 1.5)
                elif kl_mean < self.cfg["algorithm"]["desired_kl"] / 2:
                    self.learning_rate = min(lr_max, self.learning_rate * 1.5)

                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.learning_rate

            mean_value_loss /= self.cfg["runner"]["mini_epochs"]
            mean_actor_loss /= self.cfg["runner"]["mini_epochs"]
            mean_bound_loss /= self.cfg["runner"]["mini_epochs"]
            mean_entropy /= self.cfg["runner"]["mini_epochs"]
            mean_sirl_bc_loss /= self.cfg["runner"]["mini_epochs"]
            mean_sirl_bc_coef /= self.cfg["runner"]["mini_epochs"]
            if sirl_bc_update_count > 0:
                mean_sirl_bc_weight_mean /= sirl_bc_update_count
            statistics = {
                "value_loss": mean_value_loss,
                "actor_loss": mean_actor_loss,
                "bound_loss": mean_bound_loss,
                "entropy": mean_entropy,
                "kl_mean": kl_mean,
                "lr": self.learning_rate,
                "curriculum/mean_lin_vel_level": self.env.mean_lin_vel_level,
                "curriculum/mean_ang_vel_level": self.env.mean_ang_vel_level,
                "curriculum/max_lin_vel_level": self.env.max_lin_vel_level,
                "curriculum/max_ang_vel_level": self.env.max_ang_vel_level,
                "training_phase/index": float(getattr(self.env, "training_phase_index", 0)),
                "training_phase/progress": float(getattr(self.env, "training_phase_progress", 0.0)),
                "training_phase/locomotion_core": float(getattr(self.env, "reward_group_multipliers", {}).get("locomotion_core", 1.0)),
                "training_phase/approach_core": float(getattr(self.env, "reward_group_multipliers", {}).get("approach_core", 1.0)),
                "training_phase/failure_core": float(getattr(self.env, "reward_group_multipliers", {}).get("failure_core", 1.0)),
                "training_phase/intercept_core": float(getattr(self.env, "reward_group_multipliers", {}).get("intercept_core", 1.0)),
                "training_phase/control_core": float(getattr(self.env, "reward_group_multipliers", {}).get("control_core", 1.0)),
            }
            if self.sirl_enabled:
                statistics.update(sirl_rollout_stats)
                statistics["sirl/bc_loss"] = mean_sirl_bc_loss
                statistics["sirl/bc_coef"] = mean_sirl_bc_coef
                statistics["sirl/bc_weight_mean"] = mean_sirl_bc_weight_mean
                statistics["sirl/bc_weight_max"] = mean_sirl_bc_weight_max
                statistics["sirl/buffer_count"] = float(self.sirl_replay_count)
            if self.world_model_enabled:
                statistics.update(world_model_replay_stats)
                statistics.update(world_model_stats)
            self.recorder.record_statistics(statistics, it)

            should_report_progress = (
                (it + 1) == 1
                or (it + 1) == max_iterations
                or (it + 1) % progress_interval == 0
            )
            if should_report_progress:
                elapsed = time.time() - train_start_time
                avg_iter_time = elapsed / (it + 1)
                remaining_iters = max_iterations - (it + 1)
                eta = avg_iter_time * remaining_iters
                percent = 100.0 * (it + 1) / max_iterations
                rollout_mean_reward = rollout_reward_sum / self.cfg["runner"]["horizon_length"]
                rollout_done_rate = rollout_done_count / (
                    self.cfg["runner"]["horizon_length"] * self.env.num_envs
                )
                mean_entropy_value = (
                    float(mean_entropy.detach().cpu().item()) if torch.is_tensor(mean_entropy) else float(mean_entropy)
                )
                kl_mean_value = float(kl_mean.detach().cpu().item()) if torch.is_tensor(kl_mean) else float(kl_mean)
                print(
                    f"[train] {it + 1}/{max_iterations} ({percent:5.2f}%) | "
                    f"elapsed {self._format_duration(elapsed)} | "
                    f"eta {self._format_duration(eta)} | "
                    f"reward {rollout_mean_reward:.4f} | "
                    f"done {rollout_done_rate * 100.0:.2f}% | "
                    f"value_loss {mean_value_loss:.4f} | "
                    f"actor_loss {mean_actor_loss:.4f} | "
                    f"entropy {mean_entropy_value:.4f} | "
                    f"kl {kl_mean_value:.6f} | "
                    f"lr {self.learning_rate:.2e}"
                    + (
                        f" | sirl_bc {mean_sirl_bc_loss:.5f} | sirl_buf {self.sirl_replay_count}"
                        if self.sirl_enabled
                        else ""
                    )
                )

    def play(self):
        # Check if we're in record-and-exit mode (for separate process video recording)
        if self.args.record_video_mode:
            self._play_record_and_exit()
            return
        
        # Normal play mode (for manual testing)
        obs, infos = self.env.reset()
        obs = obs.to(self.device)
        play_cfg = self.cfg.get("commands", {}).get("play", {})
        if play_cfg:
            print(
                "[play command] "
                f"vx={float(play_cfg.get('lin_vel_x', 0.0)):.3f} "
                f"vy={float(play_cfg.get('lin_vel_y', 0.0)):.3f} "
                f"yaw={float(play_cfg.get('ang_vel_yaw', 0.0)):.3f} "
                f"freq={float(play_cfg.get('gait_frequency', 0.0)):.3f} "
                f"foot_yaw=({float(play_cfg.get('foot_yaw_L', 0.0)):.3f},"
                f"{float(play_cfg.get('foot_yaw_R', 0.0)):.3f}) "
                f"body=({float(play_cfg.get('body_pitch_target', 0.0)):.3f},"
                f"{float(play_cfg.get('body_roll_target', 0.0)):.3f}) "
                f"feet_offset=({float(play_cfg.get('feet_offset_x_target', 0.0)):.3f},"
                f"{float(play_cfg.get('feet_offset_y_target', 0.0)):.3f}) "
                f"fixed_yaw={bool(play_cfg.get('fixed_yaw', False))} "
                f"no_disturbance={bool(play_cfg.get('no_disturbance', False))}"
            )
            if any(key in play_cfg for key in ("target_x", "target_y", "target_local_x", "target_local_y")):
                print(
                    "[play target] "
                    f"abs=({play_cfg.get('target_x', 'auto')},"
                    f"{play_cfg.get('target_y', 'auto')},"
                    f"{play_cfg.get('target_theta', 'auto')}) "
                    f"local=({play_cfg.get('target_local_x', 'auto')},"
                    f"{play_cfg.get('target_local_y', 'auto')},"
                    f"{play_cfg.get('target_heading_offset', 'auto')})"
                )
        default_joint_angles = self.cfg.get("init_state", {}).get("default_joint_angles", {})
        init_pos = self.cfg.get("init_state", {}).get("pos", [0.0, 0.0, 0.0])
        print(
            "[play default pose] "
            f"hip_pitch={float(default_joint_angles.get('Hip_Pitch', 0.0)):.3f} "
            f"knee_pitch={float(default_joint_angles.get('Knee_Pitch', 0.0)):.3f} "
            f"ankle_pitch={float(default_joint_angles.get('Ankle_Pitch', 0.0)):.3f} "
            f"base_z={float(init_pos[2]):.3f}"
        )
        metrics_enabled = self.args.play_velocity_metrics or bool(self.args.play_metrics_csv)
        metrics = None
        metrics_report_interval = max(1, int(float(self.args.play_metrics_interval_s) / self.env.dt))
        play_step = 0
        if metrics_enabled and not (hasattr(self.env, "base_lin_vel") and hasattr(self.env, "base_ang_vel")):
            print("[play metrics] disabled: environment does not expose base velocity tensors")
            metrics_enabled = False
        if metrics_enabled:
            target_command = (
                float(play_cfg.get("lin_vel_x", 0.2)),
                float(play_cfg.get("lin_vel_y", 0.0)),
                float(play_cfg.get("ang_vel_yaw", 0.0)),
            )
            metrics = CommandVelocityMetrics(
                target_command,
                window_s=self.args.play_metrics_window_s,
                csv_path=self.args.play_metrics_csv,
                csv_sample_s=self.args.play_metrics_csv_sample_s,
            )
            print(
                "[play metrics] enabled "
                f"window={self.args.play_metrics_window_s:.2f}s "
                f"warmup={self.args.play_metrics_warmup_s:.2f}s "
                f"csv={self.args.play_metrics_csv or 'off'}"
            )
        if self.cfg["viewer"]["record_video"]:
            os.makedirs("videos", exist_ok=True)
            name = time.strftime("%Y-%m-%d-%H-%M-%S.mp4", time.localtime())
            record_time = self.cfg["viewer"]["record_interval"]
        try:
            while True:
                with torch.no_grad():
                    dist = self.model.act(obs)
                    act = dist.loc
                    obs, rew, done, infos = self.env.step(act)
                    obs, rew, done = obs.to(self.device), rew.to(self.device), done.to(self.device)
                play_step += 1
                if metrics is not None:
                    actual_velocity = (
                        float(self.env.base_lin_vel[0, 0].item()),
                        float(self.env.base_lin_vel[0, 1].item()),
                        float(self.env.base_ang_vel[0, 2].item()),
                    )
                    if hasattr(self.env, "policy_commands"):
                        policy_command = self.env.policy_commands[0, :3].detach().cpu().numpy()
                    elif hasattr(self.env, "commands"):
                        policy_command = self.env.commands[0, :3].detach().cpu().numpy()
                    else:
                        policy_command = None
                    elapsed = play_step * self.env.dt
                    tracked = elapsed >= float(self.args.play_metrics_warmup_s)
                    metrics_row, metrics_summary = metrics.update(
                        elapsed,
                        "walk",
                        actual_velocity,
                        policy_command=policy_command,
                        tracked=tracked,
                    )
                    if play_step % metrics_report_interval == 0:
                        print(f"[play metrics] t={elapsed:6.2f}s {metrics.report(metrics_row, metrics_summary)}")
                if done[0]:
                    termination = infos.get("termination", {})
                    if termination:
                        reason_order = [
                            "contact",
                            "lin_vel",
                            "ang_vel",
                            "height",
                            "timeout",
                            "clear_miss",
                            "late_chase",
                            "orbit",
                            "ball_passed_unblocked",
                            "through_legs",
                            "success",
                        ]
                        reasons = [name for name in reason_order if bool(termination[name][0].item())]
                        print(
                            "[play termination] "
                            f"reasons={','.join(reasons) if reasons else 'unknown'} "
                            f"ball_progress={float(termination['ball_progress_ratio'][0].item()):.3f} "
                            f"robot_progress={float(termination['robot_progress_ratio'][0].item()):.3f} "
                            f"heading_err={float(termination['heading_error'][0].item()):.3f} "
                            f"ball_forward={float(termination['ball_forward'][0].item()):.3f} "
                            f"block_line={float(termination['block_line'][0].item()):.3f} "
                            f"chosen_block={float(termination['chosen_block_line'][0].item()):.3f} "
                            f"support_block={float(termination['support_block_line'][0].item()):.3f}"
                        )
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
            if metrics is not None:
                metrics.close()
    
    def _play_record_and_exit(self):
        """Record video for a specified duration and save to file, then exit.
        This is used by the separate process spawned during training.
        The main process will upload the video to wandb after this process finishes."""
        # Enable video recording
        self.cfg["viewer"]["record_video"] = True
        
        # Get video duration
        video_duration = self.args.video_duration
        if video_duration is None:
            video_duration = self.cfg["runner"].get("log_video_duration", 10.0)
        
        # Get output path for video file
        video_output_path = self.args.video_output_path
        if video_output_path is None:
            print("Error: video_output_path not provided")
            return
        
        # Calculate number of frames to capture
        num_frames = int(video_duration / self.env.dt)
        
        # Clear existing frames
        if hasattr(self.env, 'camera_frames'):
            self.env.camera_frames = []
        
        # Initialize environment
        obs, infos = self.env.reset()
        obs = obs.to(self.device)
        
        # Ensure camera is initialized
        if self.cfg["viewer"]["record_video"]:
            self.env.gym.refresh_actor_root_state_tensor(self.env.sim)
            self.env.render()
        
        # Capture frames
        frames_captured = 0
        total_reward = []
        separated_reward = {}
        
        print(f"Recording video for {video_duration} seconds ({num_frames} frames)...")
        
        while frames_captured < num_frames:
            # Step the environment with current policy
            with torch.no_grad():
                dist = self.model.act(obs)
                act = dist.loc
            
            obs, rew, done, infos = self.env.step(act)
            obs, rew, done = obs.to(self.device), rew.to(self.device), done.to(self.device)
            
            # Store rewards for the first environment only
            total_reward.append(rew[0].item())
            for key, value in infos["rew_terms"].items():
                if key not in separated_reward:
                    separated_reward[key] = []
                separated_reward[key].append(value[0].item())
            
            # Render to capture frame
            if self.cfg["viewer"]["record_video"]:
                self.env.render()
            
            frames_captured += 1
            
            # Reset if episode done
            if done[0]:
                reset_obs, reset_infos = self.env.reset()
                obs = reset_obs.to(self.device)
        
        # Save video to file
        if hasattr(self.env, 'camera_frames') and len(self.env.camera_frames) > 0:
            import numpy as np
            import imageio
            
            # Convert frames to RGB format
            video_frames = []
            for frame in self.env.camera_frames:
                if len(frame.shape) == 3:
                    if frame.shape[2] == 4:
                        # BGRA to RGB
                        rgb_frame = frame[:, :, [2, 1, 0]]
                    elif frame.shape[2] == 3:
                        rgb_frame = frame
                    else:
                        rgb_frame = frame[:, :, :3]
                else:
                    continue
                
                # Ensure uint8 format
                if rgb_frame.dtype != np.uint8:
                    if rgb_frame.max() <= 1.0:
                        rgb_frame = (rgb_frame * 255).astype(np.uint8)
                    else:
                        rgb_frame = np.clip(rgb_frame, 0, 255).astype(np.uint8)
                
                video_frames.append(rgb_frame)
            
            # Save video file
            os.makedirs(os.path.dirname(video_output_path), exist_ok=True)
            fps = int(1.0 / self.env.dt)
            imageio.mimwrite(video_output_path, video_frames, fps=fps, codec='libx264')
            print(f"Video saved to {video_output_path}")
        else:
            print("Warning: No frames captured")
        
        # Save reward data to JSON file
        if self.args.rewards_output_path and len(total_reward) > 0:
            import json
            os.makedirs(os.path.dirname(self.args.rewards_output_path), exist_ok=True)
            reward_data = {
                "total_reward": total_reward,
                "separated_reward": separated_reward
            }
            with open(self.args.rewards_output_path, 'w') as f:
                json.dump(reward_data, f)
            print(f"Reward data saved to {self.args.rewards_output_path}")
        
        # Clean up
        if hasattr(self.env, 'camera_frames'):
            self.env.camera_frames = []
        
        print("Video recording complete. Exiting...")

    def interrupt_handler(self, signal, frame):
        print("\nInterrupt received, waiting for video to finish...")
        self.interrupt = True

    def _capture_training_video(self, duration, it, obs, privileged_obs):
        """Capture video frames during training for wandb logging.
        
        Args:
            duration: Duration of video in seconds
            it: Current iteration step
            obs: Current observations
            privileged_obs: Current privileged observations
            
        Returns:
            Updated obs and privileged_obs after video capture
        """
        # Clear existing frames and ensure camera is initialized
        if hasattr(self.env, 'camera_frames'):
            self.env.camera_frames = []
        
        # Ensure camera is initialized by calling render once before capturing
        # This ensures the camera exists and root_states are available
        if self.cfg["viewer"]["record_video"]:
            # Refresh root states to ensure camera position is correct
            self.env.gym.refresh_actor_root_state_tensor(self.env.sim)
            self.env.render()
        
        # Calculate number of frames to capture
        num_frames = int(duration / self.env.dt)
        
        # Capture frames by running the environment
        frames_captured = 0

        total_reward = []
        seperated_reward = {}
        
        while frames_captured < num_frames:
            # Step the environment with current policy first
            with torch.no_grad():
                dist = self.model.act(obs)
                act = dist.loc
            
            obs, rew, done, infos = self.env.step(act)
            obs, rew, done = obs.to(self.device), rew.to(self.device), done.to(self.device)
            privileged_obs = infos["privileged_obs"].to(self.device)

            # Store rewards for the first environment only
            total_reward.append(rew[0].item())
            for key, value in infos["rew_terms"].items():
                if key not in seperated_reward:
                    seperated_reward[key] = []
                seperated_reward[key].append(value[0].item())
            
            # step() already calls render() internally which captures frames
            # But we ensure render is called to capture the frame
            # The render() in step() should have already captured the frame,
            # but we call it again to be safe (it's idempotent for frame capture)
            if self.cfg["viewer"]["record_video"]:
                self.env.render()
            
            frames_captured += 1
            
            # Reset if episode done
            if done[0]:
                reset_obs, reset_infos = self.env.reset()
                obs = reset_obs.to(self.device)
                privileged_obs = reset_infos["privileged_obs"].to(self.device)
        
        # Log video to wandb
        if hasattr(self.env, 'camera_frames') and len(self.env.camera_frames) > 0:
            self.recorder.log_video(self.env.camera_frames, it, self.env.dt)
            # Clear frames to free memory
            self.env.camera_frames = []
        
        # Log video rewards
        self.recorder.log_video_rewards(total_reward, seperated_reward, it)
        
        return obs, privileged_obs

    def _get_robot_type(self, task_name):
        """Determine robot type from task name."""
        # Check if task name starts with K1 or T1
        if task_name.startswith("K1"):
            return "K1"
        elif task_name.startswith("T1"):
            return "T1"
        else:
            # Default fallback - could be extended for other robot types
            return "Unknown"
    
    def _upload_video_to_wandb(self, video_path, iteration):
        """Upload a video file to wandb.
        
        Args:
            video_path: Path to the video file
            iteration: Iteration number for logging
        """
        if not self.cfg["runner"].get("use_wandb", False):
            return
        
        import wandb
        if wandb.run is None:
            print("Warning: wandb run not initialized, cannot upload video")
            return
        
        try:
            # Use custom step metric for video logs to avoid step conflicts
            # See: https://docs.wandb.ai/models/track/log/customize-logging-axes
            # The iteration parameter is already it+1 (passed from spawn function)
            wandb.log({
                "video/iteration": iteration,  # Custom x-axis metric
                "video/training": wandb.Video(video_path, format="mp4")
            }, commit=True)
            print(f"Video uploaded to wandb at iteration {iteration}")
        except Exception as e:
            print(f"Error uploading video to wandb: {e}")
            import traceback
            traceback.print_exc()
    
    def _upload_rewards_to_wandb(self, rewards_path, iteration):
        """Load reward data from file and upload plots to wandb.
        
        Args:
            rewards_path: Path to the JSON file containing reward data
            iteration: Iteration number for logging
        """
        if not self.cfg["runner"].get("use_wandb", False):
            return
        
        import wandb
        import json
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        if wandb.run is None:
            print("Warning: wandb run not initialized, cannot upload rewards")
            return
        
        try:
            # Load reward data
            with open(rewards_path, 'r') as f:
                reward_data = json.load(f)
            
            total_reward = reward_data["total_reward"]
            separated_reward = reward_data["separated_reward"]
            
            if len(total_reward) == 0:
                print("Warning: No reward data to log")
                return
            
            # Use custom step metric for video logs to avoid step conflicts
            # See: https://docs.wandb.ai/models/track/log/customize-logging-axes
            # The iteration parameter is already it+1 (passed from spawn function)
            
            # Convert to numpy arrays
            total_reward_np = np.array(total_reward)
            timesteps = np.arange(len(total_reward_np))
            
            # Calculate statistics
            mean_total_reward = float(np.mean(total_reward_np))
            sum_total_reward = float(np.sum(total_reward_np))
            
            # Log summary statistics
            self.recorder.writer.add_scalar("video/mean_reward", mean_total_reward, iteration)
            self.recorder.writer.add_scalar("video/sum_reward", sum_total_reward, iteration)
            
            # Prepare log dictionary with custom step metric
            log_dict = {
                "video/iteration": iteration,  # Custom x-axis metric
                "video/mean_reward": mean_total_reward,
                "video/sum_reward": sum_total_reward,
            }
            
            # Create and log figure for total reward
            fig_total = plt.figure(figsize=(12, 4))
            plt.plot(timesteps, total_reward_np, linewidth=2, color='blue')
            plt.title(f'Total Reward (Mean: {mean_total_reward:.3f}, Sum: {sum_total_reward:.3f})', fontsize=12, fontweight='bold')
            plt.xlabel('Frame')
            plt.ylabel('Reward')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            plt.tight_layout()
            log_dict["video_plots/total_reward_trajectory"] = wandb.Image(fig_total)
            plt.close(fig_total)
            
            # Create and log figure for each reward term
            for key, values in separated_reward.items():
                if len(values) == 0:
                    continue
                
                values_np = np.array(values)
                mean_value = float(np.mean(values_np))
                sum_value = float(np.sum(values_np))
                
                # Create figure for this reward term
                fig_term = plt.figure(figsize=(12, 4))
                plt.plot(timesteps, values_np, linewidth=2)
                plt.title(f'{key} (Mean: {mean_value:.3f}, Sum: {sum_value:.3f})', fontsize=12)
                plt.xlabel('Frame')
                plt.ylabel('Reward')
                plt.grid(True, alpha=0.3)
                plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                plt.tight_layout()
                log_dict[f"video_plots/reward_trajectories/{key}"] = wandb.Image(fig_term)
                plt.close(fig_term)
            
            # Log everything at once with custom step metric
            wandb.log(log_dict, commit=True)
            print(f"Reward plots uploaded to wandb at iteration {iteration}")
        except Exception as e:
            print(f"Error uploading rewards to wandb: {e}")
            import traceback
            traceback.print_exc()
    
    def _spawn_video_recording_process(self, checkpoint_path, iteration, video_duration):
        """Spawn a separate process to record video and save to file.
        The main process will upload the video to wandb after the process finishes.
        
        Args:
            checkpoint_path: Path to the checkpoint file to load
            iteration: Current iteration number for wandb logging
            video_duration: Duration of video to record in seconds
        """
        if not self.cfg["runner"].get("use_wandb", False):
            return
        
        # Create video output path
        video_dir = os.path.join(self.recorder.dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        video_output_path = os.path.join(video_dir, f"video_iter_{iteration + 1}.mp4")
        rewards_output_path = os.path.join(video_dir, f"rewards_iter_{iteration + 1}.json")
        
        # Build command to run play.py in record mode
        play_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "play.py")
        cmd = [
            sys.executable,
            play_script,
            "--task", self.cfg["basic"]["task"],
            "--checkpoint", checkpoint_path,
            "--record_video_mode",
            "--video_duration", str(video_duration),
            "--video_iteration", str(iteration + 1),
            "--video_output_path", video_output_path,
            "--rewards_output_path", rewards_output_path,
        ]
        
        # Add other relevant arguments if they were provided
        if self.args.num_envs is not None:
            cmd.extend(["--num_envs", str(self.args.num_envs)])
        # Use headless from config for video recording (usually better for separate process)
        if self.cfg["basic"].get("headless") is not None:
            cmd.extend(["--headless", str(self.cfg["basic"]["headless"])])
        elif self.args.headless is not None:
            cmd.extend(["--headless", str(self.args.headless)])
        if self.args.sim_device is not None:
            cmd.extend(["--sim_device", self.args.sim_device])
        if self.args.rl_device is not None:
            cmd.extend(["--rl_device", self.args.rl_device])
        # Always forward the active training seed so video subprocesses are reproducible.
        cmd.extend(["--seed", str(self.cfg["basic"]["seed"])])
        if self.args.model is not None:
            cmd.extend(["--model", self.args.model])
        
        print(f"Spawning video recording process for iteration {iteration + 1}...")
        print(f"Command: {' '.join(cmd)}")
        
        # Verify checkpoint file exists
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint file {checkpoint_path} does not exist")
            return
        
        # Spawn process with environment variables
        env = os.environ.copy()
        # Ensure PYTHONPATH is set correctly
        if 'PYTHONPATH' not in env:
            env['PYTHONPATH'] = os.path.dirname(os.path.dirname(__file__))
        else:
            env['PYTHONPATH'] = os.path.dirname(os.path.dirname(__file__)) + os.pathsep + env['PYTHONPATH']
        
        # Create log files for the subprocess
        log_dir = os.path.join(self.recorder.dir, "video_logs")
        os.makedirs(log_dir, exist_ok=True)
        stdout_file = os.path.join(log_dir, f"video_iter_{iteration + 1}_stdout.log")
        stderr_file = os.path.join(log_dir, f"video_iter_{iteration + 1}_stderr.log")
        
        try:
            with open(stdout_file, 'w') as fout, open(stderr_file, 'w') as ferr:
                process = subprocess.Popen(
                    cmd,
                    stdout=fout,
                    stderr=ferr,
                    env=env,
                )
            
            print(f"Video recording process started (PID: {process.pid})")
            print(f"  Logs: {stdout_file} and {stderr_file}")
            print(f"  Waiting for video recording to complete...")
            
            # Wait for the process to complete
            return_code = process.wait()
            
            if return_code == 0:
                print(f"Video recording completed successfully for iteration {iteration + 1}")
                # Upload video and rewards to wandb
                if os.path.exists(video_output_path):
                    self._upload_video_to_wandb(video_output_path, iteration + 1)
                else:
                    print(f"Warning: Video file not found at {video_output_path}")
                
                # Load and log reward data
                if os.path.exists(rewards_output_path):
                    self._upload_rewards_to_wandb(rewards_output_path, iteration + 1)
                else:
                    print(f"Warning: Reward data file not found at {rewards_output_path}")
            else:
                print(f"Warning: Video recording process exited with code {return_code}")
                print(f"  Check logs: {stdout_file} and {stderr_file}")
                
        except Exception as e:
            print(f"Error spawning video recording process: {e}")
            import traceback
            traceback.print_exc()
