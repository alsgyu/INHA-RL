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
from utils.buffer import ExperienceBuffer
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
        self._load()

        self.buffer = ExperienceBuffer(self.cfg["runner"]["horizon_length"], self.env.num_envs, self.device)
        self.buffer.add_buffer("actions", (self.env.num_actions,))
        self.buffer.add_buffer("obses", (self.env.num_obs,))
        self.buffer.add_buffer("privileged_obses", (self.env.num_privileged_obs,))
        self.buffer.add_buffer("rewards", ())
        self.buffer.add_buffer("dones", (), dtype=bool)
        self.buffer.add_buffer("time_outs", (), dtype=bool)

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
        for arg in vars(self.args):
            if getattr(self.args, arg) is not None:
                if arg == "num_envs":
                    self.cfg["env"][arg] = getattr(self.args, arg)
                elif arg == "task":
                    continue
                else:
                    self.cfg["basic"][arg] = getattr(self.args, arg)
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
                self.recorder.save(
                    {
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "curriculum": self.env.curriculum_prob,
                        "ball_curriculum_level": getattr(self.env, 'ball_curriculum_global_level', 0),
                    },
                    it + 1,
                )
            
            if should_log_video:
                # If we didn't save yet, save checkpoint now for video recording
                if not should_save:
                    checkpoint_path = os.path.join(self.recorder.model_dir, f"model_{it + 1}.pth")
                    self.recorder.save(
                        {
                            "model": self.model.state_dict(),
                            "optimizer": self.optimizer.state_dict(),
                            "curriculum": self.env.curriculum_prob,
                            "ball_curriculum_level": getattr(self.env, 'ball_curriculum_global_level', 0),
                        },
                        it + 1,
                    )
                # Spawn separate process to record video (will wait for completion)
                # Note: Video will be uploaded at step it+1 (after training loop logs at step it)
                self._spawn_video_recording_process(checkpoint_path, it, log_video_duration)
            # within horizon_length, env.step() is called with same act
            for n in range(self.cfg["runner"]["horizon_length"]):
                self.buffer.update_data("obses", n, obs)
                self.buffer.update_data("privileged_obses", n, privileged_obs)
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
                self.buffer.update_data("time_outs", n, infos["time_outs"].to(self.device))
                ep_info = {"reward": rew}
                if log_reward_terms:
                    ep_info.update(infos["rew_terms"])
                if log_env_metrics and "metrics" in infos and bool(done.any().item()):
                    ep_info.update(infos["metrics"])
                self.recorder.record_episode_statistics(done, ep_info, it, n == (self.cfg["runner"]["horizon_length"] - 1))

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
                loss = (
                    value_loss
                    + actor_loss
                    + self.cfg["algorithm"]["bound_coef"] * bound_loss
                    + self.cfg["algorithm"]["entropy_coef"] * entropy.mean()
                    + 0.01 * loss_entropy
                    #+ self.cfg["algorithm"]["symmetry_coef"] * sym_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_actor_loss += actor_loss.item()
                mean_bound_loss += bound_loss.item()
                mean_entropy += entropy.mean()

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
            self.recorder.record_statistics(
                {
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
                },
                it,
            )

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
                )

    def play(self):
        # Check if we're in record-and-exit mode (for separate process video recording)
        if self.args.record_video_mode:
            self._play_record_and_exit()
            return
        
        # Normal play mode (for manual testing)
        obs, infos = self.env.reset()
        obs = obs.to(self.device)
        if self.cfg["viewer"]["record_video"]:
            os.makedirs("videos", exist_ok=True)
            name = time.strftime("%Y-%m-%d-%H-%M-%S.mp4", time.localtime())
            record_time = self.cfg["viewer"]["record_interval"]
        while True:
            with torch.no_grad():
                dist = self.model.act(obs)
                act = dist.loc
                obs, rew, done, infos = self.env.step(act)
                obs, rew, done = obs.to(self.device), rew.to(self.device), done.to(self.device)
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
