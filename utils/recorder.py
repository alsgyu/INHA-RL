import torch
from torch.utils.tensorboard import SummaryWriter
import os
import time
import wandb
import yaml
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class Recorder:

    def __init__(self, cfg, skip_wandb_init=False):
        self.cfg = cfg
        name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        # Create logs in robot-type/task-name hierarchy
        task_name = self.cfg["basic"].get("log_task", self.cfg["basic"]["task"])
        
        # Determine robot type from task name
        robot_type = self._get_robot_type(task_name)
        
        # Create hierarchical structure: logs/robot_type/task_name/timestamp
        robot_log_dir = os.path.join("logs", robot_type)
        task_log_dir = os.path.join(robot_log_dir, task_name)
        self.dir = os.path.join(task_log_dir, name)
        os.makedirs(self.dir, exist_ok=True)
        self.model_dir = os.path.join(self.dir, "nn")
        os.mkdir(self.model_dir)
        self.writer = SummaryWriter(os.path.join(self.dir, "summaries"))
        if self.cfg["runner"]["use_wandb"] and not skip_wandb_init:
            # Sanitize project name for wandb (remove invalid characters)
            project_name = self._sanitize_project_name(task_name)
            wandb.init(
                project=project_name,
                dir=self.dir,
                name=name,
                notes=self.cfg["basic"]["description"],
                config=self.cfg,
            )
            # Define custom step metric for video logs to avoid step conflicts
            # See: https://docs.wandb.ai/models/track/log/customize-logging-axes
            wandb.define_metric("video/iteration", step_metric="video/iteration")
            wandb.define_metric("video/*", step_metric="video/iteration")
            wandb.define_metric("video_plots/*", step_metric="video/iteration")

        self.episode_statistics = {}
        self.last_episode = {}
        self.last_episode["steps"] = []
        self.episode_steps = None

        with open(os.path.join(self.dir, "config.yaml"), "w") as file:
            yaml.dump(self.cfg, file)
    
    def resume_wandb_run(self, run_id, project, run_name, wandb_dir):
        """Resume an existing wandb run for logging in a separate process.
        
        Uses shared mode to allow logging from multiple processes to the same run.
        See: https://docs.wandb.ai/models/track/log/distributed-training#track-all-processes-to-a-single-run
        
        Args:
            run_id: The wandb run ID to resume
            project: The wandb project name
            run_name: The wandb run name
            wandb_dir: The wandb directory path (should be the parent directory, not the run files dir)
        """
        if not self.cfg["runner"]["use_wandb"]:
            print("Warning: wandb is disabled in config, cannot resume run")
            return
        
        # Terminate any existing wandb run
        if wandb.run is not None:
            print("Finishing existing wandb run...")
            wandb.finish()
        
        # Fix wandb_dir - it should point to the parent directory, not the run files directory
        # wandb_dir typically ends with /files, we need the parent
        original_wandb_dir = wandb_dir
        if wandb_dir.endswith("/files"):
            wandb_dir = os.path.dirname(wandb_dir)
        elif "/wandb/run-" in wandb_dir:
            # Extract the directory before /wandb/
            wandb_dir = wandb_dir.split("/wandb/")[0]
        
        print(f"Resuming wandb run: {run_id} (project: {project})")
        
        # Resume the run using shared mode for multi-process logging
        # This allows logging from a separate process while the main process is also logging
        # See: https://docs.wandb.ai/models/track/log/distributed-training#track-all-processes-to-a-single-run
        try:
            wandb.init(
                project=project,
                id=run_id,
                name=run_name,
                dir=wandb_dir,
                settings=wandb.Settings(
                    mode="shared",  # Enable shared mode for multi-process logging
                    x_label="video_recorder",  # Label to identify this process in logs
                    x_primary=False,  # This is a worker process, not the primary
                    init_timeout=120,  # Increase timeout to 120 seconds
                ),
            )
            print(f"Wandb run resumed successfully!")
            print(f"  Run ID: {wandb.run.id if wandb.run else 'None'}")
            print(f"  Run URL: {wandb.run.url if wandb.run else 'None'}")
        except Exception as e:
            print(f"Error resuming wandb run: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _sanitize_project_name(self, task_name):
        """Sanitize task name for wandb project name by removing invalid characters."""
        # Replace invalid characters with underscores
        invalid_chars = ['/', '\\', '#', '?', '%', ':']
        sanitized = task_name
        for char in invalid_chars:
            sanitized = sanitized.replace(char, '_')
        return sanitized

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

    def record_episode_statistics(self, done, ep_info, it, write_record=False):
        if self.episode_steps is None:
            self.episode_steps = torch.zeros_like(done, dtype=int)
        else:
            self.episode_steps += 1
        for val in self.episode_steps[done]:
            self.last_episode["steps"].append(val.item())
        self.episode_steps[done] = 0

        for key, value in ep_info.items():
            if self.episode_statistics.get(key) is None:
                self.episode_statistics[key] = torch.zeros_like(value)
            self.episode_statistics[key] += value
            if self.last_episode.get(key) is None:
                self.last_episode[key] = []
            for done_value in self.episode_statistics[key][done]:
                self.last_episode[key].append(done_value.item())
            self.episode_statistics[key][done] = 0

        if write_record:
            for key in self.last_episode.keys():
                path = ("" if key == "steps" or key == "reward" else "episode/") + key
                value = self._mean(self.last_episode[key])
                self.writer.add_scalar(path, value, it)
                if self.cfg["runner"]["use_wandb"]:
                    wandb.log({path: value}, step=it)
                self.last_episode[key].clear()

    def record_statistics(self, statistics, it):
        for key, value in statistics.items():
            self.writer.add_scalar(key, float(value), it)
            if self.cfg["runner"]["use_wandb"]:
                wandb.log({key: float(value)}, step=it)

    def save(self, model_dict, it):
        path = os.path.join(self.model_dir, "model_{}.pth".format(it))
        print("Saving model to {}".format(path))
        torch.save(model_dict, path)

    def _mean(self, data):
        if len(data) == 0:
            return 0.0
        else:
            return sum(data) / len(data)

    def log_video(self, frames, it, dt):
        """Log video frames to wandb.
        
        Args:
            frames: List of camera frames in RGBA format (height, width, 4)
            it: Current iteration step
            dt: Simulation timestep for calculating FPS
        """
        if not self.cfg["runner"]["use_wandb"]:
            return
        
        if len(frames) == 0:
            return
        
        import numpy as np
        
        # Convert frames from RGBA to RGB format
        # Isaac Gym returns images in BGRA format, so we need to convert
        video_array = []
        for frame in frames:
            # Handle different frame shapes
            if len(frame.shape) == 3:
                if frame.shape[2] == 4:
                    # Isaac Gym returns BGRA format, convert to RGB
                    rgb_frame = frame[:, :, [2, 1, 0]]  # BGR -> RGB
                elif frame.shape[2] == 3:
                    rgb_frame = frame
                else:
                    rgb_frame = frame[:, :, :3]
            elif len(frame.shape) == 2:
                # Grayscale, convert to RGB by repeating channels
                rgb_frame = np.stack([frame, frame, frame], axis=2)
            else:
                # Try to reshape if it's a flattened image
                if frame.size % 4 == 0:
                    h = int(np.sqrt(frame.size // 4))
                    w = frame.size // (4 * h)
                    frame_reshaped = frame.reshape(h, w, 4)
                    rgb_frame = frame_reshaped[:, :, [2, 1, 0]]  # BGR -> RGB
                else:
                    continue  # Skip invalid frames
            
            # Ensure uint8 format (0-255 range)
            if rgb_frame.dtype != np.uint8:
                if rgb_frame.max() <= 1.0:
                    rgb_frame = (rgb_frame * 255).astype(np.uint8)
                else:
                    rgb_frame = np.clip(rgb_frame, 0, 255).astype(np.uint8)
            
            video_array.append(rgb_frame)
        
        if len(video_array) == 0:
            print(f"Warning: No valid frames to log at iteration {it}")
            return
        
        # Stack frames: (time, height, width, channels)
        video_tensor = np.stack(video_array, axis=0)
        # wandb.Video expects (time, channels, height, width) format
        video_tensor = np.transpose(video_tensor, (0, 3, 1, 2))
        
        # Calculate FPS from simulation timestep
        fps = int(1.0 / dt)
        
        # Log to wandb
        if wandb.run is None:
            print("Error: wandb.run is None, cannot log video")
            return
        
        # Get current step and ensure we log at a step >= current step
        current_step = wandb.run.step
        log_step = max(it, current_step)
        
        try:
            wandb.log({"video/training": wandb.Video(video_tensor, fps=fps, format="mp4")}, step=log_step, commit=True)
            print(f"Video logged to wandb at step {log_step} (iteration {it})")
        except Exception as e:
            print(f"Error logging video to wandb: {e}")
            import traceback
            traceback.print_exc()

    def log_video_rewards(self, total_reward, separated_reward, it):
        """Log rewards collected during video capture as time-series graphs.
        
        Args:
            total_reward: List of total reward values per step (list of floats)
            separated_reward: Dictionary of reward term names to lists of floats
            it: Current iteration step
        """
        matplotlib.use('Agg')  # Use non-interactive backend

        if len(total_reward) == 0:
            print(f"Warning: No rewards to log at iteration {it}")
            return
        
        if not self.cfg["runner"]["use_wandb"]:
            print(f"Warning: wandb is disabled, skipping reward trajectory logging")
            return
        
        # Get current step and ensure we log at a step >= current step
        if wandb.run is not None:
            current_step = wandb.run.step
            log_step = max(it, current_step)
        else:
            log_step = it
        
        # Convert total reward to numpy array
        total_reward_np = np.array(total_reward)
        
        # Calculate statistics for total reward
        mean_total_reward = float(np.mean(total_reward_np))
        sum_total_reward = float(np.sum(total_reward_np))
        
        # Log summary statistics
        self.writer.add_scalar("video/mean_reward", mean_total_reward, it)
        self.writer.add_scalar("video/sum_reward", sum_total_reward, it)
        
        timesteps = np.arange(len(total_reward_np))
        
        # Log to wandb
        if self.cfg["runner"]["use_wandb"]:
            
            # Create and log figure for total reward
            fig_total = plt.figure(figsize=(12, 4))
            plt.plot(timesteps, total_reward_np, linewidth=2, color='blue')
            plt.title(f'Total Reward (Mean: {mean_total_reward:.3f}, Sum: {sum_total_reward:.3f})', fontsize=12, fontweight='bold')
            plt.xlabel('Frame')
            plt.ylabel('Reward')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            plt.tight_layout()
            wandb.log({"video_plots/total_reward_trajectory": wandb.Image(fig_total)}, step=log_step)
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
                wandb.log({f"video_plots/reward_trajectories/{key}": wandb.Image(fig_term)}, step=log_step)
                plt.close(fig_term)
            
            
