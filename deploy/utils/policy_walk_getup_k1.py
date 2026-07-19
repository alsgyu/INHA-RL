import os

import numpy as np
import torch


class Policy:
    def __init__(self, cfg, enable_getup=True, walk_policy=None, walk_policy_path=None):
        self.cfg = cfg
        self.enable_getup = enable_getup
        self.walk_policy_path = walk_policy_path or cfg["walk_policy"]["policy_path"]
        self.getup_policy_path = self._resolve_policy_path(
            cfg["getup_policy"]["policy_path"],
            allow_missing=not enable_getup,
        )
        self.walk_policy = walk_policy
        if self.walk_policy is None:
            self.walk_policy_path = self._resolve_policy_path(self.walk_policy_path)
            self.walk_policy = torch.jit.load(self.walk_policy_path)
        self.getup_policy = torch.jit.load(self.getup_policy_path) if enable_getup else None
        self.walk_policy.eval()
        if self.getup_policy is not None:
            self.getup_policy.eval()
        self._init_inference_variables()

    def _resolve_policy_path(self, configured_path, allow_missing=False):
        if configured_path and os.path.isabs(configured_path) and os.path.exists(configured_path):
            return configured_path

        deploy_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        repo_dir = os.path.abspath(os.path.join(deploy_dir, ".."))
        relative_path = configured_path[2:] if configured_path and configured_path.startswith("./") else configured_path
        candidates = [
            configured_path,
            os.path.join(deploy_dir, relative_path) if relative_path else None,
            os.path.join(repo_dir, relative_path) if relative_path else None,
            os.path.join(repo_dir, "deploy", relative_path) if relative_path else None,
        ]
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return candidate

        if allow_missing:
            return configured_path
        raise FileNotFoundError(
            f"Could not resolve policy path '{configured_path}'. "
            f"Export the model to deploy/models or pass an existing policy path."
        )

    def _init_inference_variables(self):
        self.default_dof_pos = np.array(self.cfg["common"]["default_qpos"], dtype=np.float32)
        self.walk_default_dof_pos = np.array(
            self.cfg["walk_policy"].get("default_qpos", self.cfg["common"]["default_qpos"]),
            dtype=np.float32,
        )
        self.getup_default_dof_pos = np.array(
            self.cfg["getup_policy"].get("default_qpos", self.cfg["common"]["default_qpos"]),
            dtype=np.float32,
        )
        self.dof_targets = np.copy(self.default_dof_pos)
        self.mode = "walk"
        self.recovered_time = 0.0

        self.walk_actions = np.zeros(self.cfg["walk_policy"]["num_actions"], dtype=np.float32)
        self.getup_actions = np.zeros(self.cfg["getup_policy"]["num_actions"], dtype=np.float32)
        self.walk_obs = np.zeros(self.cfg["walk_policy"]["num_observations"], dtype=np.float32)
        self.getup_obs = np.zeros(self.cfg["getup_policy"]["num_observations"], dtype=np.float32)
        self.commands = np.zeros(3, dtype=np.float32)
        self.smoothed_commands = np.zeros(3, dtype=np.float32)
        self.walk_gait_process = 0.0
        self.current_walk_gait_frequency = 0.0
        self.policy_interval = self.cfg["common"]["dt"] * self.cfg["walk_policy"]["control"]["decimation"]

    def get_policy_interval(self):
        return self.policy_interval

    def _is_fallen(self, base_rpy, projected_gravity):
        cfg = self.cfg["switching"]
        return (
            abs(float(base_rpy[0])) > float(cfg["fall_roll_threshold"])
            or abs(float(base_rpy[1])) > float(cfg["fall_pitch_threshold"])
            or float(projected_gravity[2]) > float(cfg["fall_projected_gravity_z"])
        )

    def _is_recovered(self, base_rpy, projected_gravity, base_ang_vel):
        cfg = self.cfg["switching"]
        return (
            abs(float(base_rpy[0])) < float(cfg["recover_roll_threshold"])
            and abs(float(base_rpy[1])) < float(cfg["recover_pitch_threshold"])
            and float(projected_gravity[2]) < float(cfg["recover_projected_gravity_z"])
            and float(np.linalg.norm(base_ang_vel)) < float(cfg["recover_ang_vel_threshold"])
        )

    def _update_mode(self, base_rpy, projected_gravity, base_ang_vel):
        if not self.enable_getup:
            return

        if self.mode == "walk" and self._is_fallen(base_rpy, projected_gravity):
            print("Switching policy: walk -> getup")
            self.mode = "getup"
            self.recovered_time = 0.0
            self.getup_actions[:] = 0.0
            self.smoothed_commands[:] = 0.0
            return

        if self.mode == "getup":
            if self._is_recovered(base_rpy, projected_gravity, base_ang_vel):
                self.recovered_time += self.policy_interval
            else:
                self.recovered_time = 0.0
            if self.recovered_time >= float(self.cfg["switching"]["recover_hold_s"]):
                print("Switching policy: getup -> walk")
                self.mode = "walk"
                self.walk_actions[:] = 0.0
                self.walk_gait_process = 0.0
                self.recovered_time = 0.0

    def _resolve_walk_gait_frequency(self, walk_cfg):
        cfg = walk_cfg.get("gait_frequency_by_speed", {})
        if not cfg.get("enabled", False):
            return float(walk_cfg["gait_frequency"])

        speed_low, speed_high = cfg.get("speed_range", [0.0, 1.0])
        freq_low, freq_high = cfg.get("frequency_range", [walk_cfg["gait_frequency"], walk_cfg["gait_frequency"]])
        yaw_weight = float(cfg.get("yaw_weight", 0.0))
        speed = np.linalg.norm(self.smoothed_commands[:2]) + yaw_weight * abs(float(self.smoothed_commands[2]))
        ratio = np.clip((speed - float(speed_low)) / max(float(speed_high) - float(speed_low), 1.0e-6), 0.0, 1.0)
        return float(freq_low) + ratio * (float(freq_high) - float(freq_low))

    def inference(self, time_now, dof_pos, dof_vel, base_ang_vel, projected_gravity, base_rpy, vx, vy, vyaw):
        self._update_mode(base_rpy, projected_gravity, base_ang_vel)
        if self.mode == "getup":
            return self._getup_inference(dof_pos, dof_vel, base_ang_vel, projected_gravity)
        return self._walk_inference(dof_pos, dof_vel, base_ang_vel, projected_gravity, vx, vy, vyaw)

    def _walk_inference(self, dof_pos, dof_vel, base_ang_vel, projected_gravity, vx, vy, vyaw):
        walk_cfg = self.cfg["walk_policy"]
        leg_start = int(walk_cfg.get("leg_start_index", self.cfg["common"]["joint_cnt"] - walk_cfg["num_actions"]))
        leg_end = leg_start + walk_cfg["num_actions"]

        self.commands[:] = [vx, vy, vyaw]
        clip_range = (-self.policy_interval, self.policy_interval)
        self.smoothed_commands += np.clip(self.commands - self.smoothed_commands, *clip_range)

        moving = np.linalg.norm(self.smoothed_commands) > float(walk_cfg.get("stand_command_threshold", 1.0e-5))
        gait_frequency = self._resolve_walk_gait_frequency(walk_cfg) if moving else 0.0
        self.current_walk_gait_frequency = gait_frequency
        self.walk_gait_process = np.fmod(self.walk_gait_process + self.policy_interval * gait_frequency, 1.0)

        command_block = np.array(
            [
                self.smoothed_commands[0],
                self.smoothed_commands[1],
                self.smoothed_commands[2],
                gait_frequency,
                float(walk_cfg.get("foot_yaw_L", 0.0)),
                float(walk_cfg.get("foot_yaw_R", 0.0)),
                float(walk_cfg.get("body_pitch_target", 0.0)),
                float(walk_cfg.get("body_roll_target", 0.0)),
                float(walk_cfg.get("feet_offset_x_target", 0.0)),
                float(walk_cfg.get("feet_offset_y_target", 0.0)),
            ],
            dtype=np.float32,
        )

        norm = walk_cfg["normalization"]
        command_scale = np.array(
            [
                norm["lin_vel"],
                norm["lin_vel"],
                norm["ang_vel"],
                norm["gait_frequency"],
                norm["foot_yaw"],
                norm["foot_yaw"],
                norm["body_pitch_target"],
                norm["body_roll_target"],
                norm["feet_offset_x_target"],
                norm["feet_offset_y_target"],
            ],
            dtype=np.float32,
        )
        self.walk_obs[0:3] = projected_gravity * norm["gravity"]
        self.walk_obs[3:6] = base_ang_vel * norm["ang_vel"]
        self.walk_obs[6:16] = command_block * command_scale
        self.walk_obs[16] = np.cos(2 * np.pi * self.walk_gait_process) * moving
        self.walk_obs[17] = np.sin(2 * np.pi * self.walk_gait_process) * moving
        self.walk_obs[18:30] = (dof_pos[leg_start:leg_end] - self.walk_default_dof_pos[leg_start:leg_end]) * norm["dof_pos"]
        self.walk_obs[30:42] = dof_vel[leg_start:leg_end] * norm["dof_vel"]
        self.walk_obs[42:54] = self.walk_actions

        with torch.no_grad():
            output = self.walk_policy(torch.from_numpy(self.walk_obs).unsqueeze(0)).detach().numpy()[0]
        self.walk_actions[:] = np.clip(output, -norm["clip_actions"], norm["clip_actions"])
        self.dof_targets[:] = self.default_dof_pos
        self.dof_targets[leg_start:leg_end] = (
            self.walk_default_dof_pos[leg_start:leg_end]
            + float(walk_cfg["control"]["action_scale"]) * self.walk_actions
        )
        return self.dof_targets

    def _getup_inference(self, dof_pos, dof_vel, base_ang_vel, projected_gravity):
        if self.getup_policy is None:
            raise RuntimeError("getup mode requested but getup policy is not loaded")
        getup_cfg = self.cfg["getup_policy"]
        norm = getup_cfg["normalization"]
        self.getup_obs[0:3] = projected_gravity * norm["gravity"]
        self.getup_obs[3:6] = base_ang_vel * norm["ang_vel"]
        self.getup_obs[6:9] = 0.0
        self.getup_obs[9:11] = 0.0
        self.getup_obs[11:33] = (dof_pos - self.getup_default_dof_pos) * norm["dof_pos"]
        self.getup_obs[33:55] = dof_vel * norm["dof_vel"]
        self.getup_obs[55:77] = self.getup_actions

        with torch.no_grad():
            output = self.getup_policy(torch.from_numpy(self.getup_obs).unsqueeze(0)).detach().numpy()[0]
        self.getup_actions[:] = np.clip(output, -norm["clip_actions"], norm["clip_actions"])
        self.dof_targets[:] = self.getup_default_dof_pos + float(getup_cfg["control"]["action_scale"]) * self.getup_actions
        return self.dof_targets
