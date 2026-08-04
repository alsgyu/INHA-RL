import os
import csv
import time

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import (
    get_axis_params,
    to_torch,
    quat_rotate_inverse,
    quat_from_euler_xyz,
    torch_rand_float,
    get_euler_xyz,
    quat_rotate,
)

assert gymtorch

import torch

import numpy as np
from envs.base_task import BaseTask

from utils.utils import apply_randomization


class ParameterWalkK1(BaseTask):

    def __init__(self, cfg):
        super().__init__(cfg)
        self._create_envs()
        self.gym.prepare_sim(self.sim)
        self._init_buffers()
        self._prepare_reward_function()
        
        # Optional CSV logging for first environment only
        if self.cfg.get("basic", {}).get("enable_csv_logging", False):
            self._init_csv_logging()

    @staticmethod
    def _longest_matching_key(joint_name, values):
        matches = [key for key in values.keys() if key != "default" and key in joint_name]
        if not matches:
            return None
        return max(matches, key=len)

    def _create_envs(self):
        self.num_envs = self.cfg["env"]["num_envs"]
        asset_cfg = self.cfg["asset"]
        asset_root = os.path.dirname(asset_cfg["file"])
        asset_file = os.path.basename(asset_cfg["file"])

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = asset_cfg["default_dof_drive_mode"]
        asset_options.collapse_fixed_joints = asset_cfg["collapse_fixed_joints"]
        asset_options.replace_cylinder_with_capsule = asset_cfg["replace_cylinder_with_capsule"]
        asset_options.flip_visual_attachments = asset_cfg["flip_visual_attachments"]
        asset_options.fix_base_link = asset_cfg["fix_base_link"]
        asset_options.density = asset_cfg["density"]
        asset_options.angular_damping = asset_cfg["angular_damping"]
        asset_options.linear_damping = asset_cfg["linear_damping"]
        asset_options.max_angular_velocity = asset_cfg["max_angular_velocity"]
        asset_options.max_linear_velocity = asset_cfg["max_linear_velocity"]
        asset_options.armature = asset_cfg["armature"]
        asset_options.thickness = asset_cfg["thickness"]
        asset_options.disable_gravity = asset_cfg["disable_gravity"]

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)

        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        self.dof_pos_limits = torch.zeros(self.num_dofs, 2, dtype=torch.float, device=self.device)
        self.dof_vel_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)
        self.torque_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            self.dof_pos_limits[i, 0] = dof_props_asset["lower"][i].item()
            self.dof_pos_limits[i, 1] = dof_props_asset["upper"][i].item()
            self.dof_vel_limits[i] = dof_props_asset["velocity"][i].item()
            self.torque_limits[i] = dof_props_asset["effort"][i].item()

        self.dof_stiffness = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.dof_damping = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.dof_friction = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        stiffness_cfg = self.cfg["control"]["stiffness"]
        damping_cfg = self.cfg["control"]["damping"]
        for i in range(self.num_dofs):
            name = self._longest_matching_key(self.dof_names[i], stiffness_cfg)
            if name is None:
                raise ValueError(f"PD gain of joint {self.dof_names[i]} were not defined")
            if name not in damping_cfg:
                raise ValueError(f"PD damping of joint {self.dof_names[i]} were not defined for key {name}")
            self.dof_stiffness[:, i] = stiffness_cfg[name]
            self.dof_damping[:, i] = damping_cfg[name]
        self.dof_stiffness = apply_randomization(self.dof_stiffness, self.cfg["randomization"].get("dof_stiffness"))
        self.dof_damping = apply_randomization(self.dof_damping, self.cfg["randomization"].get("dof_damping"))
        self.dof_friction = apply_randomization(self.dof_friction, self.cfg["randomization"].get("dof_friction"))

        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        penalized_contact_names = []
        for name in self.cfg["rewards"]["penalize_contacts_on"]:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg["rewards"]["terminate_contacts_on"]:
            termination_contact_names.extend([s for s in body_names if name in s])
        self.base_indice = self.gym.find_asset_rigid_body_index(robot_asset, asset_cfg["base_name"])

        # prepare penalized and termination contact indices
        self.penalized_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device)
        for i in range(len(penalized_contact_names)):
            self.penalized_contact_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, penalized_contact_names[i])
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, termination_contact_names[i])

        rbs_list = self.gym.get_asset_rigid_body_shape_indices(robot_asset)
        self.feet_indices = torch.zeros(len(asset_cfg["foot_names"]), dtype=torch.long, device=self.device)
        self.foot_shape_indices = []
        for i in range(len(asset_cfg["foot_names"])):
            indices = self.gym.find_asset_rigid_body_index(robot_asset, asset_cfg["foot_names"][i])
            self.feet_indices[i] = indices
            self.foot_shape_indices += list(range(rbs_list[indices].start, rbs_list[indices].start + rbs_list[indices].count))

        base_init_state_list = (
            self.cfg["init_state"]["pos"] + self.cfg["init_state"]["rot"] + self.cfg["init_state"]["lin_vel"] + self.cfg["init_state"]["ang_vel"]
        )
        self.base_init_state = to_torch(base_init_state_list, device=self.device)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.envs = []
        self.actor_handles = []
        self.base_mass_scaled = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            start_pose.p = gymapi.Vec3(*pos)

            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, asset_cfg["name"], i, asset_cfg["self_collisions"], 0)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)
            shape_props = self._process_rigid_shape_props(shape_props)
            self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, shape_props)
            self.gym.enable_actor_dof_force_sensors(env_handle, actor_handle)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

    def _process_rigid_body_props(self, props, i):
        for j in range(self.num_bodies):
            if j == self.base_indice:
                props[j].com.x, self.base_mass_scaled[i, 0] = apply_randomization(
                    props[j].com.x, self.cfg["randomization"].get("base_com"), return_noise=True
                )
                props[j].com.y, self.base_mass_scaled[i, 1] = apply_randomization(
                    props[j].com.y, self.cfg["randomization"].get("base_com"), return_noise=True
                )
                props[j].com.z, self.base_mass_scaled[i, 2] = apply_randomization(
                    props[j].com.z, self.cfg["randomization"].get("base_com"), return_noise=True
                )
                props[j].mass, self.base_mass_scaled[i, 3] = apply_randomization(
                    props[j].mass, self.cfg["randomization"].get("base_mass"), return_noise=True
                )
            else:
                props[j].com.x = apply_randomization(props[j].com.x, self.cfg["randomization"].get("other_com"))
                props[j].com.y = apply_randomization(props[j].com.y, self.cfg["randomization"].get("other_com"))
                props[j].com.z = apply_randomization(props[j].com.z, self.cfg["randomization"].get("other_com"))
                props[j].mass = apply_randomization(props[j].mass, self.cfg["randomization"].get("other_mass"))
            props[j].invMass = 1.0 / props[j].mass
        return props

    def _process_rigid_shape_props(self, props):
        for i in self.foot_shape_indices:
            props[i].friction = apply_randomization(0.0, self.cfg["randomization"].get("friction"))
            props[i].compliance = apply_randomization(0.0, self.cfg["randomization"].get("compliance"))
            props[i].restitution = apply_randomization(0.0, self.cfg["randomization"].get("restitution"))
        return props

    def _get_env_origins(self):
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        if self.cfg["terrain"]["type"] == "plane":
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="ij")
            spacing = self.cfg["env"]["env_spacing"]
            self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
            self.env_origins[:, 2] = 0.0
        else:
            num_cols = max(1.0, np.floor(np.sqrt(self.num_envs * self.terrain.env_length / self.terrain.env_width)))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="ij")
            self.env_origins[:, 0] = self.terrain.env_width / (num_rows + 1) * (xx.flatten()[: self.num_envs] + 1)
            self.env_origins[:, 1] = self.terrain.env_length / (num_cols + 1) * (yy.flatten()[: self.num_envs] + 1)
            self.env_origins[:, 2] = self.terrain.terrain_heights(self.env_origins)

    def _init_buffers(self):
        self.num_obs = self.cfg["env"]["num_observations"]
        self.num_privileged_obs = self.cfg["env"]["num_privileged_obs"]
        self.num_actions = self.cfg["env"]["num_actions"]
        self.dt = self.cfg["control"]["decimation"] * self.cfg["sim"]["dt"]

        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float, device=self.device)
        self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, dtype=torch.float, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.extras = {}
        self.extras["rew_terms"] = {}

        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        self.body_states = gymtorch.wrap_tensor(body_state).view(self.num_envs, self.num_bodies, 13)
        self.base_pos = self.root_states[:, 0:3]
        self.base_quat = self.root_states[:, 3:7]
        self.feet_pos = self.body_states[:, self.feet_indices, 0:3]
        self.feet_quat = self.body_states[:, self.feet_indices, 3:7]

        # initialize some data used later on
        self.common_step_counter = 0
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.action_clip_by_index = self._control_action_tensor("action_clip_by_index")
        self.action_scale_by_index = self._control_action_tensor("action_scale_by_index")
        self.action_lower_by_index = self._control_action_tensor("action_lower_by_index")
        self.action_upper_by_index = self._control_action_tensor("action_upper_by_index")
        self.action_rate_limit_by_index = self._control_action_tensor("action_rate_limit_by_index")
        self.target_lag_alpha_by_index = self._control_action_tensor("target_lag_alpha_by_index")
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.last_dof_targets = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.lagged_dof_targets = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.delay_steps_by_index = None
        if self.cfg["randomization"].get("dof_delay_steps") is not None:
            self.delay_steps_by_index = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.long, device=self.device)
        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.dof_torque_strength = apply_randomization(
            torch.ones(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device),
            self.cfg["randomization"].get("dof_torque_strength"),
        )
        self.dof_torque_bias = apply_randomization(
            torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device),
            self.cfg["randomization"].get("dof_torque_bias"),
        )
        self.commands = torch.zeros(self.num_envs, self.cfg["commands"]["num_commands"], dtype=torch.float, device=self.device)
        self.cmd_resample_time = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.gait_frequency = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.gait_process = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.desired_yaw = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.desired_pos_xy = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.filtered_lin_vel = self.base_lin_vel.clone()
        self.filtered_ang_vel = self.base_ang_vel.clone()
        self.curriculum_prob = torch.zeros(
            1 + 2 * self.cfg["commands"]["lin_vel_levels"],
            1 + 2 * self.cfg["commands"]["ang_vel_levels"],
            dtype=torch.float,
            device=self.device,
        )
        self.curriculum_prob[self.cfg["commands"]["lin_vel_levels"], self.cfg["commands"]["ang_vel_levels"]] = 1.0
        self.env_curriculum_level = torch.zeros(self.num_envs, 2, dtype=torch.long, device=self.device)
        self.mean_lin_vel_level = 0.0
        self.mean_ang_vel_level = 0.0
        self.max_lin_vel_level = 0.0
        self.max_ang_vel_level = 0.0
        command_keys = [
            "lin_vel_x",
            "lin_vel_y",
            "ang_vel_yaw",
            "gait_frequency",
            "foot_yaw_L",
            "foot_yaw_R",
            "body_pitch_target",
            "body_roll_target",
            "feet_offset_x_target",
            "feet_offset_y_target",
        ]
        self.current_command_ranges = {key: list(self.cfg["commands"][key]) for key in command_keys}
        self.current_straight_command = dict(self.cfg["commands"].get("straight", {}))
        self.current_resampling_time = list(self.cfg["commands"].get("resampling_time_s", [3.0, 8.0]))
        self.current_disturbance_scale = 1.0
        self.training_phase_index = 0
        self.training_phase_progress = 0.0
        self.pushing_forces = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device)
        self.pushing_torques = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device)
        self.feet_roll = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.feet_yaw = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.feet_yaw_rel = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.feet_pitch = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.last_feet_pos = torch.zeros_like(self.feet_pos)
        self.feet_contact = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device)
        self.feet_contact_duty_ema = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        feet_edge_count = len(self.cfg["asset"].get("feet_edge_pos", []))
        self.feet_edge_height = torch.zeros(self.num_envs, len(self.feet_indices), feet_edge_count, dtype=torch.float, device=self.device)
        self.feet_edge_contact = torch.zeros(self.num_envs, len(self.feet_indices), feet_edge_count, dtype=torch.bool, device=self.device)
        self.dof_pos_ref = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.default_dof_pos = self._dof_pos_from_joint_angles(self.cfg["init_state"]["default_joint_angles"])
        pose_blend_cfg = self.cfg.get("randomization", {}).get("init_dof_pos_pose_blend")
        self.init_dof_pos_pose_blend_target = None
        if pose_blend_cfg:
            self.init_dof_pos_pose_blend_target = self._dof_pos_from_joint_angles(pose_blend_cfg["joint_angles"])
        self.lagged_dof_targets[:] = self.default_dof_pos

    def _dof_pos_from_joint_angles(self, joint_angles):
        dof_pos = torch.zeros(1, self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            name = self._longest_matching_key(self.dof_names[i], joint_angles)
            if name is None:
                dof_pos[:, i] = joint_angles["default"]
            else:
                dof_pos[:, i] = joint_angles[name]
        return dof_pos

    def _control_action_tensor(self, name):
        values = self.cfg["control"].get(name)
        if values is None:
            return None
        if len(values) != self.num_actions:
            raise ValueError(f"control.{name} must contain {self.num_actions} values, got {len(values)}")
        return torch.tensor(values, dtype=torch.float, device=self.device).unsqueeze(0)

    def _prepare_reward_function(self):
        """Prepares a list of reward functions, whcih will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        self.reward_scales = self.cfg["rewards"]["scales"].copy()
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            self.reward_names.append(name)
            name = "_reward_" + name
            self.reward_functions.append(getattr(self, name))

    def _init_csv_logging(self):
        """Initialize CSV files for logging actions and observations of the first environment"""
        # Create logs directory if it doesn't exist
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Generate timestamp for unique filenames
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Initialize action CSV file
        self.actions_csv_path = os.path.join(log_dir, f"k1_multi3_actions_env0_{timestamp}.csv")
        self.actions_csv_file = open(self.actions_csv_path, 'w', newline='')
        self.actions_csv_writer = csv.writer(self.actions_csv_file)
        
        # Write action headers
        action_headers = [f"action_{i}" for i in range(self.num_actions)]
        self.actions_csv_writer.writerow(action_headers)
        
        # Initialize observation CSV file
        self.obs_csv_path = os.path.join(log_dir, f"k1_multi3_observations_env0_{timestamp}.csv")
        self.obs_csv_file = open(self.obs_csv_path, 'w', newline='')
        self.obs_csv_writer = csv.writer(self.obs_csv_file)
        
        # Write observation headers
        obs_headers = [f"obs_{i}" for i in range(self.num_obs)]
        self.obs_csv_writer.writerow(obs_headers)
        
        print(f"CSV logging initialized:")
        print(f"  Actions: {self.actions_csv_path}")
        print(f"  Observations: {self.obs_csv_path}")

    def close_csv_files(self):
        """Close CSV files and print summary"""
        if hasattr(self, 'actions_csv_file') and not self.actions_csv_file.closed:
            self.actions_csv_file.close()
            print(f"Actions CSV file closed: {self.actions_csv_path}")
        
        if hasattr(self, 'obs_csv_file') and not self.obs_csv_file.closed:
            self.obs_csv_file.close()
            print(f"Observations CSV file closed: {self.obs_csv_path}")

    def __del__(self):
        """Destructor to ensure CSV files are closed"""
        self.close_csv_files()

    @staticmethod
    def _wrap_to_pi(angle):
        return (angle + torch.pi) % (2 * torch.pi) - torch.pi

    def _get_base_yaw(self):
        _, _, yaw = get_euler_xyz(self.base_quat)
        return self._wrap_to_pi(yaw)

    def update_training_curriculum(self, iteration):
        command_cfg = self.cfg["commands"]
        command_keys = [
            "lin_vel_x",
            "lin_vel_y",
            "ang_vel_yaw",
            "gait_frequency",
            "foot_yaw_L",
            "foot_yaw_R",
            "body_pitch_target",
            "body_roll_target",
            "feet_offset_x_target",
            "feet_offset_y_target",
        ]
        self.current_command_ranges = {key: list(command_cfg[key]) for key in command_keys}
        self.current_straight_command = dict(command_cfg.get("straight", {}))
        self.current_resampling_time = list(command_cfg.get("resampling_time_s", [3.0, 8.0]))
        self.current_disturbance_scale = 1.0
        self.training_phase_index = 0
        self.training_phase_progress = 1.0

        if command_cfg.get("training_mode", "sampled") == "fixed":
            return

        phases = command_cfg.get("training_phases", [])
        if not phases:
            return

        active_idx = 0
        for i, phase in enumerate(phases):
            if iteration >= int(phase["start_iteration"]):
                active_idx = i
        phase = phases[active_idx]

        for key in command_keys:
            if key in phase:
                self.current_command_ranges[key] = list(phase[key])
        if "straight" in phase:
            self.current_straight_command.update(phase["straight"])
        self.current_resampling_time = list(phase.get("resampling_time_s", self.current_resampling_time))
        self.current_disturbance_scale = float(phase.get("disturbance_scale", self.current_disturbance_scale))
        self.training_phase_index = active_idx
        if active_idx < len(phases) - 1:
            next_iter = int(phases[active_idx + 1]["start_iteration"])
            span = max(next_iter - int(phase["start_iteration"]), 1)
            self.training_phase_progress = float(np.clip((iteration - int(phase["start_iteration"])) / span, 0.0, 1.0))

    def _command_range(self, key):
        return self.current_command_ranges.get(key, self.cfg["commands"][key])

    def _sample_resample_steps(self, env_count):
        low = max(1, int(float(self.current_resampling_time[0]) / self.dt))
        high = max(low + 1, int(float(self.current_resampling_time[1]) / self.dt))
        return torch.randint(low, high, (env_count,), device=self.device)

    def _reset_command_targets(self, env_ids):
        if len(env_ids) == 0:
            return
        base_yaw = self._get_base_yaw()
        self.desired_yaw[env_ids] = base_yaw[env_ids]
        self.desired_pos_xy[env_ids] = self.root_states[env_ids, 0:2]

    def _update_command_targets(self):
        self.desired_yaw[:] = self._wrap_to_pi(self.desired_yaw + self.commands[:, 2] * self.dt)
        cos_yaw = torch.cos(self.desired_yaw)
        sin_yaw = torch.sin(self.desired_yaw)
        world_vel_x = cos_yaw * self.commands[:, 0] - sin_yaw * self.commands[:, 1]
        world_vel_y = sin_yaw * self.commands[:, 0] + cos_yaw * self.commands[:, 1]
        self.desired_pos_xy[:, 0] += world_vel_x * self.dt
        self.desired_pos_xy[:, 1] += world_vel_y * self.dt

    def reset(self):
        """Reset all robots"""
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        self._resample_commands()
        self._compute_observations()
        return self.obs_buf, self.extras

    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        self._update_curriculum(env_ids)
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._reset_command_targets(env_ids)

        self.last_dof_targets[env_ids] = self.dof_pos[env_ids]
        self.lagged_dof_targets[env_ids] = self.dof_pos[env_ids]
        self.last_root_vel[env_ids] = self.root_states[env_ids, 7:13]
        self.episode_length_buf[env_ids] = 0
        self.filtered_lin_vel[env_ids] = 0.0
        self.filtered_ang_vel[env_ids] = 0.0
        self.feet_contact_duty_ema[env_ids] = 0.0
        self.cmd_resample_time[env_ids] = 0

        self.delay_steps[env_ids] = torch.randint(0, self.cfg["control"]["decimation"], (len(env_ids),), device=self.device)
        if self.delay_steps_by_index is not None:
            self.delay_steps_by_index[env_ids] = self._sample_dof_delay_steps(len(env_ids))
        self.extras["time_outs"] = self.time_out_buf

    def _sample_dof_delay_steps(self, env_count):
        delay_cfg = self.cfg["randomization"].get("dof_delay_steps")
        if delay_cfg is None:
            return None
        max_delay = max(int(self.cfg["control"]["decimation"]) - 1, 0)
        low, high = delay_cfg.get("range", [0, max_delay])
        low = max(0, min(int(low), max_delay))
        high = max(low, min(int(high), max_delay))
        return torch.randint(low, high + 1, (env_count, self.num_dofs), device=self.device)

    def _reset_dofs(self, env_ids):
        dof_pos = self.default_dof_pos.expand(len(env_ids), -1).clone()
        pose_blend_cfg = self.cfg["randomization"].get("init_dof_pos_pose_blend")
        if pose_blend_cfg and self.init_dof_pos_pose_blend_target is not None:
            blend_range = pose_blend_cfg.get("range", [0.0, 1.0])
            blend = torch_rand_float(
                float(blend_range[0]),
                float(blend_range[1]),
                (len(env_ids), 1),
                device=self.device,
            )
            probability = float(pose_blend_cfg.get("probability", 1.0))
            if probability < 1.0:
                enabled = (torch.rand(len(env_ids), 1, device=self.device) < probability).float()
                blend = blend * enabled
            target = self.init_dof_pos_pose_blend_target.expand_as(dof_pos)
            dof_pos = dof_pos + blend * (target - dof_pos)
        self.dof_pos[env_ids] = apply_randomization(dof_pos, self.cfg["randomization"].get("init_dof_pos"))
        self.dof_vel[env_ids] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)
        )

    def _reset_root_states(self, env_ids):
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :2] += self.env_origins[env_ids, :2]
        self.root_states[env_ids, :2] = apply_randomization(self.root_states[env_ids, :2], self.cfg["randomization"].get("init_base_pos_xy"))
        self.root_states[env_ids, 2] += self.terrain.terrain_heights(self.root_states[env_ids, :2])
        yaw = torch.zeros(len(env_ids), dtype=torch.float, device=self.device)
        if not self._play_fixed_yaw():
            yaw = torch.rand(len(env_ids), device=self.device) * (2 * torch.pi)
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(
            torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
            torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
            yaw,
        )
        self.root_states[env_ids, 7:9] = apply_randomization(
            torch.zeros(len(env_ids), 2, dtype=torch.float, device=self.device),
            self.cfg["randomization"].get("init_base_lin_vel_xy"),
        )
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _play_cfg(self):
        return self.cfg["commands"].get("play", {})

    def _play_fixed_yaw(self):
        return getattr(self, "is_play", False) and bool(self._play_cfg().get("fixed_yaw", False))

    def _play_no_disturbance(self):
        return getattr(self, "is_play", False) and bool(self._play_cfg().get("no_disturbance", False))

    def _apply_fixed_command(self, command_cfg, env_ids=None):
        defaults = {
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
        }
        values = [float(command_cfg.get(key, defaults[key])) for key in defaults]
        command_tensor = torch.tensor(values, dtype=torch.float, device=self.device)
        if env_ids is None:
            self.commands[:, :10] = command_tensor.unsqueeze(0)
            self.gait_frequency[:] = command_tensor[3]
        else:
            self.commands[env_ids, :10] = command_tensor.unsqueeze(0)
            self.gait_frequency[env_ids] = command_tensor[3]

    def _sample_command_value(self, command_cfg, key, count, default=0.0):
        value = command_cfg.get(key, default)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return torch_rand_float(float(value[0]), float(value[1]), (count, 1), device=self.device).squeeze(1)
        return torch.full((count,), float(value), dtype=torch.float, device=self.device)

    def _apply_straight_anchor_command(self, env_ids):
        if len(env_ids) == 0:
            return
        command_cfg = self.current_straight_command
        command_keys = [
            "lin_vel_x",
            "lin_vel_y",
            "ang_vel_yaw",
            "gait_frequency",
            "foot_yaw_L",
            "foot_yaw_R",
            "body_pitch_target",
            "body_roll_target",
            "feet_offset_x_target",
            "feet_offset_y_target",
        ]
        defaults = self.cfg["commands"].get("fixed", {})
        for command_idx, key in enumerate(command_keys):
            self.commands[env_ids, command_idx] = self._sample_command_value(
                command_cfg,
                key,
                len(env_ids),
                defaults.get(key, 0.0),
            )
        frequency_profile = command_cfg.get("gait_frequency_by_lin_vel_x")
        if frequency_profile:
            min_speed = float(frequency_profile.get("min_speed", 0.0))
            max_speed = float(frequency_profile.get("max_speed", 1.0))
            min_frequency = float(frequency_profile.get("min_frequency", self.commands[env_ids, 3].min().item()))
            max_frequency = float(frequency_profile.get("max_frequency", self.commands[env_ids, 3].max().item()))
            drive = torch.clamp(
                (torch.abs(self.commands[env_ids, 0]) - min_speed) / max(max_speed - min_speed, 1.0e-6),
                min=0.0,
                max=1.0,
            )
            self.commands[env_ids, 3] = min_frequency + drive * (max_frequency - min_frequency)
        self.gait_frequency[env_ids] = self.commands[env_ids, 3]

    def _teleport_robot(self):
        if self.terrain.type == "plane":
            return
        out_x_min = self.root_states[:, 0] < -0.75 * self.terrain.border_size
        out_x_max = self.root_states[:, 0] > self.terrain.env_width + 0.75 * self.terrain.border_size
        out_y_min = self.root_states[:, 1] < -0.75 * self.terrain.border_size
        out_y_max = self.root_states[:, 1] > self.terrain.env_length + 0.75 * self.terrain.border_size
        self.root_states[out_x_min, 0] += self.terrain.env_width + self.terrain.border_size
        self.root_states[out_x_max, 0] -= self.terrain.env_width + self.terrain.border_size
        self.root_states[out_y_min, 1] += self.terrain.env_length + self.terrain.border_size
        self.root_states[out_y_max, 1] -= self.terrain.env_length + self.terrain.border_size
        self.body_states[out_x_min, :, 0] += self.terrain.env_width + self.terrain.border_size
        self.body_states[out_x_max, :, 0] -= self.terrain.env_width + self.terrain.border_size
        self.body_states[out_y_min, :, 1] += self.terrain.env_length + self.terrain.border_size
        self.body_states[out_y_max, :, 1] -= self.terrain.env_length + self.terrain.border_size
        if out_x_min.any() or out_x_max.any() or out_y_min.any() or out_y_max.any():
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            self._refresh_feet_state()

    def _resample_commands(self):
        if getattr(self, "is_play", False):
            self._apply_fixed_command(self.cfg["commands"].get("play", {}))
            return
        if getattr(self, "manual_control", False):
            return
        env_ids = (self.episode_length_buf == self.cmd_resample_time).nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return

        if self.cfg["commands"].get("training_mode", "sampled") == "fixed":
            self._apply_fixed_command(self.cfg["commands"].get("fixed", {}), env_ids)
            self.cmd_resample_time[env_ids] += self._sample_resample_steps(len(env_ids))
            return

        if self.cfg["commands"]["curriculum"]:
            self._resample_curriculum_commands(env_ids)
        else:
            command_keys = [
                "lin_vel_x",
                "lin_vel_y",
                "ang_vel_yaw",
                "gait_frequency",
                "foot_yaw_L",
                "foot_yaw_R",
                "body_pitch_target",
                "body_roll_target",
                "feet_offset_x_target",
                "feet_offset_y_target",
            ]
            for command_idx, key in enumerate(command_keys):
                low, high = self._command_range(key)
                self.commands[env_ids, command_idx] = torch_rand_float(
                    low,
                    high,
                    (len(env_ids), 1),
                    device=self.device,
                ).squeeze(1)
            
        self.gait_frequency[env_ids] = self.commands[env_ids, 3]
        straight_count = int(self.cfg["commands"].get("straight_proportion", 0.0) * len(env_ids))
        perm = torch.randperm(len(env_ids), device=self.device)
        straight_envs = env_ids[perm[:straight_count]]
        self._apply_straight_anchor_command(straight_envs)

        remaining_envs = env_ids[perm[straight_count:]]
        still_count = int(self.cfg["commands"]["still_proportion"] * len(env_ids))
        still_envs = remaining_envs[:still_count]
        self.commands[still_envs, :10] = 0.0
        self.gait_frequency[still_envs] = 0.0
        self.cmd_resample_time[env_ids] += self._sample_resample_steps(len(env_ids))

    def _update_curriculum(self, env_ids):
        if not self.cfg["commands"]["curriculum"]:
            return
        success = self.episode_length_buf[env_ids] > np.ceil(self.cfg["rewards"]["episode_length_s"] / self.dt) * (
            1 - self.cfg["commands"]["episode_length_toler"]
        )
        success &= torch.abs(self.filtered_lin_vel[env_ids, 0] - self.commands[env_ids, 0]) < self.cfg["commands"]["lin_vel_x_toler"]
        success &= torch.abs(self.filtered_lin_vel[env_ids, 1] - self.commands[env_ids, 1]) < self.cfg["commands"]["lin_vel_y_toler"]
        success &= torch.abs(self.filtered_ang_vel[env_ids, 2] - self.commands[env_ids, 2]) < self.cfg["commands"]["ang_vel_yaw_toler"]
        for i in range(len(env_ids)):
            if success[i]:
                x = self.env_curriculum_level[env_ids[i], 0] + self.cfg["commands"]["lin_vel_levels"]
                y = self.env_curriculum_level[env_ids[i], 1] + self.cfg["commands"]["ang_vel_levels"]
                self.curriculum_prob[x, y] += self.cfg["commands"]["update_rate"]
                if x > 0:
                    self.curriculum_prob[x - 1, y] += self.cfg["commands"]["update_rate"]
                if x < self.curriculum_prob.shape[0] - 1:
                    self.curriculum_prob[x + 1, y] += self.cfg["commands"]["update_rate"]
                if y > 0:
                    self.curriculum_prob[x, y - 1] += self.cfg["commands"]["update_rate"]
                if y < self.curriculum_prob.shape[1] - 1:
                    self.curriculum_prob[x, y + 1] += self.cfg["commands"]["update_rate"]
        self.curriculum_prob.clamp_(max=1.0)

    def _resample_curriculum_commands(self, env_ids):
        grid_idx = torch.multinomial(self.curriculum_prob.flatten(), len(env_ids), replacement=True)
        lin_vel_level = grid_idx % self.curriculum_prob.shape[1] - self.cfg["commands"]["lin_vel_levels"]
        ang_vel_level = grid_idx // self.curriculum_prob.shape[1] - self.cfg["commands"]["ang_vel_levels"]
        self.env_curriculum_level[env_ids, 0] = lin_vel_level
        self.env_curriculum_level[env_ids, 1] = ang_vel_level
        self.mean_lin_vel_level = torch.mean(torch.abs(self.env_curriculum_level[:, 0]).float())
        self.mean_ang_vel_level = torch.mean(torch.abs(self.env_curriculum_level[:, 1]).float())
        self.max_lin_vel_level = torch.max(torch.abs(self.env_curriculum_level[:, 0]))
        self.max_ang_vel_level = torch.max(torch.abs(self.env_curriculum_level[:, 1]))
        self.commands[env_ids, 0] = (
            lin_vel_level + torch_rand_float(-0.5, 0.5, (len(env_ids), 1), device=self.device).squeeze(1)
        ) * self.cfg["commands"]["lin_vel_x_resolution"]
        self.commands[env_ids, 1] = (
            torch.abs(lin_vel_level)
            * torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=self.device).squeeze(1)
            * self.cfg["commands"]["lin_vel_y_resolution"]
        )
        self.commands[env_ids, 2] = (
            ang_vel_level + torch_rand_float(-0.5, 0.5, (len(env_ids), 1), device=self.device).squeeze(1)
        ) * self.cfg["commands"]["ang_vel_resolution"]
        
        # Additional parameters for curriculum mode
        self.commands[env_ids, 3] = torch_rand_float(
            self.cfg["commands"]["gait_frequency"][0], self.cfg["commands"]["gait_frequency"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        self.commands[env_ids, 4] = torch_rand_float(
            self.cfg["commands"]["foot_yaw_L"][0], self.cfg["commands"]["foot_yaw_L"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        self.commands[env_ids, 5] = torch_rand_float(
            self.cfg["commands"]["foot_yaw_R"][0], self.cfg["commands"]["foot_yaw_R"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        self.commands[env_ids, 6] = torch_rand_float(
            self.cfg["commands"]["body_pitch_target"][0], self.cfg["commands"]["body_pitch_target"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        self.commands[env_ids, 7] = torch_rand_float(
            self.cfg["commands"]["body_roll_target"][0], self.cfg["commands"]["body_roll_target"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        self.commands[env_ids, 8] = torch_rand_float(
            self.cfg["commands"]["feet_offset_x_target"][0], self.cfg["commands"]["feet_offset_x_target"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        self.commands[env_ids, 9] = torch_rand_float(
            self.cfg["commands"]["feet_offset_y_target"][0], self.cfg["commands"]["feet_offset_y_target"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)

    def step(self, actions):
        # pre physics step
        processed_actions = torch.clip(
            actions,
            -self.cfg["normalization"]["clip_actions"],
            self.cfg["normalization"]["clip_actions"],
        )
        if self.action_clip_by_index is not None:
            processed_actions = torch.minimum(
                torch.maximum(processed_actions, -self.action_clip_by_index),
                self.action_clip_by_index,
            )
        if self.action_scale_by_index is not None:
            processed_actions = processed_actions * self.action_scale_by_index
        if self.action_lower_by_index is not None:
            processed_actions = torch.maximum(processed_actions, self.action_lower_by_index)
        if self.action_upper_by_index is not None:
            processed_actions = torch.minimum(processed_actions, self.action_upper_by_index)

        action_rate_limit = self.cfg["control"].get("action_rate_limit")
        if self.action_rate_limit_by_index is not None:
            max_delta = self.action_rate_limit_by_index * self.dt
            action_delta = processed_actions - self.actions
            processed_actions = self.actions + torch.minimum(torch.maximum(action_delta, -max_delta), max_delta)
        elif action_rate_limit is not None and float(action_rate_limit) > 0.0:
            max_delta = float(action_rate_limit) * self.dt
            processed_actions = self.actions + torch.clamp(processed_actions - self.actions, -max_delta, max_delta)

        self.actions[:] = processed_actions
        dof_targets = self.default_dof_pos + self.cfg["control"]["action_scale"] * self.actions
        if self.target_lag_alpha_by_index is not None:
            self.lagged_dof_targets[:] = self.lagged_dof_targets + self.target_lag_alpha_by_index * (
                dof_targets - self.lagged_dof_targets
            )
            dof_targets = self.lagged_dof_targets
        
        # Log actions for first environment only
        if hasattr(self, 'actions_csv_writer'):
            # Convert to CPU and numpy for CSV writing
            actions_env0 = self.actions[0].cpu().numpy()
            self.actions_csv_writer.writerow(actions_env0)
            self.actions_csv_file.flush()  # Ensure data is written immediately
        
        if self.cfg.get("basic", {}).get("debug_tensors", False):
            print(actions)

        # perform physics step
        self.torques.zero_()
        for i in range(self.cfg["control"]["decimation"]):
            if self.delay_steps_by_index is None:
                self.last_dof_targets[self.delay_steps == i] = dof_targets[self.delay_steps == i]
            else:
                update_mask = self.delay_steps_by_index == i
                self.last_dof_targets[:] = torch.where(update_mask, dof_targets, self.last_dof_targets)
            dof_torques = self.dof_stiffness * (self.last_dof_targets - self.dof_pos) - self.dof_damping * self.dof_vel
            friction = torch.min(self.dof_friction, dof_torques.abs()) * torch.sign(dof_torques)
            dof_torques = (dof_torques - friction) * self.dof_torque_strength + self.dof_torque_bias
            dof_torques = torch.clip(dof_torques, min=-self.torque_limits, max=self.torque_limits)
            self.torques += dof_torques
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(dof_torques))
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)
        self.torques /= self.cfg["control"]["decimation"]
        self.render()

        # post physics step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.filtered_lin_vel[:] = self.base_lin_vel[:] * self.cfg["normalization"]["filter_weight"] + self.filtered_lin_vel[:] * (
            1.0 - self.cfg["normalization"]["filter_weight"]
        )
        
        self.filtered_ang_vel[:] = self.base_ang_vel[:] * self.cfg["normalization"]["filter_weight"] + self.filtered_ang_vel[:] * (
            1.0 - self.cfg["normalization"]["filter_weight"]
        )

        self._refresh_feet_state()

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.gait_process[:] = torch.fmod(self.gait_process + self.dt * self.gait_frequency, 1.0)
        self._update_command_targets()

        self._kick_robots()
        self._push_robots()
        self._check_termination()
        self._compute_reward()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self._reset_idx(env_ids)
        self._teleport_robot()
        self._resample_commands()

        self._compute_observations()

        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.dof_vel
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_feet_pos[:] = self.feet_pos

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _kick_robots(self):
        """Random kick the robots. Emulates an impulse by setting a randomized base velocity."""
        if self._play_no_disturbance():
            return
        if self.current_disturbance_scale <= 0.0:
            return
        if self.common_step_counter % np.ceil(self.cfg["randomization"]["kick_interval_s"] / self.dt) == 0:
            lin_kick = apply_randomization(self.root_states[:, 7:10], self.cfg["randomization"].get("kick_lin_vel"))
            ang_kick = apply_randomization(self.root_states[:, 10:13], self.cfg["randomization"].get("kick_ang_vel"))
            self.root_states[:, 7:10] = self.root_states[:, 7:10] + (lin_kick - self.root_states[:, 7:10]) * self.current_disturbance_scale
            self.root_states[:, 10:13] = self.root_states[:, 10:13] + (ang_kick - self.root_states[:, 10:13]) * self.current_disturbance_scale
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _push_robots(self):
        """Random push the robots. Emulates an impulse by setting a randomized force."""
        if self._play_no_disturbance():
            return
        if self.current_disturbance_scale <= 0.0:
            self.pushing_forces[:, self.base_indice, :].zero_()
            self.pushing_torques[:, self.base_indice, :].zero_()
            return
        if self.common_step_counter % np.ceil(self.cfg["randomization"]["push_interval_s"] / self.dt) == 0:
            self.pushing_forces[:, self.base_indice, :] = apply_randomization(
                torch.zeros_like(self.pushing_forces[:, 0, :]),
                self.cfg["randomization"].get("push_force"),
            ) * self.current_disturbance_scale
            self.pushing_torques[:, self.base_indice, :] = apply_randomization(
                torch.zeros_like(self.pushing_torques[:, 0, :]),
                self.cfg["randomization"].get("push_torque"),
            ) * self.current_disturbance_scale
        elif self.common_step_counter % np.ceil(self.cfg["randomization"]["push_interval_s"] / self.dt) == np.ceil(
            self.cfg["randomization"]["push_duration_s"] / self.dt
        ):
            self.pushing_forces[:, self.base_indice, :].zero_()
            self.pushing_torques[:, self.base_indice, :].zero_()
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.pushing_forces),
            gymtorch.unwrap_tensor(self.pushing_torques),
            gymapi.LOCAL_SPACE,
        )

    def _refresh_feet_state(self):
        self.feet_pos[:] = self.body_states[:, self.feet_indices, 0:3]
        self.feet_quat[:] = self.body_states[:, self.feet_indices, 3:7]
        roll, _, yaw = get_euler_xyz(self.feet_quat.reshape(-1, 4))
        self.feet_roll[:] = (roll.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        self.feet_yaw[:] = (yaw.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        _, pitch, _ = get_euler_xyz(self.feet_quat.reshape(-1, 4))
        self.feet_pitch[:] = (pitch.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        
        # Compute relative yaw to trunk
        _, _, base_yaw = get_euler_xyz(self.base_quat)
        self.feet_yaw_rel = (self.feet_yaw - base_yaw.unsqueeze(-1) + torch.pi) % (2 * torch.pi) - torch.pi
        
        feet_edge_relative_pos = (
            to_torch(self.cfg["asset"]["feet_edge_pos"], device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(self.num_envs, len(self.feet_indices), -1, -1)
        )
        expanded_feet_pos = self.feet_pos.unsqueeze(2).expand(-1, -1, feet_edge_relative_pos.shape[2], -1).reshape(-1, 3)
        expanded_feet_quat = self.feet_quat.unsqueeze(2).expand(-1, -1, feet_edge_relative_pos.shape[2], -1).reshape(-1, 4)
        feet_edge_pos = expanded_feet_pos + quat_rotate(expanded_feet_quat, feet_edge_relative_pos.reshape(-1, 3))
        edge_height = (feet_edge_pos[:, 2] - self.terrain.terrain_heights(feet_edge_pos)).reshape(
            self.num_envs, len(self.feet_indices), feet_edge_relative_pos.shape[2]
        )
        edge_contact_threshold = float(self.cfg["rewards"].get("feet_edge_contact_threshold", 0.01))
        self.feet_edge_height[:] = edge_height
        self.feet_edge_contact[:] = edge_height < edge_contact_threshold
        self.feet_contact[:] = torch.any(self.feet_edge_contact, dim=2)
        duty_tau_s = max(float(self.cfg["rewards"].get("feet_contact_duty_tau_s", 0.45)), self.dt)
        duty_alpha = min(self.dt / duty_tau_s, 1.0)
        self.feet_contact_duty_ema[:] = self.feet_contact_duty_ema * (1.0 - duty_alpha) + self.feet_contact.float() * duty_alpha

    def _check_termination(self):
        """Check if environments need to be reset"""
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1)
        self.reset_buf |= self.root_states[:, 7:13].square().sum(dim=-1) > self.cfg["rewards"]["terminate_vel"]
        self.reset_buf |= self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos) < self.cfg["rewards"]["terminate_height"]
        self.time_out_buf = self.episode_length_buf > np.ceil(self.cfg["rewards"]["episode_length_s"] / self.dt)
        self.reset_buf |= self.time_out_buf
        self.time_out_buf |= self.episode_length_buf == self.cmd_resample_time

    def _compute_reward(self):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.extras["rew_terms"][name] = rew
        if self.cfg["rewards"]["only_positive_rewards"]:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)

    def _compute_observations(self):
        """Computes observations"""
        commands_scale = torch.tensor(
            [
                self.cfg["normalization"]["lin_vel"],           # 0: lin_vel_x
                self.cfg["normalization"]["lin_vel"],           # 1: lin_vel_y
                self.cfg["normalization"]["ang_vel"],           # 2: ang_vel_yaw
                self.cfg["normalization"]["gait_frequency"],    # 3: gait_frequency
                self.cfg["normalization"]["foot_yaw"],          # 4: foot_yaw_L
                self.cfg["normalization"]["foot_yaw"],          # 5: foot_yaw_R
                self.cfg["normalization"]["body_pitch_target"], # 6: body_pitch_target
                self.cfg["normalization"]["body_roll_target"],  # 7: body_roll_target
                self.cfg["normalization"]["feet_offset_x_target"], # 8: feet_offset_x_target
                self.cfg["normalization"]["feet_offset_y_target"], # 9: feet_offset_y_target
            ],
            device=self.device,
        )
        if getattr(self, "is_play", False):
            self._apply_fixed_command(self.cfg["commands"].get("play", {}))
        elif self.cfg["commands"].get("training_mode", "sampled") == "fixed":
            self._apply_fixed_command(self.cfg["commands"].get("fixed", {}))
        self.obs_buf = torch.cat(
            (
                apply_randomization(self.projected_gravity, self.cfg["noise"].get("gravity")) * self.cfg["normalization"]["gravity"],
                apply_randomization(self.base_ang_vel, self.cfg["noise"].get("ang_vel")) * self.cfg["normalization"]["ang_vel"],
                self.commands[:, :10] * commands_scale,
                (torch.cos(2 * torch.pi * self.gait_process)).unsqueeze(-1),
                (torch.sin(2 * torch.pi * self.gait_process)).unsqueeze(-1),
                apply_randomization(self.dof_pos - self.default_dof_pos, self.cfg["noise"].get("dof_pos")) * self.cfg["normalization"]["dof_pos"],
                apply_randomization(self.dof_vel, self.cfg["noise"].get("dof_vel")) * self.cfg["normalization"]["dof_vel"],
                self.actions,
            ),
            dim=-1,
        )
        
        # Log observations for first environment only
        if hasattr(self, 'obs_csv_writer'):
            # Convert to CPU and numpy for CSV writing
            obs_env0 = self.obs_buf[0].cpu().numpy()
            self.obs_csv_writer.writerow(obs_env0)
            self.obs_csv_file.flush()  # Ensure data is written immediately
        
        if self.cfg.get("basic", {}).get("debug_tensors", False):
            print(self.obs_buf)
        self.privileged_obs_buf = torch.cat(
            (
                self.base_mass_scaled,
                apply_randomization(self.base_lin_vel, self.cfg["noise"].get("lin_vel")) * self.cfg["normalization"]["lin_vel"],
                apply_randomization(self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos), self.cfg["noise"].get("height")).unsqueeze(-1),
                self.pushing_forces[:, 0, :] * self.cfg["normalization"]["push_force"],
                self.pushing_torques[:, 0, :] * self.cfg["normalization"]["push_torque"],
            ),
            dim=-1,
        )
        self.extras["privileged_obs"] = self.privileged_obs_buf

    # ------------ reward functions----------------
    def _reward_survival(self):
        # Reward survival
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)

    def _reward_tracking_lin_vel_x(self):
        # Tracking of linear velocity commands (x axes)
        return torch.exp(-torch.square(self.commands[:, 0] - self.filtered_lin_vel[:, 0]) / self.cfg["rewards"]["tracking_sigma"])

    def _reward_tracking_lin_vel_y(self):
        # Tracking of linear velocity commands (y axes)
        return torch.exp(-torch.square(self.commands[:, 1] - self.filtered_lin_vel[:, 1]) / self.cfg["rewards"]["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        return torch.exp(-torch.square(self.commands[:, 2] - self.filtered_ang_vel[:, 2]) / self.cfg["rewards"]["tracking_sigma"])

    def _reward_lin_vel_x_error(self):
        moving = (torch.abs(self.commands[:, 0]) > 0.05).float()
        return torch.square(self.commands[:, 0] - self.filtered_lin_vel[:, 0]) * moving

    def _reward_heading_tracking(self):
        yaw_error = self._wrap_to_pi(self._get_base_yaw() - self.desired_yaw)
        return torch.square(yaw_error)

    def _reward_path_lateral(self):
        path_error = self.base_pos[:, 0:2] - self.desired_pos_xy
        sin_yaw = torch.sin(self.desired_yaw)
        cos_yaw = torch.cos(self.desired_yaw)
        lateral_error = -sin_yaw * path_error[:, 0] + cos_yaw * path_error[:, 1]
        error_clip = float(self.cfg["rewards"].get("path_lateral_error_clip", 0.5))
        return torch.clamp(torch.square(lateral_error), max=error_clip * error_clip)

    def _reward_straight_path_lateral(self):
        path_error = self.base_pos[:, 0:2] - self.desired_pos_xy
        sin_yaw = torch.sin(self.desired_yaw)
        cos_yaw = torch.cos(self.desired_yaw)
        lateral_error = -sin_yaw * path_error[:, 0] + cos_yaw * path_error[:, 1]
        error_clip = float(self.cfg["rewards"].get("straight_path_lateral_error_clip", 0.8))
        return torch.clamp(torch.square(lateral_error), max=error_clip * error_clip) * self._straight_walk_mask()

    def _straight_walk_mask(self):
        return (
            (torch.abs(self.commands[:, 0]) > 0.05)
            & (torch.abs(self.commands[:, 1]) < 0.05)
            & (torch.abs(self.commands[:, 2]) < 0.05)
        ).float()

    def _reward_straight_heading(self):
        yaw_error = self._wrap_to_pi(self._get_base_yaw() - self.desired_yaw)
        return torch.square(yaw_error) * self._straight_walk_mask()

    def _reward_straight_lateral_vel(self):
        return torch.square(self.filtered_lin_vel[:, 1]) * self._straight_walk_mask()

    def _reward_straight_yaw_vel(self):
        return torch.square(self.filtered_ang_vel[:, 2]) * self._straight_walk_mask()

    def _reward_base_height(self):
        # Tracking of base height
        base_height = self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos)
        return torch.square(base_height - self.cfg["rewards"]["base_height_target"])

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(torch.norm(self.contact_forces[:, self.penalized_contact_indices, :], dim=-1) > 1.0, dim=-1)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.filtered_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=-1)

    def _reward_orientation(self):
        """
        Penalize body tilt relative to commanded pitch and roll targets.

        projected_gravity is measured in the trunk frame, so this stays stable
        even when the robot has arbitrary yaw.
        """
        target_pitch = self.commands[:, 6]  # body_pitch_target
        target_roll = self.commands[:, 7]   # body_roll_target
        target_gravity_xy = torch.stack(
            (
                torch.sin(target_pitch),
                -torch.sin(target_roll) * torch.cos(target_pitch),
            ),
            dim=-1,
        )
        return torch.sum(torch.square(self.projected_gravity[:, :2] - target_gravity_xy), dim=-1)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=-1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=-1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=-1)

    def _reward_root_acc(self):
        # Penalize root accelerations
        return torch.sum(torch.square((self.last_root_vel - self.root_states[:, 7:13]) / self.dt), dim=-1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=-1)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        lower = self.dof_pos_limits[:, 0] + 0.5 * (1 - self.cfg["rewards"]["soft_dof_pos_limit"]) * (
            self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
        )
        upper = self.dof_pos_limits[:, 1] - 0.5 * (1 - self.cfg["rewards"]["soft_dof_pos_limit"]) * (
            self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
        )
        return torch.sum(((self.dof_pos < lower) | (self.dof_pos > upper)).float(), dim=-1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg["rewards"]["soft_dof_vel_limit"]).clip(min=0.0, max=1.0),
            dim=-1,
        )

    def _reward_torque_limits(self):
        # Penalize torques too close to the limit
        return torch.sum(
            (torch.abs(self.torques) - self.torque_limits * self.cfg["rewards"]["soft_torque_limit"]).clip(min=0.0),
            dim=-1,
        )

    def _reward_torque_tiredness(self):
        # Penalize torque tiredness
        return torch.sum(torch.square(self.torques / self.torque_limits).clip(max=1.0), dim=-1)

    def _reward_power(self):
        # Penalize power
        return torch.sum((self.torques * self.dof_vel).clip(min=0.0), dim=-1)

    def _reward_feet_slip(self):
        # Penalize feet velocities when contact
        return (
            torch.sum(
                torch.square((self.last_feet_pos - self.feet_pos) / self.dt).sum(dim=-1) * self.feet_contact.float(),
                dim=-1,
            )
            * (self.episode_length_buf > 1).float()
        )

    def _reward_feet_vel_z(self):
        return torch.sum(torch.square((self.last_feet_pos - self.feet_pos) / self.dt)[:, :, 2], dim=-1)

    def _reward_feet_roll(self):
        return torch.sum(torch.square(self.feet_roll), dim=-1)

    def _reward_feet_pitch(self):
        return torch.sum(torch.square(self.feet_pitch), dim=-1)

    def _reward_stance_feet_xy_vel(self):
        feet_vel = (self.last_feet_pos - self.feet_pos) / self.dt
        return (
            torch.sum(torch.sum(torch.square(feet_vel[:, :, :2]), dim=-1) * self.feet_contact.float(), dim=-1)
            * (self.episode_length_buf > 1).float()
        )

    def _reward_fast_stance_feet_xy_vel(self):
        feet_vel = (self.last_feet_pos - self.feet_pos) / self.dt
        speed_weight = torch.clamp((torch.abs(self.commands[:, 0]) - 0.70) / 0.50, min=0.0, max=1.0)
        return (
            torch.sum(torch.sum(torch.square(feet_vel[:, :, :2]), dim=-1) * self.feet_contact.float(), dim=-1)
            * speed_weight
            * self._straight_walk_mask()
            * (self.episode_length_buf > 1).float()
        )

    def _reward_stance_feet_yaw(self):
        target_yaw = self.commands[:, 4:6]
        yaw_error = (self.feet_yaw_rel - target_yaw + torch.pi) % (2 * torch.pi) - torch.pi
        return (
            torch.sum(torch.square(yaw_error) * self.feet_contact.float(), dim=-1)
            * (self.episode_length_buf > 1).float()
        )

    def _reward_stance_feet_roll(self):
        return (
            torch.sum(torch.square(self.feet_roll) * self.feet_contact.float(), dim=-1)
            * (self.episode_length_buf > 1).float()
        )

    def _reward_feet_yaw_diff(self):
        """
        Reward for tracking the commanded difference between left and right foot yaw angles.
        Instead of penalizing asymmetry, this now rewards tracking the commanded difference.
        """
        # Get commanded foot yaw difference
        commanded_diff = self.commands[:, 5] - self.commands[:, 4]  # foot_yaw_R - foot_yaw_L
        
        # Get actual foot yaw difference (relative to trunk)
        actual_diff = self.feet_yaw_rel[:, 1] - self.feet_yaw_rel[:, 0]  # right - left
        
        # Normalize difference to [-π, π]
        diff_error = (actual_diff - commanded_diff + torch.pi) % (2 * torch.pi) - torch.pi
        
        return torch.square(diff_error)

    def _reward_feet_yaw_mean(self):
        """
        Reward for tracking the commanded mean foot yaw angle.
        Instead of penalizing deviation from base yaw, this now rewards tracking the commanded mean.
        """
        # Get commanded foot yaw mean
        commanded_mean = (self.commands[:, 5] + self.commands[:, 4]) * 0.5  # (foot_yaw_R + foot_yaw_L) / 2
        
        # Get actual foot yaw mean (relative to trunk)
        actual_mean = self.feet_yaw_rel.mean(dim=-1)
        
        # Normalize mean to [-π, π]
        mean_error = (actual_mean - commanded_mean + torch.pi) % (2 * torch.pi) - torch.pi
        
        return torch.square(mean_error)

    def _reward_feet_offset_x(self):
        """Reward for tracking feet x-offset target, scaled by forward velocity"""
        # Get feet x-offset using existing helper function
        feet_x_offset, _ = self.get_feet_offset()
        
        # Get target x-offset from commands
        target_x_offset = self.commands[:, 8]  # feet_offset_x_target
        
        # Calculate error
        x_error = feet_x_offset - target_x_offset
        
        # Apply clipping similar to original feet_distance reward
        x_reward = torch.clip(torch.abs(x_error), min=0.0, max=0.1)
        
        # Get forward velocity (x-direction) from commands
        forward_vel = self.commands[:, 0]  # lin_vel_x
        
        # Get maximum forward velocity from config
        max_forward_vel = max(abs(self.cfg["commands"]["lin_vel_x"][0]), abs(self.cfg["commands"]["lin_vel_x"][1]))
        
        # Calculate velocity scaling factor: 1.0 at vel=0, 0.0 at vel=max_vel
        # Use absolute value of velocity for symmetric scaling
        # Quadratic decrease: vel_scale = (1.0 - |velocity| / max_velocity)^2
        vel_scale = torch.clamp((1.0 - torch.abs(forward_vel) / max_forward_vel) ** 2, min=0.0, max=1.0)
        
        # Scale reward by velocity factor
        x_reward = x_reward * vel_scale
        
        return x_reward

    def _reward_feet_offset_y(self):
        """Reward for tracking feet y-offset target, scaled by lateral velocity"""
        # Get feet y-offset using existing helper function
        _, feet_y_offset = self.get_feet_offset()
        
        # Get target y-offset from commands
        target_y_offset = self.commands[:, 9]  # feet_offset_y_target
        
        # Calculate error
        y_error = feet_y_offset - target_y_offset
        
        # Apply clipping similar to original feet_distance reward
        y_reward = torch.clip(torch.abs(y_error), min=0.0, max=0.1)
        
        # Get lateral velocity (y-direction) from commands
        lateral_vel = self.commands[:, 1]  # lin_vel_y
        
        # Get maximum lateral velocity from config
        max_lateral_vel = max(abs(self.cfg["commands"]["lin_vel_y"][0]), abs(self.cfg["commands"]["lin_vel_y"][1]))
        
        # Calculate velocity scaling factor: 1.0 at vel=0, 0.0 at vel=max_vel
        # Use absolute value of velocity for symmetric scaling
        # Quadratic decrease: vel_scale = (1.0 - |velocity| / max_velocity)^2
        vel_scale = torch.clamp((1.0 - torch.abs(lateral_vel) / max_lateral_vel) ** 2, min=0.0, max=1.0)
        
        # Scale reward by velocity factor
        y_reward = y_reward * vel_scale
        
        return y_reward

    def _reward_feet_swing(self):
        left_swing = (torch.abs(self.gait_process - 0.25) < 0.5 * self.cfg["rewards"]["swing_period"]) & (self.gait_frequency > 1.0e-8)
        right_swing = (torch.abs(self.gait_process - 0.75) < 0.5 * self.cfg["rewards"]["swing_period"]) & (self.gait_frequency > 1.0e-8)
        return (left_swing & ~self.feet_contact[:, 0]).float() + (right_swing & ~self.feet_contact[:, 1]).float()

    def _reward_foot_yaw_L(self):
        """Reward for tracking left foot yaw angle"""
        error = self.feet_yaw_rel[:, 0] - self.commands[:, 4]
        return torch.square(error)

    def _reward_foot_yaw_R(self):
        """Reward for tracking right foot yaw angle"""
        error = self.feet_yaw_rel[:, 1] - self.commands[:, 5]
        return torch.square(error)

    def get_feet_offset(self):
        """
        Helper function to calculate feet offsets in robot coordinates.
        Returns both x and y offsets between right and left feet (right - left).
        For y-offset, the feet_distance_ref is subtracted to get the relative offset.
        """
        # Get base yaw to transform to robot coordinates
        _, _, base_yaw = get_euler_xyz(self.base_quat)
        
        # Calculate feet positions in robot coordinates
        # Transform from world to robot coordinates
        feet_x_offset = (
            torch.cos(base_yaw) * (self.feet_pos[:, 0, 0] - self.feet_pos[:, 1, 0]) +
            torch.sin(base_yaw) * (self.feet_pos[:, 0, 1] - self.feet_pos[:, 1, 1])
        )
        
        feet_y_offset = (
            -torch.sin(base_yaw) * (self.feet_pos[:, 0, 0] - self.feet_pos[:, 1, 0]) +
            torch.cos(base_yaw) * (self.feet_pos[:, 0, 1] - self.feet_pos[:, 1, 1])
        )
        
        # Subtract feet_distance_ref from y-offset to get relative offset
        feet_y_offset = feet_y_offset - self.cfg["rewards"]["feet_distance_ref"]
        
        return feet_x_offset, feet_y_offset

    def get_feet_x_offset(self):
        """
        Helper function to calculate feet x-offset in robot coordinates.
        Returns the x-offset between right and left feet (right - left).
        """
        feet_x_offset, _ = self.get_feet_offset()
        return feet_x_offset
