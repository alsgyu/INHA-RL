"""Locomotion experiment presets for the K1 robot."""

from dataclasses import replace

from holosoma.config_types.experiment import ExperimentConfig, NightlyConfig, TrainingConfig
from holosoma.config_values import (
    action,
    algo,
    command,
    curriculum,
    observation,
    randomization,
    reward,
    robot,
    simulator,
    termination,
    terrain,
)

k1_12dof_fast_sac = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="hv-k1-manager", name="k1_12dof_fast_sac_manager"),
    algo=replace(
        algo.fast_sac, config=replace(algo.fast_sac.config, num_learning_iterations=100000, use_symmetry=True)
    ),
    simulator=simulator.isaacgym,
    robot=robot.k1_12dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=observation.k1_12dof_loco_single_wolinvel,
    action=action.k1_12dof_joint_pos,
    termination=termination.k1_12dof_termination,
    randomization=randomization.k1_12dof_randomization,
    command=command.k1_12dof_command,
    curriculum=curriculum.k1_12dof_curriculum_fast_sac,
    reward=reward.k1_12dof_loco_fast_sac,
    nightly=NightlyConfig(
        iterations=50000,
        metrics={"Episode/rew_tracking_ang_vel": [0.65, "inf"], "Episode/rew_tracking_lin_vel": [0.9, "inf"]},
    ),
)

__all__ = ["k1_12dof_fast_sac"]
