"""Locomotion curriculum presets for the K1 robot."""

from holosoma.config_values.loco.t1.curriculum import (
    t1_29dof_curriculum as k1_12dof_curriculum,
    t1_29dof_curriculum_fast_sac as k1_12dof_curriculum_fast_sac,
)

__all__ = ["k1_12dof_curriculum", "k1_12dof_curriculum_fast_sac"]
