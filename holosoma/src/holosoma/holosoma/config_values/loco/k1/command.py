"""Locomotion command presets for the K1 robot.

K1 uses the same velocity-command task definition as the Holosoma T1 FastSAC preset.
"""

from holosoma.config_values.loco.t1.command import t1_29dof_command as k1_12dof_command

__all__ = ["k1_12dof_command"]
