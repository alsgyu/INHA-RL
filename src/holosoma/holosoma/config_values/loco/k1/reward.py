"""Locomotion reward presets for the K1 robot.

The reward terms and weights follow the Holosoma T1 FastSAC locomotion presets. The only robot-specific
adaptation is projecting the T1 pose regularizer's leg-joint weights onto K1's 12 actuated joints.
"""

from dataclasses import replace

from holosoma.config_types.reward import RewardManagerCfg
from holosoma.config_values.loco.t1.reward import t1_29dof_loco, t1_29dof_loco_fast_sac

_K1_12DOF_POSE_WEIGHTS = [
    0.01,  # Left_Hip_Pitch
    1.0,  # Left_Hip_Roll
    5.0,  # Left_Hip_Yaw
    0.01,  # Left_Knee_Pitch
    5.0,  # Left_Ankle_Pitch
    5.0,  # Left_Ankle_Roll
    0.01,  # Right_Hip_Pitch
    1.0,  # Right_Hip_Roll
    5.0,  # Right_Hip_Yaw
    0.01,  # Right_Knee_Pitch
    5.0,  # Right_Ankle_Pitch
    5.0,  # Right_Ankle_Roll
]


def _project_pose_weights(cfg: RewardManagerCfg) -> RewardManagerCfg:
    terms = dict(cfg.terms)
    pose_term = terms["pose"]
    terms["pose"] = replace(pose_term, params={**pose_term.params, "pose_weights": _K1_12DOF_POSE_WEIGHTS})
    return replace(cfg, terms=terms)


k1_12dof_loco = _project_pose_weights(t1_29dof_loco)
k1_12dof_loco_fast_sac = _project_pose_weights(t1_29dof_loco_fast_sac)

__all__ = ["k1_12dof_loco", "k1_12dof_loco_fast_sac"]
