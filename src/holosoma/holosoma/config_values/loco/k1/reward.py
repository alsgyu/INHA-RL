"""Locomotion reward presets for the K1 robot.

The reward terms and weights follow the Holosoma T1 FastSAC locomotion presets. The only robot-specific
adaptation is projecting the T1 pose regularizer's leg-joint weights onto K1's 12 actuated joints.
"""

from dataclasses import replace

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg
from holosoma.config_values.loco.t1.reward import t1_29dof_loco, t1_29dof_loco_fast_sac

_K1_12DOF_POSE_WEIGHTS = [
    0.5,  # Left_Hip_Pitch   (was 0.01 — too low, caused leg drift during standing)
    1.0,  # Left_Hip_Roll
    5.0,  # Left_Hip_Yaw
    0.5,  # Left_Knee_Pitch  (was 0.01 — too low, caused leg drift during standing)
    5.0,  # Left_Ankle_Pitch
    5.0,  # Left_Ankle_Roll
    0.5,  # Right_Hip_Pitch  (was 0.01 — too low, caused leg drift during standing)
    1.0,  # Right_Hip_Roll
    5.0,  # Right_Hip_Yaw
    0.5,  # Right_Knee_Pitch (was 0.01 — too low, caused leg drift during standing)
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

# ---------------------------------------------------------------------------
# Stability terms adapted from INHA-LAB (verified on real K1 hardware).
# These are appended on top of the T1 FastSAC baseline to fix:
#   1. Zigzag walking (heading_drift)
#   2. Leg drift during standing (stand_feet_flat)
# ---------------------------------------------------------------------------
_STABILITY_TERMS: dict[str, RewardTermCfg] = {
    "heading_drift": RewardTermCfg(
        func="holosoma.managers.reward.terms.locomotion:heading_drift",
        weight=-1.0,
        params={"deadband": 0.1},
    ),
    "stand_feet_flat": RewardTermCfg(
        func="holosoma.managers.reward.terms.locomotion:stand_feet_flat",
        weight=-1.0,
        params={"command_threshold": 0.05, "speed_threshold": 0.2},
    ),
    "dof_acc": RewardTermCfg(
        func="holosoma.managers.reward.terms.locomotion:dof_acc_l2",
        weight=-1.0e-7,
    ),
}

k1_12dof_loco_fast_sac = replace(
    k1_12dof_loco_fast_sac,
    terms={**k1_12dof_loco_fast_sac.terms, **_STABILITY_TERMS},
)

__all__ = ["k1_12dof_loco", "k1_12dof_loco_fast_sac"]
