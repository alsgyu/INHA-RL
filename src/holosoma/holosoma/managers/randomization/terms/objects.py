"""Object reset-time randomization term (pose jitter).

The object physics-DR *startup* terms (mass / material / inertia / damping) live in this module
on the full feature branch; the object-DR follow-up branch adds them. This scene-objects branch
carries only the reset-time pose-jitter term, which resets alongside the object reset it perturbs.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from loguru import logger

from holosoma.managers.randomization.terms._shared import _ensure_env_ids_tensor
from holosoma.simulator.shared.object_registry import ObjectType
from holosoma.utils.rotations import quat_from_angle_axis, quat_mul
from holosoma.utils.sampler import DistributionLike, DistributionSpec, TermSampler


def _resolve_object_names(env: Any, object_names: Sequence[str] | None) -> list[str]:
    """Free-body (INDIVIDUAL) actor names a per-object randomizer should target.

    Defaults to every registered free body so a preset stays object-count-agnostic
    (robot-only scenes resolve to an empty list -> the term no-ops). An explicit
    ``object_names`` narrows it to specific bodies. Static (SCENE) bodies are never
    included — they have no manipulable physics to randomize here.
    """
    if object_names is not None:
        return list(object_names)
    return env.simulator.object_registry.get_names_by_type(ObjectType.INDIVIDUAL)


def _mujoco_object_root_body(simulator: Any, name: str) -> str:
    """The compiled-model (prefixed) root-body name of registered object ``name``.

    Resolves from whichever scene-manager map holds the object: a STANDALONE rigid object is
    recorded in ``rigid_object_root_bodies`` (``{name}_{root}``), while a body that came from a
    1->N scene FILE is recorded in ``scene_file_bodies`` (the body is ``{file}_{body}``, which is
    the actor name itself). Raises KeyError if ``name`` is neither.
    """
    sm = simulator.scene_manager
    if name in sm.rigid_object_root_bodies:
        return sm.rigid_object_root_bodies[name]
    if name in getattr(sm, "scene_file_bodies", {}):
        return sm.scene_file_bodies[name][0]  # (prefixed_root_body, is_static)
    raise KeyError(f"Object '{name}' is not a registered standalone or scene-file body.")


def _mujoco_object_bodies(simulator: Any, name: str) -> list[tuple[int, str]]:
    """``(body_id, body_name)`` for every compiled-model body owned by object ``name``.

    Reads the CPU ``mujoco.MjModel`` (``simulator.backend.model``), which both MuJoCo
    backends expose, so it feeds the WarpBackend and ClassicBackend DR paths alike. Collects
    the object's root body plus its descendant subtree via ``body_parentid`` — correct for both
    a standalone object (single root, possibly with nested child bodies) and a scene-file body
    (its own root in the shared file tree, NOT its file-siblings). Resolving the subtree
    structurally (not by a ``{name}_`` name prefix) is what keeps scene-file siblings separate.
    """
    import mujoco

    model = simulator.backend.model
    root_body = _mujoco_object_root_body(simulator, name)
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body)
    if root_id == -1:
        raise ValueError(f"Root body '{root_body}' for object '{name}' not found in compiled model.")

    # Subtree = root + every body whose parent chain reaches root (walk parentid up to root/world).
    subtree = {root_id}
    for bid in range(model.nbody):
        anc = bid
        while anc not in (0, root_id) and anc != model.body_parentid[anc]:
            anc = int(model.body_parentid[anc])
            if anc == root_id:
                subtree.add(bid)
                break
    return [(bid, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)) for bid in sorted(subtree)]


def _jitter_spec(range_cfg: float | DistributionLike) -> DistributionSpec | None:
    """Build a :class:`DistributionSpec` for a pose-jitter component, or ``None`` if it's a no-op.

    A scalar ``r`` is the symmetric uniform band ``[-r, r]``; ``0.0`` (or ``[0, 0]``) is a no-op
    -> ``None``. A ``[lo, hi]`` pair (uniform) or spec dict is taken as-is. A non-uniform jitter
    (e.g. gaussian) must be given as a spec dict, not a bare float.
    """
    if isinstance(range_cfg, (int, float)):
        r = float(range_cfg)
        if r == 0.0:
            return None
        return DistributionSpec(kind="uniform", low=-r, high=r)
    spec = DistributionSpec.parse(range_cfg)
    # Treat an explicit zero-width band as a no-op, regardless of kind: a [c, c] band is a point
    # mass at c, and only c == 0 leaves the baseline pose unchanged. (An explicit (mean, std)
    # gaussian has low/high == None, so the `== 0.0` test is False and it is never a no-op here.)
    if spec.low == 0.0 and spec.high == 0.0:
        return None
    return spec


def jitter_object_pose_on_reset(
    env,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    *,
    sampler: TermSampler,
    xy_range: float | DistributionLike = 0.0,
    yaw_range: float | DistributionLike = 0.0,
    object_names: Sequence[str] | None = None,
    enabled: bool = True,
    **_,
) -> None:
    """Jitter free-body XY position and yaw around the baseline reset pose, per env.

    The random yaw is COMPOSED onto the initial orientation (which may be non-identity).
    Only the pose is jittered; each body keeps its configured initial velocity
    (``RigidObjectConfig.linear_velocity``/``angular_velocity``, zero by default), matching
    the baseline reset (``BaseTask._reset_objects_callback``) this term overwrites. Static
    (SCENE) bodies are never jittered. No-ops on a robot-only scene or when both ranges are
    zero.

    Parameters
    ----------
    xy_range : float | [lo, hi] | spec dict
        XY offset (metres). A FLOAT ``r`` is the symmetric uniform half-width ``[-r, r]``; a
        ``[lo, hi]`` pair (uniform) or ``{kind, low, high, mean, std}`` spec dict is taken directly.
    yaw_range : float | [lo, hi] | spec dict
        Yaw delta (radians); same forms as ``xy_range``.
    object_names : Sequence[str] | None
        Free bodies to jitter. ``None`` (default) jitters every registered free body.
    """
    if not enabled:
        return

    xy_spec = _jitter_spec(xy_range)
    yaw_spec = _jitter_spec(yaw_range)
    if xy_spec is None and yaw_spec is None:
        return

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    # Only INDIVIDUAL (free) bodies are jitterable; never write a static body's pose. An explicitly
    # requested static name is a config mistake — warn instead of silently skipping it.
    static = set(env.simulator.object_registry.get_names_by_type(ObjectType.SCENE))
    resolved = _resolve_object_names(env, object_names)
    dropped_static = [n for n in resolved if n in static]
    if dropped_static and object_names is not None:
        logger.warning(f"pose jitter skips static (SCENE) bodies: {dropped_static}")
    names = [n for n in resolved if n not in static]
    if not names:
        return

    # Baseline reset pose + velocity (env origins already applied), object-major / env-minor:
    # pose [len(names)*len(env_ids), 7] = [x,y,z,qx,qy,qz,qw]; velocity [..., 6] = [vx,vy,vz,wx,wy,wz].
    base = env.simulator.get_actor_initial_poses(names, idx)
    base_vel = env.simulator.get_actor_initial_velocities(names, idx)
    n_rows = base.shape[0]

    states = torch.zeros(n_rows, 13, device=env.device)
    states[:, :7] = base
    states[:, 7:] = base_vel  # keep the configured initial velocity; jitter only the pose

    # Draws are keyed per (env, object) and laid out object-major to match the row order
    # [name0_env0, name0_env1, ..., name1_env0, ...]: draw [n_env, n_names] then transpose -> flatten.
    n_names = len(names)
    obj_ids = torch.arange(n_names)[None, :]  # [1, n_names] coord -> trailing object dimension

    def _draw_object_major(spec: DistributionSpec, axis: int) -> torch.Tensor:
        # one value per (env, object) -> object-major column vector of length n_rows. ``axis`` is the
        # int STREAM coord (x/y/yaw kept independent); obj ids land on the trailing dimension.
        per_env_obj = sampler.draw(spec, env_ids=idx, coords=(axis, obj_ids), device=env.device)  # [n_env, n_names]
        return per_env_obj.transpose(0, 1).reshape(-1)  # [n_names, n_env] -> n_rows

    if xy_spec is not None:
        states[:, 0] += _draw_object_major(xy_spec, axis=0)
        states[:, 1] += _draw_object_major(xy_spec, axis=1)

    if yaw_spec is not None:
        yaw = _draw_object_major(yaw_spec, axis=2)
        z_axis = torch.zeros(n_rows, 3, device=env.device)
        z_axis[:, 2] = 1.0
        yaw_quat = quat_from_angle_axis(yaw, z_axis, w_last=True)  # xyzw
        # Compose the yaw delta onto the baseline orientation (world-frame yaw): q = dq * q0.
        states[:, 3:7] = quat_mul(yaw_quat, base[:, 3:7].clone(), w_last=True)

    env.simulator.set_actor_states(names, idx, states)
