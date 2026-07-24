import argparse
import glob
import os
import time

import numpy as np
import torch
import yaml

from utils.models.BaseAC import BaseActorCritic
from utils.command_metrics import CommandVelocityMetrics
from deploy.utils.policy_walk_getup_k1 import Policy


LEG_START_INDEX = 10


class ClampedActor(torch.nn.Module):
    def __init__(self, actor, action_clip):
        super().__init__()
        self.actor = actor
        self.action_clip = None if action_clip is None else float(action_clip)

    def forward(self, obs):
        action = self.actor(obs)
        if self.action_clip is not None:
            action = torch.clamp(action, -self.action_clip, self.action_clip)
        return action


def quat_to_mat(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def merge_dicts(base, override):
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_task_config(cfg_file, visited=None):
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
    return merge_dicts(load_task_config(parent, visited), cfg)


def robot_type(task_name):
    return task_name.split("/", 1)[0] if "/" in task_name else "Unknown"


def resolve_checkpoint(task, checkpoint):
    if checkpoint in (None, "", "deploy"):
        return None
    if checkpoint not in ("-1", -1):
        return checkpoint

    task_cfg = load_task_config(os.path.join("envs", f"{task}.yaml"))
    task_names = [task_cfg.get("basic", {}).get("log_task", task)]
    for fallback in (task_cfg.get("basic", {}).get("task"), task):
        if fallback and fallback not in task_names:
            task_names.append(fallback)

    for task_name in task_names:
        pattern = os.path.join("logs", robot_type(task_name), task_name, "**", "*.pth")
        matches = sorted(glob.glob(pattern, recursive=True), key=os.path.getmtime)
        if matches:
            return matches[-1]
    return None


def load_checkpoint_actor(task, checkpoint, action_clip_override=None):
    checkpoint_path = resolve_checkpoint(task, checkpoint)
    if checkpoint_path is None:
        return None, None, None

    task_cfg = load_task_config(os.path.join("envs", f"{task}.yaml"))
    model = BaseActorCritic(
        task_cfg["env"]["num_actions"],
        task_cfg["env"]["num_observations"],
        task_cfg["env"]["num_privileged_obs"],
    )
    try:
        model_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        model_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(model_dict["model"], strict=False)
    model.actor.eval()
    action_clip = action_clip_override
    learner = str(model_dict.get("learner", ""))
    if action_clip is None and learner.startswith("sirl_worldmodel"):
        action_clip = model_dict.get("deploy_action_clip")
        if action_clip is None:
            wm_cfg = task_cfg.get("algorithm", {}).get("sirl_worldmodel", {})
            action_clip = wm_cfg.get("deploy_action_clip", wm_cfg.get("collect_action_clip", wm_cfg.get("action_clip")))
    actor = ClampedActor(model.actor, action_clip) if action_clip is not None else model.actor
    actor.eval()
    return actor, checkpoint_path, action_clip


def quat_to_euler(q):
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=np.float32)


def quat_from_euler(roll, pitch, yaw):
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float32,
    )


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def estimate_body_velocity(prev_pos, prev_yaw, data, dt):
    current_pos = np.array(data.qpos[0:3], dtype=np.float64)
    current_yaw = float(quat_to_euler(np.array(data.qpos[3:7], dtype=np.float32))[2])
    world_vel = (current_pos - prev_pos) / dt
    cos_yaw = np.cos(current_yaw)
    sin_yaw = np.sin(current_yaw)
    body_vx = cos_yaw * world_vel[0] + sin_yaw * world_vel[1]
    body_vy = -sin_yaw * world_vel[0] + cos_yaw * world_vel[1]
    body_vyaw = wrap_to_pi(current_yaw - prev_yaw) / dt
    return current_pos, current_yaw, np.array([body_vx, body_vy, body_vyaw], dtype=np.float64), world_vel


def command_path_error(start_xy, start_yaw, time_s, current_xy, current_yaw, vx, vy, vyaw):
    desired_yaw = start_yaw + vyaw * time_s
    if abs(vyaw) < 1.0e-6:
        local_dx = vx * time_s
        local_dy = vy * time_s
    else:
        local_dx = (vx * np.sin(vyaw * time_s) + vy * (np.cos(vyaw * time_s) - 1.0)) / vyaw
        local_dy = (vx * (1.0 - np.cos(vyaw * time_s)) + vy * np.sin(vyaw * time_s)) / vyaw
    cos_yaw = np.cos(start_yaw)
    sin_yaw = np.sin(start_yaw)
    desired_xy = start_xy + np.array(
        [
            cos_yaw * local_dx - sin_yaw * local_dy,
            sin_yaw * local_dx + cos_yaw * local_dy,
        ],
        dtype=np.float64,
    )
    path_error = np.asarray(current_xy, dtype=np.float64) - desired_xy
    along_error = cos_yaw * path_error[0] + sin_yaw * path_error[1]
    lateral_error = -sin_yaw * path_error[0] + cos_yaw * path_error[1]
    yaw_error = wrap_to_pi(current_yaw - desired_yaw)
    return along_error, lateral_error, yaw_error


def fallen_rpy(pose):
    if pose == "front":
        return 0.0, 1.45, 0.0
    if pose == "back":
        return 3.0, 0.0, 0.0
    if pose == "left":
        return 1.45, 0.0, 0.0
    if pose == "right":
        return -1.45, 0.0, 0.0
    raise ValueError(f"Unknown fall pose: {pose}")


def set_root_pose(data, qpos, height, rpy):
    data.qpos[0:3] = qpos
    data.qpos[2] = height
    data.qpos[3:7] = quat_from_euler(*rpy)
    data.qvel[0:6] = 0.0


def resolve_target_pose(args, start_xy, start_yaw):
    has_absolute = args.target_x is not None or args.target_y is not None
    has_local = args.target_local_x is not None or args.target_local_y is not None
    if not has_absolute and not has_local:
        return None
    if has_absolute and (args.target_x is None or args.target_y is None):
        raise ValueError("--target_x and --target_y must be provided together.")
    if has_absolute and has_local:
        raise ValueError("Use either absolute target_x/target_y or local target_local_x/target_local_y, not both.")

    if has_absolute:
        target_x = float(args.target_x)
        target_y = float(args.target_y)
    else:
        local_x = float(args.target_local_x or 0.0)
        local_y = float(args.target_local_y or 0.0)
        cos_yaw = np.cos(start_yaw)
        sin_yaw = np.sin(start_yaw)
        target_x = float(start_xy[0] + cos_yaw * local_x - sin_yaw * local_y)
        target_y = float(start_xy[1] + sin_yaw * local_x + cos_yaw * local_y)

    if args.target_theta is not None:
        target_theta = float(args.target_theta)
    else:
        target_theta = float(start_yaw + (args.target_heading_offset or 0.0))
    return np.array([target_x, target_y, wrap_to_pi(target_theta)], dtype=np.float64)


def configure_ground_contact(mujoco, model, friction, torsional_friction, rolling_friction, condim):
    ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    if ground_id < 0:
        return False
    model.geom_friction[ground_id, 0] = float(friction)
    model.geom_friction[ground_id, 1] = float(torsional_friction)
    model.geom_friction[ground_id, 2] = float(rolling_friction)
    model.geom_condim[ground_id] = int(condim)
    return True


def sensor_vector(mujoco, model, data, name, fallback):
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id < 0:
        return fallback
    adr = int(model.sensor_adr[sensor_id])
    dim = int(model.sensor_dim[sensor_id])
    if dim <= 0:
        return fallback
    return np.array(data.sensordata[adr : adr + dim], dtype=np.float32)


def apply_default_pose_overrides(cfg, args):
    pitch_overrides = {
        0: args.default_hip_pitch,
        3: args.default_knee_pitch,
        4: args.default_ankle_pitch,
        6: args.default_hip_pitch,
        9: args.default_knee_pitch,
        10: args.default_ankle_pitch,
    }
    if not any(value is not None for value in pitch_overrides.values()):
        return

    for section_name in ("common", "walk_policy"):
        qpos = cfg[section_name]["default_qpos"]
        for leg_offset, value in pitch_overrides.items():
            if value is not None:
                qpos[LEG_START_INDEX + leg_offset] = float(value)


def leg_default_summary(qpos):
    return (
        qpos[LEG_START_INDEX],
        qpos[LEG_START_INDEX + 3],
        qpos[LEG_START_INDEX + 4],
        qpos[LEG_START_INDEX + 6],
        qpos[LEG_START_INDEX + 9],
        qpos[LEG_START_INDEX + 10],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="deploy/configs/Walk_GetUp_k1.yaml")
    parser.add_argument("--xml", default="resources/K1/K1_22dof.xml")
    parser.add_argument("--task", default="K1/ParameterWalk")
    parser.add_argument("--checkpoint", default="-1", help="Walk .pth checkpoint. Use -1 for latest, or deploy to use deploy/models .pt.")
    parser.add_argument("--checkpoint_action_clip", type=float, default=None, help="Optional action clamp for .pth walk checkpoints.")
    parser.add_argument("--walk_policy", default=None, help="Explicit TorchScript .pt walk policy path. Overrides --checkpoint.")
    parser.add_argument("--duration_s", type=float, default=30.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--cmd_pose6",
        "--cmds",
        nargs=6,
        type=float,
        metavar=("ROBOT_X", "ROBOT_Y", "ROBOT_THETA", "TARGET_X", "TARGET_Y", "TARGET_THETA"),
        help="Six pose command values: robot_x robot_y robot_theta target_x target_y target_theta.",
    )
    parser.add_argument("--robot_x", type=float, default=0.0)
    parser.add_argument("--robot_y", type=float, default=0.0)
    parser.add_argument("--robot_theta", type=float, default=0.0)
    parser.add_argument("--vx", type=float, default=0.2)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--vyaw", type=float, default=0.0)
    parser.add_argument("--target_x", type=float, default=None)
    parser.add_argument("--target_y", type=float, default=None)
    parser.add_argument("--target_theta", type=float, default=None)
    parser.add_argument("--target_local_x", type=float, default=None)
    parser.add_argument("--target_local_y", type=float, default=None)
    parser.add_argument("--target_heading_offset", type=float, default=0.0)
    parser.add_argument("--force_fall_after_s", type=float, default=-1.0)
    parser.add_argument("--fall_pose", choices=["front", "back", "left", "right"], default="front")
    parser.add_argument("--start_fallen", action="store_true")
    parser.add_argument("--enable_getup", action="store_true")
    parser.add_argument("--walk_only", action="store_true")
    parser.add_argument("--default_hip_pitch", type=float)
    parser.add_argument("--default_knee_pitch", type=float)
    parser.add_argument("--default_ankle_pitch", type=float)
    parser.add_argument("--default_base_height", type=float, default=0.70)
    parser.add_argument("--ground_friction", type=float, default=1.0)
    parser.add_argument("--ground_torsional_friction", type=float, default=0.05)
    parser.add_argument("--ground_rolling_friction", type=float, default=0.001)
    parser.add_argument("--ground_condim", type=int, default=6)
    parser.add_argument("--command_slew_rate", type=float, default=None)
    parser.add_argument("--walk_action_clip", type=float, default=None)
    parser.add_argument("--disable_velocity_adapter", action="store_true")
    parser.add_argument("--metrics_window_s", type=float, default=3.0)
    parser.add_argument("--metrics_warmup_s", type=float, default=1.0)
    parser.add_argument("--metrics_csv", default=None)
    parser.add_argument("--metrics_csv_sample_s", type=float, default=0.05)
    args = parser.parse_args()
    if args.cmd_pose6 is not None:
        (
            args.robot_x,
            args.robot_y,
            args.robot_theta,
            args.target_x,
            args.target_y,
            args.target_theta,
        ) = args.cmd_pose6

    import mujoco

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    apply_default_pose_overrides(cfg, args)
    if args.command_slew_rate is not None:
        cfg["walk_policy"]["command_slew_rate"] = float(args.command_slew_rate)
    if args.walk_action_clip is not None:
        cfg["walk_policy"]["normalization"]["clip_actions"] = float(args.walk_action_clip)
    if args.disable_velocity_adapter:
        cfg["walk_policy"].setdefault("velocity_command_adapter", {})["enabled"] = False

    torch.set_num_threads(1)
    enable_getup = (not args.walk_only) and (args.enable_getup or args.start_fallen or args.force_fall_after_s >= 0.0)
    checkpoint_actor = None
    checkpoint_action_clip = None
    policy_source = args.walk_policy or cfg["walk_policy"]["policy_path"]
    checkpoint_warning = None
    if args.walk_policy:
        cfg["walk_policy"]["policy_path"] = args.walk_policy
    else:
        checkpoint_actor, checkpoint_path, checkpoint_action_clip = load_checkpoint_actor(
            args.task,
            args.checkpoint,
            action_clip_override=args.checkpoint_action_clip,
        )
        if checkpoint_actor is not None:
            policy_source = checkpoint_path
        elif args.checkpoint not in (None, "", "deploy"):
            checkpoint_warning = f"Could not find checkpoint '{args.checkpoint}' for {args.task}; falling back to deploy policy."
    policy = Policy(
        cfg,
        enable_getup=enable_getup,
        walk_policy=checkpoint_actor,
        walk_policy_path=policy_source,
    )
    model = mujoco.MjModel.from_xml_path(args.xml)
    ground_configured = configure_ground_contact(
        mujoco,
        model,
        args.ground_friction,
        args.ground_torsional_friction,
        args.ground_rolling_friction,
        args.ground_condim,
    )
    data = mujoco.MjData(model)

    default_qpos = np.array(cfg["common"]["default_qpos"], dtype=np.float32)
    stiffness = np.array(cfg["common"]["stiffness"], dtype=np.float32)
    damping = np.array(cfg["common"]["damping"], dtype=np.float32)
    torque_limit = np.array(cfg["common"]["torque_limit"], dtype=np.float32)
    if model.actuator_forcerange.shape[0] == len(torque_limit):
        torque_limit = np.minimum(torque_limit, np.abs(model.actuator_forcerange[:, 1]))

    data.qpos[0:3] = np.array([args.robot_x, args.robot_y, args.default_base_height], dtype=np.float32)
    data.qpos[3:7] = quat_from_euler(0.0, 0.0, args.robot_theta)
    data.qpos[7 : 7 + len(default_qpos)] = default_qpos
    data.qvel[:] = 0.0
    if args.start_fallen:
        set_root_pose(data, np.array([args.robot_x, args.robot_y, 0.25], dtype=np.float32), 0.25, fallen_rpy(args.fall_pose))
    mujoco.mj_forward(model, data)

    viewer = None
    if not args.headless:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(model, data)

    sim_dt = float(model.opt.timestep)
    policy_dt = float(policy.get_policy_interval())
    next_policy_t = 0.0
    target_qpos = np.copy(default_qpos)
    forced_fall = args.start_fallen
    start_xy = np.copy(data.qpos[0:2])
    start_yaw = float(quat_to_euler(np.array(data.qpos[3:7], dtype=np.float32))[2])
    target_pose = resolve_target_pose(args, start_xy, start_yaw)
    path_start_time = float(data.time)
    last_report = -1.0
    mode_time = 0.0
    metric_pos = np.array(data.qpos[0:3], dtype=np.float64)
    metric_yaw = float(quat_to_euler(np.array(data.qpos[3:7], dtype=np.float32))[2])
    actual_velocity = np.zeros(3, dtype=np.float64)
    world_velocity = np.zeros(3, dtype=np.float64)
    metrics = CommandVelocityMetrics(
        (0.0, 0.0, 0.0) if target_pose is not None else (args.vx, args.vy, args.vyaw),
        window_s=args.metrics_window_s,
        csv_path=args.metrics_csv,
        csv_sample_s=args.metrics_csv_sample_s,
    )
    if target_pose is None:
        print(f"[mujoco] walk command vx={args.vx:.3f} vy={args.vy:.3f} vyaw={args.vyaw:.3f} getup_enabled={enable_getup}")
    else:
        print(
            f"[mujoco] robot pose x={args.robot_x:.3f} y={args.robot_y:.3f} theta={args.robot_theta:.3f} "
        )
        print(
            f"[mujoco] target pose x={target_pose[0]:.3f} y={target_pose[1]:.3f} "
            f"theta={target_pose[2]:.3f} getup_enabled={enable_getup}"
        )
    if checkpoint_warning:
        print(f"[mujoco] warning: {checkpoint_warning}")
    print(f"[mujoco] walk policy source={policy.walk_policy_path}")
    if checkpoint_actor is not None and checkpoint_action_clip is not None:
        print(f"[mujoco] checkpoint action clip={float(checkpoint_action_clip):.3f}")
    print(
        f"[mujoco] walk action_clip={cfg['walk_policy']['normalization']['clip_actions']:.3f} "
        f"command_slew_rate={float(cfg['walk_policy'].get('command_slew_rate', 1.0)):.3f} "
        f"velocity_adapter={bool(cfg['walk_policy'].get('velocity_command_adapter', {}).get('enabled', False))}"
    )
    print(
        f"[mujoco] model nq={model.nq} nv={model.nv} nu={model.nu} "
        f"actuated_dof={model.nu} ground_configured={ground_configured} "
        f"ground_friction=({args.ground_friction:.2f},{args.ground_torsional_friction:.3f},"
        f"{args.ground_rolling_friction:.4f}) ground_condim={args.ground_condim}"
    )
    l_hip, l_knee, l_ankle, r_hip, r_knee, r_ankle = leg_default_summary(default_qpos)
    print(
        "[mujoco] default pose "
        f"L(hip={l_hip:.3f}, knee={l_knee:.3f}, ankle={l_ankle:.3f}) "
        f"R(hip={r_hip:.3f}, knee={r_knee:.3f}, ankle={r_ankle:.3f}) "
        f"base_z={args.default_base_height:.3f}"
    )

    while data.time < args.duration_s:
        if enable_getup and args.force_fall_after_s >= 0.0 and (not forced_fall) and data.time >= args.force_fall_after_s:
            set_root_pose(data, data.qpos[0:3].copy(), 0.25, fallen_rpy(args.fall_pose))
            mujoco.mj_forward(model, data)
            policy.mode = "getup"
            forced_fall = True
            mode_time = 0.0
            metric_pos = np.array(data.qpos[0:3], dtype=np.float64)
            metric_yaw = float(quat_to_euler(np.array(data.qpos[3:7], dtype=np.float32))[2])
            actual_velocity[:] = 0.0
            world_velocity[:] = 0.0
            start_xy = np.copy(data.qpos[0:2])
            start_yaw = metric_yaw
            path_start_time = float(data.time)
            metrics.reset_window()
            print(f"[mujoco] forced fall at t={data.time:.2f}s pose={args.fall_pose}")

        root_quat = np.array(data.qpos[3:7], dtype=np.float32)
        base_rpy = quat_to_euler(root_quat)
        projected_gravity = quat_to_mat(root_quat).T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
        base_ang_vel = sensor_vector(
            mujoco,
            model,
            data,
            "angular-velocity",
            np.array(data.qvel[3:6], dtype=np.float32),
        )

        if data.time >= next_policy_t:
            next_policy_t += policy_dt
            previous_mode = policy.mode
            if target_pose is None:
                target_qpos[:] = policy.inference(
                    time_now=float(data.time),
                    dof_pos=np.array(data.qpos[7 : 7 + len(default_qpos)], dtype=np.float32),
                    dof_vel=np.array(data.qvel[6 : 6 + len(default_qpos)], dtype=np.float32),
                    base_ang_vel=base_ang_vel,
                    projected_gravity=projected_gravity,
                    base_rpy=base_rpy,
                    vx=args.vx,
                    vy=args.vy,
                    vyaw=args.vyaw,
                )
            else:
                target_qpos[:] = policy.target_pose_inference(
                    time_now=float(data.time),
                    dof_pos=np.array(data.qpos[7 : 7 + len(default_qpos)], dtype=np.float32),
                    dof_vel=np.array(data.qvel[6 : 6 + len(default_qpos)], dtype=np.float32),
                    base_ang_vel=base_ang_vel,
                    projected_gravity=projected_gravity,
                    base_rpy=base_rpy,
                    base_pos=np.array(data.qpos[0:3], dtype=np.float32),
                    target_x=target_pose[0],
                    target_y=target_pose[1],
                    target_theta=target_pose[2],
                )
            if policy.mode != previous_mode:
                mode_time = 0.0
                start_xy = np.copy(data.qpos[0:2])
                start_yaw = metric_yaw
                path_start_time = float(data.time)
                metrics.reset_window()

        dof_pos = np.array(data.qpos[7 : 7 + len(default_qpos)], dtype=np.float32)
        dof_vel = np.array(data.qvel[6 : 6 + len(default_qpos)], dtype=np.float32)
        torque = stiffness * (target_qpos - dof_pos) - damping * dof_vel
        data.ctrl[:] = np.clip(torque, -torque_limit, torque_limit)
        mujoco.mj_step(model, data)
        metric_pos, metric_yaw, actual_velocity, world_velocity = estimate_body_velocity(metric_pos, metric_yaw, data, sim_dt)
        mode_time += sim_dt

        tracked = policy.mode == "walk" and mode_time >= args.metrics_warmup_s
        metrics_row, metrics_summary = metrics.update(
            data.time,
            policy.mode,
            actual_velocity,
            policy_command=np.array(policy.smoothed_commands, dtype=np.float64),
            tracked=tracked,
        )

        if viewer is not None:
            viewer.sync()
            time.sleep(sim_dt)

        if data.time - last_report >= 1.0:
            last_report = data.time
            xy_error = data.qpos[0:2] - start_xy
            if target_pose is None:
                path_time = max(float(data.time) - path_start_time, 0.0)
                path_along_error, path_lateral_error, path_yaw_error = command_path_error(
                    start_xy,
                    start_yaw,
                    path_time,
                    data.qpos[0:2],
                    metric_yaw,
                    args.vx,
                    args.vy,
                    args.vyaw,
                )
                path_text = f"path_err=({path_along_error:+.3f},{path_lateral_error:+.3f}) yaw_err={path_yaw_error:+.3f} "
            else:
                target_dist = np.linalg.norm(target_pose[0:2] - data.qpos[0:2])
                target_heading_err = wrap_to_pi(target_pose[2] - metric_yaw)
                path_text = f"target_err=(dist={target_dist:.3f},yaw={target_heading_err:+.3f}) "
            world_speed_xy = np.linalg.norm(world_velocity[:2])
            report_rpy = quat_to_euler(np.array(data.qpos[3:7], dtype=np.float32))
            metrics_text = metrics.report(metrics_row, metrics_summary)
            print(
                f"[mujoco] t={data.time:5.2f}s mode={policy.mode:5s} "
                f"xy=({data.qpos[0]:+.2f},{data.qpos[1]:+.2f}) "
                f"drift_y={xy_error[1]:+.3f} "
                f"{path_text}"
                f"world_v=({world_velocity[0]:+.3f},{world_velocity[1]:+.3f}) "
                f"world_speed={world_speed_xy:.3f} "
                f"rpy=({report_rpy[0]:+.2f},{report_rpy[1]:+.2f},{report_rpy[2]:+.2f}) "
                f"{metrics_text}"
            )

    if viewer is not None:
        viewer.close()
    metrics.close()


if __name__ == "__main__":
    main()
