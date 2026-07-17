import argparse
import time

import numpy as np
import torch
import yaml

from utils.command_metrics import CommandVelocityMetrics
from deploy.utils.policy_walk_getup_k1 import Policy


LEG_START_INDEX = 10


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
    return current_pos, current_yaw, np.array([body_vx, body_vy, body_vyaw], dtype=np.float64)


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
    parser.add_argument("--duration_s", type=float, default=30.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--vx", type=float, default=0.2)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--vyaw", type=float, default=0.0)
    parser.add_argument("--force_fall_after_s", type=float, default=-1.0)
    parser.add_argument("--fall_pose", choices=["front", "back", "left", "right"], default="front")
    parser.add_argument("--start_fallen", action="store_true")
    parser.add_argument("--enable_getup", action="store_true")
    parser.add_argument("--walk_only", action="store_true")
    parser.add_argument("--default_hip_pitch", type=float)
    parser.add_argument("--default_knee_pitch", type=float)
    parser.add_argument("--default_ankle_pitch", type=float)
    parser.add_argument("--default_base_height", type=float, default=0.70)
    parser.add_argument("--metrics_window_s", type=float, default=3.0)
    parser.add_argument("--metrics_warmup_s", type=float, default=1.0)
    parser.add_argument("--metrics_csv", default=None)
    parser.add_argument("--metrics_csv_sample_s", type=float, default=0.05)
    args = parser.parse_args()

    import mujoco

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    apply_default_pose_overrides(cfg, args)

    torch.set_num_threads(1)
    enable_getup = (not args.walk_only) and (args.enable_getup or args.start_fallen or args.force_fall_after_s >= 0.0)
    policy = Policy(cfg, enable_getup=enable_getup)
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    default_qpos = np.array(cfg["common"]["default_qpos"], dtype=np.float32)
    stiffness = np.array(cfg["common"]["stiffness"], dtype=np.float32)
    damping = np.array(cfg["common"]["damping"], dtype=np.float32)
    torque_limit = np.array(cfg["common"]["torque_limit"], dtype=np.float32)
    if model.actuator_forcerange.shape[0] == len(torque_limit):
        torque_limit = np.minimum(torque_limit, np.abs(model.actuator_forcerange[:, 1]))

    data.qpos[0:3] = np.array([0.0, 0.0, args.default_base_height], dtype=np.float32)
    data.qpos[3:7] = quat_from_euler(0.0, 0.0, 0.0)
    data.qpos[7 : 7 + len(default_qpos)] = default_qpos
    data.qvel[:] = 0.0
    if args.start_fallen:
        set_root_pose(data, np.array([0.0, 0.0, 0.25], dtype=np.float32), 0.25, fallen_rpy(args.fall_pose))
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
    last_report = -1.0
    mode_time = 0.0
    metric_pos = np.array(data.qpos[0:3], dtype=np.float64)
    metric_yaw = float(quat_to_euler(np.array(data.qpos[3:7], dtype=np.float32))[2])
    actual_velocity = np.zeros(3, dtype=np.float64)
    metrics = CommandVelocityMetrics(
        (args.vx, args.vy, args.vyaw),
        window_s=args.metrics_window_s,
        csv_path=args.metrics_csv,
        csv_sample_s=args.metrics_csv_sample_s,
    )
    print(f"[mujoco] walk command vx={args.vx:.3f} vy={args.vy:.3f} vyaw={args.vyaw:.3f} getup_enabled={enable_getup}")
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
            metrics.reset_window()
            print(f"[mujoco] forced fall at t={data.time:.2f}s pose={args.fall_pose}")

        root_quat = np.array(data.qpos[3:7], dtype=np.float32)
        base_rpy = quat_to_euler(root_quat)
        projected_gravity = quat_to_mat(root_quat).T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
        base_ang_vel = np.array(data.qvel[3:6], dtype=np.float32)

        if data.time >= next_policy_t:
            next_policy_t += policy_dt
            previous_mode = policy.mode
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
            if policy.mode != previous_mode:
                mode_time = 0.0
                metrics.reset_window()

        dof_pos = np.array(data.qpos[7 : 7 + len(default_qpos)], dtype=np.float32)
        dof_vel = np.array(data.qvel[6 : 6 + len(default_qpos)], dtype=np.float32)
        torque = stiffness * (target_qpos - dof_pos) - damping * dof_vel
        data.ctrl[:] = np.clip(torque, -torque_limit, torque_limit)
        mujoco.mj_step(model, data)
        metric_pos, metric_yaw, actual_velocity = estimate_body_velocity(metric_pos, metric_yaw, data, sim_dt)
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
            metrics_text = metrics.report(metrics_row, metrics_summary)
            print(
                f"[mujoco] t={data.time:5.2f}s mode={policy.mode:5s} "
                f"xy=({data.qpos[0]:+.2f},{data.qpos[1]:+.2f}) "
                f"drift_y={xy_error[1]:+.3f} rpy=({base_rpy[0]:+.2f},{base_rpy[1]:+.2f},{base_rpy[2]:+.2f}) "
                f"{metrics_text}"
            )

    if viewer is not None:
        viewer.close()
    metrics.close()


if __name__ == "__main__":
    main()
