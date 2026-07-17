import argparse
import time

import numpy as np
import torch
import yaml

from deploy.utils.policy_walk_getup_k1 import Policy


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
    args = parser.parse_args()

    import mujoco

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    torch.set_num_threads(1)
    policy = Policy(cfg)
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    default_qpos = np.array(cfg["common"]["default_qpos"], dtype=np.float32)
    stiffness = np.array(cfg["common"]["stiffness"], dtype=np.float32)
    damping = np.array(cfg["common"]["damping"], dtype=np.float32)
    torque_limit = np.array(cfg["common"]["torque_limit"], dtype=np.float32)
    if model.actuator_forcerange.shape[0] == len(torque_limit):
        torque_limit = np.minimum(torque_limit, np.abs(model.actuator_forcerange[:, 1]))

    data.qpos[0:3] = np.array([0.0, 0.0, 0.70], dtype=np.float32)
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

    while data.time < args.duration_s:
        if args.force_fall_after_s >= 0.0 and (not forced_fall) and data.time >= args.force_fall_after_s:
            set_root_pose(data, data.qpos[0:3].copy(), 0.25, fallen_rpy(args.fall_pose))
            policy.mode = "getup"
            forced_fall = True
            print(f"[mujoco] forced fall at t={data.time:.2f}s pose={args.fall_pose}")

        root_quat = np.array(data.qpos[3:7], dtype=np.float32)
        base_rpy = quat_to_euler(root_quat)
        projected_gravity = quat_to_mat(root_quat).T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
        base_ang_vel = np.array(data.qvel[3:6], dtype=np.float32)

        if data.time >= next_policy_t:
            next_policy_t += policy_dt
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

        dof_pos = np.array(data.qpos[7 : 7 + len(default_qpos)], dtype=np.float32)
        dof_vel = np.array(data.qvel[6 : 6 + len(default_qpos)], dtype=np.float32)
        torque = stiffness * (target_qpos - dof_pos) - damping * dof_vel
        data.ctrl[:] = np.clip(torque, -torque_limit, torque_limit)
        mujoco.mj_step(model, data)

        if viewer is not None:
            viewer.sync()
            time.sleep(sim_dt)

        if data.time - last_report >= 1.0:
            last_report = data.time
            xy_error = data.qpos[0:2] - start_xy
            print(
                f"[mujoco] t={data.time:5.2f}s mode={policy.mode:5s} "
                f"xy=({data.qpos[0]:+.2f},{data.qpos[1]:+.2f}) "
                f"drift_y={xy_error[1]:+.3f} rpy=({base_rpy[0]:+.2f},{base_rpy[1]:+.2f},{base_rpy[2]:+.2f})"
            )

    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
