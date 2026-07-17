import argparse
import os

import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import get_euler_xyz, quat_from_euler_xyz

from utils.command_metrics import CommandVelocityMetrics
from utils.runner import get_task_class, load_config


LEG_START_INDEX = 10


def _set_dofs(env, dof_pos):
    env.dof_pos[:] = dof_pos
    env.dof_vel[:] = 0.0
    if hasattr(env, "prev_dof_pos"):
        env.prev_dof_pos[:] = env.dof_pos
    if hasattr(env, "custom_dof_vel"):
        env.custom_dof_vel[:] = 0.0
    if hasattr(env, "filtered_custom_dof_vel"):
        env.filtered_custom_dof_vel[:] = 0.0
    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))


def set_standing_state(env):
    env_ids = torch.arange(env.num_envs, device=env.device)
    env.root_states[env_ids] = env.base_init_state
    env.root_states[env_ids, 0:2] = env.env_origins[env_ids, 0:2]
    env.root_states[env_ids, 2] = env.terrain.terrain_heights(env.root_states[env_ids, 0:2]) + 0.70
    env.root_states[env_ids, 3:7] = quat_from_euler_xyz(
        torch.zeros(env.num_envs, device=env.device),
        torch.zeros(env.num_envs, device=env.device),
        torch.zeros(env.num_envs, device=env.device),
    )
    env.root_states[env_ids, 7:13] = 0.0
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
    _set_dofs(env, env.default_dof_pos.repeat(env.num_envs, 1))


def fallen_rpy(pose, count, device):
    roll = torch.zeros(count, dtype=torch.float, device=device)
    pitch = torch.zeros(count, dtype=torch.float, device=device)
    if pose == "front":
        pitch[:] = 1.45
    elif pose == "back":
        roll[:] = 3.0
    elif pose == "left":
        roll[:] = 1.45
    elif pose == "right":
        roll[:] = -1.45
    else:
        raise ValueError(f"Unknown fall pose: {pose}")
    yaw = torch.zeros(count, dtype=torch.float, device=device)
    return roll, pitch, yaw


def set_fallen_state(env, pose):
    env_ids = torch.arange(env.num_envs, device=env.device)
    roll, pitch, yaw = fallen_rpy(pose, env.num_envs, env.device)
    env.root_states[env_ids, 2] = env.terrain.terrain_heights(env.root_states[env_ids, 0:2]) + 0.25
    env.root_states[env_ids, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)
    env.root_states[env_ids, 7:13] = 0.0
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
    _set_dofs(env, env.default_dof_pos.repeat(env.num_envs, 1))


def make_walk_obs(env, walk_actions, gait_process, vx, vy, vyaw):
    gait_frequency = 1.3 if abs(vx) + abs(vy) + abs(vyaw) > 1.0e-5 else 0.0
    commands = torch.zeros(env.num_envs, 10, dtype=torch.float, device=env.device)
    commands[:, 0] = vx
    commands[:, 1] = vy
    commands[:, 2] = vyaw
    commands[:, 3] = gait_frequency
    commands_scale = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=env.device)
    leg_slice = slice(LEG_START_INDEX, LEG_START_INDEX + 12)
    return torch.cat(
        (
            env.projected_gravity,
            env.base_ang_vel,
            commands * commands_scale,
            torch.cos(2 * torch.pi * gait_process).unsqueeze(-1) * (gait_frequency > 1.0e-8),
            torch.sin(2 * torch.pi * gait_process).unsqueeze(-1) * (gait_frequency > 1.0e-8),
            env.dof_pos[:, leg_slice] - env.default_dof_pos[:, leg_slice],
            env.dof_vel[:, leg_slice] * 0.1,
            walk_actions,
        ),
        dim=-1,
    )


def make_getup_obs(env, getup_actions):
    return torch.cat(
        (
            env.projected_gravity,
            env.base_ang_vel,
            torch.zeros(env.num_envs, 3, dtype=torch.float, device=env.device),
            torch.zeros(env.num_envs, 2, dtype=torch.float, device=env.device),
            env.dof_pos - env.default_dof_pos,
            env.dof_vel * 0.1,
            getup_actions,
        ),
        dim=-1,
    )


def is_fallen(env):
    roll, pitch, _ = get_euler_xyz(env.base_quat)
    return (torch.abs(roll) > 0.75) | (torch.abs(pitch) > 0.75) | (env.projected_gravity[:, 2] > -0.55)


def is_recovered(env):
    roll, pitch, _ = get_euler_xyz(env.base_quat)
    return (
        (torch.abs(roll) < 0.25)
        & (torch.abs(pitch) < 0.25)
        & (env.projected_gravity[:, 2] < -0.92)
        & (torch.norm(env.root_states[:, 7:13], dim=-1) < 0.35)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--walk_policy", default="deploy/models/parameter_walk_k1.pt")
    parser.add_argument("--getup_policy", default="deploy/models/get_up_k1.pt")
    parser.add_argument("--task", default="K1/GetUp")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--headless", type=bool, default=False)
    parser.add_argument("--sim_device", default="cuda:0")
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--duration_s", type=float, default=30.0)
    parser.add_argument("--vx", type=float, default=0.2)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--vyaw", type=float, default=0.0)
    parser.add_argument("--force_fall_after_s", type=float, default=5.0)
    parser.add_argument("--fall_pose", choices=["front", "back", "left", "right"], default="front")
    parser.add_argument("--start_fallen", action="store_true")
    parser.add_argument("--metrics_window_s", type=float, default=3.0)
    parser.add_argument("--metrics_warmup_s", type=float, default=1.0)
    parser.add_argument("--metrics_csv", default=None)
    parser.add_argument("--metrics_csv_sample_s", type=float, default=0.05)
    args = parser.parse_args()

    if not os.path.exists(args.walk_policy):
        raise FileNotFoundError(f"Missing walk policy: {args.walk_policy}. Run export_model.py for K1/ParameterWalk first.")
    if not os.path.exists(args.getup_policy):
        raise FileNotFoundError(f"Missing getup policy: {args.getup_policy}. Run export_model.py for K1/GetUp first.")

    cfg = load_config(os.path.join("envs", f"{args.task}.yaml"))
    cfg["env"]["num_envs"] = args.num_envs
    cfg["basic"]["headless"] = args.headless
    cfg["basic"]["sim_device"] = args.sim_device
    cfg["basic"]["rl_device"] = args.rl_device
    cfg["rewards"]["episode_length_s"] = max(args.duration_s + 5.0, cfg["rewards"]["episode_length_s"])

    task_name = cfg["basic"]["task"].split("/")[-1]
    task_class = get_task_class(task_name)
    env = task_class(cfg)
    env.is_play = True
    env.reset()
    if args.start_fallen:
        set_fallen_state(env, args.fall_pose)
        mode = "getup"
    else:
        set_standing_state(env)
        mode = "walk"

    walk_policy = torch.jit.load(args.walk_policy, map_location=env.device).eval()
    getup_policy = torch.jit.load(args.getup_policy, map_location=env.device).eval()
    walk_actions = torch.zeros(env.num_envs, 12, dtype=torch.float, device=env.device)
    getup_actions = torch.zeros(env.num_envs, 22, dtype=torch.float, device=env.device)
    recovered_time = 0.0
    gait_process = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    forced_fall = args.start_fallen
    report_interval = max(1, int(1.0 / env.dt))
    mode_time = 0.0
    metrics = CommandVelocityMetrics(
        (args.vx, args.vy, args.vyaw),
        window_s=args.metrics_window_s,
        csv_path=args.metrics_csv,
        csv_sample_s=args.metrics_csv_sample_s,
    )

    for step_idx in range(int(args.duration_s / env.dt)):
        sim_time = step_idx * env.dt
        if args.force_fall_after_s >= 0.0 and (not forced_fall) and sim_time >= args.force_fall_after_s:
            set_fallen_state(env, args.fall_pose)
            mode = "getup"
            mode_time = 0.0
            metrics.reset_window()
            recovered_time = 0.0
            forced_fall = True
            print(f"[gym] forced fall at t={sim_time:.2f}s pose={args.fall_pose}")

        if mode == "walk" and bool(is_fallen(env).any().item()):
            mode = "getup"
            mode_time = 0.0
            metrics.reset_window()
            recovered_time = 0.0
            getup_actions.zero_()
            print(f"[gym] switching policy: walk -> getup at t={sim_time:.2f}s")
        elif mode == "getup":
            if bool(is_recovered(env).all().item()):
                recovered_time += env.dt
            else:
                recovered_time = 0.0
            if recovered_time >= 1.0:
                mode = "walk"
                mode_time = 0.0
                metrics.reset_window()
                walk_actions.zero_()
                gait_process.zero_()
                print(f"[gym] switching policy: getup -> walk at t={sim_time:.2f}s")

        actions = torch.zeros(env.num_envs, 22, dtype=torch.float, device=env.device)
        with torch.no_grad():
            if mode == "getup":
                getup_obs = make_getup_obs(env, getup_actions)
                getup_actions[:] = torch.clamp(getup_policy(getup_obs), -1.0, 1.0)
                actions[:] = getup_actions
            else:
                gait_frequency = 1.3 if abs(args.vx) + abs(args.vy) + abs(args.vyaw) > 1.0e-5 else 0.0
                gait_process[:] = torch.fmod(gait_process + env.dt * gait_frequency, 1.0)
                walk_obs = make_walk_obs(env, walk_actions, gait_process, args.vx, args.vy, args.vyaw)
                walk_actions[:] = torch.clamp(walk_policy(walk_obs), -1.0, 1.0)
                actions[:, LEG_START_INDEX : LEG_START_INDEX + 12] = walk_actions

        env.step(actions)
        mode_time += env.dt

        actual_velocity = (
            env.base_lin_vel[0, 0].item(),
            env.base_lin_vel[0, 1].item(),
            env.base_ang_vel[0, 2].item(),
        )
        tracked = mode == "walk" and mode_time >= args.metrics_warmup_s
        policy_command = (args.vx, args.vy, args.vyaw) if mode == "walk" else (0.0, 0.0, 0.0)
        metrics_row, metrics_summary = metrics.update(
            sim_time + env.dt,
            mode,
            actual_velocity,
            policy_command=policy_command,
            tracked=tracked,
        )

        if step_idx % report_interval == 0:
            roll, pitch, yaw = get_euler_xyz(env.base_quat)
            metrics_text = metrics.report(metrics_row, metrics_summary)
            print(
                f"[gym] t={sim_time:5.2f}s mode={mode:5s} "
                f"pos=({env.base_pos[0,0].item():+.2f},{env.base_pos[0,1].item():+.2f},{env.base_pos[0,2].item():+.2f}) "
                f"rpy=({roll[0].item():+.2f},{pitch[0].item():+.2f},{yaw[0].item():+.2f}) "
                f"{metrics_text}"
            )

    metrics.close()


if __name__ == "__main__":
    main()
