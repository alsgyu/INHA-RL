#!/usr/bin/env python3
"""Filter K1 real robot trajectory npz files into expert replay snippets.

The deploy recorder often captures mixed runs: stable forward walking, hand-caught
recovery, and final falls in one file. This script keeps only contiguous, upright,
straight-command transitions and writes npz files with explicit next_obs so that
separate stable windows are not accidentally stitched together during training.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def _array_len(data):
    if "obs" not in data:
        raise ValueError("npz must contain obs")
    return int(data["obs"].shape[0])


def _first_dim_masked(value, indices, total_len):
    arr = np.asarray(value)
    if arr.ndim > 0 and arr.shape[0] == total_len:
        return arr[indices]
    return arr


def _command_array(data, total_len):
    if "public_command" in data:
        return np.asarray(data["public_command"], dtype=np.float32)
    if "remote_command" in data:
        return np.asarray(data["remote_command"], dtype=np.float32)
    if "command" in data:
        command = np.asarray(data["command"], dtype=np.float32)
        if command.ndim == 2 and command.shape[1] >= 3:
            return command[:, :3]
    return np.zeros((total_len, 3), dtype=np.float32)


def _stage_mask(data, total_len):
    if "stage" not in data:
        return np.ones(total_len, dtype=bool)
    stage = np.asarray(data["stage"])
    return stage.astype(str) == "rl"


def _finite_mask(*arrays):
    mask = np.ones(arrays[0].shape[0], dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array).all(axis=tuple(range(1, array.ndim))) if array.ndim > 1 else np.isfinite(array)
    return mask


def _contiguous_transition_runs(transition_mask):
    true_idx = np.flatnonzero(transition_mask)
    if true_idx.size == 0:
        return []
    runs = []
    start = int(true_idx[0])
    prev = int(true_idx[0])
    for idx in true_idx[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev))
        start = prev = idx
    runs.append((start, prev))
    return runs


def _select_expert_transition_indices(data, args):
    n = _array_len(data)
    if n < 2:
        return np.zeros(0, dtype=np.int64), []

    obs = np.asarray(data["obs"], dtype=np.float32)
    action = np.asarray(data["action"] if "action" in data else data["actions"], dtype=np.float32)
    command = _command_array(data, n)
    time_s = np.asarray(data["time_s"], dtype=np.float64) if "time_s" in data else np.arange(n, dtype=np.float64) * args.default_dt
    projected_gravity = np.asarray(data["projected_gravity"], dtype=np.float32) if "projected_gravity" in data else obs[:, :3]
    base_rpy = np.asarray(data["base_rpy"], dtype=np.float32) if "base_rpy" in data else np.zeros((n, 3), dtype=np.float32)
    base_ang_vel = np.asarray(data["base_ang_vel"], dtype=np.float32) if "base_ang_vel" in data else obs[:, 3:6]
    motion_alpha = np.asarray(data["motion_start_alpha"], dtype=np.float32) if "motion_start_alpha" in data else np.ones(n, dtype=np.float32)
    cmd_alpha = np.asarray(data["motion_command_alpha"], dtype=np.float32) if "motion_command_alpha" in data else np.ones(n, dtype=np.float32)
    gait_frequency = np.asarray(data["gait_frequency"], dtype=np.float32) if "gait_frequency" in data else np.ones(n, dtype=np.float32)
    command_age = np.asarray(data["command_age"], dtype=np.float32) if "command_age" in data else np.full(n, args.min_command_age_s, dtype=np.float32)
    heading_error = np.asarray(data["heading_error_yaw"], dtype=np.float32) if "heading_error_yaw" in data else np.zeros(n, dtype=np.float32)

    leg_error = np.zeros(n, dtype=np.float32)
    if "filtered_dof_target" in data and "dof_pos" in data:
        leg_error = np.max(np.abs(np.asarray(data["filtered_dof_target"]) - np.asarray(data["dof_pos"])), axis=1)
    elif "dof_target" in data and "dof_pos" in data:
        leg_error = np.max(np.abs(np.asarray(data["dof_target"]) - np.asarray(data["dof_pos"])), axis=1)

    command_norm = np.linalg.norm(command[:, :3], axis=1)
    raw_action = np.asarray(data["raw_action"], dtype=np.float32) if "raw_action" in data else action
    yaw_unwrapped = np.unwrap(base_rpy[:, 2].astype(np.float64))
    yaw_drift = yaw_unwrapped - yaw_unwrapped[0]

    stable = _stage_mask(data, n)
    stable &= motion_alpha >= args.min_motion_alpha
    stable &= cmd_alpha >= args.min_command_alpha
    stable &= command_age >= args.min_command_age_s
    stable &= command_norm >= args.min_command_norm
    stable &= command[:, 0] >= args.min_vx
    stable &= command[:, 0] <= args.max_vx
    stable &= np.abs(command[:, 1]) <= args.max_abs_vy_command
    stable &= np.abs(command[:, 2]) <= args.max_abs_yaw_command
    stable &= gait_frequency >= args.min_gait_frequency
    stable &= np.abs(projected_gravity[:, 0]) <= args.max_abs_gravity_x
    stable &= np.abs(projected_gravity[:, 1]) <= args.max_abs_gravity_y
    stable &= np.abs(base_rpy[:, 0]) <= args.max_abs_roll
    stable &= np.abs(base_rpy[:, 1]) <= args.max_abs_pitch
    stable &= np.abs(base_ang_vel[:, 0]) <= args.max_abs_roll_rate
    stable &= np.abs(base_ang_vel[:, 1]) <= args.max_abs_pitch_rate
    stable &= np.abs(base_ang_vel[:, 2]) <= args.max_abs_yaw_rate
    stable &= np.abs(heading_error) <= args.max_abs_heading_error
    stable &= np.abs(yaw_drift) <= args.max_abs_yaw_drift
    stable &= np.max(np.abs(action), axis=1) <= args.max_abs_action
    stable &= np.max(np.abs(raw_action), axis=1) <= args.max_abs_raw_action
    stable &= leg_error <= args.max_abs_leg_error
    stable &= _finite_mask(obs, action, command, projected_gravity, base_rpy, base_ang_vel)

    dt = np.diff(time_s)
    positive_dt = dt[dt > 1.0e-6]
    median_dt = float(np.median(positive_dt)) if positive_dt.size else args.default_dt
    max_dt = max(args.max_dt_s, median_dt * args.max_dt_multiplier)
    consecutive = (dt > 0.0) & (dt <= max_dt)
    transition_mask = stable[:-1] & stable[1:] & consecutive

    trim_steps = max(0, int(round(args.trim_s / max(median_dt, 1.0e-6))))
    min_steps = max(1, int(round(args.min_segment_s / max(median_dt, 1.0e-6))))
    selected = []
    kept_runs = []
    for start, end in _contiguous_transition_runs(transition_mask):
        trimmed_start = start + trim_steps
        trimmed_end = end - trim_steps
        if trimmed_end < trimmed_start:
            continue
        if trimmed_end - trimmed_start + 1 < min_steps:
            continue
        selected.extend(range(trimmed_start, trimmed_end + 1))
        kept_runs.append((trimmed_start, trimmed_end))

    return np.asarray(selected, dtype=np.int64), kept_runs


def _write_filtered_npz(path, data, indices, runs, args):
    total_len = _array_len(data)
    out = {}
    for key in data.files:
        if key == "metadata_json":
            out[key] = np.asarray(data[key])
        else:
            out[key] = _first_dim_masked(data[key], indices, total_len)
    out["obs"] = np.asarray(data["obs"], dtype=np.float32)[indices]
    action_key = "action" if "action" in data.files else "actions"
    out["action"] = np.asarray(data[action_key], dtype=np.float32)[indices]
    out["next_obs"] = np.asarray(data["obs"], dtype=np.float32)[indices + 1]
    out["done"] = np.zeros(indices.shape[0], dtype=np.float32)
    if "command" not in out:
        out["command"] = _command_array(data, total_len)[indices]
    meta = {
        "source": str(path),
        "filter": "k1_real_expert",
        "selected_transitions": int(indices.shape[0]),
        "runs": [(int(a), int(b)) for a, b in runs],
        "args": vars(args),
    }
    out["filter_metadata_json"] = np.asarray(json.dumps(meta, sort_keys=True))

    output_dir = Path(args.out_dir) if args.out_dir else Path(path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(path).stem}{args.suffix}.npz"
    np.savez_compressed(output_path, **out)
    return output_path


def _summarize(path, data, indices, runs):
    n = _array_len(data)
    command = _command_array(data, n)
    gx = np.asarray(data["projected_gravity"] if "projected_gravity" in data else data["obs"][:, :3])[:, 0]
    gy = np.asarray(data["projected_gravity"] if "projected_gravity" in data else data["obs"][:, :3])[:, 1]
    rpy = np.asarray(data["base_rpy"], dtype=np.float32) if "base_rpy" in data else np.zeros((n, 3), dtype=np.float32)
    prefix = f"[real-filter] {path}"
    print(
        f"{prefix}: total={n} selected={indices.shape[0]} runs={len(runs)} "
        f"vx=({float(np.min(command[:,0])):+.2f},{float(np.max(command[:,0])):+.2f}) "
        f"cmd_y_abs_max={float(np.max(np.abs(command[:,1]))):.2f} "
        f"cmd_yaw_abs_max={float(np.max(np.abs(command[:,2]))):.2f} "
        f"grav_x_abs_max={float(np.max(np.abs(gx))):.3f} "
        f"grav_y_abs_max={float(np.max(np.abs(gy))):.3f} "
        f"rpy_pitch_abs_max={float(np.max(np.abs(rpy[:,1]))):.3f}"
    )
    if runs:
        print(f"{prefix}: kept_runs={runs[:8]}{' ...' if len(runs) > 8 else ''}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Real trajectory npz files.")
    parser.add_argument("--out_dir", default=None, help="Output directory. Defaults to each input file directory.")
    parser.add_argument("--suffix", default=".expert", help="Suffix appended before .npz.")
    parser.add_argument("--min_vx", type=float, default=0.08)
    parser.add_argument("--max_vx", type=float, default=0.65)
    parser.add_argument("--min_command_norm", type=float, default=0.07)
    parser.add_argument("--max_abs_vy_command", type=float, default=0.05)
    parser.add_argument("--max_abs_yaw_command", type=float, default=0.08)
    parser.add_argument("--min_motion_alpha", type=float, default=0.98)
    parser.add_argument("--min_command_alpha", type=float, default=0.98)
    parser.add_argument("--min_command_age_s", type=float, default=0.35)
    parser.add_argument("--min_gait_frequency", type=float, default=1.2)
    parser.add_argument("--max_abs_gravity_x", type=float, default=0.16)
    parser.add_argument("--max_abs_gravity_y", type=float, default=0.12)
    parser.add_argument("--max_abs_roll", type=float, default=0.18)
    parser.add_argument("--max_abs_pitch", type=float, default=0.24)
    parser.add_argument("--max_abs_roll_rate", type=float, default=1.4)
    parser.add_argument("--max_abs_pitch_rate", type=float, default=1.4)
    parser.add_argument("--max_abs_yaw_rate", type=float, default=1.6)
    parser.add_argument("--max_abs_heading_error", type=float, default=0.35)
    parser.add_argument("--max_abs_yaw_drift", type=float, default=0.70)
    parser.add_argument("--max_abs_action", type=float, default=0.72)
    parser.add_argument("--max_abs_raw_action", type=float, default=0.95)
    parser.add_argument("--max_abs_leg_error", type=float, default=0.22)
    parser.add_argument("--min_segment_s", type=float, default=0.80)
    parser.add_argument("--trim_s", type=float, default=0.12)
    parser.add_argument("--default_dt", type=float, default=0.02)
    parser.add_argument("--max_dt_s", type=float, default=0.08)
    parser.add_argument("--max_dt_multiplier", type=float, default=4.0)
    args = parser.parse_args()

    outputs = []
    for text_path in args.paths:
        path = Path(text_path)
        with np.load(path, allow_pickle=False) as data:
            indices, runs = _select_expert_transition_indices(data, args)
            _summarize(path, data, indices, runs)
            if indices.size == 0:
                print(f"[real-filter] {path}: no expert output written")
                continue
            output_path = _write_filtered_npz(path, data, indices, runs, args)
            outputs.append(str(output_path))
            print(f"[real-filter] wrote {output_path}")
    if outputs:
        print("[real-filter] expert_replay_path=" + ",".join(outputs))


if __name__ == "__main__":
    main()
