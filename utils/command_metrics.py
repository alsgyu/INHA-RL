import csv
import os
from collections import deque

import numpy as np


def _fmt_vec(values):
    return f"({values[0]:+.3f},{values[1]:+.3f},{values[2]:+.3f})"


class CommandVelocityMetrics:
    HEADERS = (
        "time_s",
        "mode",
        "tracked",
        "target_vx",
        "target_vy",
        "target_vyaw",
        "policy_vx",
        "policy_vy",
        "policy_vyaw",
        "actual_vx",
        "actual_vy",
        "actual_vyaw",
        "err_vx",
        "err_vy",
        "err_vyaw",
        "abs_err_vx",
        "abs_err_vy",
        "abs_err_vyaw",
        "window_actual_vx",
        "window_actual_vy",
        "window_actual_vyaw",
        "window_mae_vx",
        "window_mae_vy",
        "window_mae_vyaw",
        "total_mae_vx",
        "total_mae_vy",
        "total_mae_vyaw",
    )

    def __init__(self, target_command, window_s=3.0, csv_path=None, csv_sample_s=0.05):
        self.target_command = np.asarray(target_command, dtype=np.float64)
        self.window_s = max(float(window_s), 1.0e-6)
        self.csv_sample_s = max(float(csv_sample_s), 0.0)
        self.window_rows = deque()
        self.total_count = 0
        self.total_abs_error = np.zeros(3, dtype=np.float64)
        self._next_csv_time = -np.inf
        self._csv_file = None
        self._csv_writer = None

        if csv_path:
            csv_dir = os.path.dirname(os.path.abspath(csv_path))
            os.makedirs(csv_dir, exist_ok=True)
            self._csv_file = open(csv_path, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.HEADERS)
            self._csv_writer.writeheader()

    def close(self):
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    def reset_window(self):
        self.window_rows.clear()

    def update(self, time_s, mode, actual_velocity, policy_command=None, tracked=True):
        actual = np.asarray(actual_velocity, dtype=np.float64)
        policy = self.target_command if policy_command is None else np.asarray(policy_command, dtype=np.float64)
        error = actual - policy
        abs_error = np.abs(error)

        if tracked:
            self.window_rows.append((float(time_s), actual.copy(), abs_error.copy()))
            while self.window_rows and float(time_s) - self.window_rows[0][0] > self.window_s:
                self.window_rows.popleft()
            self.total_count += 1
            self.total_abs_error += abs_error

        summary = self.summary()
        row = self._make_row(time_s, mode, tracked, policy, actual, error, abs_error, summary)
        if self._csv_writer is not None and float(time_s) + 1.0e-9 >= self._next_csv_time:
            self._csv_writer.writerow(row)
            self._next_csv_time = float(time_s) + self.csv_sample_s
        return row, summary

    def summary(self):
        if not self.window_rows:
            return None

        actual = np.stack([row[1] for row in self.window_rows], axis=0)
        abs_error = np.stack([row[2] for row in self.window_rows], axis=0)
        total_mae = self.total_abs_error / max(self.total_count, 1)
        return {
            "count": len(self.window_rows),
            "actual_mean": actual.mean(axis=0),
            "mae": abs_error.mean(axis=0),
            "total_count": self.total_count,
            "total_mae": total_mae,
        }

    def report(self, row, summary):
        target = np.array([row["target_vx"], row["target_vy"], row["target_vyaw"]], dtype=np.float64)
        policy = np.array([row["policy_vx"], row["policy_vy"], row["policy_vyaw"]], dtype=np.float64)
        actual = np.array([row["actual_vx"], row["actual_vy"], row["actual_vyaw"]], dtype=np.float64)
        error = np.array([row["err_vx"], row["err_vy"], row["err_vyaw"]], dtype=np.float64)

        parts = [f"cmd={_fmt_vec(target)}"]
        if np.max(np.abs(policy - target)) > 1.0e-4:
            parts.append(f"policy_cmd={_fmt_vec(policy)}")
        parts.append(f"actual={_fmt_vec(actual)}")
        parts.append(f"err={_fmt_vec(error)}")

        if summary is None:
            parts.append("mae=warming")
        else:
            parts.append(f"avg{self.window_s:.1f}s={_fmt_vec(summary['actual_mean'])}")
            parts.append(f"mae{self.window_s:.1f}s={_fmt_vec(summary['mae'])}")
        return " ".join(parts)

    def _make_row(self, time_s, mode, tracked, policy, actual, error, abs_error, summary):
        row = {
            "time_s": float(time_s),
            "mode": mode,
            "tracked": int(bool(tracked)),
            "target_vx": float(self.target_command[0]),
            "target_vy": float(self.target_command[1]),
            "target_vyaw": float(self.target_command[2]),
            "policy_vx": float(policy[0]),
            "policy_vy": float(policy[1]),
            "policy_vyaw": float(policy[2]),
            "actual_vx": float(actual[0]),
            "actual_vy": float(actual[1]),
            "actual_vyaw": float(actual[2]),
            "err_vx": float(error[0]),
            "err_vy": float(error[1]),
            "err_vyaw": float(error[2]),
            "abs_err_vx": float(abs_error[0]),
            "abs_err_vy": float(abs_error[1]),
            "abs_err_vyaw": float(abs_error[2]),
        }

        if summary is None:
            for key in self.HEADERS:
                row.setdefault(key, "")
            return row

        row.update(
            {
                "window_actual_vx": float(summary["actual_mean"][0]),
                "window_actual_vy": float(summary["actual_mean"][1]),
                "window_actual_vyaw": float(summary["actual_mean"][2]),
                "window_mae_vx": float(summary["mae"][0]),
                "window_mae_vy": float(summary["mae"][1]),
                "window_mae_vyaw": float(summary["mae"][2]),
                "total_mae_vx": float(summary["total_mae"][0]),
                "total_mae_vy": float(summary["total_mae"][1]),
                "total_mae_vyaw": float(summary["total_mae"][2]),
            }
        )
        return row
