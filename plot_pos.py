#!/usr/bin/env python3
"""
Plot the waypoint triangle and the robot's recorded trajectory,
subsampled to roughly one point every 0.5 seconds.

Usage:
    python3 plot_trajectory.py robot_positions.csv
"""

import os
import sys
import csv
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------
# Waypoints (must match the controller exactly)
# ---------------------------------------------------------------
# WAYPOINTS = np.array([
#     [0.00, 0.00, 0.0],
#     [1.00, 0.00, 0.0],
#     [1.00, 2.00, math.pi],
#     [0.00, 0.00, 0.0],
# ], dtype=float)

WAYPOINTS = np.array([
    [0.00, 0.00, 0.0],
    [1.626, 0.00, math.pi / 2],
    [1.626, 1.626, math.pi],
    [0.00, 1.626, -math.pi/2],
    [0.00, 0.00, 0.0],
])

# WAYPOINTS = np.array([
#     [0.00, 0.00, 0.0],
#     [0.476244, 0.000000, 0.0],
#     [1.149756, 0.000000, 0.0],
#     [1.626000, 0.476244, math.pi / 4],
#     [1.626000, 1.149756, math.pi / 2],
#     [1.149756, 1.626000, 3 * math.pi / 4],
#     [0.476244, 1.626000, math.pi],
#     [0.000000, 1.149756, -3 * math.pi / 4],
#     [0.000000, 0.476244, -math.pi / 2],
#     [0.476244, 0.000000, -math.pi / 4],
# ], dtype=float)

PLOT_INTERVAL = 0.5   # seconds between plotted robot positions


def load_csv(path):
    """Read timestamp, x, y from the CSV (skips the header)."""
    ts, xs, ys = [], [], []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts.append(float(row['timestamp_sec']))
            xs.append(float(row['x']))
            ys.append(float(row['y']))
    return np.array(ts), np.array(xs), np.array(ys)


def subsample_by_time(ts, xs, ys, interval):
    """
    Keep the first sample, then keep a sample only when at least
    `interval` seconds have passed since the last kept sample.
    Always keep the last sample so the endpoint is visible.
    """
    if len(ts) == 0:
        return ts, xs, ys

    keep = [0]
    last_t = ts[0]
    for i in range(1, len(ts)):
        if ts[i] - last_t >= interval:
            keep.append(i)
            last_t = ts[i]

    # Ensure the final sample is included
    if keep[-1] != len(ts) - 1:
        keep.append(len(ts) - 1)

    keep = np.array(keep, dtype=int)
    return ts[keep], xs[keep], ys[keep]


def load_waypoint_error_summary(path):
    """Average position and heading error for each waypoint from a CSV log."""
    if not path or not os.path.exists(path):
        return {}

    waypoint_error_data = {}
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(float(row['waypoint_index']))
            except (KeyError, TypeError, ValueError):
                continue

            entry = waypoint_error_data.setdefault(
                idx,
                {'position': [], 'heading': [], 'x': [], 'y': []}
            )

            for key, column_name in [('position', 'position_error_m'),
                                     ('heading', 'heading_error_rad'),
                                     ('x', 'error_x'),
                                     ('y', 'error_y')]:
                if column_name not in row:
                    continue
                try:
                    value = float(row[column_name])
                except ValueError:
                    continue
                entry[key].append(value)

    summary = {}
    for idx, values in waypoint_error_data.items():
        summary[idx] = {
            'position': float(np.mean(values['position'])) if values['position'] else np.nan,
            'heading': float(np.mean(values['heading'])) if values['heading'] else np.nan,
            'x': float(np.mean(values['x'])) if values['x'] else np.nan,
            'y': float(np.mean(values['y'])) if values['y'] else np.nan,
        }
    return summary


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'robot_position_octogon_2.csv'
    error_csv_path = None
    if len(sys.argv) > 2:
        error_csv_path = sys.argv[2]
    elif os.path.exists('waypoint_results.csv'):
        error_csv_path = 'waypoint_results.csv'

    ts, xs, ys = load_csv(csv_path)
    ts_s, xs_s, ys_s = subsample_by_time(ts, xs, ys, PLOT_INTERVAL)
    avg_errors = load_waypoint_error_summary(error_csv_path)

    print(f'Loaded {len(ts)} samples; plotting {len(ts_s)} '
          f'(~{PLOT_INTERVAL}s apart)')

    if avg_errors:
        print('\nAverage error by waypoint:')
        for idx in sorted(avg_errors):
            err = avg_errors[idx]
            print(f'  WP{idx}: position_error={err["position"]:.4f} m, '
                  f'heading_error={err["heading"]:.4f} rad, '
                  f'error_xy=({err["x"]:.4f}, {err["y"]:.4f}) m')

    fig, ax = plt.subplots(figsize=(7, 7))

    # ---- Waypoint triangle (straight lines, closed) ----
    wpx = WAYPOINTS[:, 0]
    wpy = WAYPOINTS[:, 1]
    ax.plot(wpx, wpy, '-o', color='tab:blue', linewidth=2,
            markersize=8, label='Waypoint path')
    for i, (x, y, _) in enumerate(WAYPOINTS):
        label = f'WP{i}\n({x:.1f},{y:.1f})'
        if i in avg_errors:
            err = avg_errors[i]
            label += f'\navg err={err["position"]:.2f} m'
        ax.annotate(label,
                    xy=(x, y), xytext=(8, 8),
                    textcoords='offset points', fontsize=9,
                    color='tab:blue')

    # ---- Robot trajectory (subsampled) ----
    ax.plot(xs_s, ys_s, '-', color='tab:red', linewidth=1.2,
            alpha=0.6, label=f'Robot trajectory (~{PLOT_INTERVAL}s)')
    ax.scatter(xs_s, ys_s, color='tab:red', s=25, zorder=3,
               label=f'Robot positions (~{PLOT_INTERVAL}s)')

    # Mark start and end of the plotted run
    ax.scatter(xs_s[0], ys_s[0], marker='s', s=90, color='green',
               zorder=4, label='Start')
    ax.scatter(xs_s[-1], ys_s[-1], marker='X', s=110, color='black',
               zorder=4, label='End')

    # ---- Cosmetic ----
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Waypoint triangle and robot trajectory (0.5 s sampling)')
    ax.legend(loc='best')

    all_x = np.concatenate([wpx, xs_s])
    all_y = np.concatenate([wpy, ys_s])
    pad = 0.2
    ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
    ax.set_ylim(all_y.min() - pad, all_y.max() + pad)

    plt.tight_layout()
    plt.savefig('trajectory_plot.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()