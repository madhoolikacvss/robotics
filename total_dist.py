#!/usr/bin/env python3
"""
Compute the robot's total moving distance from a CSV of (timestamp, x, y).

Unit convention: 1 unit = 1 meter (waypoints were defined 1:1 with meters).

Usage:
    python3 total_distance.py robot_positions.csv
"""

import sys
import csv
import numpy as np

UNIT_TO_METERS = 1.0      # 1 CSV unit = 1 meter
PLOT_INTERVAL = 0.5       # seconds, for the subsampled comparison


def load_csv(path):
    ts, xs, ys = [], [], []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts.append(float(row['timestamp_sec']))
            xs.append(float(row['x']))
            ys.append(float(row['y']))
    return np.array(ts), np.array(xs), np.array(ys)


def path_length(xs, ys):
    """Sum of Euclidean distances between consecutive points."""
    if len(xs) < 2:
        return 0.0
    dx = np.diff(xs)
    dy = np.diff(ys)
    return float(np.sum(np.hypot(dx, dy)))


def subsample_by_time(ts, xs, ys, interval):
    if len(ts) == 0:
        return ts, xs, ys
    keep = [0]
    last_t = ts[0]
    for i in range(1, len(ts)):
        if ts[i] - last_t >= interval:
            keep.append(i)
            last_t = ts[i]
    if keep[-1] != len(ts) - 1:
        keep.append(len(ts) - 1)
    keep = np.array(keep, dtype=int)
    return ts[keep], xs[keep], ys[keep]


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'robot_position.csv'
    ts, xs, ys = load_csv(csv_path)

    if len(ts) < 2:
        print('Not enough samples to compute a distance.')
        return

    # ---- Raw distance (all samples) ----
    raw_dist_units = path_length(xs, ys)
    raw_dist_m = raw_dist_units * UNIT_TO_METERS

    # ---- Subsampled distance (0.5 s) ----
    ts_s, xs_s, ys_s = subsample_by_time(ts, xs, ys, PLOT_INTERVAL)
    sub_dist_units = path_length(xs_s, ys_s)
    sub_dist_m = sub_dist_units * UNIT_TO_METERS

    # ---- Run info ----
    duration = ts[-1] - ts[0]
    avg_speed = raw_dist_m / duration if duration > 0 else 0.0

    print('=' * 50)
    print(f'CSV file            : {csv_path}')
    print(f'Samples             : {len(ts)}')
    print(f'Duration            : {duration:.2f} s')
    print(f'Unit conversion     : 1 unit = {UNIT_TO_METERS} m')
    print('-' * 50)
    print(f'Raw path length     : {raw_dist_units:.4f} units '
          f'= {raw_dist_m:.4f} m')
    print(f'0.5 s subsampled    : {sub_dist_units:.4f} units '
          f'= {sub_dist_m:.4f} m')
    print(f'Average speed (raw) : {avg_speed:.4f} m/s')
    print('=' * 50)

    # ---- Optional: per-waypoint segment distances ----
    # If your waypoints are (0,0) -> (1,0) -> (1,2) -> (0,0),
    # the planned segment lengths are 1.0, 2.0, and sqrt(1^2+2^2) = 2.236.
    waypoints = np.array([
        [0.00, 0.00],
        [1.00, 0.00],
        [1.00, 2.00],
        [0.00, 0.00],
    ])
    print('\nPlanned waypoint segments (1:1 meters):')
    total_planned = 0.0
    for i in range(len(waypoints) - 1):
        d = np.linalg.norm(waypoints[i + 1] - waypoints[i])
        total_planned += d
        print(f'  WP{i} -> WP{i+1}: {d:.4f} m')
    print(f'  Planned total      : {total_planned:.4f} m')
    if total_planned > 0:
        print(f'  Actual / planned   : '
              f'{raw_dist_m / total_planned * 100:.1f}%')


if __name__ == '__main__':
    main()