#!/usr/bin/env python3
"""
Plot the Dijkstra waypoint path and the robot's recorded trajectory,
subsampled to roughly one point every 0.5 seconds.

Usage:
    python3 plot_pos.py robot_positions.csv
"""

import sys
import csv
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

# Import the local ROS-independent HW4 planning package.
sys.path.insert(0, str(Path(__file__).resolve().parent / 'hw4'))
from hw4.visibility_graph import build_visibility_graph
from hw4.dijkstra import shortest_path
from hw4.path_waypoints import path_to_waypoints
from hw4.map_geometry import OBSTACLE


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


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'hw4_robot_position.csv'
    nodes, graph = build_visibility_graph()
    path, distance = shortest_path(graph)
    waypoints = np.array(path_to_waypoints(nodes, path), dtype=float)
    print(f'Dijkstra path: {" -> ".join(path)} ({distance:.3f} m)')
    ts, xs, ys = load_csv(csv_path)
    ts_s, xs_s, ys_s = subsample_by_time(ts, xs, ys, PLOT_INTERVAL)

    print(f'Loaded {len(ts)} samples; plotting {len(ts_s)} '
          f'(~{PLOT_INTERVAL}s apart)')

    fig, ax = plt.subplots(figsize=(7, 7))

    # ---- Physical obstacle measured from the tag map ----
    ax.add_patch(Rectangle(
        (OBSTACLE.x_min, OBSTACLE.y_min),
        OBSTACLE.x_max - OBSTACLE.x_min,
        OBSTACLE.y_max - OBSTACLE.y_min,
        facecolor='#475569',
        edgecolor='#334155',
        linewidth=2,
        zorder=2,
        label='Physical obstacle',
    ))

    # ---- Current Dijkstra waypoint path ----
    wpx = waypoints[:, 0]
    wpy = waypoints[:, 1]
    ax.plot(wpx, wpy, '-o', color='tab:blue', linewidth=2,
            markersize=8, label='Dijkstra path')
    for i, (x, y, _) in enumerate(waypoints):
        ax.annotate(f'WP{i}\n({x:.1f},{y:.1f})',
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
    ax.set_title('Dijkstra path and robot trajectory (0.5 s sampling)')
    ax.legend(loc='best')
    ax.set_xlim(0.0, 2.44)
    ax.set_ylim(0.0, 2.44)

    plt.tight_layout()
    plt.savefig('trajectory_plot.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()