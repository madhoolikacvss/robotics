"""Run: python hw4/plot_rrt.py [--seed N] [--no-show]."""

import argparse
from pathlib import Path

from hw4.map_geometry import (
    EXPANDED_OBSTACLE, FT_TO_M, OBSTACLE, USABLE_WORKSPACE, WORKSPACE,
)
from hw4.rrt import path_length, plan_rrt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for reproducible planning')
    parser.add_argument('--no-show', action='store_true',
                        help='save the plot without opening a window')
    args = parser.parse_args()

    import matplotlib
    if args.no_show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    path = plan_rrt(seed=args.seed)
    fig, ax = plt.subplots(figsize=(9, 9))

    def draw_rectangle(bounds, **kwargs):
        ax.add_patch(Rectangle(
            (bounds.x_min / FT_TO_M, bounds.y_min / FT_TO_M),
            (bounds.x_max - bounds.x_min) / FT_TO_M,
            (bounds.y_max - bounds.y_min) / FT_TO_M, **kwargs,
        ))

    draw_rectangle(WORKSPACE, facecolor='#edf0f3', edgecolor='#334155',
                   linewidth=2, label='Physical workspace')
    draw_rectangle(USABLE_WORKSPACE, facecolor='white', edgecolor='#64748b',
                   linestyle='--', linewidth=1.5, label='Usable center workspace')
    draw_rectangle(EXPANDED_OBSTACLE, facecolor='#ffedd5', edgecolor='#ea580c',
                   linestyle='--', linewidth=1.5, label='Expanded obstacle')
    draw_rectangle(OBSTACLE, facecolor='#475569', edgecolor='#334155',
                   label='Physical obstacle')

    x_coords = [point[0] / FT_TO_M for point in path]
    y_coords = [point[1] / FT_TO_M for point in path]
    ax.plot(x_coords, y_coords, color='#16a34a', linewidth=3,
            marker='o', markersize=3, label='RRT path', zorder=3)
    ax.scatter(x_coords[0], y_coords[0], color='#16a34a', s=70,
               label='Start', zorder=4)
    ax.scatter(x_coords[-1], y_coords[-1], color='#9333ea', s=70,
               label='Goal', zorder=4)

    ax.set(title=f'RRT path: {len(path)} points, {path_length(path):.3f} m',
           xlabel='X (ft)', ylabel='Y (ft)', aspect='equal')
    ax.set_xlim(WORKSPACE.x_min / FT_TO_M - 0.25,
                WORKSPACE.x_max / FT_TO_M + 0.25)
    ax.set_ylim(WORKSPACE.y_min / FT_TO_M - 0.25,
                WORKSPACE.y_max / FT_TO_M + 0.25)
    ax.grid(alpha=0.15)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2,
              fontsize=9)
    fig.tight_layout()
    output = Path(__file__).resolve().parent / 'output' / 'rrt_path.png'
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=160)
    print(f'Saved {output}')
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
