"""Run: python hw4/plot_visibility_graph.py [--no-show]."""

import argparse
from pathlib import Path

from hw4.map_geometry import (
    EXPANDED_OBSTACLE, FT_TO_M, OBSTACLE, USABLE_WORKSPACE, WORKSPACE,
)
from hw4.visibility_graph import build_visibility_graph
from hw4.dijkstra import shortest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--shortest-path', action='store_true',
                        help='Run Dijkstra and highlight the shortest route.')
    args = parser.parse_args()
    import matplotlib
    if args.no_show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    nodes, graph = build_visibility_graph()
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
    first = True
    for a, neighbors in graph.items():
        for b, distance in neighbors.items():
            if a >= b:
                continue
            ax.plot([nodes[a][0] / FT_TO_M, nodes[b][0] / FT_TO_M],
                    [nodes[a][1] / FT_TO_M, nodes[b][1] / FT_TO_M],
                    color='#2563eb', linewidth=2, zorder=3,
                    label='Valid visibility edge' if first else None)
            first = False
            print(f'{a} -- {b}: {distance:.4f} m ({distance / FT_TO_M:.3f} ft)')
    if 'G' not in graph['S']:
        ax.plot([nodes['S'][0] / FT_TO_M, nodes['G'][0] / FT_TO_M],
                [nodes['S'][1] / FT_TO_M, nodes['G'][1] / FT_TO_M],
                color='#dc2626', linestyle=':', linewidth=1.5,
                label='Blocked S-G connection (excluded)', zorder=2)
    offsets = {'S': (-35, -20), 'G': (8, 10), 'A': (-22, -20),
               'B': (10, -20), 'C': (10, 10), 'D': (-22, 10)}
    for label, (x, y) in nodes.items():
        color = '#16a34a' if label == 'S' else '#9333ea' if label == 'G' else '#2563eb'
        ax.scatter(x / FT_TO_M, y / FT_TO_M, color=color, s=70, zorder=4)
        text = {'S': 'S: Start', 'G': 'G: Goal'}.get(label, label)
        ax.annotate(text, (x / FT_TO_M, y / FT_TO_M), xytext=offsets[label],
                    textcoords='offset points', fontsize=11, weight='bold')
    subtitle = 'All valid connections shown; no shortest path selected'
    if args.shortest_path:
        path, length = shortest_path(graph)
        ax.plot([nodes[label][0] / FT_TO_M for label in path],
                [nodes[label][1] / FT_TO_M for label in path],
                color='#16a34a', linewidth=4, zorder=3.5,
                label='Dijkstra shortest path')
        subtitle = f'{" -> ".join(path)} | {length:.3f} m ({length / FT_TO_M:.3f} ft)'
        print(f'Shortest path: {subtitle}')
        for label in path:
            print(f'  {label}: {nodes[label]} m')
    count = sum(map(len, graph.values())) // 2
    ax.set(title=f'Visibility graph: {len(nodes)} nodes, {count} edges\n'
                 f'{subtitle}',
           xlabel='X (ft)', ylabel='Y (ft)', aspect='equal')
    ax.set_xlim(WORKSPACE.x_min / FT_TO_M - 0.25, WORKSPACE.x_max / FT_TO_M + 0.25)
    ax.set_ylim(WORKSPACE.y_min / FT_TO_M - 0.25, WORKSPACE.y_max / FT_TO_M + 0.25)
    ax.grid(alpha=0.15)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=9)
    fig.tight_layout()
    filename = 'shortest_path.png' if args.shortest_path else 'visibility_graph.png'
    output = Path(__file__).resolve().parent / 'output' / filename
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=160)
    print(f'Saved {output}')
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
