"""Plot the generated coverage route and tag-derived workspace geometry."""

from pathlib import Path


def plot_coverage_path(waypoints):
    """Save and display the route, tag poses, physical walls, and safe inset."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from .coverage_path import route_length
    from .map_geometry import USABLE_WORKSPACE, WORKSPACE
    from .tag_map import WORKSPACE_TAGS

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.add_patch(Rectangle(
        (WORKSPACE.x_min, WORKSPACE.y_min),
        WORKSPACE.x_max - WORKSPACE.x_min,
        WORKSPACE.y_max - WORKSPACE.y_min,
        facecolor='#edf0f3', edgecolor='#334155', linewidth=2,
        label='Physical workspace from tags',
    ))
    ax.add_patch(Rectangle(
        (USABLE_WORKSPACE.x_min, USABLE_WORKSPACE.y_min),
        USABLE_WORKSPACE.x_max - USABLE_WORKSPACE.x_min,
        USABLE_WORKSPACE.y_max - USABLE_WORKSPACE.y_min,
        facecolor='white', edgecolor='#dc2626', linestyle='--',
        linewidth=1.8, label='Safe center boundary (26.55 cm inset)',
    ))

    x_values = [point[0] for point in waypoints]
    y_values = [point[1] for point in waypoints]
    ax.plot(x_values, y_values, color='#16a34a', linewidth=2.2,
            marker='o', markersize=3.5, label='Generated coverage route',
            zorder=3)
    ax.scatter(x_values[0], y_values[0], color='#2563eb', s=70,
               label='Route start', zorder=4)
    ax.scatter(x_values[-1], y_values[-1], color='#9333ea', s=70,
               label='Route end', zorder=4)

    tag_x = [tag['pose'][0] for tag in WORKSPACE_TAGS.values()]
    tag_y = [tag['pose'][1] for tag in WORKSPACE_TAGS.values()]
    ax.scatter(tag_x, tag_y, marker='s', color='#ea580c', s=38,
               label='AprilTags', zorder=5)
    for tag_id, tag in WORKSPACE_TAGS.items():
        tx, ty = tag['pose'][:2]
        ax.annotate(tag_id, (tx, ty), xytext=(4, 4),
                    textcoords='offset points', fontsize=8,
                    color='#9a3412')

    padding = 0.12
    ax.set_xlim(WORKSPACE.x_min - padding, WORKSPACE.x_max + padding)
    ax.set_ylim(WORKSPACE.y_min - padding, WORKSPACE.y_max + padding)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(
        f'HW5 coverage route: {len(waypoints)} waypoints, '
        f'{route_length(waypoints):.2f} m'
    )
    ax.grid(alpha=0.2)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
              ncol=2, fontsize=9)
    fig.tight_layout()
    output = Path.cwd() / 'hw5_coverage_path.png'
    fig.savefig(output, dpi=160)
    print(f'Saved coverage plot to {output}')
    plt.show()
    plt.close(fig)
