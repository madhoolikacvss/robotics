"""Generate a tag-map-based serpentine coverage route."""

import math

from .map_geometry import (
    ROBOT_SIDE,
    USABLE_WORKSPACE,
    WALL_CENTER_CLEARANCE,
)


# Leave lane overlap so modest tracking error is less likely to create gaps.
LANE_SPACING = 0.8 * ROBOT_SIDE


def generate_coverage_path(workspace=USABLE_WORKSPACE, lane_spacing=LANE_SPACING):
    """Return (x, y, yaw) waypoints for back-and-forth horizontal passes.

    The first and last passes reach the safe inset's lower and upper edges.
    The x/y limits come from workspace tag poses through map_geometry; no
    route coordinates or room dimensions are hardcoded here.
    """
    if not math.isfinite(lane_spacing) or lane_spacing <= 0.0:
        raise ValueError('Lane spacing must be positive and finite.')

    lane_count = max(1, math.ceil(
        (workspace.y_max - workspace.y_min) / lane_spacing
    ))
    actual_spacing = (workspace.y_max - workspace.y_min) / lane_count
    points = []
    for lane in range(lane_count + 1):
        y = workspace.y_min + lane * actual_spacing
        if lane % 2 == 0:
            points.extend(((workspace.x_min, y), (workspace.x_max, y)))
        else:
            points.extend(((workspace.x_max, y), (workspace.x_min, y)))

    waypoints = []
    for index, (x, y) in enumerate(points):
        if index + 1 < len(points):
            next_x, next_y = points[index + 1]
            yaw = math.atan2(next_y - y, next_x - x)
        elif index > 0:
            previous_x, previous_y = points[index - 1]
            yaw = math.atan2(y - previous_y, x - previous_x)
        else:
            yaw = 0.0
        waypoints.append((x, y, yaw))
    return waypoints


def route_length(waypoints):
    """Return total Euclidean distance along the generated waypoint route."""
    return sum(
        math.hypot(current[0] - previous[0], current[1] - previous[1])
        for previous, current in zip(waypoints, waypoints[1:])
    )


def main():
    waypoints = generate_coverage_path()
    print(
        f'Workspace bounds from tags: '
        f'x=[{USABLE_WORKSPACE.x_min:.3f}, {USABLE_WORKSPACE.x_max:.3f}] m, '
        f'y=[{USABLE_WORKSPACE.y_min:.3f}, {USABLE_WORKSPACE.y_max:.3f}] m'
    )
    print(
        f'Wall center clearance: {WALL_CENTER_CLEARANCE:.4f} m; lane spacing: '
        f'{LANE_SPACING:.3f} m; route length: {route_length(waypoints):.3f} m'
    )
    for index, (x, y, yaw) in enumerate(waypoints):
        print(f'{index:02d}: x={x:.3f} m, y={y:.3f} m, yaw={yaw:.3f} rad')


if __name__ == '__main__':
    main()
