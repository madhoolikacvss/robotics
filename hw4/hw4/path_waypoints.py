"""Convert a planned route into controller waypoints."""

import math


def path_to_waypoints(nodes, path):
    """Return (x meters, y meters, yaw radians) for each route node.
    """
    if len(path) < 2:
        raise ValueError('A route needs at least two nodes to define a heading.')

    waypoints = []
    for current, following in zip(path, path[1:]):
        x, y = nodes[current]
        next_x, next_y = nodes[following]
        yaw = math.atan2(next_y - y, next_x - x)
        waypoints.append((x, y, yaw))

    goal_x, goal_y = nodes[path[-1]]
    waypoints.append((goal_x, goal_y, waypoints[-1][2]))
    return tuple(waypoints)
