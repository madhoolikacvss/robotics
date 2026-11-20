"""Build a weighted visibility graph without searching for a path."""

from itertools import combinations
import math

from .map_geometry import EXPANDED_OBSTACLE, GOAL, START, USABLE_WORKSPACE


def contains_point(rectangle, point):
    """Include the rectangle boundary; reject non-finite coordinates."""
    x, y = point
    return (
        math.isfinite(x) and math.isfinite(y)
        and rectangle.x_min <= x <= rectangle.x_max
        and rectangle.y_min <= y <= rectangle.y_max
    )


def crosses_interior(start, end, rectangle):
    enter, leave = 0.0, 1.0
    for a, b, low, high in (
        (start[0], end[0], rectangle.x_min, rectangle.x_max),
        (start[1], end[1], rectangle.y_min, rectangle.y_max),
    ):
        delta = b - a
        if delta == 0.0:
            if not low < a < high:
                return False
            continue
        t_low, t_high = sorted(((low - a) / delta, (high - a) / delta))
        enter = max(enter, t_low)
        leave = min(leave, t_high)
    # Ignore roundoff at an exact corner tangency (dimensionless t).
    return leave - enter > 1e-12


def segment_is_free(start, end, workspace=USABLE_WORKSPACE,
                    obstacle=EXPANDED_OBSTACLE):
    """Require workspace containment and no expanded-obstacle interior."""
    # A rectangle is convex: containing both endpoints contains the segment.
    return (
        contains_point(workspace, start)
        and contains_point(workspace, end)
        and not crosses_interior(start, end, obstacle)
    )


def build_visibility_graph(start=START, goal=GOAL,
                           workspace=USABLE_WORKSPACE,
                           obstacle=EXPANDED_OBSTACLE):
    if not (
        workspace.x_min < obstacle.x_min < obstacle.x_max < workspace.x_max
        and workspace.y_min < obstacle.y_min < obstacle.y_max < workspace.y_max
    ):
        raise ValueError('Expanded obstacle must be inside the usable workspace.')
    for label, point in (('Start', start), ('Goal', goal)):
        if not segment_is_free(point, point, workspace, obstacle):
            raise ValueError(f'{label} must be in free space.')
    nodes = {'S': start, 'G': goal}
    nodes.update(zip(('A', 'B', 'C', 'D'), obstacle.corners))
    graph = {label: {} for label in nodes}
    for a, b in combinations(nodes, 2):
        if segment_is_free(nodes[a], nodes[b], workspace, obstacle):
            distance = math.dist(nodes[a], nodes[b])
            graph[a][b] = distance
            graph[b][a] = distance
    return nodes, graph
