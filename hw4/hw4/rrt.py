"""Plan a collision-free route with a clearance-aware path cost."""

import argparse
import math
import random
from typing import List, Optional, Set, Tuple

from .map_geometry import (
    EXPANDED_OBSTACLE, GOAL, START, USABLE_WORKSPACE, Rectangle,
)
from .visibility_graph import segment_is_free


Point = Tuple[float, float]
CLEARANCE_SAMPLE_SPACING = 0.02


def point_clearance(point: Point,
                    obstacle: Rectangle = EXPANDED_OBSTACLE,
                    workspace: Rectangle = USABLE_WORKSPACE) -> float:
    """Return the distance from a point to the nearest obstacle or wall."""
    x, y = point
    dx = max(obstacle.x_min - x, 0.0, x - obstacle.x_max)
    dy = max(obstacle.y_min - y, 0.0, y - obstacle.y_max)
    obstacle_clearance = math.hypot(dx, dy)
    wall_clearance = min(
        x - workspace.x_min,
        workspace.x_max - x,
        y - workspace.y_min,
        workspace.y_max - y,
    )
    return min(obstacle_clearance, wall_clearance)


def edge_clearance(start: Point, end: Point,
                   obstacle: Rectangle = EXPANDED_OBSTACLE,
                   workspace: Rectangle = USABLE_WORKSPACE) -> float:
    """Approximate the minimum boundary clearance along an edge."""
    length = math.dist(start, end)
    if length == 0.0:
        return point_clearance(start, obstacle, workspace)

    intervals = max(1, math.ceil(length / CLEARANCE_SAMPLE_SPACING))
    minimum = math.inf
    for index in range(intervals + 1):
        fraction = index / intervals
        x = start[0] + fraction * (end[0] - start[0])
        y = start[1] + fraction * (end[1] - start[1])
        minimum = min(minimum, point_clearance(
            (x, y), obstacle, workspace
        ))
    return minimum


def edge_clearance_cost(start: Point, end: Point,
                        obstacle: Rectangle = EXPANDED_OBSTACLE,
                        workspace: Rectangle = USABLE_WORKSPACE) -> float:
    """Backward-compatible name for minimum edge clearance."""
    return edge_clearance(start, end, obstacle, workspace)


def path_minimum_clearance(path: Tuple[Point, ...],
                           obstacle: Rectangle = EXPANDED_OBSTACLE,
                           workspace: Rectangle = USABLE_WORKSPACE) -> float:
    """Return the minimum clearance at every point on a sampled path."""
    minimum = math.inf
    for point in path:
        minimum = min(minimum, point_clearance(point, obstacle, workspace))
    for first, second in zip(path, path[1:]):
        minimum = min(
            minimum,
            edge_clearance(first, second, obstacle, workspace),
        )
    return minimum


def plan_rrt(start: Point = START, goal: Point = GOAL,
             workspace: Rectangle = USABLE_WORKSPACE,
             obstacle: Rectangle = EXPANDED_OBSTACLE,
             step_size: float = 0.15, goal_sample_rate: float = 0.1,
             lambda_clearance: float = 1, max_iterations: int = 1000,
             seed: Optional[int] = None) -> Tuple[Point, ...]:
    """Return an RRT route in meters, including both endpoints.

    The tree maximizes the minimum boundary clearance along each route.
    Path length is used only to break ties between routes with the same
    minimum clearance. The obstacle and workspace walls remain hard collision
    boundaries.

    ``lambda_clearance`` is retained for caller compatibility but is ignored;
    clearance is not combined with edge length as a weighted cost.
    """
    for label, point in (('Start', start), ('Goal', goal)):
        if len(point) != 2 or not all(math.isfinite(value) for value in point):
            raise ValueError(f'{label} must contain two finite coordinates.')
        if not segment_is_free(point, point, workspace, obstacle):
            raise ValueError(f'{label} must be in free space.')
    if not math.isfinite(step_size) or step_size <= 0.0:
        raise ValueError('Step size must be finite and positive.')
    if not math.isfinite(goal_sample_rate) or not 0.0 <= goal_sample_rate <= 1.0:
        raise ValueError('Goal sample rate must be between 0 and 1.')
    if not math.isfinite(lambda_clearance) or lambda_clearance < 0.0:
        raise ValueError('Clearance weight must be finite and non-negative.')
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) \
            or max_iterations <= 0:
        raise ValueError('Maximum iterations must be a positive integer.')

    rng = random.Random(seed)
    points: List[Point] = [start]
    parents: List[Optional[int]] = [None]
    children: List[Set[int]] = [set()]
    costs: List[Tuple[float, float]] = [(point_clearance(
        start, obstacle, workspace
    ), 0.0)]
    neighbor_radius = 2.0 * step_size
    goal_index: Optional[int] = None
    best_path: Optional[Tuple[Point, ...]] = None
    best_objective = (-math.inf, -math.inf)

    for _ in range(max_iterations):
        if rng.random() < goal_sample_rate:
            sample = goal
        else:
            sample = (
                rng.uniform(workspace.x_min, workspace.x_max),
                rng.uniform(workspace.y_min, workspace.y_max),
            )

        nearest_index = min(
            range(len(points)),
            key=lambda index: (
                (points[index][0] - sample[0]) ** 2
                + (points[index][1] - sample[1]) ** 2
            ),
        )
        nearest = points[nearest_index]
        dx, dy = sample[0] - nearest[0], sample[1] - nearest[1]
        distance = math.hypot(dx, dy)
        if distance <= 1e-12:
            continue

        travel = min(step_size, distance)
        candidate = (
            nearest[0] + dx * travel / distance,
            nearest[1] + dy * travel / distance,
        )
        if not segment_is_free(nearest, candidate, workspace, obstacle):
            continue

        near_indices = [
            index for index, point in enumerate(points)
            if math.dist(point, candidate) <= neighbor_radius
        ]
        parent_index = None
        candidate_cost: Optional[Tuple[float, float]] = None
        for index in near_indices:
            if not segment_is_free(points[index], candidate, workspace, obstacle):
                continue
            edge_length = math.dist(points[index], candidate)
            cost = (
                min(costs[index][0], edge_clearance(
                    points[index], candidate, obstacle, workspace
                )),
                costs[index][1] + edge_length,
            )
            if candidate_cost is None or _better_bottleneck_cost(
                    cost, candidate_cost):
                parent_index, candidate_cost = index, cost

        if parent_index is None:
            continue

        points.append(candidate)
        parents.append(parent_index)
        children.append(set())
        children[parent_index].add(len(points) - 1)
        costs.append(candidate_cost)
        candidate_index = len(points) - 1

        for index in near_indices:
            if index == parent_index or _is_ancestor(index, candidate_index, parents):
                continue
            if not segment_is_free(candidate, points[index], workspace, obstacle):
                continue
            edge_length = math.dist(candidate, points[index])
            cost = (
                min(candidate_cost[0], edge_clearance(
                    candidate, points[index], obstacle, workspace
                )),
                candidate_cost[1] + edge_length,
            )
            if not _better_bottleneck_cost(cost, costs[index]):
                continue

            old_parent = parents[index]
            if old_parent is not None:
                children[old_parent].remove(index)
            parents[index] = candidate_index
            children[candidate_index].add(index)
            costs[index] = cost
            _propagate_bottleneck_costs(
                index, points, parents, children, costs, obstacle, workspace
            )

        goal_distance = math.dist(candidate, goal)
        if goal_distance <= 1e-12:
            goal_index = candidate_index
        elif goal_distance <= step_size \
                and segment_is_free(candidate, goal, workspace, obstacle):
            goal_index = len(points)
            points.append(goal)
            parents.append(candidate_index)
            children.append(set())
            children[candidate_index].add(goal_index)
            costs.append((
                min(costs[candidate_index][0], edge_clearance(
                    candidate, goal, obstacle, workspace
                )),
                costs[candidate_index][1] + goal_distance,
            ))

        if goal_index is not None:
            path = _trace_path(points, parents, goal_index)
            objective_clearance = _path_objective(
                path, lambda_clearance, obstacle, workspace
            )
            objective_length = -path_length(path)
            if objective_clearance > best_objective[0] + 1e-12 or (
                abs(objective_clearance - best_objective[0]) <= 1e-12
                and objective_length > best_objective[1]
            ):
                best_objective = (objective_clearance, objective_length)
                best_path = path

    if best_path is not None:
        return best_path

    raise RuntimeError(
        f'RRT did not reach the goal within {max_iterations} iterations.'
    )


def _is_ancestor(ancestor: int, node: int,
                 parents: List[Optional[int]]) -> bool:
    while node is not None:
        if node == ancestor:
            return True
        node = parents[node]
    return False


def _better_bottleneck_cost(candidate: Tuple[float, float],
                            incumbent: Tuple[float, float]) -> bool:
    """Compare bottleneck first, then shorter path length."""
    if candidate[0] != incumbent[0]:
        return candidate[0] > incumbent[0]
    return candidate[1] + 1e-12 < incumbent[1]


def _propagate_bottleneck_costs(root: int, points: List[Point],
                                parents: List[Optional[int]],
                                children: List[Set[int]],
                                costs: List[Tuple[float, float]],
                                obstacle: Rectangle,
                                workspace: Rectangle) -> None:
    """Refresh cost tuples for a rewired node and every descendant."""
    descendants = list(children[root])
    while descendants:
        descendant = descendants.pop()
        parent = parents[descendant]
        if parent is None:
            continue
        edge_length = math.dist(points[parent], points[descendant])
        clearance = edge_clearance(
            points[parent], points[descendant], obstacle, workspace
        )
        costs[descendant] = (
            min(costs[parent][0], clearance),
            costs[parent][1] + edge_length,
        )
        descendants.extend(children[descendant])


def _trace_path(points: List[Point], parents: List[Optional[int]],
                index: int) -> Tuple[Point, ...]:
    path = []
    while index is not None:
        path.append(points[index])
        index = parents[index]
    return tuple(reversed(path))


def _path_objective(path: Tuple[Point, ...],
                    lambda_clearance: float,
                    obstacle: Rectangle,
                    workspace: Rectangle) -> float:
    """Return the maximum-minimum-clearance objective C_min."""
    del lambda_clearance
    return path_minimum_clearance(path, obstacle, workspace)


def path_length(path: Tuple[Point, ...]) -> float:
    """Return the total Euclidean length of a route in meters."""
    return sum(math.dist(first, second) for first, second in zip(path, path[1:]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for reproducible planning')
    parser.add_argument('--step-size', type=float, default=0.15,
                        help='maximum extension length in meters')
    parser.add_argument('--goal-sample-rate', type=float, default=0.1,
                        help='fraction of samples directed at the goal')
    parser.add_argument('--lambda-clearance', type=float, default=0.05,
                        help=argparse.SUPPRESS)
    parser.add_argument('--max-iterations', type=int, default=10000)
    args = parser.parse_args()

    try:
        path = plan_rrt(
            step_size=args.step_size,
            goal_sample_rate=args.goal_sample_rate,
            lambda_clearance=args.lambda_clearance,
            max_iterations=args.max_iterations,
            seed=args.seed,
        )
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))

    length = path_length(path)
    minimum_clearance = path_minimum_clearance(
        path, EXPANDED_OBSTACLE, USABLE_WORKSPACE
    )
    print(
        f'RRT path: {len(path)} points, length={length:.3f} m, '
        f'C_min={minimum_clearance:.3f} m'
    )
    for index, (x, y) in enumerate(path):
        print(f'  {index}: ({x:.6f}, {y:.6f}) m')


if __name__ == '__main__':
    main()
