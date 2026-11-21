"""Plan a collision-free route with RRT's implicit Voronoi bias."""

import argparse
import math
import random
from typing import List, Optional, Tuple

from .map_geometry import (
    EXPANDED_OBSTACLE, GOAL, START, USABLE_WORKSPACE, Rectangle,
)
from .visibility_graph import segment_is_free


Point = Tuple[float, float]


def plan_rrt(start: Point = START, goal: Point = GOAL,
             workspace: Rectangle = USABLE_WORKSPACE,
             obstacle: Rectangle = EXPANDED_OBSTACLE,
             step_size: float = 0.15, goal_sample_rate: float = 0.1,
             max_iterations: int = 10000,
             seed: Optional[int] = None) -> Tuple[Point, ...]:
    """Return an RRT route in meters, including both endpoints.

    Uniform samples select their nearest tree node, so nodes with larger
    Voronoi regions are selected more often. Goal samples help connect the
    growing tree to the destination without removing that Voronoi bias.
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
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) \
            or max_iterations <= 0:
        raise ValueError('Maximum iterations must be a positive integer.')

    if segment_is_free(start, goal, workspace, obstacle):
        return start, goal

    rng = random.Random(seed)
    points: List[Point] = [start]
    parents: List[Optional[int]] = [None]

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

        points.append(candidate)
        parents.append(nearest_index)
        candidate_index = len(points) - 1
        if math.dist(candidate, goal) <= step_size \
                and segment_is_free(candidate, goal, workspace, obstacle):
            points.append(goal)
            parents.append(candidate_index)
            return _trace_path(points, parents, len(points) - 1)

    raise RuntimeError(
        f'RRT did not reach the goal within {max_iterations} iterations.'
    )


def _trace_path(points: List[Point], parents: List[Optional[int]],
                index: int) -> Tuple[Point, ...]:
    path = []
    while index is not None:
        path.append(points[index])
        index = parents[index]
    return tuple(reversed(path))


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
    parser.add_argument('--max-iterations', type=int, default=10000)
    args = parser.parse_args()

    try:
        path = plan_rrt(
            step_size=args.step_size,
            goal_sample_rate=args.goal_sample_rate,
            max_iterations=args.max_iterations,
            seed=args.seed,
        )
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))

    print(f'RRT path: {len(path)} points, {path_length(path):.3f} m')
    for index, (x, y) in enumerate(path):
        print(f'  {index}: ({x:.6f}, {y:.6f}) m')


if __name__ == '__main__':
    main()
