from hw4.map_geometry import EXPANDED_OBSTACLE, GOAL, START, USABLE_WORKSPACE
from hw4.rrt import (
    edge_clearance,
    edge_clearance_cost,
    path_length,
    path_minimum_clearance,
    plan_rrt,
)
from hw4.visibility_graph import segment_is_free


def test_rrt_returns_collision_free_path_and_is_reproducible():
    path = plan_rrt(seed=7)

    assert path[0] == START
    assert path[-1] == GOAL
    assert path == plan_rrt(seed=7)
    assert path_length(path) > 0.0
    assert all(
        segment_is_free(first, second, USABLE_WORKSPACE, EXPANDED_OBSTACLE)
        for first, second in zip(path, path[1:])
    )


def test_edge_clearance_measures_clearance_near_obstacle():
    y = (EXPANDED_OBSTACLE.y_min + EXPANDED_OBSTACLE.y_max) / 2.0
    near_edge = (
        (EXPANDED_OBSTACLE.x_min - 0.1, y - 0.05),
        (EXPANDED_OBSTACLE.x_min - 0.1, y + 0.05),
    )
    far_edge = (
        (USABLE_WORKSPACE.x_min, y - 0.05),
        (USABLE_WORKSPACE.x_min, y + 0.05),
    )

    assert edge_clearance(*near_edge) < edge_clearance(*far_edge)


def test_edge_clearance_measures_clearance_near_workspace_wall():
    y = (USABLE_WORKSPACE.y_min + USABLE_WORKSPACE.y_max) / 2.0
    wall_edge = (
        (USABLE_WORKSPACE.x_min + 0.02, y - 0.05),
        (USABLE_WORKSPACE.x_min + 0.02, y + 0.05),
    )
    center_edge = (
        ((USABLE_WORKSPACE.x_min + USABLE_WORKSPACE.x_max) / 2.0, y - 0.05),
        ((USABLE_WORKSPACE.x_min + USABLE_WORKSPACE.x_max) / 2.0, y + 0.05),
    )

    assert edge_clearance(*wall_edge) < edge_clearance(*center_edge)


def test_rrt_continues_sampling_and_improves_minimum_clearance():
    short_run = plan_rrt(
        lambda_clearance=0.05,
        max_iterations=500,
        seed=7,
    )
    long_run = plan_rrt(
        lambda_clearance=0.05,
        max_iterations=2000,
        seed=7,
    )

    short_clearance = path_minimum_clearance(short_run)
    long_clearance = path_minimum_clearance(long_run)

    assert long_clearance >= short_clearance - 1e-12
