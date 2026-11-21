from hw4.map_geometry import EXPANDED_OBSTACLE, GOAL, START, USABLE_WORKSPACE
from hw4.rrt import path_length, plan_rrt
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
