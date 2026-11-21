import math

import pytest

from hw4.rrt_waypoint_motion import RRTWaypointMotion


def test_waypoint_completion_uses_position_not_waypoint_yaw():
    motion = RRTWaypointMotion(((1.0, 2.0, math.pi), (3.0, 4.0, -math.pi)))

    linear, angular, action, _, _, _, distance = motion.command_for_pose(
        1.0, 2.0, 0.0
    )

    assert action == 'reached'
    assert linear == 0.0
    assert angular == 0.0
    assert distance == 0.0
    assert motion.waypoints == ((1.0, 2.0), (3.0, 4.0))


def test_waypoint_motion_rejects_non_finite_coordinates():
    with pytest.raises(ValueError, match='finite'):
        RRTWaypointMotion(((1.0, math.inf),))
