import math

import pytest

from waypoint_follower import (
    FollowerConfig,
    Pose2D,
    Waypoint,
    WaypointFollower,
    load_waypoints,
    normalize_angle,
)


def test_load_waypoints_skips_comments_and_blank_lines(tmp_path):
    path = tmp_path / "points.csv"
    path.write_text("# x,y,theta\n\n0, 0, 0\n1.5, -2, 3.14\n")
    assert load_waypoints(str(path)) == [
        Waypoint(0.0, 0.0, 0.0),
        Waypoint(1.5, -2.0, 3.14),
    ]


def test_load_waypoints_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Waypoint file not found"):
        load_waypoints(str(tmp_path / "missing.csv"))


def test_load_waypoints_rejects_wrong_number_of_fields(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("1,2\n")
    with pytest.raises(ValueError, match=r"Line 1: expected 3 values, got 2"):
        load_waypoints(str(path))


def test_load_waypoints_rejects_non_numeric_values(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("1,not-a-number,0\n")
    with pytest.raises(ValueError, match="Line 1: bad floats"):
        load_waypoints(str(path))


def test_load_waypoints_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("# only a comment\n\n")
    with pytest.raises(ValueError, match="No waypoints parsed"):
        load_waypoints(str(path))


def test_angle_wrap_handles_boundary():
    assert normalize_angle(math.radians(181)) == pytest.approx(math.radians(-179))
    assert normalize_angle(math.radians(-181)) == pytest.approx(math.radians(179))


def test_forward_waypoint_commands_both_wheels_forward():
    follower = WaypointFollower(
        [Waypoint(1.0, 0.0, 0.0)],
        FollowerConfig(wheel_speed_per_command=1.0),
        Pose2D(0.0, 0.0, 0.0),
    )
    left, right = follower.step(0.05)
    assert left == pytest.approx(right)
    assert 0.0 < left <= 0.5


def test_waypoint_behind_robot_turns_without_driving_forward():
    follower = WaypointFollower(
        [Waypoint(-1.0, 0.0, 0.0)],
        FollowerConfig(wheel_speed_per_command=1.0),
        Pose2D(0.0, 0.0, 0.0),
    )
    left, right = follower.step(0.05)
    # A target exactly behind can be reached by turning either direction.
    assert left * right < 0
    assert abs(left) <= 0.5
    assert abs(right) <= 0.5


def test_command_outputs_are_always_within_motor_node_limits():
    follower = WaypointFollower(
        [Waypoint(100.0, 100.0, math.pi)],
        FollowerConfig(wheel_speed_per_command=0.1),
    )
    for _ in range(100):
        left, right = follower.step(0.05)
        assert -0.5 <= left <= 0.5
        assert -0.5 <= right <= 0.5


def test_zero_distance_waypoint_rotates_to_requested_heading():
    follower = WaypointFollower(
        [Waypoint(0.0, 0.0, math.pi / 2)],
        FollowerConfig(wheel_speed_per_command=1.0),
        Pose2D(0.0, 0.0, 0.0),
    )
    left, right = follower.step(0.05)
    assert left < 0 < right
    assert left == pytest.approx(-right)


def test_waypoint_completes_when_position_and_heading_already_match():
    follower = WaypointFollower(
        [Waypoint(0.0, 0.0, 0.0)],
        initial_pose=Pose2D(0.0, 0.0, 0.0),
    )
    left, right = follower.step(0.05)
    assert follower.done
    assert (left, right) == (0.0, 0.0)


def test_previous_forward_command_updates_open_loop_pose():
    follower = WaypointFollower(
        [Waypoint(10.0, 0.0, 0.0)],
        FollowerConfig(wheel_speed_per_command=1.0),
        Pose2D(0.0, 0.0, 0.0),
    )
    follower.left_command = 0.2
    follower.right_command = 0.2
    follower.step(0.1)
    assert follower.pose.x == pytest.approx(0.02)
    assert follower.pose.y == pytest.approx(0.0)
    assert follower.pose.theta == pytest.approx(0.0)


def test_negative_dt_is_rejected():
    follower = WaypointFollower([Waypoint(1.0, 0.0, 0.0)])
    with pytest.raises(ValueError, match="dt cannot be negative"):
        follower.step(-0.1)
