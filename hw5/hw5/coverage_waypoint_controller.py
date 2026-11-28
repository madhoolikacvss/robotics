"""Follow the tag-map-derived coverage route using HW3/HW4 control behavior."""

import math
import argparse
import sys
from pathlib import Path

# Direct local preview mode avoids importing ROS. Run this file with
# --visibility to plot the route without starting the robot controller.
if __name__ == '__main__' and '--visibility' in sys.argv:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hw5.coverage_path import generate_coverage_path
    from hw5.coverage_plot import plot_coverage_path

    plot_coverage_path(generate_coverage_path())
    raise SystemExit(0)

import rclpy
from geometry_msgs.msg import Twist
from hw3.ekf_waypoint_controller import EKFWaypointController
from hw3.velocity_mapping import K_V
from hw3.waypoint_motion import (
    HEADING_TOLERANCE,
    POSITION_TOLERANCE,
    wrap_angle,
)

from .coverage_path import generate_coverage_path, route_length
from .coverage_plot import plot_coverage_path
from .map_geometry import USABLE_WORKSPACE


def contains_point(rectangle, point):
    """Return whether a point is inside or on a rectangle boundary."""
    x, y = point
    return (rectangle.x_min <= x <= rectangle.x_max
            and rectangle.y_min <= y <= rectangle.y_max)


class CoverageWaypointController(EKFWaypointController):
    """Track generated coverage waypoints with HW4's periodic checks."""

    def __init__(self, visibility_override=False):
        waypoints = generate_coverage_path()
        super().__init__()
        self.motion.waypoints = waypoints
        configured_visibility = self.declare_parameter(
            'visibility', False
        ).value
        self.visibility = bool(visibility_override or configured_visibility)
        self.drive_interval = float(
            self.declare_parameter('drive_interval', 1.0).value
        )
        self.heading_check_pause = float(
            self.declare_parameter('heading_check_pause', 0.5).value
        )
        if (not math.isfinite(self.drive_interval) or self.drive_interval <= 0
                or not math.isfinite(self.heading_check_pause)
                or self.heading_check_pause <= 0):
            raise ValueError(
                'Driving interval and heading-check pause must be positive '
                'and finite.'
            )
        self.drive_started = None
        self.pause_until = None
        self.boundary_stopped = False
        self.get_logger().info(
            f'HW5 coverage route: {len(waypoints)} waypoints, '
            f'{route_length(waypoints):.3f} m; '
            f'usable workspace x=[{USABLE_WORKSPACE.x_min:.3f}, '
            f'{USABLE_WORKSPACE.x_max:.3f}], '
            f'y=[{USABLE_WORKSPACE.y_min:.3f}, '
            f'{USABLE_WORKSPACE.y_max}] m.'
        )
        if self.visibility:
            plot_coverage_path(waypoints)

    def boundary_stop(self, reason):
        if not self.boundary_stopped:
            self.get_logger().error(
                f'Workspace stop: {reason}. Restart required.'
            )
        self.boundary_stopped = True
        self.publish_stop()

    def control_loop(self):
        now = self._now_seconds()
        if self.boundary_stopped:
            self.publish_stop()
            return

        pose = self.latest_estimated_position
        if pose is not None:
            x, y, yaw = pose
            if (not all(math.isfinite(value) for value in pose)
                    or not contains_point(USABLE_WORKSPACE, (x, y))):
                self.boundary_stop(
                    f'estimated center ({x:.3f}, {y:.3f}) outside '
                    f'x=[{USABLE_WORKSPACE.x_min:.3f}, '
                    f'{USABLE_WORKSPACE.x_max:.3f}], '
                    f'y=[{USABLE_WORKSPACE.y_min:.3f}, '
                    f'{USABLE_WORKSPACE.y_max:.3f}]'
                )
                return

        if (self.motion.finished or pose is None
                or self.last_pose_received_time is None
                or now - self.last_pose_received_time > self.pose_timeout):
            self.drive_started = None
            self.pause_until = None
            super().control_loop()
            return

        if self.pause_until is not None:
            self.desired_command = Twist()
            self.cmd_pub.publish(Twist())
            if (now < self.pause_until
                    or self.last_pose_received_time < self.pause_until):
                return
            target = self.motion.waypoints[self.motion.current_waypoint_idx]
            error = wrap_angle(
                math.atan2(target[1] - y, target[0] - x) - yaw
            )
            self.get_logger().info(
                f'Periodic heading check: error={math.degrees(error):.1f}deg'
            )
            self.motion.reset_turn()
            self.motion.turn_complete = abs(error) <= HEADING_TOLERANCE
            self.pause_until = None
            self.drive_started = None

        if (self.drive_started is not None
                and now - self.drive_started >= self.drive_interval):
            self.desired_command = Twist()
            self.cmd_pub.publish(Twist())
            self.pause_until = now + self.heading_check_pause
            self.drive_started = None
            self.get_logger().info('Pausing for periodic heading check.')
            return

        target = self.motion.waypoints[self.motion.current_waypoint_idx]
        error = wrap_angle(
            math.atan2(target[1] - y, target[0] - x) - yaw
        )
        would_drive = (
            self.motion.phase == 'position'
            and math.hypot(target[0] - x, target[1] - y)
            > POSITION_TOLERANCE
            and (self.motion.turn_complete
                 or abs(error) <= HEADING_TOLERANCE)
        )
        if would_drive:
            projected = (
                x + K_V * 0.25 * math.cos(yaw),
                y + K_V * 0.25 * math.sin(yaw),
            )
            if not contains_point(USABLE_WORKSPACE, projected):
                self.boundary_stop(
                    'forward projection leaves usable workspace'
                )
                return

        super().control_loop()
        if self.desired_command.linear.x > 0:
            # Preserve HW4 behavior: hold the heading between timed checks.
            self.motion.turn_complete = True
            if self.drive_started is None:
                self.drive_started = now
        else:
            self.drive_started = None


def main(args=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--visibility', action='store_true')
    cli_args, ros_args = parser.parse_known_args(
        sys.argv[1:] if args is None else args
    )
    rclpy.init(args=ros_args)
    node = CoverageWaypointController(visibility_override=cli_args.visibility)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()
