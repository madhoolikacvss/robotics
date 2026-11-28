"""Follow the tag-map-derived coverage route using HW3/HW4 control behavior."""

import math
import argparse
import sys
from pathlib import Path

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
from .map_geometry import USABLE_WORKSPACE, WORKSPACE
from .tag_map import WORKSPACE_TAGS


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
            self.plot_coverage_path(waypoints)

    @staticmethod
    def plot_coverage_path(waypoints):
        """Show and save the route, tag poses, physical walls, and safe inset."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        fig, ax = plt.subplots(figsize=(9, 8))
        ax.add_patch(Rectangle(
            (WORKSPACE.x_min, WORKSPACE.y_min),
            WORKSPACE.x_max - WORKSPACE.x_min,
            WORKSPACE.y_max - WORKSPACE.y_min,
            facecolor='#edf0f3', edgecolor='#334155', linewidth=2,
            label='Physical workspace from tags',
        ))
        ax.add_patch(Rectangle(
            (USABLE_WORKSPACE.x_min, USABLE_WORKSPACE.y_min),
            USABLE_WORKSPACE.x_max - USABLE_WORKSPACE.x_min,
            USABLE_WORKSPACE.y_max - USABLE_WORKSPACE.y_min,
            facecolor='white', edgecolor='#dc2626', linestyle='--',
            linewidth=1.8, label='Safe center boundary (26.55 cm inset)',
        ))

        x_values = [point[0] for point in waypoints]
        y_values = [point[1] for point in waypoints]
        ax.plot(x_values, y_values, color='#16a34a', linewidth=2.2,
                marker='o', markersize=3.5, label='Generated coverage route',
                zorder=3)
        ax.scatter(x_values[0], y_values[0], color='#2563eb', s=70,
                   label='Route start', zorder=4)
        ax.scatter(x_values[-1], y_values[-1], color='#9333ea', s=70,
                   label='Route end', zorder=4)

        tag_x = [tag['pose'][0] for tag in WORKSPACE_TAGS.values()]
        tag_y = [tag['pose'][1] for tag in WORKSPACE_TAGS.values()]
        ax.scatter(tag_x, tag_y, marker='s', color='#ea580c', s=38,
                   label='AprilTags', zorder=5)
        for tag_id, tag in WORKSPACE_TAGS.items():
            tx, ty = tag['pose'][:2]
            ax.annotate(tag_id, (tx, ty), xytext=(4, 4),
                        textcoords='offset points', fontsize=8,
                        color='#9a3412')

        padding = 0.12
        ax.set_xlim(WORKSPACE.x_min - padding, WORKSPACE.x_max + padding)
        ax.set_ylim(WORKSPACE.y_min - padding, WORKSPACE.y_max + padding)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(
            f'HW5 coverage route: {len(waypoints)} waypoints, '
            f'{route_length(waypoints):.2f} m'
        )
        ax.grid(alpha=0.2)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
                  ncol=2, fontsize=9)
        fig.tight_layout()
        output = Path.cwd() / 'hw5_coverage_path.png'
        fig.savefig(output, dpi=160)
        print(f'Saved coverage plot to {output}')
        # Hold execution until the user closes the preview, so movement does
        # not begin while the route and mapped boundaries are being inspected.
        plt.show()
        plt.close(fig)

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
