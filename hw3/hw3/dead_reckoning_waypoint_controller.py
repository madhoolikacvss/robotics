#!/usr/bin/env python3
"""Follow waypoints using an open-loop pose integrated from /cmd_vel."""

import csv
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node

from .waypoint_motion import (
    CONTROL_PERIOD,
    POSITION_TOLERANCE,
    WaypointMotion,
    estimate_angular_displacement,
    wrap_angle,
)


class DeadReckoningWaypointController(Node):
    """Run the shared waypoint controller without external pose feedback."""

    def __init__(self):
        super().__init__('dead_reckoning_waypoint_controller')

        self.world_frame = str(
            self.declare_parameter('world_frame', 'odom').value
        )
        self.x = float(self.declare_parameter('initial_x', 0.0).value)
        self.y = float(self.declare_parameter('initial_y', 0.0).value)
        self.yaw = float(self.declare_parameter('initial_yaw', 0.0).value)
        self.yaw = wrap_angle(self.yaw)

        self.motion = WaypointMotion()
        self.last_time = self._now_seconds()
        self.last_command = Twist()
        self.linear_command_started_at: Optional[
            tuple[float, str, float]
        ] = None
        self.angular_command_started_at: Optional[
            tuple[float, str, float]
        ] = None
        self.angular_command_direction = 0.0
        self.angular_command_elapsed = 0.0
        self.last_log_time: Optional[float] = None

        self.command_duration_log_path = Path(
            self.declare_parameter(
                'command_duration_log_path', 'command_durations.csv'
            ).value
        ).expanduser()
        self.command_duration_log_file = self.command_duration_log_path.open(
            'a', newline='', encoding='utf-8'
        )
        self.command_duration_log_writer = csv.writer(
            self.command_duration_log_file
        )
        if self.command_duration_log_path.stat().st_size == 0:
            self.command_duration_log_writer.writerow([
                'command_type', 'command_value_start',
                'command_value_stop', 'started_at_utc', 'stopped_at_utc',
                'duration_seconds',
            ])
            self.command_duration_log_file.flush()

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pose_pub = self.create_publisher(
            PoseStamped, '/dead_reckoned_pose', 10
        )
        self.timer = self.create_timer(CONTROL_PERIOD, self.control_loop)

        self.get_logger().info(
            'Dead-reckoning waypoint controller started. It integrates '
            '/cmd_vel without sensor feedback; run it instead of the '
            'AprilTag waypoint controller. '
            f'Position tolerance={POSITION_TOLERANCE:.2f}m.'
        )

    def _now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _integrate_command(self, dt: float):
        if dt <= 0.0:
            return
        linear = self.last_command.linear.x
        angular_command = self.last_command.angular.z
        direction = (
            math.copysign(1.0, angular_command)
            if abs(angular_command) > 1e-9 else 0.0
        )
        if direction != self.angular_command_direction:
            self.angular_command_elapsed = 0.0
            self.angular_command_direction = direction
        angular_displacement, self.angular_command_elapsed = (
            estimate_angular_displacement(
                angular_command, self.angular_command_elapsed, dt
            )
        )
        angular = angular_displacement / dt
        next_yaw = self.yaw + angular_displacement
        if abs(angular) > 1e-9:
            self.x += linear / angular * (
                math.sin(next_yaw) - math.sin(self.yaw)
            )
            self.y -= linear / angular * (
                math.cos(next_yaw) - math.cos(self.yaw)
            )
        else:
            self.x += linear * math.cos(self.yaw) * dt
            self.y += linear * math.sin(self.yaw) * dt
        self.yaw = wrap_angle(next_yaw)

    def _publish_command(self, command: Twist):
        now = time.monotonic()
        now_utc = datetime.now(timezone.utc).isoformat()
        previous = self.last_command

        if abs(previous.linear.x) > 1e-9 and (
            abs(command.linear.x) <= 1e-9
            or previous.linear.x * command.linear.x < 0.0
        ):
            self._write_command_duration(
                'linear', previous.linear.x,
                self.linear_command_started_at, now, now_utc,
            )
            self.linear_command_started_at = None
        if abs(command.linear.x) > 1e-9 and (
            abs(previous.linear.x) <= 1e-9
            or previous.linear.x * command.linear.x < 0.0
        ):
            self.linear_command_started_at = (
                now, now_utc, command.linear.x
            )

        if abs(previous.angular.z) > 1e-9 and (
            abs(command.angular.z) <= 1e-9
            or previous.angular.z * command.angular.z < 0.0
        ):
            self._write_command_duration(
                'angular', previous.angular.z,
                self.angular_command_started_at, now, now_utc,
            )
            self.angular_command_started_at = None
        if abs(command.angular.z) > 1e-9 and (
            abs(previous.angular.z) <= 1e-9
            or previous.angular.z * command.angular.z < 0.0
        ):
            self.angular_command_started_at = (
                now, now_utc, command.angular.z
            )

        self.last_command = command
        self.cmd_pub.publish(command)

    def _write_command_duration(
        self,
        command_type: str,
        command_value: float,
        started_at: Optional[tuple[float, str, float]],
        stopped_at: float,
        stopped_at_utc: str,
    ):
        if started_at is None:
            return
        self.command_duration_log_writer.writerow([
            command_type,
            f'{started_at[2]:.6f}',
            f'{command_value:.6f}',
            started_at[1],
            stopped_at_utc,
            f'{stopped_at - started_at[0]:.6f}',
        ])
        self.command_duration_log_file.flush()

    def destroy_node(self):
        self.command_duration_log_file.close()
        return super().destroy_node()

    def _publish_pose(self, now: float):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self.world_frame
        pose.pose.position.x = self.x
        pose.pose.position.y = self.y
        pose.pose.position.z = 0.0
        pose.pose.orientation.z = math.sin(self.yaw / 2.0)
        pose.pose.orientation.w = math.cos(self.yaw / 2.0)
        self.pose_pub.publish(pose)

        if (
            self.last_log_time is None
            or now - self.last_log_time >= 1.0
            or now < self.last_log_time
        ):
            self.get_logger().info(
                f'Dead-reckoned pose=({self.x:.3f}, {self.y:.3f}, '
                f'{self.yaw:.3f}), '
                f'waypoint={self.motion.current_waypoint_idx}'
            )
            self.last_log_time = now

    def control_loop(self):
        now = self._now_seconds()
        dt = now - self.last_time
        self.last_time = now
        if dt < 0.0:
            self.get_logger().warning(
                'ROS time moved backwards; skipping dead-reckoning update.'
            )
            self._publish_command(Twist())
            return

        self._integrate_command(dt)
        self._publish_pose(now)

        if self.motion.finished:
            self._publish_command(Twist())
            return

        (
            linear, angular, action, _, _, heading_error, distance_error,
        ) = self.motion.command_for_pose(self.x, self.y, self.yaw)
        linear, angular = self.motion.shape_turn(
            linear, angular, heading_error
        )

        command = Twist()
        command.linear.x = linear
        command.angular.z = angular
        self._publish_command(command)

        if action == 'reached':
            waypoint_index = self.motion.current_waypoint_idx
            self.get_logger().info(
                f'Reached dead-reckoned waypoint {waypoint_index}: '
                f'position error={distance_error:.3f}m.'
            )
            self.motion.advance_waypoint()
            self._publish_command(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = DeadReckoningWaypointController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_command(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
