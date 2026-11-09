#!/usr/bin/env python3
"""Continuously control a differential-drive robot using AprilTag poses."""

import csv
import json
import math
from pathlib import Path
from typing import Optional

import rclpy
import tf_transformations
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import String

from .waypoint_motion import (
    CONTROL_PERIOD,
    POSITION_TOLERANCE,
    WaypointMotion,
)


class EKFWaypointController(Node):
    def __init__(self):
        super().__init__('ekf_waypoint_controller')

        self.world_frame = str(
            self.declare_parameter('world_frame', 'odom').value
        )

        self.motion = WaypointMotion()
        self.pose_timeout = float(
            self.declare_parameter('pose_timeout', 0.50).value
        )
        if self.pose_timeout <= 0.0:
            raise ValueError('pose_timeout must be positive.')
        self.final_waypoint_settle_time = float(
            self.declare_parameter(
                'final_waypoint_settle_time', 1.0
            ).value
        )
        if self.final_waypoint_settle_time < 0.0:
            raise ValueError(
                'final_waypoint_settle_time must be non-negative.'
            )

        self.control_period = CONTROL_PERIOD
        self.last_control_log_time: Optional[float] = None
        self.desired_command = Twist()
        self.stale_stop_logged = False
        self.final_waypoint_wait_started: Optional[float] = None
        self.latest_estimated_position: Optional[tuple[float, float, float]] = None
        self.last_pose_received_time: Optional[float] = None
        self.frame_mismatch_logged = False
        self.latest_tag_positions: dict[str, tuple[float, float]] = {}
        self.tag_frame_mismatch_logged = False
        self.tag_position_log_written = False

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(
            PoseStamped, '/robot_pose_estimate', self.pose_callback, 10
        )
        self.create_subscription(
            String, '/tag_positions', self.tag_positions_callback, 10
        )

        self.position_log_path = Path(
            self.declare_parameter(
                'position_log_path', 'robot_position.csv'
            ).value
        ).expanduser()
        self.position_log_file = self.position_log_path.open(
            'a', newline='', encoding='utf-8'
        )
        self.position_log_writer = csv.writer(self.position_log_file)
        if self.position_log_path.stat().st_size == 0:
            self.position_log_writer.writerow(
                ['timestamp_sec', 'frame', 'x', 'y', 'yaw_rad']
            )
            self.position_log_file.flush()

        self.waypoint_log_path = Path(
            self.declare_parameter(
                'waypoint_log_path', 'waypoint_results.csv'
            ).value
        ).expanduser()
        self.waypoint_log_file = self.waypoint_log_path.open(
            'a', newline='', encoding='utf-8'
        )
        self.waypoint_log_writer = csv.writer(self.waypoint_log_file)
        if self.waypoint_log_path.stat().st_size == 0:
            self.waypoint_log_writer.writerow([
                'timestamp_sec', 'frame', 'waypoint_index',
                'target_x', 'target_y', 'target_yaw_rad',
                'estimated_x', 'estimated_y', 'estimated_yaw_rad',
                'error_x', 'error_y', 'position_error_m',
                'heading_error_rad',
            ])
            self.waypoint_log_file.flush()

        self.tag_position_log_path = Path(
            self.declare_parameter(
                'tag_position_log_path', 'tag_positions.csv'
            ).value
        ).expanduser()
        self.tag_position_log_file = self.tag_position_log_path.open(
            'w', newline='', encoding='utf-8'
        )
        self.tag_position_log_writer = csv.writer(
            self.tag_position_log_file
        )
        self.tag_position_log_writer.writerow([
            'timestamp_sec', 'frame', 'tag', 'x', 'y',
        ])
        self.tag_position_log_file.flush()

        self.timer = self.create_timer(
            self.control_period, self.control_loop
        )
        self.position_log_timer = self.create_timer(
            0.5, self.write_position_to_file
        )

        self.get_logger().info(
            'EKF waypoint controller started: '
            f'position tolerance={POSITION_TOLERANCE:.2f}m, '
            f'pose timeout={self.pose_timeout:.2f}s. '
            f'Using /robot_pose_estimate and writing positions to '
            f'{self.position_log_path} and '
            f'waypoint results to {self.waypoint_log_path}; tag positions '
            f'will be written to {self.tag_position_log_path} on completion.'
        )

    def _now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def write_position_to_file(self):
        """Append the latest EKF pose to the CSV log every 0.5 seconds."""
        if self.latest_estimated_position is None:
            return

        timestamp = self._now_seconds()
        x, y, yaw = self.latest_estimated_position
        self.position_log_writer.writerow([
            f'{timestamp:.9f}', self.world_frame,
            f'{x:.6f}', f'{y:.6f}', f'{yaw:.6f}',
        ])
        self.position_log_file.flush()

    def write_waypoint_result(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        dx: float,
        dy: float,
        distance_error: float,
        heading_error: float,
    ):
        """Append the estimated pose and errors for a completed waypoint."""
        waypoint = self.motion.waypoints[self.motion.current_waypoint_idx]
        self.waypoint_log_writer.writerow([
            f'{self._now_seconds():.9f}',
            self.world_frame,
            self.motion.current_waypoint_idx,
            f'{waypoint[0]:.6f}',
            f'{waypoint[1]:.6f}',
            f'{waypoint[2]:.6f}',
            f'{robot_x:.6f}',
            f'{robot_y:.6f}',
            f'{robot_yaw:.6f}',
            f'{dx:.6f}',
            f'{dy:.6f}',
            f'{distance_error:.6f}',
            f'{heading_error:.6f}',
        ])
        self.waypoint_log_file.flush()

    def destroy_node(self):
        self.position_log_file.close()
        self.waypoint_log_file.close()
        self.tag_position_log_file.close()
        return super().destroy_node()

    def publish_stop(self):
        self.desired_command = Twist()
        self.motion.reset_turn()
        self.cmd_pub.publish(Twist())

    def pose_callback(self, msg: PoseStamped):
        """Store a recent EKF pose expressed in the waypoint world frame."""
        if msg.header.frame_id != self.world_frame:
            self.latest_estimated_position = None
            self.last_pose_received_time = None
            if not self.frame_mismatch_logged:
                self.get_logger().error(
                    f'Rejecting EKF pose in frame {msg.header.frame_id!r}; '
                    f'waypoints use {self.world_frame!r}.'
                )
                self.frame_mismatch_logged = True
            return

        orientation = msg.pose.orientation
        _, _, yaw = tf_transformations.euler_from_quaternion([
            orientation.x, orientation.y, orientation.z, orientation.w,
        ])
        self.latest_estimated_position = (
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(yaw),
        )
        self.last_pose_received_time = self._now_seconds()
        self.frame_mismatch_logged = False

    def tag_positions_callback(self, msg: String):
        """Store the latest EKF map of tag positions."""
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as error:
            self.get_logger().error(
                f'Rejecting malformed /tag_positions message: {error}.'
            )
            return

        if (
            not isinstance(payload, dict)
            or payload.get('frame') != self.world_frame
        ):
            if not self.tag_frame_mismatch_logged:
                self.get_logger().error(
                    'Rejecting tag positions that do not match waypoint '
                    f'frame {self.world_frame!r}.'
                )
                self.tag_frame_mismatch_logged = True
            return

        tags = payload.get('tags')
        if not isinstance(tags, list):
            self.get_logger().error(
                'Rejecting /tag_positions message with an invalid tags list.'
            )
            return

        positions = {}
        for tag in tags:
            if not isinstance(tag, dict):
                self.get_logger().error(
                    'Rejecting /tag_positions message with an invalid tag.'
                )
                return
            name, x, y = tag.get('name'), tag.get('x'), tag.get('y')
            if (
                not isinstance(name, str)
                or not isinstance(x, (int, float))
                or not isinstance(y, (int, float))
                or not math.isfinite(x)
                or not math.isfinite(y)
            ):
                self.get_logger().error(
                    'Rejecting /tag_positions message with invalid tag '
                    'coordinates.'
                )
                return
            positions[name] = (float(x), float(y))

        self.latest_tag_positions = positions
        self.tag_frame_mismatch_logged = False

    def write_tag_positions(self):
        """Write all estimated tag positions once waypoint motion completes."""
        if self.tag_position_log_written:
            return

        timestamp = self._now_seconds()
        for tag, (x, y) in sorted(self.latest_tag_positions.items()):
            self.tag_position_log_writer.writerow([
                f'{timestamp:.9f}', self.world_frame, tag,
                f'{x:.6f}', f'{y:.6f}',
            ])
        self.tag_position_log_file.flush()
        self.tag_position_log_written = True

        if self.latest_tag_positions:
            self.get_logger().info(
                f'Wrote {len(self.latest_tag_positions)} tag positions to '
                f'{self.tag_position_log_path}.'
            )
        else:
            self.get_logger().warning(
                'Waypoint motion completed before any tag positions were '
                f'received; {self.tag_position_log_path} contains no tag rows.'
            )

    def control_loop(self):
        now = self._now_seconds()

        if self.motion.finished:
            self.write_tag_positions()
            self.publish_stop()
            return

        if (
            self.latest_estimated_position is None
            or self.last_pose_received_time is None
            or now - self.last_pose_received_time > self.pose_timeout
        ):
            self.publish_stop()
            if not self.stale_stop_logged:
                self.get_logger().warning(
                    'No recent EKF pose; stopping until '
                    '/robot_pose_estimate updates.'
                )
                self.stale_stop_logged = True
            return

        if self.final_waypoint_wait_started is not None:
            if (
                now - self.final_waypoint_wait_started
                < self.final_waypoint_settle_time
            ):
                self.publish_stop()
                return
            self.final_waypoint_wait_started = None

        self.stale_stop_logged = False
        robot_x, robot_y, robot_yaw = self.latest_estimated_position

        previous_phase = self.motion.phase
        (
            linear, angular, action, dx, dy, heading_error, distance_error,
        ) = self.motion.command_for_pose(robot_x, robot_y, robot_yaw)
        if previous_phase == 'position' and self.motion.phase == 'heading':
            self.get_logger().info(
                f'Waypoint {self.motion.current_waypoint_idx}: position '
                f'tolerance reached (error={distance_error:.3f}m); '
                'aligning heading.'
            )

        linear, angular = self.motion.shape_turn(
            linear, angular, heading_error
        )
        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        self.desired_command = cmd
        self.cmd_pub.publish(cmd)

        if (
            self.last_control_log_time is None
            or now - self.last_control_log_time >= 1.0
            or now < self.last_control_log_time
        ):
            self.get_logger().info(
                f'EKF pose=({robot_x:.3f}, {robot_y:.3f}, '
                f'{robot_yaw:.3f}) '
                f'waypoint={self.motion.current_waypoint_idx} '
                f'distance_err={distance_error:.3f} '
                f'heading_err={heading_error:.3f} '
                f'v={cmd.linear.x:.3f}, w={cmd.angular.z:.3f}'
            )
            self.last_control_log_time = now

        if action == 'reached':
            if (
                self.motion.current_waypoint_idx
                == len(self.motion.waypoints) - 1
                and self.final_waypoint_wait_started is None
            ):
                self.final_waypoint_wait_started = now
                self.publish_stop()
                self.get_logger().info(
                    'Final waypoint reached; holding position for '
                    f'{self.final_waypoint_settle_time:.1f}s to allow '
                    'camera feedback.'
                )
                return

            self.write_waypoint_result(
                robot_x, robot_y, robot_yaw, dx, dy,
                distance_error, heading_error,
            )
            self.get_logger().info(
                f'Reached waypoint {self.motion.current_waypoint_idx}: '
                f'position error={distance_error:.3f}m, '
                f'heading error={heading_error:.3f}rad.'
            )
            self.motion.advance_waypoint()

            if self.motion.finished:
                self.write_tag_positions()
                self.publish_stop()
                self.get_logger().info('All waypoints reached.')
                return

            self.publish_stop()


def main(args=None):
    rclpy.init(args=args)
    node = EKFWaypointController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
