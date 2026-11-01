#!/usr/bin/env python3
"""Continuously control a differential-drive robot using AprilTag poses."""

import csv
import math
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import PoseStamped, Twist
from tf2_ros import Buffer, TransformListener, TransformException


def wrap_angle(angle: float) -> float:
    """Normalize an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def quaternion_to_rotation_matrix(x: float, y: float, z: float, w: float):
    """Convert a ROS quaternion (x, y, z, w) to a 3x3 rotation matrix."""
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        return np.eye(3)

    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def transform_to_matrix(transform_stamped):
    """Convert a TransformStamped into a 4x4 homogeneous transform matrix."""
    t = transform_stamped.transform.translation
    q = transform_stamped.transform.rotation
    matrix = np.eye(4)
    matrix[:3, :3] = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
    matrix[:3, 3] = [t.x, t.y, t.z]
    return matrix


def pose_matrix_xyz_rpy(
    x: float, y: float, z: float,
    roll: float, pitch: float, yaw: float
):
    """Build a 4x4 pose matrix using R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rotation = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=float)

    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = [x, y, z]
    return matrix


def inverse_transform(matrix):
    """Invert a rigid homogeneous transform."""
    result = np.eye(4)
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    result[:3, :3] = rotation.T
    result[:3, 3] = -rotation.T @ translation
    return result


def rotation_matrix_to_yaw(rotation) -> float:
    """Extract planar yaw from a rotation matrix."""
    return math.atan2(rotation[1, 0], rotation[0, 0])


class AprilTagWaypointController(Node):
    def __init__(self):
        super().__init__('apriltag_waypoint_controller')

        # ---------------------------------------------------------------
        # 1. FRAME NAMES
        # ---------------------------------------------------------------
        self.world_frame = 'odom'
        self.base_frame = 'base_link'
        self.camera_frame = 'camera_frame'

        # ---------------------------------------------------------------
        # 2. KNOWN APRILTAG WORLD POSES
        # ---------------------------------------------------------------
        self.tag_poses_world = {
            'tag_0': (1.75, 0.00, 0.05, math.pi / 2, 0.0, -math.pi / 2),
            'tag_1': (1.0, 2.3, 0.05, math.pi / 2, 0.0, 0.0),
            'tag_2': (0.0, -0.35, 0.05, math.pi / 2, 0.0, math.pi),
            # 'tag_2': (-0.2, 0.0, 0.05, math.pi / 2, 0.0, math.pi / 2),
            'tag_3': (0.3, 1.41 , 0.05, math.pi / 2, 0.0, math.pi / 2),
            'tag_4': (0.38, 2.0 , 0.05, math.pi / 2, 0.0, math.pi/2),
        }

        # ---------------------------------------------------------------
        # 3. CAMERA EXTRINSICS
        # ---------------------------------------------------------------
        self.camera_xyz = (0.07, 0.0, 0.12)
        self.camera_rpy = (-math.pi / 2, 0.0, -math.pi / 2)
        self.T_base_camera = pose_matrix_xyz_rpy(
            *self.camera_xyz, *self.camera_rpy
        )

        # ---------------------------------------------------------------
        # 4. WAYPOINTS: [x, y, heading] in world_frame
        # ---------------------------------------------------------------
        self.waypoints = np.array([
            [0.00, 0.00, 0],
            [1.00, 0.00, 0],
            [1.00, 2.00, math.pi],
            [0.00, 0.00, 0.0],
        ], dtype=float)

        # ---------------------------------------------------------------
        # 5. MOTION AND OBSERVATION SETTINGS
        # ---------------------------------------------------------------
        self.k_v = 0.12
        self.k_omega = 0.35
        self.k_heading = 0.10
        self.k_distance = 0.8

        self.control_period = 0.02

        self.tag_max_age = 0.60
        self.position_outlier_threshold = 0.30

        self.position_tolerance = 0.08
        self.heading_tolerance = 0.13

        # ---------------------------------------------------------------
        # 6. STATE
        # ---------------------------------------------------------------
        self.current_waypoint_idx = 0
        self.finished = False

        self.waypoint_phase = 'position'
        self.last_vision_stamp = 0.0
        self.last_tag_age_log_time: Optional[float] = None
        self.last_localization_log_time: Optional[float] = None
        self.last_outlier_log_time: Optional[float] = None
        self.desired_command = Twist()
        self.heading_error = 0.0
        self.turn_mode = 'idle'
        self.turn_direction = 0.0
        self.turn_until = 0.0
        self.turn_pause_until = 0.0
        self.coarse_turn_rate = 2.2
        self.coarse_turn_capture_angle = math.radians(15.0)
        self.fine_turn_pulse_duration = 0.05
        self.fine_turn_pause_duration = 0.20
        self.stale_stop_logged = False
        self.last_used_tag_stamp = {name: 0.0 for name in self.tag_poses_world}
        self.latest_estimated_position: Optional[tuple[float, float, float]] = None

        # ---------------------------------------------------------------
        # 7. ROS COMMUNICATION
        # ---------------------------------------------------------------
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pose_pub = self.create_publisher(
            PoseStamped, '/robot_pose_estimate', 10
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

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(
            self.control_period, self.control_loop
        )
        self.position_log_timer = self.create_timer(
            1.0, self.write_position_to_file
        )

        self.get_logger().info(
            'Continuous AprilTag waypoint controller started: '
            f'position tolerance={self.position_tolerance:.2f}m, '
            f'max tag age={self.tag_max_age:.2f}s. '
            f'Writing estimated poses to {self.position_log_path} and '
            f'waypoint results to {self.waypoint_log_path}.'
        )

    def _now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def write_position_to_file(self):
        """Append the latest estimated pose to the CSV log once per second."""
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
        waypoint = self.waypoints[self.current_waypoint_idx]
        self.waypoint_log_writer.writerow([
            f'{self._now_seconds():.9f}',
            self.world_frame,
            self.current_waypoint_idx,
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
        return super().destroy_node()

    def publish_stop(self):
        self.desired_command = Twist()
        self.turn_mode = 'idle'
        self.turn_direction = 0.0
        self.turn_until = 0.0
        self.turn_pause_until = 0.0
        self.cmd_pub.publish(Twist())

    def publish_control_command(self, now: float, allow_new_turn: bool):
        """Run a timed coarse turn, then adjust with separated short nudges."""
        command = self.desired_command
        is_pure_turn = (
            abs(command.linear.x) < 1e-6
            and abs(command.angular.z) > 1e-6
        )

        if not is_pure_turn:
            self.turn_mode = 'idle'
            self.turn_direction = 0.0
            self.cmd_pub.publish(command)
            return

        direction = math.copysign(1.0, command.angular.z)
        if direction != self.turn_direction:
            self.turn_mode = 'idle'
            self.turn_direction = direction

        output = Twist()
        if self.turn_mode == 'coarse':
            turn_complete = (
                now >= self.turn_until
                or abs(self.heading_error) <= self.coarse_turn_capture_angle
                or self.heading_error * direction <= 0.0
            )
            if turn_complete:
                self.turn_mode = 'fine_pause'
                self.turn_pause_until = now + self.fine_turn_pause_duration
            else:
                output.angular.z = direction * self.k_omega
        elif self.turn_mode == 'fine_pulse':
            if now < self.turn_until:
                output.angular.z = direction * self.k_omega
            else:
                self.turn_mode = 'fine_pause'
                self.turn_pause_until = now + self.fine_turn_pause_duration
        elif self.turn_mode == 'fine_pause':
            if now >= self.turn_pause_until:
                self.turn_mode = 'idle'

        if self.turn_mode == 'idle' and allow_new_turn:
            if abs(self.heading_error) > self.coarse_turn_capture_angle:
                self.turn_mode = 'coarse'
                self.turn_until = now + (
                    abs(self.heading_error) / self.coarse_turn_rate
                )
                output.angular.z = direction * self.k_omega
            else:
                self.turn_mode = 'fine_pulse'
                self.turn_until = now + self.fine_turn_pulse_duration
                output.angular.z = direction * self.k_omega

        self.cmd_pub.publish(output)

    def get_new_known_tag(self):
        """Return a new, recent observation for a known tag, if available."""
        now_seconds = self._now_seconds()
        newest_observation = None

        for tag_frame in self.tag_poses_world:
            try:
                tf_msg = self.tf_buffer.lookup_transform(
                    self.camera_frame,
                    tag_frame,
                    Time(),
                    timeout=Duration(seconds=0.0),
                )
            except TransformException:
                continue

            stamp = tf_msg.header.stamp
            stamp_seconds = stamp.sec + stamp.nanosec * 1e-9
            if stamp_seconds <= 0.0:
                continue

            age = now_seconds - stamp_seconds
            if age < -0.10 or age > self.tag_max_age:
                continue

            if stamp_seconds <= self.last_used_tag_stamp[tag_frame]:
                continue

            if (
                newest_observation is None
                or stamp_seconds > newest_observation[2]
            ):
                newest_observation = (
                    tag_frame, transform_to_matrix(tf_msg), stamp_seconds
                )

        if newest_observation is not None:
            tag_frame, _, stamp_seconds = newest_observation
            age = now_seconds - stamp_seconds
            if (
                self.last_tag_age_log_time is None
                or now_seconds - self.last_tag_age_log_time >= 1.0
                or now_seconds < self.last_tag_age_log_time
            ):
                self.get_logger().info(
                    f'AprilTag image age: {age:.3f}s '
                    f'(image stamp={stamp_seconds:.6f}, '
                    f'current ROS time={now_seconds:.6f}, tag={tag_frame}).'
                )
                self.last_tag_age_log_time = now_seconds

        return newest_observation or (None, None, None)

    def estimate_world_robot_pose(self, tag_frame: str, T_camera_tag):
        """Estimate T_world_base from known tag pose and observed camera-to-tag transform."""
        T_world_tag = pose_matrix_xyz_rpy(
            *self.tag_poses_world[tag_frame]
        )
        T_base_tag = self.T_base_camera @ T_camera_tag
        return T_world_tag @ inverse_transform(T_base_tag)

    def calculate_motion_command(
        self, robot_x: float, robot_y: float, robot_yaw: float
    ):
        """Calculate a proportional command for the current waypoint."""
        waypoint = self.waypoints[self.current_waypoint_idx]

        dx = float(waypoint[0] - robot_x)
        dy = float(waypoint[1] - robot_y)
        distance_error = math.hypot(dx, dy)

        cmd = Twist()

        if self.waypoint_phase == 'position':
            if distance_error <= self.position_tolerance:
                self.waypoint_phase = 'heading'
                self.get_logger().info(
                    f'Waypoint {self.current_waypoint_idx}: position tolerance '
                    f'reached (error={distance_error:.3f}m); aligning heading.'
                )

            if self.waypoint_phase == 'position':
                desired_heading = math.atan2(dy, dx)
                heading_error = wrap_angle(desired_heading - robot_yaw)
                cmd.angular.z = float(np.clip(
                    self.k_heading * heading_error,
                    -self.k_omega,
                    self.k_omega,
                ))

                if abs(heading_error) > self.heading_tolerance:
                    cmd.angular.z = math.copysign(
                        self.k_omega, heading_error
                    )
                    action = 'turn'
                else:
                    cmd.linear.x = min(
                        self.k_v,
                        max(0.0, self.k_distance * distance_error),
                    )
                    action = 'drive'

                return (
                    cmd, action, dx, dy, heading_error,
                    distance_error
                )

        if self.waypoint_phase == 'heading':
            heading_error = wrap_angle(
                float(waypoint[2]) - robot_yaw
            )
            cmd.angular.z = float(np.clip(
                self.k_heading * heading_error,
                -self.k_omega,
                self.k_omega,
            ))
            if abs(heading_error) > self.heading_tolerance:
                cmd.angular.z = math.copysign(
                    self.k_omega, heading_error
                )
                action = 'final_heading_turn'
            else:
                action = 'reached'

            return (
                cmd, action, dx, dy, heading_error,
                distance_error
            )

        raise ValueError(
            f"Unknown waypoint phase: {self.waypoint_phase!r}"
        )

    def control_loop(self):
        now = self._now_seconds()

        if self.finished or self.current_waypoint_idx >= len(self.waypoints):
            self.finished = True
            self.publish_stop()
            return

        tag_frame, T_camera_tag, tag_stamp = self.get_new_known_tag()
        if T_camera_tag is None:
            if (
                self.waypoint_phase == 'heading'
                and self.turn_mode in ('coarse', 'fine_pulse')
            ):
                self.publish_control_command(now, allow_new_turn=False)
                if not self.stale_stop_logged:
                    self.get_logger().warning(
                        'No fresh AprilTag pose; completing the active timed '
                        'heading turn, then waiting for vision to update.'
                    )
                    self.stale_stop_logged = True
                return

            vision_age = now - self.last_vision_stamp
            if (
                self.last_vision_stamp > 0.0
                and -0.10 <= vision_age <= self.tag_max_age
            ):
                self.publish_control_command(now, allow_new_turn=False)
                return

            self.publish_stop()
            if not self.stale_stop_logged:
                self.get_logger().warning(
                    'No fresh AprilTag pose; stopping until vision updates.'
                )
                self.stale_stop_logged = True
            return

        self.last_used_tag_stamp[tag_frame] = tag_stamp

        T_world_robot = self.estimate_world_robot_pose(
            tag_frame, T_camera_tag
        )
        robot_x = float(T_world_robot[0, 3])
        robot_y = float(T_world_robot[1, 3])
        robot_yaw = rotation_matrix_to_yaw(T_world_robot[:3, :3])

        if self.latest_estimated_position is not None:
            previous_x, previous_y, _ = self.latest_estimated_position
            position_change = math.hypot(
                robot_x - previous_x, robot_y - previous_y
            )
            if position_change >= self.position_outlier_threshold:
                if (
                    self.last_outlier_log_time is None
                    or now - self.last_outlier_log_time >= 1.0
                    or now < self.last_outlier_log_time
                ):
                    self.get_logger().warning(
                        f'Rejecting AprilTag pose from {tag_frame}: '
                        f'position changed by {position_change:.3f}m '
                        f'(limit={self.position_outlier_threshold:.2f}m).'
                    )
                    self.last_outlier_log_time = now
                return

        self.last_vision_stamp = tag_stamp
        self.stale_stop_logged = False
        self.latest_estimated_position = (robot_x, robot_y, robot_yaw)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = f'{self.world_frame}|{tag_frame}'
        pose_msg.pose.position.x = robot_x
        pose_msg.pose.position.y = robot_y
        pose_msg.pose.position.z = 0.0
        pose_msg.pose.orientation.z = math.sin(robot_yaw / 2.0)
        pose_msg.pose.orientation.w = math.cos(robot_yaw / 2.0)
        self.pose_pub.publish(pose_msg)

        cmd, action, dx, dy, heading_error, distance_error = (
            self.calculate_motion_command(robot_x, robot_y, robot_yaw)
        )
        self.desired_command = cmd
        self.heading_error = heading_error
        self.publish_control_command(now, allow_new_turn=True)

        if (
            self.last_localization_log_time is None
            or now - self.last_localization_log_time >= 1.0
            or now < self.last_localization_log_time
        ):
            self.get_logger().info(
                f'LOCALIZED tag={tag_frame} '
                f'pose=({robot_x:.3f}, {robot_y:.3f}, {robot_yaw:.3f}) '
                f'waypoint={self.current_waypoint_idx} '
                f'distance_err={distance_error:.3f} '
                f'heading_err={heading_error:.3f} '
                f'v={cmd.linear.x:.3f}, w={cmd.angular.z:.3f}'
            )
            self.last_localization_log_time = now

        if action == 'reached':
            self.write_waypoint_result(
                robot_x, robot_y, robot_yaw, dx, dy,
                distance_error, heading_error,
            )
            self.get_logger().info(
                f'Reached waypoint {self.current_waypoint_idx}: '
                f'position error={distance_error:.3f}m, '
                f'heading error={heading_error:.3f}rad.'
            )
            self.current_waypoint_idx += 1

            if self.current_waypoint_idx >= len(self.waypoints):
                self.finished = True
                self.publish_stop()
                self.get_logger().info('All waypoints reached.')
                return

            self.waypoint_phase = 'position'
            self.publish_stop()


def main(args=None):
    rclpy.init(args=args)
    node = AprilTagWaypointController()

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
