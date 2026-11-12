#!/usr/bin/env python3
"""Estimate the robot's planar pose from velocity commands and AprilTags."""

import json
import math
from typing import Optional

import numpy as np
import rclpy
import tf_transformations
import yaml
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from .waypoint_motion import estimate_angular_displacement


def wrap_angle(angle: float) -> float:
    """Normalize an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class PlanarRobotEKF:
    """EKF-SLAM for a unicycle robot and position-only landmark observations."""

    def __init__(
        self,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        control_noise: np.ndarray,
        measurement_noise: np.ndarray,
        gate_threshold: float,
    ):
        self.state = initial_state.astype(float)
        self.covariance = initial_covariance.astype(float)
        self.control_noise = control_noise
        self.measurement_noise = measurement_noise
        self.gate_threshold = gate_threshold
        self.landmark_indices: dict[str, int] = {}

    def predict(
        self, linear_velocity: float, angular_velocity: float, dt: float
    ):
        """Propagate the pose and covariance using the commanded twist."""
        if dt <= 0.0:
            return

        x, y, yaw = self.state[:3]
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        self.state[:3] = np.array([
            x + linear_velocity * cos_yaw * dt,
            y + linear_velocity * sin_yaw * dt,
            wrap_angle(yaw + angular_velocity * dt),
        ])

        robot_jacobian = np.array([
            [1.0, 0.0, -linear_velocity * sin_yaw * dt],
            [0.0, 1.0, linear_velocity * cos_yaw * dt],
            [0.0, 0.0, 1.0],
        ])
        state_jacobian = np.eye(self.state.size)
        state_jacobian[:3, :3] = robot_jacobian
        control_jacobian = np.zeros((self.state.size, 2))
        control_jacobian[:3, :] = np.array([
            [cos_yaw * dt, 0.0],
            [sin_yaw * dt, 0.0],
            [0.0, dt],
        ])
        self.covariance = (
            state_jacobian @ self.covariance @ state_jacobian.T
            + control_jacobian @ self.control_noise @ control_jacobian.T
        )

    def add_landmark(self, tag_frame: str, observed_tag_base: np.ndarray):
        """Initialize a newly seen landmark in the relative world frame."""
        if tag_frame in self.landmark_indices:
            return

        _, _, yaw = self.state[:3]
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        rotation = np.array([
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw],
        ])
        landmark_world = self.state[:2] + rotation @ observed_tag_base
        robot_jacobian = np.array([
            [1.0, 0.0,
             -sin_yaw * observed_tag_base[0]
             - cos_yaw * observed_tag_base[1]],
            [0.0, 1.0,
             cos_yaw * observed_tag_base[0]
             - sin_yaw * observed_tag_base[1]],
        ])
        observation_jacobian = rotation
        old_size = self.state.size
        cross_covariance = (
            robot_jacobian @ self.covariance[:3, :]
        )
        landmark_covariance = (
            robot_jacobian @ self.covariance[:3, :3]
            @ robot_jacobian.T
            + observation_jacobian @ self.measurement_noise
            @ observation_jacobian.T
        )

        self.state = np.concatenate((self.state, landmark_world))
        expanded_covariance = np.zeros((old_size + 2, old_size + 2))
        expanded_covariance[:old_size, :old_size] = self.covariance
        expanded_covariance[old_size:, :old_size] = cross_covariance
        expanded_covariance[:old_size, old_size:] = cross_covariance.T
        expanded_covariance[old_size:, old_size:] = landmark_covariance
        self.covariance = expanded_covariance
        self.landmark_indices[tag_frame] = old_size

    def update(
        self, tag_frame: str, observed_tag_base: np.ndarray
    ) -> Optional[float]:
        """Correct robot and landmark estimates from a tag position.

        Return the squared Mahalanobis distance, or None when the observation
        is outside the configured validation gate.
        """
        x, y, yaw = self.state[:3]
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        landmark_index = self.landmark_indices[tag_frame]
        dx = float(self.state[landmark_index] - x)
        dy = float(self.state[landmark_index + 1] - y)
        predicted = np.array([
            cos_yaw * dx + sin_yaw * dy,
            -sin_yaw * dx + cos_yaw * dy,
        ])
        innovation = observed_tag_base - predicted
        measurement_jacobian = np.zeros((2, self.state.size))
        measurement_jacobian[:, :3] = np.array([
            [-cos_yaw, -sin_yaw, predicted[1]],
            [sin_yaw, -cos_yaw, -predicted[0]],
        ])
        measurement_jacobian[:, landmark_index:landmark_index + 2] = (
            np.array([
                [cos_yaw, sin_yaw],
                [-sin_yaw, cos_yaw],
            ])
        )
        innovation_covariance = (
            measurement_jacobian @ self.covariance
            @ measurement_jacobian.T + self.measurement_noise
        )
        mahalanobis_squared = float(
            innovation.T @ np.linalg.solve(innovation_covariance, innovation)
        )
        if mahalanobis_squared > self.gate_threshold:
            return None

        kalman_gain = np.linalg.solve(
            innovation_covariance,
            measurement_jacobian @ self.covariance,
        ).T
        self.state += kalman_gain @ innovation
        self.state[2] = wrap_angle(float(self.state[2]))
        covariance_update = (
            np.eye(self.state.size) - kalman_gain @ measurement_jacobian
        )
        self.covariance = (
            covariance_update @ self.covariance @ covariance_update.T
            + kalman_gain @ self.measurement_noise @ kalman_gain.T
        )
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        return mahalanobis_squared


class AprilTagEKFLocalizer(Node):
    """Estimate robot pose and an unknown AprilTag map."""

    def __init__(self):
        super().__init__('ekf_localizer')

        self.world_frame = str(
            self.declare_parameter('world_frame', 'odom').value
        )
        self.camera_frame = str(
            self.declare_parameter('camera_frame', 'camera_frame').value
        )
        self.tag_max_age = float(
            self.declare_parameter('tag_max_age', 0.60).value
        )
        self.command_timeout = float(
            self.declare_parameter('command_timeout', 0.50).value
        )
        initial_x = float(
            self.declare_parameter('initial_x', 0.0).value
        )
        initial_y = float(
            self.declare_parameter('initial_y', 0.0).value
        )
        initial_yaw = float(
            self.declare_parameter('initial_yaw', 0.0).value
        )
        initial_position_std = float(
            self.declare_parameter('initial_position_std', 0.1).value
        )
        initial_yaw_std = float(
            self.declare_parameter('initial_yaw_std', 0.1).value
        )
        linear_velocity_std = float(
            self.declare_parameter('linear_velocity_std', 0.02).value # 0.01
        )
        angular_velocity_std = float(
            self.declare_parameter('angular_velocity_std', 0.20).value # 0.25
        )
        tag_position_x_std = float(
            self.declare_parameter('tag_position_x_std', 0.08).value # 0.013
        )
        tag_position_y_std = float(
            self.declare_parameter('tag_position_y_std', 0.08).value # 0.017
        )
        self.gate_threshold = float(
            self.declare_parameter('mahalanobis_gate', 12).value
        )
        positive_parameters = (
            self.tag_max_age, self.command_timeout, linear_velocity_std,
            angular_velocity_std, tag_position_x_std, tag_position_y_std,
            self.gate_threshold,
        )
        if any(value <= 0.0 for value in positive_parameters):
            raise ValueError('EKF noise, age, timeout, and gate parameters '
                             'must be positive.')
        if initial_position_std < 0.0 or initial_yaw_std < 0.0:
            raise ValueError('Initial pose standard deviations must be '
                             'non-negative.')

        self.ekf = PlanarRobotEKF(
            initial_state=np.array([
                initial_x, initial_y, wrap_angle(initial_yaw),
            ]),
            initial_covariance=np.diag([
                initial_position_std ** 2,
                initial_position_std ** 2,
                initial_yaw_std ** 2,
            ]),
            control_noise=np.diag([
                linear_velocity_std ** 2,
                angular_velocity_std ** 2,
            ]),
            measurement_noise=np.diag([
                tag_position_x_std ** 2,
                tag_position_y_std ** 2,
            ]),
            gate_threshold=self.gate_threshold,
        )

        camera_xyz = np.array([0.09, 0.0, 0.127])
        camera_rpy = (-math.pi / 2.0, 0.0, -math.pi / 2.0)
        self.rotation_base_camera = tf_transformations.euler_matrix(
            *camera_rpy, axes='sxyz'
        )[:3, :3]
        self.translation_base_camera = camera_xyz

        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.angular_command_direction = 0.0
        self.angular_command_elapsed = 0.0
        self.last_command_time: Optional[float] = None
        now = self._now_seconds()
        self.last_prediction_time = now
        self.last_used_tag_stamp: dict[str, float] = {}
        self.last_log_time: Optional[float] = None

        self.pose_publisher = self.create_publisher(
            PoseStamped, '/robot_pose_estimate', 10
        )
        self.tag_positions_publisher = self.create_publisher(
            String, '/tag_positions', 10
        )
        self.create_subscription(
            Twist, '/cmd_vel', self.command_callback, 10
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(0.02, self.filter_step)

        self.get_logger().info(
            'AprilTag EKF localizer started; subscribes to /cmd_vel and '
            'publishes /robot_pose_estimate.'
        )

    def _now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def command_callback(self, msg: Twist):
        """Store the most recently commanded planar velocity."""
        self.linear_velocity = float(msg.linear.x)
        self.angular_velocity = float(msg.angular.z)
        self.last_command_time = self._now_seconds()

    def _predict(self, now: float):
        """Advance the filter to the current time."""
        dt = now - self.last_prediction_time
        if dt < 0.0:
            self.get_logger().warning(
                'ROS time moved backwards; resetting the EKF time baseline.'
            )
            self.last_prediction_time = now
            self.last_command_time = None
            self.linear_velocity = 0.0
            self.angular_velocity = 0.0
            self.angular_command_direction = 0.0
            self.angular_command_elapsed = 0.0
            self.last_used_tag_stamp.clear()
            return
        if dt == 0.0:
            return

        command_is_fresh = (
            self.last_command_time is not None
            and now - self.last_command_time <= self.command_timeout
        )
        linear_velocity = self.linear_velocity if command_is_fresh else 0.0
        angular_command = self.angular_velocity if command_is_fresh else 0.0
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
        angular_velocity = angular_displacement / dt
        self.ekf.predict(linear_velocity, angular_velocity, dt)
        self.last_prediction_time = now

    def _observe_visible_tags(self, now: float) -> list[str]:
        """Discover and process each new, recent tag observation."""
        accepted_tags = []
        frame_graph = yaml.safe_load(
            self.tf_buffer.all_frames_as_yaml()
        ) or {}
        if not isinstance(frame_graph, dict):
            raise ValueError('TF buffer returned an invalid frame graph.')

        tag_frames = sorted(
            frame for frame in frame_graph if frame.startswith('tag_')
        )
        for tag_frame in tag_frames:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.camera_frame,
                    tag_frame,
                    Time(),
                    timeout=Duration(seconds=0.0),
                )
                stamp = transform.header.stamp
                # Debug
                stamp_seconds = stamp.sec + stamp.nanosec * 1e-9

                self.get_logger().info(
                    f'{tag_frame}: '
                    f'now={now:.3f}, '
                    f'tf_stamp={stamp_seconds:.3f}, '
                    f'age={now - stamp_seconds:.3f}'
                )
                # Debug end
            except TransformException:
                continue

            stamp = transform.header.stamp
            stamp_seconds = stamp.sec + stamp.nanosec * 1e-9
            age = now - stamp_seconds
            last_used_stamp = self.last_used_tag_stamp.get(tag_frame, 0.0)
            if stamp_seconds <= 0.0:
                reason = f'invalid timestamp {stamp_seconds:.9f}'
            elif stamp_seconds <= last_used_stamp:
                reason = (
                    f'timestamp {stamp_seconds:.9f} is not newer than '
                    f'last-used timestamp {last_used_stamp:.9f}'
                )
            elif age < -0.10:
                reason = f'transform is {-age:.3f}s in the future'
            elif age > self.tag_max_age:
                reason = (
                    f'transform age {age:.3f}s exceeds '
                    f'tag_max_age {self.tag_max_age:.3f}s'
                )
            else:
                reason = None

            if reason is not None:
                self.get_logger().debug(
                    f'Skipping {tag_frame} observation: {reason}.'
                )
                continue

            self.last_used_tag_stamp[tag_frame] = stamp_seconds
            translation = transform.transform.translation
            camera_to_tag = np.array([
                translation.x, translation.y, translation.z,
            ])
            tag_position_base = (
                self.translation_base_camera
                + self.rotation_base_camera @ camera_to_tag
            )
            if tag_frame not in self.ekf.landmark_indices:
                self.ekf.add_landmark(tag_frame, tag_position_base[:2])
            else:
                mahalanobis_squared = self.ekf.update(
                    tag_frame, tag_position_base[:2]
                )
                if mahalanobis_squared is None:
                    self.get_logger().warning(
                        f'Rejected {tag_frame} position measurement: '
                        'outside the Mahalanobis validation gate.'
                    )
                    continue
            accepted_tags.append(tag_frame)

        return accepted_tags

    def _publish_pose(self, now: float, accepted_tags: list[str]):
        """Publish the current EKF pose in the configured world frame."""
        x, y, yaw = self.ekf.state[:3]
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self.world_frame
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = 0.0
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        self.pose_publisher.publish(pose)

        if (
            self.last_log_time is None
            or now - self.last_log_time >= 1.0
            or now < self.last_log_time
        ):
            self.get_logger().info(
                f'EKF pose=({x:.3f}, {y:.3f}, {yaw:.3f}), '
                f'command=(v={self.linear_velocity:.3f}, '
                f'w={self.angular_velocity:.3f}), '
                f'position_std=({math.sqrt(self.ekf.covariance[0, 0]):.3f}, '
                f'{math.sqrt(self.ekf.covariance[1, 1]):.3f}), '
                f'tags={",".join(accepted_tags) or "none"}'
            )
            self.last_log_time = now

    def _publish_tag_positions(self):
        """Publish the current estimated positions of all mapped tags."""
        tags = []
        for tag_frame, index in sorted(self.ekf.landmark_indices.items()):
            tags.append({
                'name': tag_frame,
                'x': float(self.ekf.state[index]),
                'y': float(self.ekf.state[index + 1]),
            })
        message = String()
        message.data = json.dumps({
            'frame': self.world_frame,
            'tags': tags,
        })
        self.tag_positions_publisher.publish(message)

    def filter_step(self):
        """Run one prediction, measurement-update, and publication cycle."""
        now = self._now_seconds()
        self._predict(now)
        accepted_tags = self._observe_visible_tags(now)
        self._publish_pose(now, accepted_tags)
        self._publish_tag_positions()


def main(args=None):
    """Run the standalone AprilTag EKF localization node."""
    rclpy.init(args=args)
    node = AprilTagEKFLocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
