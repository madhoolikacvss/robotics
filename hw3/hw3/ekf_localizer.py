#!/usr/bin/env python3
"""Estimate the robot's planar pose from velocity commands and AprilTags."""

import json
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import rclpy
from apriltag_msgs.msg import AprilTagDetectionArray
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.duration import Duration
from rclpy.logging import LoggingSeverity
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from .camera_transform import camera_to_base_translation
from .waypoint_motion import (
    LINEAR_VELOCITY_SCALE,
    estimate_angular_displacement,
)

POSE_PUBLISH_TIMEOUT = 2.0
HISTORY_SECONDS = 2.0
MAX_HISTORY_ENTRIES = 256


def wrap_angle(angle: float) -> float:
    """Normalize an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class MotionInterval:
    """Command and angular-model state for one prediction interval."""

    linear_velocity: float
    angular_command: float
    direction: float
    elapsed_start: float


@dataclass
class EKFHistoryEntry:
    """Complete EKF and motion-model snapshot at one ROS timestamp."""

    timestamp: float
    incoming: Optional[MotionInterval]
    observations: list[tuple[str, np.ndarray]] = field(default_factory=list)
    state: np.ndarray = field(default_factory=lambda: np.empty(0))
    covariance: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    landmark_indices: dict[str, int] = field(default_factory=dict)
    angular_elapsed: float = 0.0
    angular_direction: float = 0.0
    prior_state: np.ndarray = field(default_factory=lambda: np.empty(0))
    prior_covariance: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0))
    )
    prior_landmark_indices: dict[str, int] = field(default_factory=dict)
    prior_angular_elapsed: float = 0.0
    prior_angular_direction: float = 0.0
    accepted_tags: list[str] = field(default_factory=list)
    diagnostics: list[
        tuple[str, bool, float, np.ndarray, np.ndarray]
    ] = field(default_factory=list)


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
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

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
        self.covariance = 0.5 * (
            expanded_covariance + expanded_covariance.T
        )
        self.landmark_indices[tag_frame] = old_size

    def update_batch(
        self, observations: list[tuple[str, np.ndarray]]
    ) -> list[tuple[str, bool, float, np.ndarray]]:
        """Correct the state from simultaneous tag position observations."""
        x, y, yaw = self.state[:3]
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        linearized_measurements = []
        diagnostics = []
        for tag_frame, observed_tag_base in observations:
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
                innovation.T
                @ np.linalg.solve(innovation_covariance, innovation)
            )
            accepted = mahalanobis_squared <= self.gate_threshold
            diagnostics.append((
                tag_frame,
                accepted,
                mahalanobis_squared,
                innovation,
                np.diag(innovation_covariance).copy(),
            ))
            if accepted:
                linearized_measurements.append(
                    (measurement_jacobian, innovation)
                )

        if not linearized_measurements:
            return diagnostics

        measurement_jacobian = np.vstack([
            item[0] for item in linearized_measurements
        ])
        innovation = np.concatenate([
            item[1] for item in linearized_measurements
        ])
        measurement_noise = np.kron(
            np.eye(len(linearized_measurements)), self.measurement_noise
        )
        innovation_covariance = (
            measurement_jacobian @ self.covariance
            @ measurement_jacobian.T + measurement_noise
        )
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
            + kalman_gain @ measurement_noise @ kalman_gain.T
        )
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        return diagnostics


class AprilTagEKFLocalizer(Node):
    """Estimate robot pose and an unknown AprilTag map."""

    def __init__(self):
        super().__init__('ekf_localizer')
        self.get_logger().set_level(LoggingSeverity.DEBUG)

        self.world_frame = str(
            self.declare_parameter('world_frame', 'odom').value
        )
        self.camera_frame = str(
            self.declare_parameter('camera_frame', 'camera_frame').value
        )
        self.tag_max_age = float(
            self.declare_parameter('tag_max_age', HISTORY_SECONDS).value
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
            self.declare_parameter('initial_position_std', 0.001).value
        )
        initial_yaw_std = float(
            self.declare_parameter('initial_yaw_std', 0.001).value
        )
        linear_velocity_std = float(
            self.declare_parameter('linear_velocity_std', 0.02).value # 0.01
        )
        angular_velocity_std = float(
            self.declare_parameter('angular_velocity_std', 0.3).value # 0.25
        )
        tag_position_x_std = float(
            self.declare_parameter('tag_position_x_std', 0.03).value # 0.013
        )
        tag_position_y_std = float(
            self.declare_parameter('tag_position_y_std', 0.03).value # 0.017
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

        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.angular_command_direction = 0.0
        self.angular_command_elapsed = 0.0
        self.last_command_time: Optional[float] = None
        now = self._now_seconds()
        self.last_prediction_time = now
        self.last_pose_publish_time = now
        self.pending_tag_observations: dict[int, dict[str, float]] = {}
        self.last_log_time: Optional[float] = None
        self.history: deque[EKFHistoryEntry] = deque()
        initial_entry = EKFHistoryEntry(timestamp=now, incoming=None)
        self._save_history_state(initial_entry)
        self._save_history_state(initial_entry, prior=True)
        self.history.append(initial_entry)

        self.pose_publisher = self.create_publisher(
            PoseStamped, '/robot_pose_estimate', 10
        )
        self.tag_positions_publisher = self.create_publisher(
            String, '/tag_positions', 10
        )
        self.create_subscription(
            Twist, '/cmd_vel', self.command_callback, 10
        )
        self.create_subscription(
            AprilTagDetectionArray,
            '/detections',
            self.detections_callback,
            10,
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(0.02, self.filter_step)

        self.get_logger().info(
            'AprilTag EKF localizer started; subscribes to /cmd_vel and '
            '/detections, and publishes /robot_pose_estimate.'
        )

    def _now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _check_covariance(self, operation: str):
        """Log the first covariance symmetry or positive-definiteness issue."""
        covariance = self.ekf.covariance
        asymmetry = float(np.max(np.abs(covariance - covariance.T)))
        minimum_eigenvalue = float(np.linalg.eigvalsh(
            0.5 * (covariance + covariance.T)
        )[0])
        if minimum_eigenvalue < -1e-12 or asymmetry > 1e-12:
            self.get_logger().warning(
                f'EKF covariance issue after {operation}: '
                f'min_eigenvalue={minimum_eigenvalue:.3e}, '
                f'max_asymmetry={asymmetry:.3e}.'
            )

    def command_callback(self, msg: Twist):
        """Store the most recently commanded planar velocity."""
        self._predict(self._now_seconds())
        self.linear_velocity = float(msg.linear.x)
        self.angular_velocity = float(msg.angular.z)
        self.last_command_time = self._now_seconds()

    def detections_callback(self, msg: AprilTagDetectionArray):
        """Queue detections by their original camera-frame timestamp."""
        if not msg.detections:
            return
        stamp = msg.header.stamp
        stamp_nanoseconds = stamp.sec * 1_000_000_000 + stamp.nanosec
        arrival_time = self._now_seconds()
        frame_observations = self.pending_tag_observations.setdefault(
            stamp_nanoseconds, {}
        )
        for detection in msg.detections:
            frame_observations.setdefault(
                f'tag_{detection.id}', arrival_time
            )

    def _save_history_state(
        self, entry: EKFHistoryEntry, prior: bool = False
    ):
        """Copy the complete filter and angular-model state into an entry."""
        if prior:
            entry.prior_state = self.ekf.state.copy()
            entry.prior_covariance = self.ekf.covariance.copy()
            entry.prior_landmark_indices = self.ekf.landmark_indices.copy()
            entry.prior_angular_elapsed = self.angular_command_elapsed
            entry.prior_angular_direction = self.angular_command_direction
            return
        entry.state = self.ekf.state.copy()
        entry.covariance = self.ekf.covariance.copy()
        entry.landmark_indices = self.ekf.landmark_indices.copy()
        entry.angular_elapsed = self.angular_command_elapsed
        entry.angular_direction = self.angular_command_direction

    def _restore_history_state(
        self, entry: EKFHistoryEntry, prior: bool = False
    ):
        """Restore the EKF and angular-model state from a history entry."""
        if prior:
            self.ekf.state = entry.prior_state.copy()
            self.ekf.covariance = entry.prior_covariance.copy()
            self.ekf.landmark_indices = entry.prior_landmark_indices.copy()
            self.angular_command_elapsed = entry.prior_angular_elapsed
            self.angular_command_direction = entry.prior_angular_direction
            return
        self.ekf.state = entry.state.copy()
        self.ekf.covariance = entry.covariance.copy()
        self.ekf.landmark_indices = entry.landmark_indices.copy()
        self.angular_command_elapsed = entry.angular_elapsed
        self.angular_command_direction = entry.angular_direction

    def _advance_interval(self, interval: MotionInterval, dt: float):
        """Replay a portion of a recorded command using its angular model."""
        if dt <= 0.0:
            return
        self.angular_command_direction = interval.direction
        self.angular_command_elapsed = interval.elapsed_start
        angular_displacement, self.angular_command_elapsed = (
            estimate_angular_displacement(
                interval.angular_command,
                interval.elapsed_start,
                dt,
            )
        )
        self.ekf.predict(
            interval.linear_velocity * LINEAR_VELOCITY_SCALE,
            angular_displacement / dt,
            dt,
        )
        self._check_covariance('predict')

    def _apply_entry_observations(self, entry: EKFHistoryEntry):
        """Apply one timestamp's landmark initializations and gated update."""
        new_observations = [
            observation for observation in entry.observations
            if observation[0] not in self.ekf.landmark_indices
        ]
        mapped_observations = [
            observation for observation in entry.observations
            if observation[0] in self.ekf.landmark_indices
        ]
        entry.accepted_tags = []
        entry.diagnostics = []
        for tag_frame, observed_tag_base in new_observations:
            self.ekf.add_landmark(tag_frame, observed_tag_base)
            self._check_covariance(f'add_landmark({tag_frame})')
            entry.accepted_tags.append(tag_frame)
        if mapped_observations:
            entry.diagnostics = self.ekf.update_batch(mapped_observations)
            self._check_covariance('update_batch')
            entry.accepted_tags.extend(
                tag
                for tag, accepted, _, _, _ in entry.diagnostics
                if accepted
            )

    def _replay_from(self, start_index: int):
        """Rebuild snapshots and reapply all later timestamped observations."""
        entries = list(self.history)
        if start_index == 0:
            entry = entries[0]
            self._restore_history_state(entry, prior=True)
            self._apply_entry_observations(entry)
            self._save_history_state(entry)
            start_index = 1
        else:
            self._restore_history_state(entries[start_index - 1])

        for index in range(start_index, len(entries)):
            entry = entries[index]
            previous = entries[index - 1]
            if entry.incoming is not None:
                self._advance_interval(
                    entry.incoming, entry.timestamp - previous.timestamp
                )
            self._save_history_state(entry, prior=True)
            self._apply_entry_observations(entry)
            self._save_history_state(entry)

    def _record_prediction(
        self, timestamp: float, interval: MotionInterval
    ):
        """Save the prediction endpoint and bound retained history."""
        entry = EKFHistoryEntry(timestamp=timestamp, incoming=interval)
        self._save_history_state(entry, prior=True)
        self._save_history_state(entry)
        self.history.append(entry)
        cutoff = timestamp - HISTORY_SECONDS
        while (
            len(self.history) > 2
            and self.history[1].timestamp < cutoff
        ):
            self.history.popleft()
        while len(self.history) > MAX_HISTORY_ENTRIES:
            self.history.popleft()
        if self.history:
            self.history[0].incoming = None

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
            self.pending_tag_observations.clear()
            self.history.clear()
            entry = EKFHistoryEntry(timestamp=now, incoming=None)
            self._save_history_state(entry)
            self._save_history_state(entry, prior=True)
            self.history.append(entry)
            self.last_prediction_time = now
            return
        if dt == 0.0:
            return

        start = self.last_prediction_time
        while start < now:
            end = now
            if self.last_command_time is not None:
                expiry = self.last_command_time + self.command_timeout
                if start < expiry < end:
                    end = expiry
            command_is_fresh = (
                self.last_command_time is not None
                and start - self.last_command_time < self.command_timeout
            )
            linear_velocity = self.linear_velocity if command_is_fresh else 0.0
            angular_command = (
                self.angular_velocity if command_is_fresh else 0.0
            )
            direction = (
                math.copysign(1.0, angular_command)
                if abs(angular_command) > 1e-9 else 0.0
            )
            if direction != self.angular_command_direction:
                self.angular_command_elapsed = 0.0
                self.angular_command_direction = direction
            interval = MotionInterval(
                linear_velocity=linear_velocity,
                angular_command=angular_command,
                direction=direction,
                elapsed_start=self.angular_command_elapsed,
            )
            self._advance_interval(interval, end - start)
            self._record_prediction(end, interval)
            start = end
        self.last_prediction_time = now

    def _observe_visible_tags(self, now: float) -> list[str]:
        """Transform and timestamp-align queued camera observations."""
        accepted_tags: list[str] = []
        for stamp_nanoseconds in sorted(list(self.pending_tag_observations)):
            pending = self.pending_tag_observations[stamp_nanoseconds]
            if not pending:
                del self.pending_tag_observations[stamp_nanoseconds]
                continue
            stamp_seconds = stamp_nanoseconds * 1e-9
            age = now - stamp_seconds
            arrival_time = min(pending.values())
            latency = arrival_time - stamp_seconds
            oldest_timestamp = self.history[0].timestamp
            if stamp_seconds <= 0.0:
                self.get_logger().warning(
                    f'Rejecting AprilTag frame: measurement timestamp='
                    f'{stamp_seconds:.9f}, arrival={arrival_time:.9f}, '
                    f'latency={latency:.3f}s, historical state timestamp='
                    'unavailable (invalid measurement timestamp).'
                )
                del self.pending_tag_observations[stamp_nanoseconds]
                continue
            if stamp_seconds < oldest_timestamp:
                self.get_logger().warning(
                    f'Rejecting AprilTag measurement: measurement timestamp='
                    f'{stamp_seconds:.9f}, arrival={arrival_time:.9f}, '
                    f'latency={latency:.3f}s, historical state timestamp='
                    f'{oldest_timestamp:.9f} (oldest available); timestamp '
                    'is older than EKF history.'
                )
                del self.pending_tag_observations[stamp_nanoseconds]
                continue
            if age < -0.10:
                self.get_logger().warning(
                    f'Rejecting AprilTag measurement: measurement timestamp='
                    f'{stamp_seconds:.9f}, arrival={arrival_time:.9f}, '
                    f'latency={latency:.3f}s, historical state timestamp='
                    'unavailable (future timestamp); measurement is '
                    f'{-age:.3f}s in the future.'
                )
                del self.pending_tag_observations[stamp_nanoseconds]
                continue
            if age > self.tag_max_age:
                self.get_logger().warning(
                    f'Rejecting stale AprilTag measurement: measurement '
                    f'timestamp={stamp_seconds:.9f}, '
                    f'arrival={arrival_time:.9f}, latency={latency:.3f}s, '
                    f'historical state timestamp={stamp_seconds:.9f} '
                    f'(not applied); age={age:.3f}s exceeds '
                    f'tag_max_age={self.tag_max_age:.3f}s.'
                )
                del self.pending_tag_observations[stamp_nanoseconds]
                continue
            if stamp_seconds > self.history[-1].timestamp:
                continue

            observations = []
            completed_tags = []
            for tag_frame in sorted(pending):
                try:
                    transform = self.tf_buffer.lookup_transform(
                        self.camera_frame,
                        tag_frame,
                        Time(
                            nanoseconds=stamp_nanoseconds,
                            clock_type=self.get_clock().clock_type,
                        ),
                        timeout=Duration(seconds=0.0),
                    )
                except TransformException:
                    continue
                translation = transform.transform.translation
                camera_to_tag, tag_position_base = (
                    camera_to_base_translation((
                        translation.x,
                        translation.y,
                        translation.z,
                    ))
                )
                self.get_logger().info(
                    f'AprilTag read {tag_frame} at {stamp_seconds:.9f}: '
                    f'camera_raw=({translation.x:.3f}, '
                    f'{translation.y:.3f}, {translation.z:.3f})m; '
                    f'camera_corrected=({camera_to_tag[0]:.3f}, '
                    f'{camera_to_tag[1]:.3f}, {camera_to_tag[2]:.3f})m; '
                    f'base_xy_used=({tag_position_base[0]:.3f}, '
                    f'{tag_position_base[1]:.3f})m.'
                )
                observations.append((tag_frame, tag_position_base[:2]))
                completed_tags.append(tag_frame)

            if not observations:
                continue
            arrival_time = min(pending[tag] for tag in completed_tags)
            for tag_frame in completed_tags:
                del pending[tag_frame]
            if not pending:
                del self.pending_tag_observations[stamp_nanoseconds]
            accepted_tags.extend(self._apply_timestamped_observations(
                stamp_nanoseconds, stamp_seconds, arrival_time, observations
            ))
        return accepted_tags

    def _apply_timestamped_observations(
        self,
        stamp_nanoseconds: int,
        timestamp: float,
        arrival_time: float,
        observations: list[tuple[str, np.ndarray]],
    ) -> list[str]:
        """Insert one frame's observations and replay the affected history."""
        entries = list(self.history)
        exact_index = next((
            index for index, entry in enumerate(entries)
            if abs(entry.timestamp - timestamp) < 1e-9
        ), None)
        if exact_index is not None:
            index = exact_index
            entry = entries[index]
            known_tags = {tag for tag, _ in entry.observations}
            observations = [
                (tag, value) for tag, value in observations
                if tag not in known_tags
            ]
            if not observations:
                return []
            entry.observations.extend(observations)
        else:
            next_index = next(
                index for index, entry in enumerate(entries)
                if entry.timestamp > timestamp
            )
            previous = entries[next_index - 1]
            next_entry = entries[next_index]
            original_interval = next_entry.incoming
            if original_interval is None:
                self.get_logger().error(
                    'Cannot align an AprilTag measurement: missing recorded '
                    'prediction interval.'
                )
                return []
            self._restore_history_state(previous)
            self._advance_interval(
                original_interval, timestamp - previous.timestamp
            )
            partial_interval = MotionInterval(
                linear_velocity=original_interval.linear_velocity,
                angular_command=original_interval.angular_command,
                direction=original_interval.direction,
                elapsed_start=original_interval.elapsed_start,
            )
            entry = EKFHistoryEntry(
                timestamp=timestamp,
                incoming=partial_interval,
                observations=list(observations),
            )
            self._save_history_state(entry, prior=True)
            self._save_history_state(entry)
            next_entry.incoming = MotionInterval(
                linear_velocity=original_interval.linear_velocity,
                angular_command=original_interval.angular_command,
                direction=original_interval.direction,
                elapsed_start=self.angular_command_elapsed,
            )
            self.history.insert(next_index, entry)
            index = next_index

        latency = arrival_time - timestamp
        # self.get_logger().info(
        #     f'AprilTag measurement timestamp={timestamp:.9f}, '
        #     f'arrival={arrival_time:.9f}, latency={latency:.3f}s, '
        #     f'historical state timestamp={timestamp:.9f}; '
        #     f'tags={",".join(tag for tag, _ in observations)}.'
        # )
        self._replay_from(index)
        updated_entry = list(self.history)[index]
        yaw_correction = wrap_angle(float(
            updated_entry.state[2] - updated_entry.prior_state[2]
        ))
        if abs(yaw_correction) >= 0.02:
            accepted_innovations = [
                f'{tag_frame}=({innovation[0]:.3f}, '
                f'{innovation[1]:.3f})m, d^2={mahalanobis_squared:.3f}'
                for tag_frame, accepted, mahalanobis_squared, innovation, _
                in updated_entry.diagnostics
                if accepted and tag_frame in {
                    item[0] for item in observations
                }
            ]
            innovation_summary = "; ".join(accepted_innovations) or "none"
            self.get_logger().warning(
                f'AprilTag measurements changed EKF yaw by '
                f'{yaw_correction:+.3f}rad at {timestamp:.9f}; '
                f'tags={",".join(tag for tag, _ in observations)}; '
                f'accepted innovations={innovation_summary}.'
            )
        for (
            tag_frame,
            accepted,
            mahalanobis_squared,
            innovation,
            innovation_covariance_diagonal,
        ) in (
            updated_entry.diagnostics
        ):
            if tag_frame not in {tag for tag, _ in observations}:
                continue
            innovation_x, innovation_y = innovation
            if not accepted:
                self.get_logger().warning(
                    f'Rejected {tag_frame} position measurement at '
                    f'{timestamp:.9f}: outside the Mahalanobis validation '
                    f'gate; innovation=({innovation_x:.3f}, '
                    f'{innovation_y:.3f})m, '
                    f'Mahalanobis d^2={mahalanobis_squared:.3f}, '
                    f'diag(S)=({innovation_covariance_diagonal[0]:.3e}, '
                    f'{innovation_covariance_diagonal[1]:.3e}).'
                )
        for tag_frame in updated_entry.accepted_tags:
            if tag_frame in {tag for tag, _ in observations}:
                if tag_frame not in {
                    item[0] for item in updated_entry.diagnostics
                }:
                    self.get_logger().info(
                        f'Initialized landmark {tag_frame} at timestamp '
                        f'{stamp_nanoseconds}; no pose correction or '
                        'innovation.'
                    )
                else:
                    pass
                    # self.get_logger().debug(
                    #     f'Updated {tag_frame} at timestamp '
                    #     f'{stamp_nanoseconds}.'
                    # )
        return updated_entry.accepted_tags

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
                'cov_xx': float(self.ekf.covariance[index, index]),
                'cov_xy': float(self.ekf.covariance[index, index + 1]),
                'cov_yy': float(self.ekf.covariance[index + 1, index + 1]),
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
        if (
            accepted_tags
            or now - self.last_pose_publish_time >= POSE_PUBLISH_TIMEOUT
            or now < self.last_pose_publish_time
        ):
            self._publish_pose(now, accepted_tags)
            self.last_pose_publish_time = now
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
