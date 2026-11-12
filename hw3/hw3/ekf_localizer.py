#!/usr/bin/env python3
"""Estimate the robot's planar pose from velocity commands and AprilTags.

State: [x, y, yaw, (lx, ly, phi) for each tag]
  (lx, ly) is the tag position in the odom frame.
  phi is the direction of the tag's outward normal in the odom frame.
No tag locations are predefined; every landmark is initialized from its
first sighting.
"""

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

HISTORY_SECONDS = 2.0
MAX_HISTORY_ENTRIES = 256
LANDMARK_SIZE = 3  # lx, ly, phi
TAG_READ_TIMEOUT = 1.5


def wrap_angle(angle: float) -> float:
    """Normalize an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def quaternion_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Convert a quaternion to a 3x3 rotation matrix."""
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-9:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


@dataclass
class TagMeasurement:
    """One tag sighting expressed in the robot base frame."""

    tag: str
    position: np.ndarray        # (2,) tag xy in base frame
    position_noise: np.ndarray  # (2, 2)
    psi: float                  # tag-normal direction in base frame
    psi_var: float
    psi_valid: bool


@dataclass
class TagDiagnostic:
    tag: str
    position_accepted: bool
    position_d2: float
    position_innovation: np.ndarray
    orientation_used: bool
    orientation_accepted: bool
    orientation_d2: float
    orientation_innovation: float


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
    observations: list[TagMeasurement] = field(default_factory=list)
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
    diagnostics: list[TagDiagnostic] = field(default_factory=list)
    recovery: bool = False


class PlanarRobotEKF:
    """EKF-SLAM: unicycle robot, landmarks with position and normal angle."""

    def __init__(
        self,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        linear_noise_density: float,
        linear_noise_gain: float,
        position_noise_density: float,
        yaw_noise_density: float,
        turn_noise_gain: float,
        yaw_noise_per_linear: float,
        position_gate: float,
        orientation_gate: float,
    ):
        self.state = initial_state.astype(float)
        self.covariance = initial_covariance.astype(float)
        self.linear_noise_density = linear_noise_density
        self.linear_noise_gain = linear_noise_gain
        self.position_noise_density = position_noise_density
        self.yaw_noise_density = yaw_noise_density
        self.turn_noise_gain = turn_noise_gain
        self.yaw_noise_per_linear = yaw_noise_per_linear
        self.position_gate = position_gate
        self.orientation_gate = orientation_gate
        self.landmark_indices: dict[str, int] = {}

    def predict(
        self, linear_velocity: float, angular_velocity: float, dt: float
    ):
        """Propagate the pose and covariance using the modeled twist.

        Process noise is continuous-time white noise, so the covariance added
        is proportional to dt and independent of how the interval is split.
        """
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

        state_jacobian = np.eye(self.state.size)
        state_jacobian[0, 2] = -linear_velocity * sin_yaw * dt
        state_jacobian[1, 2] = linear_velocity * cos_yaw * dt

        sigma_v = (
            self.linear_noise_density
            + self.linear_noise_gain * abs(linear_velocity)
        )
        sigma_w = (
            self.yaw_noise_density
            + self.turn_noise_gain * abs(angular_velocity)
            + self.yaw_noise_per_linear * abs(linear_velocity)
        )
        noise_input = np.zeros((self.state.size, 2))
        noise_input[:3, :] = np.array([
            [cos_yaw, 0.0],
            [sin_yaw, 0.0],
            [0.0, 1.0],
        ])
        noise = np.diag([sigma_v ** 2, sigma_w ** 2]) * dt
        self.covariance = (
            state_jacobian @ self.covariance @ state_jacobian.T
            + noise_input @ noise @ noise_input.T
        )
        self.covariance[0, 0] += self.position_noise_density ** 2 * dt
        self.covariance[1, 1] += self.position_noise_density ** 2 * dt
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    def inflate_pose(self, position_std: float, yaw_std: float):
        """Add pose uncertainty (keeps P positive semidefinite)."""
        self.covariance[0, 0] += position_std ** 2
        self.covariance[1, 1] += position_std ** 2
        self.covariance[2, 2] += yaw_std ** 2

    def add_landmark(self, measurement: TagMeasurement):
        """Initialize a newly seen landmark from the current pose."""
        tag_frame = measurement.tag
        if tag_frame in self.landmark_indices:
            return

        x, y, yaw = self.state[:3]
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        bx, by = float(measurement.position[0]), float(measurement.position[1])
        if measurement.psi_valid:
            psi = measurement.psi
            psi_var = measurement.psi_var
        else:
            psi = 0.0
            psi_var = math.pi ** 2  # unknown orientation

        landmark = np.array([
            x + cos_yaw * bx - sin_yaw * by,
            y + sin_yaw * bx + cos_yaw * by,
            wrap_angle(yaw + psi),
        ])
        robot_jacobian = np.array([
            [1.0, 0.0, -sin_yaw * bx - cos_yaw * by],
            [0.0, 1.0, cos_yaw * bx - sin_yaw * by],
            [0.0, 0.0, 1.0],
        ])
        observation_jacobian = np.array([
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ])
        observation_noise = np.zeros((3, 3))
        observation_noise[:2, :2] = measurement.position_noise
        observation_noise[2, 2] = psi_var

        old_size = self.state.size
        cross_covariance = robot_jacobian @ self.covariance[:3, :]
        landmark_covariance = (
            robot_jacobian @ self.covariance[:3, :3] @ robot_jacobian.T
            + observation_jacobian @ observation_noise
            @ observation_jacobian.T
        )

        self.state = np.concatenate((self.state, landmark))
        expanded = np.zeros((old_size + 3, old_size + 3))
        expanded[:old_size, :old_size] = self.covariance
        expanded[old_size:, :old_size] = cross_covariance
        expanded[:old_size, old_size:] = cross_covariance.T
        expanded[old_size:, old_size:] = landmark_covariance
        self.covariance = 0.5 * (expanded + expanded.T)
        self.landmark_indices[tag_frame] = old_size

    def update_batch(
        self, observations: list[TagMeasurement]
    ) -> list[TagDiagnostic]:
        """Correct the state from simultaneous tag observations.

        Position (2 DOF) and orientation (1 DOF) are gated independently.
        """
        x, y, yaw = self.state[:3]
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        size = self.state.size
        rows = []
        innovations = []
        noises = []
        diagnostics = []

        for obs in observations:
            index = self.landmark_indices[obs.tag]
            dx = float(self.state[index] - x)
            dy = float(self.state[index + 1] - y)
            predicted = np.array([
                cos_yaw * dx + sin_yaw * dy,
                -sin_yaw * dx + cos_yaw * dy,
            ])

            h_pos = np.zeros((2, size))
            h_pos[:, :3] = np.array([
                [-cos_yaw, -sin_yaw, predicted[1]],
                [sin_yaw, -cos_yaw, -predicted[0]],
            ])
            h_pos[:, index:index + 2] = np.array([
                [cos_yaw, sin_yaw],
                [-sin_yaw, cos_yaw],
            ])
            innovation_pos = obs.position - predicted
            s_pos = h_pos @ self.covariance @ h_pos.T + obs.position_noise
            d2_pos = float(
                innovation_pos @ np.linalg.solve(s_pos, innovation_pos)
            )
            pos_ok = d2_pos <= self.position_gate

            ori_ok = False
            d2_ori = 0.0
            innovation_ori = 0.0
            h_ori = None
            if obs.psi_valid:
                h_ori = np.zeros((1, size))
                h_ori[0, 2] = -1.0
                h_ori[0, index + 2] = 1.0
                predicted_psi = wrap_angle(float(self.state[index + 2] - yaw))
                innovation_ori = wrap_angle(obs.psi - predicted_psi)
                s_ori = float(
                    (h_ori @ self.covariance @ h_ori.T)[0, 0]
                ) + obs.psi_var
                d2_ori = innovation_ori ** 2 / s_ori
                ori_ok = d2_ori <= self.orientation_gate

            diagnostics.append(TagDiagnostic(
                tag=obs.tag,
                position_accepted=pos_ok,
                position_d2=d2_pos,
                position_innovation=innovation_pos,
                orientation_used=obs.psi_valid,
                orientation_accepted=ori_ok,
                orientation_d2=d2_ori,
                orientation_innovation=innovation_ori,
            ))
            if pos_ok:
                rows.append(h_pos)
                innovations.append(innovation_pos)
                noises.append(obs.position_noise)
            if ori_ok:
                rows.append(h_ori)
                innovations.append(np.array([innovation_ori]))
                noises.append(np.array([[obs.psi_var]]))

        if not rows:
            return diagnostics

        measurement_jacobian = np.vstack(rows)
        innovation = np.concatenate(innovations)
        measurement_noise = np.zeros((innovation.size, innovation.size))
        offset = 0
        for block in noises:
            width = block.shape[0]
            measurement_noise[offset:offset + width,
                              offset:offset + width] = block
            offset += width

        innovation_covariance = (
            measurement_jacobian @ self.covariance @ measurement_jacobian.T
            + measurement_noise
        )
        kalman_gain = np.linalg.solve(
            innovation_covariance,
            measurement_jacobian @ self.covariance,
        ).T
        self.state += kalman_gain @ innovation
        self.state[2] = wrap_angle(float(self.state[2]))
        for index in self.landmark_indices.values():
            self.state[index + 2] = wrap_angle(float(self.state[index + 2]))
        covariance_update = (
            np.eye(size) - kalman_gain @ measurement_jacobian
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

        def param(name, default):
            return self.declare_parameter(name, default).value

        self.world_frame = str(param('world_frame', 'odom'))
        self.camera_frame = str(param('camera_frame', 'camera_frame'))
        self.tag_max_age = float(param('tag_max_age', HISTORY_SECONDS))
        self.command_timeout = float(param('command_timeout', 0.50))
        initial_x = float(param('initial_x', 0.0))
        initial_y = float(param('initial_y', 0.0))
        initial_yaw = float(param('initial_yaw', 0.0))
        initial_position_std = float(param('initial_position_std', 0.001))
        initial_yaw_std = float(param('initial_yaw_std', 0.001))

        # Process noise densities (units per sqrt(second)).
        linear_noise_density = float(param('linear_noise_density', 0.03))
        linear_noise_gain = float(param('linear_noise_gain', 0.10))
        position_noise_density = float(param('position_noise_density', 0.02))
        yaw_noise_density = float(param('yaw_noise_density', 0.05))
        turn_noise_gain = float(param('turn_noise_gain', 0.25))
        yaw_noise_per_linear = float(param('yaw_noise_per_linear', 0.10))

        # Range-dependent tag measurement noise. Base x is forward/range.
        self.range_std_base = float(param('tag_range_std_base', 0.05))
        self.range_std_per_m = float(param('tag_range_std_per_m', 0.02))
        self.lateral_std_base = float(param('tag_lateral_std_base', 0.04))
        self.lateral_std_per_m = float(param('tag_lateral_std_per_m', 0.01))
        self.psi_std_base = float(param('tag_psi_std_base', 0.08))
        self.psi_std_per_m = float(param('tag_psi_std_per_m', 0.03))
        self.psi_frontal_std = float(param('tag_psi_frontal_std', 0.25))
        self.psi_frontal_angle = float(param('tag_psi_frontal_angle', 0.35))
        self.use_tag_orientation = bool(param('use_tag_orientation', True))

        self.gate_threshold = float(param('mahalanobis_gate', 16.0))
        self.orientation_gate = float(param('orientation_gate', 9.0))
        self.recovery_after_frames = int(param('recovery_after_frames', 5))
        self.recovery_position_std = float(param('recovery_position_std', 0.3))
        self.recovery_yaw_std = float(param('recovery_yaw_std', 0.6))

        positive_parameters = (
            self.tag_max_age, self.command_timeout,
            linear_noise_density, position_noise_density,
            yaw_noise_density, self.range_std_base, self.lateral_std_base,
            self.psi_std_base, self.psi_frontal_angle,
            self.gate_threshold, self.orientation_gate,
        )
        if any(value <= 0.0 for value in positive_parameters):
            raise ValueError('EKF noise, age, timeout, and gate parameters '
                             'must be positive.')
        if initial_position_std < 0.0 or initial_yaw_std < 0.0:
            raise ValueError('Initial pose standard deviations must be '
                             'non-negative.')

        self.get_logger().info(
            'Loaded EKF parameters: '
            f'init_std=({initial_position_std}, {initial_yaw_std}), '
            f'process=(lin={linear_noise_density}, '
            f'lin_gain={linear_noise_gain}, pos={position_noise_density}, '
            f'yaw={yaw_noise_density}, turn_gain={turn_noise_gain}, '
            f'yaw_per_v={yaw_noise_per_linear}), '
            f'tag_range=({self.range_std_base}+{self.range_std_per_m}*r), '
            f'tag_lateral=({self.lateral_std_base}+'
            f'{self.lateral_std_per_m}*r), '
            f'tag_psi=({self.psi_std_base}+{self.psi_std_per_m}*r), '
            f'use_orientation={self.use_tag_orientation}, '
            f'gates=(pos={self.gate_threshold}, '
            f'ori={self.orientation_gate}), '
            f'recovery=({self.recovery_after_frames} frames, '
            f'{self.recovery_position_std}m, {self.recovery_yaw_std}rad)'
        )

        self.ekf = PlanarRobotEKF(
            initial_state=np.array([
                initial_x, initial_y, wrap_angle(initial_yaw),
            ]),
            initial_covariance=np.diag([
                initial_position_std ** 2,
                initial_position_std ** 2,
                initial_yaw_std ** 2,
            ]),
            linear_noise_density=linear_noise_density,
            linear_noise_gain=linear_noise_gain,
            position_noise_density=position_noise_density,
            yaw_noise_density=yaw_noise_density,
            turn_noise_gain=turn_noise_gain,
            yaw_noise_per_linear=yaw_noise_per_linear,
            position_gate=self.gate_threshold,
            orientation_gate=self.orientation_gate,
        )

        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.angular_command_direction = 0.0
        self.angular_command_elapsed = 0.0
        self.last_command_time: Optional[float] = None
        now = self._now_seconds()
        self.last_prediction_time = now
        self.pending_tag_observations: dict[int, dict[str, float]] = {}
        self.tag_freshness_reference_time = now
        self.tag_read_timeout_logged = False
        self.last_log_time: Optional[float] = None
        self.reject_streak = 0
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
        arrival_time = self._now_seconds()
        self.tag_freshness_reference_time = arrival_time
        self.tag_read_timeout_logged = False
        stamp = msg.header.stamp
        stamp_nanoseconds = stamp.sec * 1_000_000_000 + stamp.nanosec
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

    def _apply_entry_observations(self, entry: EKFHistoryEntry):
        """Apply one timestamp's gated update, then landmark initialization.

        Mapped tags correct the pose first so new landmarks are initialized
        from the corrected pose.
        """
        if entry.recovery:
            self.ekf.inflate_pose(
                self.recovery_position_std, self.recovery_yaw_std
            )
        new_observations = [
            obs for obs in entry.observations
            if obs.tag not in self.ekf.landmark_indices
        ]
        mapped_observations = [
            obs for obs in entry.observations
            if obs.tag in self.ekf.landmark_indices
        ]
        entry.accepted_tags = []
        entry.diagnostics = []
        if mapped_observations:
            entry.diagnostics = self.ekf.update_batch(mapped_observations)
            entry.accepted_tags.extend(
                d.tag for d in entry.diagnostics
                if d.position_accepted or d.orientation_accepted
            )
        for obs in new_observations:
            self.ekf.add_landmark(obs)
            entry.accepted_tags.append(obs.tag)

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
            self.tag_freshness_reference_time = now
            self.tag_read_timeout_logged = False
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

    def _build_measurement(self, tag_frame: str, transform):
        """Turn a camera->tag transform into a base-frame measurement."""
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        camera_to_tag, tag_position_base = camera_to_base_translation((
            translation.x, translation.y, translation.z,
        ))
        position = np.asarray(tag_position_base, dtype=float)[:2]
        tag_range = float(math.hypot(position[0], position[1]))

        sigma_range = self.range_std_base + self.range_std_per_m * tag_range
        sigma_lateral = (
            self.lateral_std_base + self.lateral_std_per_m * tag_range
        )
        position_noise = np.diag([sigma_range ** 2, sigma_lateral ** 2])

        # Tag normal in the camera frame, flipped to point at the camera.
        tag_rotation = quaternion_to_matrix(
            rotation.x, rotation.y, rotation.z, rotation.w
        )
        normal = tag_rotation[:, 2]
        to_camera = -np.array([translation.x, translation.y, translation.z])
        to_camera_norm = float(np.linalg.norm(to_camera))
        if normal @ to_camera < 0.0:
            normal = -normal
        horizontal = math.hypot(normal[0], normal[2])
        # Camera optical frame: z forward, x right. Base: x forward, y left.
        psi = math.atan2(-normal[0], normal[2])
        view_angle = 0.0
        if to_camera_norm > 1e-6:
            view_angle = math.acos(
                max(-1.0, min(1.0, float(normal @ to_camera) / to_camera_norm))
            )
        psi_valid = (
            self.use_tag_orientation
            and horizontal > 0.3
            and to_camera_norm > 1e-6
        )
        psi_var = (
            (self.psi_std_base + self.psi_std_per_m * tag_range) ** 2
            + (
                self.psi_frontal_std
                * max(0.0, 1.0 - view_angle / self.psi_frontal_angle)
            ) ** 2
        )

        self.get_logger().info(
            f'AprilTag read {tag_frame}: '
            f'camera_raw=({translation.x:.3f}, {translation.y:.3f}, '
            f'{translation.z:.3f})m; '
            f'camera_corrected=({camera_to_tag[0]:.3f}, '
            f'{camera_to_tag[1]:.3f}, {camera_to_tag[2]:.3f})m; '
            f'base_xy_used=({position[0]:.3f}, {position[1]:.3f})m; '
            f'psi={psi:.3f}rad (valid={psi_valid}, '
            f'view={math.degrees(view_angle):.0f}deg, '
            f'std={math.sqrt(psi_var):.3f}).'
        )
        return TagMeasurement(
            tag=tag_frame,
            position=position,
            position_noise=position_noise,
            psi=psi,
            psi_var=psi_var,
            psi_valid=psi_valid,
        )

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

            observations: list[TagMeasurement] = []
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
                observations.append(
                    self._build_measurement(tag_frame, transform)
                )
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
        observations: list[TagMeasurement],
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
            known_tags = {obs.tag for obs in entry.observations}
            observations = [
                obs for obs in observations if obs.tag not in known_tags
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

        self._replay_from(index)
        updated_entry = list(self.history)[index]
        observed_tags = {obs.tag for obs in observations}
        frame_diagnostics = [
            d for d in updated_entry.diagnostics if d.tag in observed_tags
        ]

        # If every mapped tag keeps getting rejected, the pose (not the tags)
        # is what is wrong. Inflate pose uncertainty and re-run the update.
        if frame_diagnostics:
            if any(
                d.position_accepted or d.orientation_accepted
                for d in frame_diagnostics
            ):
                self.reject_streak = 0
            else:
                self.reject_streak += 1
                if (
                    self.reject_streak >= self.recovery_after_frames
                    and not updated_entry.recovery
                ):
                    self.get_logger().warning(
                        f'All mapped tags rejected for {self.reject_streak} '
                        f'frames; inflating pose covariance at '
                        f'{timestamp:.9f} to re-localize.'
                    )
                    updated_entry.recovery = True
                    self.reject_streak = 0
                    self._replay_from(index)
                    updated_entry = list(self.history)[index]
                    frame_diagnostics = [
                        d for d in updated_entry.diagnostics
                        if d.tag in observed_tags
                    ]

        yaw_correction = wrap_angle(float(
            updated_entry.state[2] - updated_entry.prior_state[2]
        ))
        if abs(yaw_correction) >= 0.02:
            summary = []
            for d in frame_diagnostics:
                part = []
                if d.position_accepted:
                    part.append(
                        f'pos=({d.position_innovation[0]:.3f}, '
                        f'{d.position_innovation[1]:.3f})m '
                        f'd^2={d.position_d2:.2f}'
                    )
                if d.orientation_accepted:
                    part.append(
                        f'ori={d.orientation_innovation:+.3f}rad '
                        f'd^2={d.orientation_d2:.2f}'
                    )
                if part:
                    summary.append(f'{d.tag}[' + ' '.join(part) + ']')
            self.get_logger().warning(
                f'AprilTag measurements changed EKF yaw by '
                f'{yaw_correction:+.3f}rad at {timestamp:.9f}; '
                f'tags={",".join(sorted(observed_tags))}; '
                f'accepted={"; ".join(summary) or "none"}.'
            )
        for d in frame_diagnostics:
            if not d.position_accepted:
                self.get_logger().warning(
                    f'Rejected {d.tag} position measurement at '
                    f'{timestamp:.9f}: innovation=('
                    f'{d.position_innovation[0]:.3f}, '
                    f'{d.position_innovation[1]:.3f})m, '
                    f'Mahalanobis d^2={d.position_d2:.3f}.'
                )
            if d.orientation_used and not d.orientation_accepted:
                self.get_logger().warning(
                    f'Rejected {d.tag} orientation measurement at '
                    f'{timestamp:.9f}: innovation='
                    f'{d.orientation_innovation:+.3f}rad, '
                    f'd^2={d.orientation_d2:.3f}.'
                )
        diagnosed_tags = {d.tag for d in updated_entry.diagnostics}
        for tag_frame in updated_entry.accepted_tags:
            if tag_frame in observed_tags and tag_frame not in diagnosed_tags:
                self.get_logger().info(
                    f'Initialized landmark {tag_frame} at timestamp '
                    f'{stamp_nanoseconds}; no pose correction or '
                    'innovation.'
                )
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
                f'std=({math.sqrt(self.ekf.covariance[0, 0]):.3f}, '
                f'{math.sqrt(self.ekf.covariance[1, 1]):.3f}, '
                f'{math.sqrt(self.ekf.covariance[2, 2]):.3f}), '
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
                'phi': float(self.ekf.state[index + 2]),
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
            now - self.tag_freshness_reference_time > TAG_READ_TIMEOUT
        ):
            if not self.tag_read_timeout_logged:
                self.get_logger().warning(
                    f'No AprilTag read for {TAG_READ_TIMEOUT:.1f}s; '
                    'publishing the predicted robot pose.'
                )
                self.tag_read_timeout_logged = True
        else:
            self.tag_read_timeout_logged = False
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
