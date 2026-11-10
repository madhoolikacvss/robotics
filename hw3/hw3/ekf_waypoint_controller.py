#!/usr/bin/env python3
"""Continuously control a differential-drive robot using AprilTag poses."""

import csv
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Optional
from uuid import uuid4

import rclpy
import tf_transformations
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import String, Float64

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
        self.forward_pulse_distance = float(self.declare_parameter(
            'forward_pulse_distance', 0.50
        ).value)
        self.forward_pulse_duration = float(self.declare_parameter(
            'forward_pulse_duration', 0.20
        ).value)
        self.forward_observation_pause = float(self.declare_parameter(
            'forward_observation_pause', 0.40
        ).value)
        if any(not math.isfinite(v) or v <= 0.0 for v in (
            self.forward_pulse_distance, self.forward_pulse_duration,
            self.forward_observation_pause,
        )):
            raise ValueError('Forward pulse parameters must be finite and positive.')
        self.forward_pulse_started: Optional[float] = None
        self.forward_pause_started: Optional[float] = None
        self.angular_burst_duration = float(self.declare_parameter(
            'angular_burst_duration', 0.10
        ).value)
        self.angular_observation_pause = float(self.declare_parameter(
            'angular_observation_pause', 0.40
        ).value)
        if any(not math.isfinite(v) or v <= 0.0 for v in (
            self.angular_burst_duration, self.angular_observation_pause,
        )):
            raise ValueError('Angular burst parameters must be finite and positive.')
        self.angular_burst_started = None
        self.angular_burst_direction = 0.0
        self.angular_pause_started = None
        self.latest_accepted_tag_stamp = None
        self.last_control_log_time: Optional[float] = None
        self.desired_command = Twist()
        self.stale_stop_logged = False
        self.final_waypoint_wait_started: Optional[float] = None
        self.final_waypoint_settled = False
        self.latest_estimated_position: Optional[tuple[float, float, float]] = None
        self.last_pose_received_time: Optional[float] = None
        self.frame_mismatch_logged = False
        self.latest_tag_positions: dict[str, tuple[float, ...]] = {}
        self.latest_map_timestamp: Optional[float] = None
        self.lap_end_indices = list(self.declare_parameter(
            'lap_end_waypoint_indices', [1]
        ).value)
        if (
            self.lap_end_indices != sorted(set(self.lap_end_indices))
            or any(i <= 0 or i >= len(self.motion.waypoints)
                   for i in self.lap_end_indices)
        ):
            raise ValueError('Lap end indices must be increasing valid waypoint indices.')
        self.completed_laps = 0
        self.tag_frame_mismatch_logged = False
        self.tag_position_log_written = False
        self.last_saved_map_timestamp = None
        self.snapshot_id = 0

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(
            Float64, '/accepted_tag_observation_stamp',
            self.accepted_tag_callback, 10,
        )
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
        self.tag_position_log_path = self.tag_position_log_path.resolve()
        self.tag_position_log_path.parent.mkdir(parents=True, exist_ok=True)

        self.map_snapshot_log_path = Path(self.declare_parameter(
            'map_snapshot_log_path', f'map_snapshots_{uuid4().hex[:12]}.csv'
        ).value).expanduser().resolve()
        self.map_snapshot_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.map_snapshot_log_file = self.map_snapshot_log_path.open(
            'x', newline='', encoding='utf-8'
        )
        self.map_snapshot_writer = csv.writer(self.map_snapshot_log_file)
        self.map_snapshot_writer.writerow([
            'timestamp_sec', 'map_timestamp_sec', 'lap', 'waypoint_index',
            'frame', 'landmark_count', 'tag', 'x', 'y',
            'cov_xx', 'cov_xy', 'cov_yy',
            'snapshot_id', 'event',
        ])
        self.map_snapshot_log_file.flush()
        os.fsync(self.map_snapshot_log_file.fileno())
        self.get_logger().info(
            f'Per-lap maps: {self.map_snapshot_log_path}; '
            f'lap end waypoint indices: {self.lap_end_indices}'
        )

        self.timer = self.create_timer(
            self.control_period, self.control_loop
        )
        self.position_log_timer = self.create_timer(
            0.5, self.write_position_to_file
        )
        self.map_log_timer = self.create_timer(1.0, self.log_current_map)

        self.get_logger().info(
            'EKF waypoint controller started: '
            f'position tolerance={POSITION_TOLERANCE:.2f}m, '
            f'pose timeout={self.pose_timeout:.2f}s. '
            f'Using /robot_pose_estimate and writing positions to '
            f'{self.position_log_path} and '
            f'waypoint results to {self.waypoint_log_path}; tag positions '
            f'are saved every second to {self.tag_position_log_path} '
            'and at completion; map snapshots include periodic and lap events.'
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
        if self.latest_tag_positions and not self.motion.finished:
            self.write_tag_positions(force=True)
            self.write_map_snapshot('shutdown')
        self.position_log_file.close()
        self.waypoint_log_file.close()
        self.map_snapshot_log_file.close()
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
        map_timestamp = payload.get('timestamp_sec')
        if not isinstance(map_timestamp, (int, float)) or not math.isfinite(map_timestamp):
            self.get_logger().error('Rejecting map with missing or invalid timestamp.')
            return
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
            covariance = [tag.get(key) for key in ('cov_xx', 'cov_xy', 'cov_yy')]
            if not all(isinstance(v, (int, float)) and math.isfinite(v)
                       for v in covariance):
                self.get_logger().error('Rejecting map with missing or invalid covariance.')
                return
            positions[name] = (float(x), float(y), *map(float, covariance))

        self.latest_tag_positions = positions
        self.latest_map_timestamp = float(map_timestamp)
        self.tag_frame_mismatch_logged = False

    def write_lap_snapshot(self):
        """Save the latest received map at each configured lap boundary."""
        index = self.motion.current_waypoint_idx
        if index not in self.lap_end_indices:
            return
        self.completed_laps += 1
        self.write_map_snapshot('lap_end')
        self.write_tag_positions(force=True)

    def write_map_snapshot(self, event):
        """Append a complete map with a unique snapshot ID and event label."""
        self.snapshot_id += 1
        prefix = [
            self._now_seconds(), self.latest_map_timestamp,
            self.completed_laps, self.motion.current_waypoint_idx, self.world_frame,
            len(self.latest_tag_positions),
        ]
        for tag, values in sorted(self.latest_tag_positions.items()):
            self.map_snapshot_writer.writerow([
                *prefix, tag, *values, self.snapshot_id, event,
            ])
        if not self.latest_tag_positions:
            # Preserve evidence of the lap even if no map was received.
            self.map_snapshot_writer.writerow([
                *prefix, '', '', '', '', '', '', self.snapshot_id, event,
            ])
        self.map_snapshot_log_file.flush()
        os.fsync(self.map_snapshot_log_file.fileno())

    def log_current_map(self):
        """Checkpoint fresh maps during motion without mixing in post-run updates."""
        if self.motion.finished or not self.latest_tag_positions:
            return
        if self.latest_map_timestamp == self.last_saved_map_timestamp:
            return
        self.write_tag_positions(force=True)
        self.write_map_snapshot('periodic')

    def write_tag_positions(self, force=False):
        """Atomically replace the latest map, preserving the prior file on failure."""
        if self.tag_position_log_written and not force:
            return
        if not self.latest_tag_positions:
            return
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', newline='', encoding='utf-8', delete=False,
                dir=self.tag_position_log_path.parent, suffix='.tmp',
            ) as output:
                temp_path = Path(output.name)
                writer = csv.writer(output)
                writer.writerow([
                    'timestamp_sec', 'frame', 'tag', 'x', 'y',
                    'cov_xx', 'cov_xy', 'cov_yy', 'map_timestamp_sec',
                ])
                timestamp = self._now_seconds()
                for tag, values in sorted(self.latest_tag_positions.items()):
                    writer.writerow([
                        timestamp, self.world_frame, tag, *values,
                        self.latest_map_timestamp,
                    ])
                output.flush()
                os.fsync(output.fileno())
            os.replace(temp_path, self.tag_position_log_path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
        self.last_saved_map_timestamp = self.latest_map_timestamp
        if not force:
            self.tag_position_log_written = True
            self.get_logger().info(
                f'Saved {len(self.latest_tag_positions)} final tag positions to '
                f'{self.tag_position_log_path}.'
            )

    def shape_forward_approach(self, now, linear, angular, action, distance):
        """Limit near-target forward bursts; pauses allow camera processing.

        A timed pause does not guarantee that a fresh tag was visible.
        No angular command is modified outside an observation pause.
        """
        if self.forward_pause_started is not None:
            if now < self.forward_pause_started:
                self.forward_pause_started = now
            if now - self.forward_pause_started < self.forward_observation_pause:
                return 0.0, 0.0, 'observation_pause'
            self.forward_pause_started = None

        if action != 'drive' or linear <= 0.0 or distance > self.forward_pulse_distance:
            self.forward_pulse_started = None
            return linear, angular, action

        if self.forward_pulse_started is None or now < self.forward_pulse_started:
            self.forward_pulse_started = now
        if now - self.forward_pulse_started >= self.forward_pulse_duration:
            self.forward_pulse_started = None
            self.forward_pause_started = now
            self.get_logger().info('Forward approach: stopping for camera observation.')
            return 0.0, 0.0, 'observation_pause'
        return linear, angular, action

    def accepted_tag_callback(self, msg):
        if math.isfinite(msg.data):
            self.latest_accepted_tag_stamp = float(msg.data)

    def shape_angular_burst(self, now, linear, angular, action):
        """Stop between turn bursts until a post-stop known-tag update arrives."""
        if self.angular_pause_started is not None:
            if now < self.angular_pause_started:
                self.angular_pause_started = now
                self.latest_accepted_tag_stamp = None
            stamp = self.latest_accepted_tag_stamp
            if (now - self.angular_pause_started < self.angular_observation_pause
                    or stamp is None
                    or stamp <= self.angular_pause_started + 0.10
                    or not 0.0 <= now - stamp <= self.pose_timeout):
                return 0.0, 0.0, 'angular_observation_pause'
            self.angular_pause_started = None

        if self.angular_burst_started is not None:
            if (now < self.angular_burst_started
                    or now - self.angular_burst_started >= self.angular_burst_duration
                    or angular * self.angular_burst_direction <= 0.0
                    or linear != 0.0):
                self.angular_burst_started = None
                self.angular_pause_started = now
                self.get_logger().info(
                    'Turn burst stopped; waiting for fresh accepted camera feedback.'
                )
                return 0.0, 0.0, 'angular_observation_pause'
        elif angular != 0.0:
            self.angular_burst_started = now
            self.angular_burst_direction = math.copysign(1.0, angular)
        return linear, angular, action

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
            self.forward_pulse_started = None
            if self.angular_burst_started is not None:
                self.angular_burst_started = None
                self.angular_pause_started = now
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
            self.final_waypoint_settled = True

        self.stale_stop_logged = False
        robot_x, robot_y, robot_yaw = self.latest_estimated_position

        previous_phase = self.motion.phase
        (
            linear, angular, action, dx, dy, heading_error, distance_error,
        ) = self.motion.command_for_pose(robot_x, robot_y, robot_yaw)
        if action != 'reached':
            self.final_waypoint_settled = False
        if previous_phase == 'position' and self.motion.phase == 'heading':
            self.get_logger().info(
                f'Waypoint {self.motion.current_waypoint_idx}: position '
                f'tolerance reached (error={distance_error:.3f}m); '
                'aligning heading.'
            )

        linear, angular = self.motion.shape_turn(
            linear, angular, heading_error
        )
        linear, angular, action = self.shape_forward_approach(
            now, linear, angular, action, distance_error
        )
        linear, angular, action = self.shape_angular_burst(
            now, linear, angular, action
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
                and not self.final_waypoint_settled
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
            self.write_lap_snapshot()
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
