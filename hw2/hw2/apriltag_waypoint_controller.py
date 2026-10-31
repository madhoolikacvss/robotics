#!/usr/bin/env python3
"""
Stop-and-go AprilTag waypoint controller for a differential-drive ROS 2 robot.

Control cycle:
  1. Stop and wait for the observation pause to finish.
  2. Wait for a NEW AprilTag TF observation captured after the previous move.
  3. Estimate the robot pose from the known tag pose and camera extrinsics.
  4. Calculate one velocity command.
  5. Execute that command for a fixed movement window.
  6. Stop, wait, and localize again.

Unlike the previous controller, this node does not rotate to search for tags.
If no fresh known-tag transform is available, it remains stopped.

IMPORTANT:
  - TAG_POSES_WORLD must contain measured physical tag poses in the world frame.
  - WAYPOINTS are [x, y, heading] in the same world frame.
  - T_base_camera maps camera-frame coordinates into base_link.
  - Confirm camera_frame and tag frame names match the TF tree.
  - Test with the robot lifted and at low speeds first.
"""

import math
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
        # Frame name -> (x, y, z, roll, pitch, yaw)
        # These are physical tag poses, not waypoint positions.
        # ---------------------------------------------------------------
        self.tag_poses_world = {
            'tag_0': (1.20, 0.00, 0.05, math.pi / 2, 0.0, -math.pi / 2),
            'tag_1': (1.00, 1.20, 0.05, 0.0, 0.0, math.pi / 2),
            'tag_2': (0.00, 1.20, 0.05, 0.0, 0.0, math.pi / 2),
            'tag_3': (0.00, -0.85, 0.05, math.pi / 2, -math.pi / 2, -math.pi / 2),
        }

        # ---------------------------------------------------------------
        # 3. CAMERA EXTRINSICS
        # T_base_camera maps camera-frame coordinates into base_link.
        # Preserved from the supplied script.
        # ---------------------------------------------------------------
        self.camera_xyz = (0.07, 0.0, 0.12)
        self.camera_rpy = (-math.pi / 2, 0.0, -math.pi / 2)
        self.T_base_camera = pose_matrix_xyz_rpy(
            *self.camera_xyz, *self.camera_rpy
        )

        # ---------------------------------------------------------------
        # 4. WAYPOINTS: [x, y, heading] in world_frame
        # Preserved from the supplied script.
        # ---------------------------------------------------------------
        self.waypoints = np.array([
            # [0.00, 0.00, 0.0],
            [1.00, 0.00, 0.0],
            [1.00, 1.00, math.pi],
            [0.00, 0.00, 0.0],
        ], dtype=float)

        # ---------------------------------------------------------------
        # 5. MOTION AND OBSERVATION SETTINGS
        # ---------------------------------------------------------------
        self.k_v = 0.12
        self.k_omega = 0.70
        self.k_heading = 1.8
        self.k_distance = 0.8

        self.control_period = 0.10

        # Each movement command is held for this long, then the robot stops.
        self.move_duration = 2.0

        # After stopping, wait at least this long before using a new image.
        self.observation_pause = 2.0

        # TF may arrive late due to camera/detector latency. This is only
        # an upper age bound; old observations are never reused across cycles.
        self.tag_max_age = 5.0

        self.position_tolerance = 0.035
        self.heading_tolerance = 0.10

        # Forward speed is still capped by k_v. At k_v=0.12 m/s,
        # a full 2-second drive window can cover up to about 0.24 m.

        # ---------------------------------------------------------------
        # 6. STATE
        # ---------------------------------------------------------------
        self.current_waypoint_idx = 0
        self.finished = False

        # Start in OBSERVE. At startup, do not impose an extra 2-second pause;
        # wait for the first currently available recent tag observation.
        self.phase = 'observe'
        self.phase_started_at = self._now_seconds()
        self.observe_ready_at = self.phase_started_at
        self.require_stamp_after = 0.0

        self.active_command = Twist()
        self.last_used_tag_stamp = {name: 0.0 for name in self.tag_poses_world}
        self.last_no_tag_log_time = 0.0

        # ---------------------------------------------------------------
        # 7. ROS COMMUNICATION
        # ---------------------------------------------------------------
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pose_pub = self.create_publisher(
            PoseStamped, '/robot_pose_estimate', 10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(
            self.control_period, self.control_loop
        )

        self.get_logger().info(
            'Stop-and-go AprilTag controller started: '
            f'move={self.move_duration:.1f}s, '
            f'observation pause={self.observation_pause:.1f}s, '
            f'max TF age={self.tag_max_age:.1f}s. '
            'The robot will remain stopped if no fresh known tag is available.'
        )

    def _now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def publish_stop(self):
        self.cmd_pub.publish(Twist())

    def get_new_known_tag(self):
        """
        Return the first available known tag with a recent, unused observation.

        Tags need not be visible simultaneously. Missing tag frames are
        silently skipped. The timestamp must be newer than the last observation
        used for that tag and newer than the time the previous movement ended.
        """
        now_seconds = self._now_seconds()

        for tag_frame in self.tag_poses_world:
            # Tags are not expected to be visible simultaneously. Check each
            # known tag independently and silently skip frames absent from TF.
            try:
                available = self.tf_buffer.can_transform(
                    self.camera_frame,
                    tag_frame,
                    Time(),
                    timeout=Duration(seconds=0.02),
                )
            except TransformException:
                continue

            if not available:
                continue

            try:
                tf_msg = self.tf_buffer.lookup_transform(
                    self.camera_frame,
                    tag_frame,
                    Time(),
                    timeout=Duration(seconds=0.02),
                )
            except TransformException:
                # The transform may disappear between can_transform and lookup.
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

            if stamp_seconds <= self.require_stamp_after:
                continue

            return tag_frame, transform_to_matrix(tf_msg), stamp_seconds

        return None, None, None

    def estimate_world_robot_pose(self, tag_frame: str, T_camera_tag):
        """
        Estimate T_world_base from the known tag pose and observed
        camera-to-tag transform:

        T_world_base = T_world_tag @ inverse(T_base_camera @ T_camera_tag)
        """
        T_world_tag = pose_matrix_xyz_rpy(
            *self.tag_poses_world[tag_frame]
        )
        T_base_tag = self.T_base_camera @ T_camera_tag
        return T_world_tag @ inverse_transform(T_base_tag)

    def calculate_motion_command(
        self, robot_x: float, robot_y: float, robot_yaw: float
    ):
        """Calculate the next fixed-duration command from one localized pose."""
        waypoint = self.waypoints[self.current_waypoint_idx]

        dx = float(waypoint[0] - robot_x)
        dy = float(waypoint[1] - robot_y)
        distance_error = math.hypot(dx, dy)

        if distance_error > 1e-6:
            desired_heading = math.atan2(dy, dx)
        else:
            desired_heading = float(waypoint[2])

        heading_error = wrap_angle(desired_heading - robot_yaw)
        cmd = Twist()

        # First navigate to the waypoint position.
        if distance_error > self.position_tolerance:
            if abs(heading_error) > self.heading_tolerance:
                # Turn in place for one movement window, then stop and relocalize.
                cmd.linear.x = 0.0
                cmd.angular.z = float(np.clip(
                    self.k_heading * heading_error,
                    -self.k_omega,
                    self.k_omega,
                ))
                action = 'turn'
            else:
                desired_speed = min(
                    self.k_v,
                    max(0.0, self.k_distance * distance_error),
                )
                cmd.linear.x = desired_speed
                cmd.angular.z = float(np.clip(
                    self.k_heading * heading_error,
                    -self.k_omega,
                    self.k_omega,
                ))
                action = 'drive'
        else:
            # At the waypoint position: align to its requested final heading.
            final_heading_error = wrap_angle(
                float(waypoint[2]) - robot_yaw
            )
            if abs(final_heading_error) > self.heading_tolerance:
                cmd.linear.x = 0.0
                cmd.angular.z = float(np.clip(
                    self.k_heading * final_heading_error,
                    -self.k_omega,
                    self.k_omega,
                ))
                action = 'final_heading_turn'
            else:
                action = 'reached'

        return cmd, action, dx, dy, heading_error, distance_error

    def begin_observation_phase(self, now: float, require_new_after_stop: bool):
        """Stop and wait before accepting the next distinct tag observation."""
        self.phase = 'observe'
        self.phase_started_at = now
        self.observe_ready_at = now + self.observation_pause

        if require_new_after_stop:
            # Require an image/TF observation timestamped after the stop began.
            self.require_stamp_after = now

        self.active_command = Twist()
        self.publish_stop()

        self.get_logger().info(
            f'Stopped. Waiting {self.observation_pause:.1f}s before '
            'localizing again.'
        )

    def start_movement(self, cmd: Twist, tag_frame: str, stamp_seconds: float,
                       now: float, action: str):
        """Consume this observation and execute its command for move_duration."""
        self.last_used_tag_stamp[tag_frame] = stamp_seconds
        self.active_command = cmd
        self.phase = 'move'
        self.phase_started_at = now

        self.get_logger().info(
            f'New localization from {tag_frame} '
            f'(TF age={max(0.0, now - stamp_seconds):.2f}s). '
            f'Action={action}; executing for {self.move_duration:.1f}s: '
            f'v={cmd.linear.x:.3f}, w={cmd.angular.z:.3f}'
        )

    def control_loop(self):
        now = self._now_seconds()

        if self.finished or self.current_waypoint_idx >= len(self.waypoints):
            self.finished = True
            self.publish_stop()
            return

        # ---------------------------------------------------------------
        # MOVE PHASE: keep the selected command active for a fixed window.
        # No new localization is used while the robot is moving.
        # ---------------------------------------------------------------
        if self.phase == 'move':
            elapsed = now - self.phase_started_at

            if elapsed < self.move_duration:
                self.cmd_pub.publish(self.active_command)
                return

            # Movement window is over. Stop immediately and begin observing.
            self.begin_observation_phase(
                now, require_new_after_stop=True
            )
            return

        # ---------------------------------------------------------------
        # OBSERVE PHASE: remain stopped. Wait for the pause and a new TF.
        # ---------------------------------------------------------------
        self.publish_stop()

        if now < self.observe_ready_at:
            return

        tag_frame, T_camera_tag, tag_stamp = self.get_new_known_tag()

        if T_camera_tag is None:
            if now - self.last_no_tag_log_time >= 1.0:
                self.get_logger().info(
                    'Still stopped; waiting for a new AprilTag observation '
                    'captured after the previous movement.'
                )
                self.last_no_tag_log_time = now
            return

        T_world_robot = self.estimate_world_robot_pose(
            tag_frame, T_camera_tag
        )
        robot_x = float(T_world_robot[0, 3])
        robot_y = float(T_world_robot[1, 3])
        robot_yaw = rotation_matrix_to_yaw(T_world_robot[:3, :3])

        # Publish the pose estimate for visualization/logging.
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

        self.get_logger().info(
            f'LOCALIZED tag={tag_frame} '
            f'pose=({robot_x:.3f}, {robot_y:.3f}, {robot_yaw:.3f}) '
            f'waypoint={self.current_waypoint_idx} '
            f'error_xy=({dx:.3f}, {dy:.3f}) '
            f'heading_error={heading_error:.3f} '
            f'distance={distance_error:.3f}'
        )

        if action == 'reached':
            self.last_used_tag_stamp[tag_frame] = tag_stamp
            self.get_logger().info(
                f'Reached waypoint {self.current_waypoint_idx}: '
                f'position error={distance_error:.3f}m.'
            )
            self.current_waypoint_idx += 1

            if self.current_waypoint_idx >= len(self.waypoints):
                self.finished = True
                self.publish_stop()
                self.get_logger().info('All waypoints reached.')
                return

            # Require a new observation before planning for the next waypoint.
            self.begin_observation_phase(
                now, require_new_after_stop=True
            )
            return

        self.start_movement(
            cmd, tag_frame, tag_stamp, now, action
        )


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
