#!/usr/bin/env python3
"""
Stop-and-go AprilTag waypoint controller for a differential-drive ROS 2 robot.

Control cycle:
  1. Stop and wait for the observation pause to finish.
  2. Wait for a NEW AprilTag TF observation captured after the previous move.
  3. Estimate the robot pose from the known tag pose and camera extrinsics.
  4. Calculate one velocity command and its required movement duration (both linear & angular).
  5. Execute that command for the precise movement duration.
  6. Stop, wait, and localize again.
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
        # ---------------------------------------------------------------
        self.tag_poses_world = {
            'tag_0': (1.60, 0.00, 0.05, math.pi / 2, 0.0, -math.pi / 2),
            'tag_1': (0.90, 1.14, 0.05, math.pi / 2, 0.0, 0.0),
            'tag_2': (-0.17, 0.0, 0.05, math.pi / 2, 0.0, math.pi / 2),
            'tag_3': (0.55, -1.10 , 0.05, math.pi / 2, 0.0, -math.pi),
            'tag_4': (0.0, 1.14 , 0.05, math.pi / 2, 0.0, math.pi/2),
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
            [1.00, 0.00, 0],
            [0.80, 1.00, math.pi],
            [0.00, 0.00, 0.0],
        ], dtype=float)

        # ---------------------------------------------------------------
        # 5. MOTION AND OBSERVATION SETTINGS
        # ---------------------------------------------------------------
        self.k_v = 0.12
        self.k_omega = 0.35
        self.k_heading = 1.8
        self.k_distance = 0.8

        self.control_period = 0.02

        # Maximum movement window cap
        self.move_duration = 2.0

        # After stopping, wait at least this long before using a new image.
        self.observation_pause = 2.0

        self.tag_max_age = 5.0

        # Increased tolerance to 0.2 meters as requested
        self.position_tolerance = 0.2 # 0.4 IS ONLY FOR THE HEADING ISOLATION - CHANGE ITTTTtTTTTTT!!!!!!!!!!!!!!!!
        self.heading_tolerance = 0.2

        # ---------------------------------------------------------------
        # 6. STATE
        # ---------------------------------------------------------------
        self.current_waypoint_idx = 0
        self.finished = False

        # Waypoint sub-phase is independent of the operational observe/move phase.
        # It remains 'heading' throughout final alignment, even if x/y drifts.
        self.waypoint_phase = 'position'

        self.phase = 'observe'
        self.phase_started_at = self._now_seconds()
        self.observe_ready_at = self.phase_started_at
        self.require_stamp_after = 0.0

        self.active_command = Twist()
        self.current_move_duration = self.move_duration
        self.last_used_tag_stamp = {name: 0.0 for name in self.tag_poses_world}
        self.last_no_tag_log_time = 0.0


        # Short, adaptive turn bursts
#        self.minimum_turn_duration = 0.05
#        self.maximum_turn_duration = 0.20
#
#        # Minimum motor command that produces a turn.
#        self.minimum_angular_motor_command = 0.3
#
#        # Calibrate this experimentally: actual robot angular velocity (rad/s)
#        # when the motor command reaches its minimum effective turning command.
#        self.minimum_turn_angular_velocity = 0.3  # Set after measuring

        
        # ---------------------------------------------------------------
        # TURN CALIBRATION
        # ---------------------------------------------------------------
        # HW1 calibration:
        #   omega ~= k_omega * (u_right - u_left)
        #
        # The existing velocity mapping turns cmd.angular.z = +/-0.35
        # into approximately u_left = -0.3, u_right = +0.3 because
        # of its angular motor-command floor.
        #
        # Therefore, the estimated effective angular rates are:
        #   k_omega * (0.3 - (-0.3)) = k_omega * 0.6
        #
        # These are INITIAL ESTIMATES, not verified HW2 measurements.
        self.hw1_k_omega_right = 3.85
        self.hw1_k_omega_left = 3.80
        self.estimated_motor_command_magnitude = 0.30
        
        self.estimated_turn_rate_ccw = (
            self.hw1_k_omega_right
            * 2.0
            * self.estimated_motor_command_magnitude
        )  # approximately 2.31 rad/s
        
        self.estimated_turn_rate_cw = (
            self.hw1_k_omega_left
            * 2.0
            * self.estimated_motor_command_magnitude
        )  # approximately 2.28 rad/s
        
        # Keep a small lower bound for commands shorter than one motor update.
        self.minimum_turn_duration = 0.05
        
        # Safety limit, not a burst limit. This allows turns up to about
        # 180 degrees at the estimated angular rates.
        self.maximum_turn_duration = 2.0
        
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
            f'max move duration={self.move_duration:.1f}s, '
            f'position tolerance={self.position_tolerance:.2f}m.'
        )

    def _now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def publish_stop(self):
        self.cmd_pub.publish(Twist())

    def get_new_known_tag(self):
        """Return the first available known tag with a recent, unused observation."""
        now_seconds = self._now_seconds()

        for tag_frame in self.tag_poses_world:
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
        """Estimate T_world_base from known tag pose and observed camera-to-tag transform."""
        T_world_tag = pose_matrix_xyz_rpy(
            *self.tag_poses_world[tag_frame]
        )
        T_base_tag = self.T_base_camera @ T_camera_tag
        return T_world_tag @ inverse_transform(T_base_tag)

    def calculate_motion_command(
        self, robot_x: float, robot_y: float, robot_yaw: float
    ):
        """
        Calculate a movement command for the current waypoint sub-phase.

        waypoint_phase='position': navigate to the waypoint's x/y.
        waypoint_phase='heading': align to its specified final heading.

        The operational self.phase ('observe'/'move') is kept separate.
        """
        waypoint = self.waypoints[self.current_waypoint_idx]

        dx = float(waypoint[0] - robot_x)
        dy = float(waypoint[1] - robot_y)
        distance_error = math.hypot(dx, dy)

        cmd = Twist()
        move_duration = 0.0

        # ---------------------------------------------------------------
        # WAYPOINT SUB-PHASE 1: Navigate to the waypoint's x/y position.
        # ---------------------------------------------------------------
        if self.waypoint_phase == 'position':
            if distance_error <= self.position_tolerance:
                # Lock this waypoint into heading alignment. Do not advance
                # the waypoint index yet; get a fresh observation first.
                self.waypoint_phase = 'heading'
                self.get_logger().info(
                    f'Waypoint {self.current_waypoint_idx}: position tolerance '
                    f'reached (error={distance_error:.3f}m). '
                    'Switching to final-heading alignment.'
                )
                return (
                    cmd, 'phase_transition', dx, dy, 0.0,
                    distance_error, 0.0
                )

            desired_heading = math.atan2(dy, dx)
            travel_heading_error = wrap_angle(desired_heading - robot_yaw)

            if abs(travel_heading_error) > self.heading_tolerance:
                cmd.linear.x = 0.0
                cmd.angular.z = math.copysign(
                    self.k_omega, travel_heading_error
                )
                action = 'turn'
                move_duration = self.calculate_turn_duration(
                    travel_heading_error
                )
            else:
                # Drive toward the waypoint, with heading correction.
                desired_speed = min(
                    self.k_v,
                    max(0.0, self.k_distance * distance_error),
                )
                cmd.linear.x = desired_speed
                cmd.angular.z = float(np.clip(
                    self.k_heading * travel_heading_error,
                    -self.k_omega,
                    self.k_omega,
                ))
                action = 'drive'

                # Dynamic duration for drive capped at move_duration.
                if abs(cmd.linear.x) > 1e-6:
                    move_duration = min(
                        self.move_duration,
                        distance_error / abs(cmd.linear.x),
                    )

            return (
                cmd, action, dx, dy, travel_heading_error,
                distance_error, move_duration
            )

        # ---------------------------------------------------------------
        # WAYPOINT SUB-PHASE 2: Align to the specified final heading.
        # Ignore x/y error for action selection because the robot may drift
        # while turning. The position error is retained for diagnostics.
        # ---------------------------------------------------------------
        if self.waypoint_phase == 'heading':
            final_heading_error = wrap_angle(
                float(waypoint[2]) - robot_yaw
            )

            if abs(final_heading_error) > self.heading_tolerance:
                cmd.linear.x = 0.0
                cmd.angular.z = math.copysign(
                    self.k_omega, final_heading_error
                )
                action = 'final_heading_turn'
                move_duration = self.calculate_turn_duration(
                    final_heading_error
                )
            else:
                action = 'reached'
                move_duration = 0.0

            return (
                cmd, action, dx, dy, final_heading_error,
                distance_error, move_duration
            )

        raise ValueError(
            f"Unknown waypoint phase: {self.waypoint_phase!r}"
        )

    def calculate_turn_duration(self, angular_error: float) -> float:
      """
      Estimate the duration for one complete timed turn.
  
      Uses direction-specific angular-rate estimates derived from HW1
      calibration and the existing velocity_mapping.py command floor.
  
      The estimates must be validated through AprilTag observations.
      No IMU is used.
      """
      if abs(angular_error) <= self.heading_tolerance:
          return 0.0
  
      if angular_error > 0.0:
          # Positive yaw: counterclockwise turn.
          angular_speed = self.estimated_turn_rate_ccw
      else:
          # Negative yaw: clockwise turn.
          angular_speed = self.estimated_turn_rate_cw
  
      if angular_speed <= 0.0:
          raise ValueError("Estimated turn rate must be positive.")
  
      required_duration = abs(angular_error) / angular_speed
  
      return min(
          self.maximum_turn_duration,
          max(self.minimum_turn_duration, required_duration),
      )
        
        
    def begin_observation_phase(self, now: float, require_new_after_stop: bool):
        """Stop and wait before accepting the next distinct tag observation."""
        self.phase = 'observe'
        self.phase_started_at = now
        self.observe_ready_at = now + self.observation_pause

        if require_new_after_stop:
            self.require_stamp_after = now

        self.active_command = Twist()
        self.publish_stop()

        self.get_logger().info(
            f'Stopped. Waiting {self.observation_pause:.1f}s before localizing again.'
        )

    def start_movement(self, cmd: Twist, tag_frame: str, stamp_seconds: float,
                       now: float, action: str, move_duration: float):
        """Consume observation and execute command for the calculated duration."""
        self.last_used_tag_stamp[tag_frame] = stamp_seconds
        self.active_command = cmd
        self.current_move_duration = move_duration
        self.phase = 'move'
        self.phase_started_at = now

        self.get_logger().info(
            f'New localization from {tag_frame} '
            f'(TF age={max(0.0, now - stamp_seconds):.2f}s). '
            f'Action={action}; executing for **{move_duration:.3f}s**: '
            f'v={cmd.linear.x:.3f}, w={cmd.angular.z:.3f}'
        )

    def control_loop(self):
        now = self._now_seconds()

        if self.finished or self.current_waypoint_idx >= len(self.waypoints):
            self.finished = True
            self.publish_stop()
            return

        # ---------------------------------------------------------------
        # MOVE PHASE: keep command active for the precise calculated duration.
        # ---------------------------------------------------------------
        if self.phase == 'move':
            elapsed = now - self.phase_started_at

            if elapsed < self.current_move_duration:
                self.cmd_pub.publish(self.active_command)
                return

            self.begin_observation_phase(
                now, require_new_after_stop=True
            )
            return

        # ---------------------------------------------------------------
        # OBSERVE PHASE: remain stopped. Wait for pause and new TF.
        # ---------------------------------------------------------------
        self.publish_stop()

        if now < self.observe_ready_at:
            return

        tag_frame, T_camera_tag, tag_stamp = self.get_new_known_tag()

        if T_camera_tag is None:
            if now - self.last_no_tag_log_time >= 1.0:
                self.get_logger().info(
                    'Still stopped; waiting for a new AprilTag observation.'
                )
                self.last_no_tag_log_time = now
            return

        T_world_robot = self.estimate_world_robot_pose(
            tag_frame, T_camera_tag
        )
        robot_x = float(T_world_robot[0, 3])
        robot_y = float(T_world_robot[1, 3])
        robot_yaw = rotation_matrix_to_yaw(T_world_robot[:3, :3])

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = f'{self.world_frame}|{tag_frame}'
        pose_msg.pose.position.x = robot_x
        pose_msg.pose.position.y = robot_y
        pose_msg.pose.position.z = 0.0
        pose_msg.pose.orientation.z = math.sin(robot_yaw / 2.0)
        pose_msg.pose.orientation.w = math.cos(robot_yaw / 2.0)
        self.pose_pub.publish(pose_msg)

        cmd, action, dx, dy, heading_error, distance_error, move_duration = (
            self.calculate_motion_command(robot_x, robot_y, robot_yaw)
        )

        self.get_logger().info(
            f'LOCALIZED tag={tag_frame} '
            f'pose=({robot_x:.3f}, {robot_y:.3f}, {robot_yaw:.3f}) '
            f'waypoint={self.current_waypoint_idx} '
            f'distance_err={distance_error:.3f} '
            f'calc_duration={move_duration:.3f}s'
        )

        if action == 'phase_transition':
            # Position tolerance was reached. The waypoint index stays the
            # same; self.waypoint_phase is now 'heading'. Stop and wait for
            # a fresh AprilTag observation before calculating a turn.
            self.last_used_tag_stamp[tag_frame] = tag_stamp
            self.begin_observation_phase(
                now, require_new_after_stop=True
            )
            return

        if action == 'reached':
            # The final heading is aligned. Only now is this waypoint complete.
            self.last_used_tag_stamp[tag_frame] = tag_stamp
            self.get_logger().info(
                f'Reached waypoint {self.current_waypoint_idx}: '
                f'final heading aligned; '
                f'position error={distance_error:.3f}m, '
                f'heading error={heading_error:.3f}rad.'
            )
            self.current_waypoint_idx += 1

            if self.current_waypoint_idx >= len(self.waypoints):
                self.finished = True
                self.publish_stop()
                self.get_logger().info('All waypoints reached.')
                return

            # Start position navigation for the next waypoint.
            self.waypoint_phase = 'position'
            self.begin_observation_phase(
                now, require_new_after_stop=True
            )
            return

        self.start_movement(
            cmd, tag_frame, tag_stamp, now, action, move_duration
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
