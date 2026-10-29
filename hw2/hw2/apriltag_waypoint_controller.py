#!/usr/bin/env python3
"""
Iterative AprilTag waypoint controller for a differential-drive ROS 2 robot.

How it works:
  1. Each known AprilTag has a measured pose in the world frame.
  2. The controller checks the configured tag TF frames and uses a visible,
     recent tag to estimate the robot's world pose.
  3. It compares that pose with the current waypoint.
  4. It publishes a short velocity command, then repeats the observation.

IMPORTANT SETUP:
  - Fill in TAG_POSES_WORLD with the measured world pose of each tag.
  - Fill in WAYPOINTS with your desired [x, y, heading] targets.
  - Replace CAMERA_XYZ / CAMERA_RPY with the calibrated camera-to-robot pose.
  - Confirm TAG_FRAME_PREFIX / frame names match your AprilTag TF publisher.
  - All tag poses and waypoints must use the same world coordinate frame.
  - Tag pose is [x, y, z, roll, pitch, yaw], with angles in radians.
  - Waypoints are [x, y, heading], with heading in radians.
  - Test with the robot lifted and at low speeds first.
"""

import math
from typing import Optional

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import Twist
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


def pose_matrix_xyz_rpy(x: float, y: float, z: float,
                        roll: float, pitch: float, yaw: float):
    """Build a 4x4 pose matrix from translation and fixed-axis roll/pitch/yaw."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
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
        self.world_frame = 'odom'       # Name of your shared world frame
        self.base_frame = 'base_link'   # Robot body frame
        self.camera_frame = 'camera_frame'

        # ---------------------------------------------------------------
        # 2. KNOWN APRILTAG WORLD POSES
        # ---------------------------------------------------------------
        # Dictionary: TF child frame name -> (x, y, z, roll, pitch, yaw).
        # Frame names must match the configured names published by apriltag_ros.
        #
        # These are the measured poses of the PHYSICAL TAGS in the world
        # coordinate frame, NOT waypoint locations.
        #
        # Position is in meters; roll/pitch/yaw are in radians.
        # The orientation must describe the tag TF frame's axes in the
        # world frame, using the same convention as your TF publisher.
        self.tag_poses_world = {
            'tag_0': (1.20, 0.00, 0.05, 0.0, math.pi / 2, 0.0),             
            'tag_1': (1.00, 2.20, 0.05, 0.0, 0.0, math.pi / 2),     
            'tag_2': (0.80, 2.00, 0.05, 0.0, -math.pi / 2, 0.0),        
            'tag_3': (-0.20, -0.20, 0.05, 0.0, 0.0, math.pi / 4),   
        }

        # ---------------------------------------------------------------
        # 3. CAMERA EXTRINSICS -- REPLACE WITH CALIBRATED VALUES
        # ---------------------------------------------------------------
        # T_base_camera maps camera-frame coordinates into base_link.
        # Temporary example only. A real optical camera frame often has a
        # nonzero rotation because its axes differ from base_link's axes.
        self.camera_xyz = (0.08, 0.0, 0.12)
        self.camera_rpy = (0.0, 0.0, 0.0)
        self.T_base_camera = pose_matrix_xyz_rpy(
            *self.camera_xyz, *self.camera_rpy
        )

        # ---------------------------------------------------------------
        # 4. WAYPOINTS -- REPLACE THESE SAMPLE VALUES
        # ---------------------------------------------------------------
        # Each waypoint is [x, y, heading] in meters/radians, in world_frame.
        # The robot does NOT have to start at waypoint 0.
        self.waypoints = np.array([
            [0.00, 0.00, 0],               
            [1.00, 0.00, 0],       
            [1.00, 2.00, math.pi],               
            [0.00, 0.00, 0],      
        ], dtype=float)

        # ---------------------------------------------------------------
        # 5. CONTROL / CALIBRATION SETTINGS
        # ---------------------------------------------------------------
        # These are controller velocity limits, not the motor mapping slopes
        # or deadzones in VelocityToMotorNode.
        self.k_v = 0.12             # m/s maximum forward speed
        self.k_omega = 0.70         # rad/s maximum turn speed
        self.control_period = 0.10  # seconds (10 Hz)
        self.k_heading = 1.8
        self.k_distance = 0.8
        self.max_step_distance = 0.025  # nominal max movement per tick (m)
        self.position_tolerance = 0.035 # m
        self.heading_tolerance = 0.10   # rad
        self.tag_timeout = 0.40         # seconds
        self.search_turn_duration = 0.25
        self.search_turn_sign = 1.0     # +1 left, -1 right

        # ---------------------------------------------------------------
        # ROS communication
        # ---------------------------------------------------------------
        self.current_waypoint_idx = 0
        self.searching_until: Optional[float] = None
        self.searching = False
        self.finished = False

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(self.control_period, self.control_loop)

        self.get_logger().info(
            'AprilTag waypoint controller started. Replace sample tag poses, '
            'waypoints, and camera extrinsics before driving.'
        )

    def get_visible_known_tag(self):
        """
        Return (tag_frame, T_camera_tag) for the first known, recent tag TF found.
        """
        now_seconds = self.get_clock().now().nanoseconds * 1e-9

        for tag_frame in self.tag_poses_world:
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
            if age > self.tag_timeout or age < -0.10:
                continue

            return tag_frame, transform_to_matrix(tf_msg)

        return None, None

    def estimate_world_robot_pose(self, tag_frame: str, T_camera_tag):
        """
        Estimate T_world_base from the known tag world pose and observed
        camera-to-tag transform.

        T_world_base = T_world_tag @ inverse(T_base_camera @ T_camera_tag)
        """
        tag_pose = self.tag_poses_world[tag_frame]
        T_world_tag = pose_matrix_xyz_rpy(*tag_pose)

        # T_base_tag: tag coordinates transformed into the robot base frame.
        T_base_tag = self.T_base_camera @ T_camera_tag

        # T_world_robot/base: robot pose in the world coordinate frame.
        return T_world_tag @ inverse_transform(T_base_tag)

    def publish_stop(self):
        self.cmd_pub.publish(Twist())

    def begin_search_turn(self, now: float):
        """Turn in place briefly when no known, recent tag is visible."""
        if not self.searching:
            self.searching = True
            self.searching_until = now + self.search_turn_duration
            self.get_logger().warning(
                'No known AprilTag TF is visible/recent. Searching by turning '
                f'{"left" if self.search_turn_sign > 0 else "right"} in place.'
            )

        cmd = Twist()
        if self.searching_until is not None and now < self.searching_until:
            cmd.linear.x = 0.0
            cmd.angular.z = self.search_turn_sign * self.k_omega
            self.cmd_pub.publish(cmd)
        else:
            self.publish_stop()
            self.searching = False
            self.searching_until = None

    def control_loop(self):
        if self.finished:
            self.publish_stop()
            return

        if self.current_waypoint_idx >= len(self.waypoints):
            self.finished = True
            self.publish_stop()
            self.get_logger().info('All waypoints reached; robot stopped.')
            return

        tag_frame, T_camera_tag = self.get_visible_known_tag()
        now = self.get_clock().now().nanoseconds * 1e-9

        if T_camera_tag is None:
            self.begin_search_turn(now)
            return

        # A tag is visible again; stop search behavior and localize.
        self.searching = False
        self.searching_until = None

        T_world_robot = self.estimate_world_robot_pose(tag_frame, T_camera_tag)
        robot_x = float(T_world_robot[0, 3])
        robot_y = float(T_world_robot[1, 3])
        robot_yaw = rotation_matrix_to_yaw(T_world_robot[:3, :3])

        waypoint = self.waypoints[self.current_waypoint_idx]
        dx = float(waypoint[0] - robot_x)
        dy = float(waypoint[1] - robot_y)
        distance_error = math.hypot(dx, dy)

        desired_heading = (
            math.atan2(dy, dx) if distance_error > 1e-6 else float(waypoint[2])
        )
        heading_error = wrap_angle(desired_heading - robot_yaw)

        cmd = Twist()

        # First reach the waypoint position, then align to its requested heading.
        if distance_error > self.position_tolerance:
            if abs(heading_error) > self.heading_tolerance:
                # Turn in place if the waypoint is not sufficiently ahead.
                angular = self.k_heading * heading_error
                cmd.angular.z = float(np.clip(
                    angular, -self.k_omega, self.k_omega
                ))
                cmd.linear.x = 0.0
            else:
                desired_speed = min(
                    self.k_v,
                    max(0.0, self.k_distance * distance_error),
                )
                cmd.linear.x = min(
                    desired_speed,
                    self.max_step_distance / self.control_period,
                )
                cmd.angular.z = float(np.clip(
                    self.k_heading * heading_error, -self.k_omega, self.k_omega
                ))
        else:
            final_heading_error = wrap_angle(float(waypoint[2]) - robot_yaw)
            if abs(final_heading_error) > self.heading_tolerance:
                cmd.linear.x = 0.0
                cmd.angular.z = float(np.clip(
                    self.k_heading * final_heading_error,
                    -self.k_omega,
                    self.k_omega,
                ))
            else:
                self.publish_stop()
                self.get_logger().info(
                    f'Reached waypoint {self.current_waypoint_idx}: '
                    f'position error={distance_error:.3f} m; tag frame={tag_frame}'
                )
                self.current_waypoint_idx += 1
                return

        self.cmd_pub.publish(cmd)
        self.get_logger().debug(
            f'tag={tag_frame} wp={self.current_waypoint_idx} '
            f'pose=({robot_x:.3f}, {robot_y:.3f}, {robot_yaw:.3f}) '
            f'error=({dx:.3f}, {dy:.3f}, {heading_error:.3f}) '
            f'cmd=(v={cmd.linear.x:.3f}, w={cmd.angular.z:.3f})'
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
