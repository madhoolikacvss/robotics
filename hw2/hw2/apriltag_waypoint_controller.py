#!/usr/bin/env python3
"""
Iterative AprilTag waypoint controller for a differential-drive ROS 2 robot.

Control cycle:
  1. Read the latest camera -> AprilTag transform.
  2. Estimate the robot pose in the world using the known world -> tag pose
     and the temporary robot -> camera extrinsic transform.
  3. Make one small motion command toward the waypoint.
  4. Re-read the tag and correct again on the next timer tick.
  5. If the tag is not visible / its transform is stale, stop translating and
     rotate in place briefly before trying again.

Assumptions:
  - TF lookup target=camera_frame, source=tag_frame returns T_camera_tag.
  - world_frame is the coordinate frame in which waypoints are specified.
  - waypoint heading defines the local forward direction of the waypoint.
  - The tag is 0.20 m behind the waypoint along that heading by default.
  - k_v and k_omega are calibrated robot speeds at the chosen command level.
    They should be loaded from calibration.json or set below to measured values.
  - The robot accepts geometry_msgs/Twist on /cmd_vel.

IMPORTANT:
  - Replace camera_frame, tag_frame, and the temporary camera extrinsic values.
  - Check the sign of tag_offset_from_waypoint for your physical tag placement.
  - Test with the robot lifted / wheels clear first, then at very low speeds.
"""

import json
import math
from pathlib import Path

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import Twist
from tf2_ros import Buffer, TransformListener, TransformException


def wrap_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def quaternion_to_rotation_matrix(x, y, z, w):
    """Convert ROS quaternion (x, y, z, w) to a 3x3 rotation matrix."""
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if norm < 1e-12:
        return np.eye(3)
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),         2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),         1 - 2*(x*x + y*y)],
    ], dtype=float)


def rotation_matrix_to_yaw(rotation):
    """Extract planar yaw from a rotation matrix."""
    return math.atan2(rotation[1, 0], rotation[0, 0])


def transform_to_matrix(transform_stamped):
    """Return homogeneous T_parent_child from a TransformStamped."""
    t = transform_stamped.transform.translation
    q = transform_stamped.transform.rotation
    matrix = np.eye(4)
    matrix[:3, :3] = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
    matrix[:3, 3] = [t.x, t.y, t.z]
    return matrix


def pose_matrix(x, y, yaw, z=0.0):
    """Homogeneous planar pose matrix T_world_pose."""
    c, s = math.cos(yaw), math.sin(yaw)
    matrix = np.eye(4)
    matrix[:3, :3] = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])
    matrix[:3, 3] = [x, y, z]
    return matrix


def inverse_transform(matrix):
    """Inverse of a rigid homogeneous transform."""
    result = np.eye(4)
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    result[:3, :3] = rotation.T
    result[:3, 3] = -rotation.T @ translation
    return result


class AprilTagWaypointController(Node):
    def __init__(self):
        super().__init__('apriltag_waypoint_controller')

        # ---- Frames: change these to match `ros2 run tf2_tools view_frames` ----
        self.declare_parameter('world_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame', 'camera_frame')
        self.declare_parameter('tag_frame', 'tag36h11:0')

        self.world_frame = self.get_parameter('world_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.tag_frame = self.get_parameter('tag_frame').value

        # ---- Replace with your real camera pose relative to base_link ----
        # T_base_camera maps camera-frame coordinates into base_link coordinates.
        # TEMPORARY example only: camera 8 cm forward, 0 cm left, 12 cm up,
        # with camera axes assumed already aligned to base_link axes.
        # A real optical camera frame usually needs a rotation as well!
        self.temp_camera_xyz = (0.08, 0.0, 0.12)
        self.temp_camera_rpy = (0.0, 0.0, 0.0)
        self.T_base_camera = self.make_extrinsic(
            self.temp_camera_xyz, self.temp_camera_rpy
        )

        # ---- Calibrated motion values ----
        # Option A: set these directly to the measured values from calibration.
        # k_v: m/s when the motor commands are at the calibrated forward level.
        # k_omega: rad/s for the calibrated in-place turn.
        self.k_v = 0.12
        self.k_omega = 0.70
        self.load_calibration_if_available()

        # Motion is deliberately incremental: one command is applied per timer tick,
        # then the next tag observation recalculates the command.
        self.control_period = 0.10
        self.max_linear_command = self.k_v
        self.max_angular_command = self.k_omega
        self.k_heading = 1.8
        self.k_distance = 0.8
        self.max_step_distance = 0.025       # nominal max forward movement per tick
        self.position_tolerance = 0.035      # meters
        self.heading_tolerance = 0.10        # radians
        self.tag_timeout = 0.40              # seconds
        self.search_turn_duration = 0.25     # seconds per short search turn
        self.search_turn_sign = 1.0          # +1 = left; change to -1 to search right

        # Tag is 20 cm behind waypoint along waypoint heading:
        # T_world_tag = T_world_waypoint * translation(-0.20, 0, 0).
        # If your physical placement means the opposite direction, change the sign.
        self.tag_offset_from_waypoint = -0.20

        # Waypoints [x metres, y metres, heading radians], matching the example
        # waypoints in the provided controller. Replace with your assignment's list.
        self.waypoints = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, math.pi / 2],
            [1.0, 0.5, 0.0],
            [1.0, 1.0, -math.pi / 2],
            [0.5, 0.5, -math.pi / 4],
            [0.0, 0.0, 0.0],
        ], dtype=float)

        # Each waypoint must have a corresponding tag frame. For a deployment with
        # unique tags, configure this list to the actual TF frame names.
        self.tag_frames = [
            'tag36h11:0',
            'tag36h11:1',
            'tag36h11:2',
            'tag36h11:3',
            'tag36h11:4',
            'tag36h11:5',
            'tag36h11:6',
        ]

        self.current_waypoint_idx = 0
        self.searching_until = None
        self.searching = False
        self.finished = False

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(self.control_period, self.control_loop)

        self.get_logger().info(
            'AprilTag iterative controller started. Verify frames, tag IDs, '
            'camera extrinsics, tag placement, and calibration before driving.'
        )

    @staticmethod
    def make_extrinsic(xyz, rpy):
        """Build T_base_camera from temporary translation and RPY values."""
        x, y, z = xyz
        roll, pitch, yaw = rpy
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        rotation = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr],
        ])
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = [x, y, z]
        return matrix

    def load_calibration_if_available(self):
        """Optionally read k_v and k_omega from a calibration JSON file."""
        self.declare_parameter('calibration_file', '')
        filename = self.get_parameter('calibration_file').value
        if not filename:
            return
        path = Path(filename).expanduser()
        try:
            with path.open('r', encoding='utf-8') as f:
                calibration = json.load(f)
            self.k_v = float(calibration.get('k_v', self.k_v))
            self.k_omega = float(calibration.get('k_omega', self.k_omega))
            self.get_logger().info(
                f'Loaded calibration: k_v={self.k_v:.4f} m/s, '
                f'k_omega={self.k_omega:.4f} rad/s from {path}'
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self.get_logger().error(
                f'Could not load calibration file {path}: {exc}. '
                'Using built-in temporary values.'
            )

    def get_tag_transform(self, tag_frame):
        """Get T_camera_tag; return None if unavailable or stale."""
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.camera_frame,
                tag_frame,
                Time(),
                timeout=Duration(seconds=0.03),
            )
        except TransformException:
            return None

        stamp = tf_msg.header.stamp
        stamp_seconds = stamp.sec + stamp.nanosec * 1e-9
        now_seconds = self.get_clock().now().nanoseconds * 1e-9
        if stamp_seconds <= 0.0:
            return None
        age = now_seconds - stamp_seconds
        if age > self.tag_timeout or age < -0.10:
            return None

        return transform_to_matrix(tf_msg)

    def estimate_world_robot_pose(self, waypoint, T_camera_tag):
        """
        Estimate T_world_robot from the known waypoint/tag pose.

        T_world_tag = T_world_waypoint * T_waypoint_tag
        T_base_tag  = T_base_camera * T_camera_tag
        T_world_robot = T_world_tag * inverse(T_base_tag)

        This uses the tag's known placement and its observed pose, rather than
        integrating wheel motion. It therefore re-corrects the pose every tick.
        """
        wx, wy, wyaw = waypoint

        # Tag is offset along the waypoint's local x axis.
        T_world_waypoint = pose_matrix(wx, wy, wyaw)
        T_waypoint_tag = pose_matrix(self.tag_offset_from_waypoint, 0.0, 0.0)
        T_world_tag = T_world_waypoint @ T_waypoint_tag

        T_base_tag = self.T_base_camera @ T_camera_tag
        T_world_robot = T_world_tag @ inverse_transform(T_base_tag)
        return T_world_robot

    def publish_stop(self):
        self.cmd_pub.publish(Twist())

    def begin_search_turn(self, now):
        """Brief in-place turn only; never command forward motion while searching."""
        if not self.searching:
            self.searching = True
            self.searching_until = now + self.search_turn_duration
            self.get_logger().warn(
                f'AprilTag not detected/stale. Searching by turning '
                f'{"left" if self.search_turn_sign > 0 else "right"} in place.'
            )

        cmd = Twist()
        if now < self.searching_until:
            cmd.linear.x = 0.0
            cmd.angular.z = self.search_turn_sign * min(
                abs(self.max_angular_command), self.k_omega
            )
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

        waypoint = self.waypoints[self.current_waypoint_idx]
        tag_frame = self.tag_frames[self.current_waypoint_idx]

        T_camera_tag = self.get_tag_transform(tag_frame)
        now = self.get_clock().now().nanoseconds * 1e-9

        if T_camera_tag is None:
            self.begin_search_turn(now)
            return

        # Tag has returned: stop any search turn and immediately re-evaluate pose.
        self.searching = False
        self.searching_until = None

        T_world_robot = self.estimate_world_robot_pose(waypoint, T_camera_tag)
        robot_x = float(T_world_robot[0, 3])
        robot_y = float(T_world_robot[1, 3])
        robot_yaw = rotation_matrix_to_yaw(T_world_robot[:3, :3])

        dx = float(waypoint[0] - robot_x)
        dy = float(waypoint[1] - robot_y)
        distance_error = math.hypot(dx, dy)
        desired_heading = math.atan2(dy, dx) if distance_error > 1e-6 else waypoint[2]
        heading_error = wrap_angle(desired_heading - robot_yaw)

        # First reach waypoint position; then align to its requested orientation.
        cmd = Twist()
        if distance_error > self.position_tolerance:
            if abs(heading_error) > self.heading_tolerance:
                # Turn in place toward waypoint. No forward motion during large errors.
                angular = self.k_heading * heading_error
                cmd.angular.z = float(np.clip(
                    angular, -self.max_angular_command, self.max_angular_command
                ))
                cmd.linear.x = 0.0
            else:
                # Short forward step, then re-observe and correct next timer tick.
                desired_speed = min(
                    self.max_linear_command,
                    max(0.0, self.k_distance * distance_error),
                )
                # Limit nominal travel per tick; next observation corrects the pose.
                cmd.linear.x = min(
                    desired_speed,
                    self.max_step_distance / self.control_period,
                )
                cmd.angular.z = float(np.clip(
                    self.k_heading * heading_error,
                    -self.max_angular_command,
                    self.max_angular_command,
                ))
        else:
            final_heading_error = wrap_angle(float(waypoint[2]) - robot_yaw)
            if abs(final_heading_error) > self.heading_tolerance:
                cmd.linear.x = 0.0
                cmd.angular.z = float(np.clip(
                    self.k_heading * final_heading_error,
                    -self.max_angular_command,
                    self.max_angular_command,
                ))
            else:
                self.publish_stop()
                self.get_logger().info(
                    f'Reached waypoint {self.current_waypoint_idx}: '
                    f'position error={distance_error:.3f} m'
                )
                self.current_waypoint_idx += 1
                return

        self.cmd_pub.publish(cmd)
        self.get_logger().debug(
            f'wp={self.current_waypoint_idx} pose=({robot_x:.3f}, {robot_y:.3f}, '
            f'{robot_yaw:.3f}) error=({dx:.3f}, {dy:.3f}, {heading_error:.3f}) '
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
