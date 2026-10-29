#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from tf2_ros import TransformBroadcaster, Buffer, TransformListener, TransformException
import numpy as np
from math import sin, cos
import tf2_ros

# ---------------------------------------------------------------
# Tag world coordinates (constant, measured when placing the tags)
# World frame is anchored to Tag 0: Tag 0 is at origin, +x along its top edge
# ---------------------------------------------------------------
TAG_WORLD = {
    0: (0.0, 0.0, 0.0),                 # Tag 0: origin
    1: (1.0, 0.0, 0.0),                 # Tag 1
    2: (1.0, 2.0, np.pi / 2),           # Tag 2 (rotated 90 deg)
    3: (0.5, 0.5, 3 * np.pi / 4),       # intermediate tag
}

# Camera mounting (base_link -> camera_optical_frame)
# measured: 4 cm forward, 0 cm lateral, 12.5 cm high
T_BASE_CAM = np.eye(4)
T_BASE_CAM[0, 3] = 0.04
T_BASE_CAM[1, 3] = 0.0
T_BASE_CAM[2, 3] = 0.125
# forward-facing camera: optical z -> base x, optical x -> base -y, optical y -> base -z
# rotation = Rz(-pi/2) @ Rx(-pi/2)
def _rot_y(yaw):
    c, s = cos(yaw), sin(yaw)
    R = np.eye(4)
    R[0,0], R[0,1] = c, -s
    R[1,0], R[1,1] = s,  c
    return R

def _rot_x(pitch):
    c, s = cos(pitch), sin(pitch)
    R = np.eye(4)
    R[1,1], R[1,2] = c, -s
    R[2,1], R[2,2] = s,  c
    return R

T_BASE_CAM = T_BASE_CAM @ _rot_y(-np.pi/2) @ _rot_x(-np.pi/2)


def make_T_world_tag(x, y, yaw):
    T = np.eye(4)
    T[0,0] = cos(yaw); T[0,1] = -sin(yaw); T[0,3] = x
    T[1,0] = sin(yaw); T[1,1] =  cos(yaw); T[1,3] = y
    return T

T_WORLD_TAG = {tid: make_T_world_tag(*vals) for tid, vals in TAG_WORLD.items()}


def transform_to_mat44(tf_msg):
    """Convert TransformStamped to 4x4 numpy matrix."""
    t = tf_msg.transform.translation
    q = tf_msg.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    # quaternion to rotation matrix
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def extract_xy_yaw(T):
    """Extract (x, y, yaw) from a 4x4 matrix (planar assumption)."""
    x = T[0, 3]
    y = T[1, 3]
    yaw = np.arctan2(T[1, 0], T[0, 0])
    return np.array([x, y, yaw])


class PIDcontroller:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.target = None
        self.I = np.array([0.0, 0.0])
        self.lastError = np.array([0.0, 0.0])
        self.timestep = 0.1
        self.maximumValue = 0.2

    def setTarget(self, state):
        self.I = np.array([0.0, 0.0])
        self.lastError = np.array([0.0, 0.0])
        self.target = np.array(state)

    def getError(self, currentState, targetState, drive_backwards):
        delta_x = targetState[0] - currentState[0]
        delta_y = targetState[1] - currentState[1]
        distance = np.sqrt(delta_x**2 + delta_y**2)
        angle_to_target = np.arctan2(delta_y, delta_x)
        if drive_backwards:
            desired_heading = angle_to_target + np.pi
            desired_heading = (desired_heading + np.pi) % (2 * np.pi) - np.pi
            distance = -distance
        else:
            desired_heading = angle_to_target
        heading_error = desired_heading - currentState[2]
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
        if abs(distance) < 0.05:
            heading_error = targetState[2] - currentState[2]
            heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
            distance = 0.0
        return np.array([distance, heading_error])

    def setMaximumUpdate(self, mv):
        self.maximumValue = mv

    def update(self, currentState, drive_backwards):
        e = self.getError(currentState, self.target, drive_backwards)
        P = self.Kp * e
        self.I = self.I + self.Ki * e * self.timestep
        I = self.I
        D = self.Kd * (e - self.lastError)
        result = P + I + D
        self.lastError = e
        if abs(result[0]) > self.maximumValue:
            result[0] = np.sign(result[0]) * self.maximumValue
        max_angular = 1.5
        if abs(result[1]) > max_angular:
            result[1] = np.sign(result[1]) * max_angular
        if abs(e[0]) < 0.05:
            result[0] = 0.0
        return result


class Hw1SolutionNode(Node):
    def __init__(self):
        super().__init__('hw1_solution_node')

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # TF buffer/listener for reading apriltag_ros output
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.odom_frame = 'odom'
        self.base_frame = 'base_link'

        self.waypoints = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 2.0, np.pi],
            [0.0, 0.0, 0.0],
        ])

        self.pid = PIDcontroller(0.5, 0.01, 0.005)
        self.current_state = np.array([0.0, 0.0, 0.0])
        self.current_waypoint_idx = 0
        self.waypoint_reached = False
        self.tolerance = 0.1
        self.angle_tolerance = 0.1
        self.drive_backwards = False
        self.dt = 0.1
        self.control_timer = self.create_timer(self.dt, self.control_loop)
        self.stage = 'rotate_to_goal'
        self.stage_pid = PIDcontroller(0.5, 0.01, 0.005)
        self.left_rotation_vel = 1.35
        self.right_rotation_vel = 1.53

        # tag freshness threshold
        self.tag_timeout = 0.5  # seconds
        # visual pose smoothing
        self.vis_alpha = 0.7
        self.last_vis_pose = None

        # logging
        self.total_distance = 0.0
        self.total_rotation = 0.0
        self.trajectory = []  # (time, x, y, yaw)
        self.waypoint_errors = []  # (wp_idx, pos_err, yaw_err)

    # -----------------------------------------------------------
    # Localization: AprilTag -> world pose
    # -----------------------------------------------------------
    def get_pose_from_tags(self):
        now = self.get_clock().now()
        for tag_id in TAG_WORLD:
            try:
                T_cam_tag_msg = self.tf_buffer.lookup_transform(
                    'camera_optical_frame', f'tag_{tag_id}', rclpy.time.Time())
            except TransformException:
                continue

            # freshness check
            stamp = rclpy.time.Time.from_msg(T_cam_tag_msg.header.stamp)
            age = (now - stamp).nanoseconds / 1e9
            if age > self.tag_timeout:
                continue

            T_cam_tag = transform_to_mat44(T_cam_tag_msg)
            T_world_base = T_WORLD_TAG[tag_id] @ np.linalg.inv(T_cam_tag) @ T_BASE_CAM
            return extract_xy_yaw(T_world_base)

        return None

    # -----------------------------------------------------------
    # Pose update: visual overwrite with smoothing, dead-reckoning fallback
    # -----------------------------------------------------------
    def update_pose(self, linear_vel, angular_vel):
        visual_pose = self.get_pose_from_tags()
        if visual_pose is not None:
            if self.last_vis_pose is None:
                self.current_state = visual_pose.copy()
            else:
                # wrap yaw difference for smoothing
                dyaw = visual_pose[2] - self.current_state[2]
                dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
                self.current_state[0] = self.vis_alpha * visual_pose[0] + (1 - self.vis_alpha) * self.current_state[0]
                self.current_state[1] = self.vis_alpha * visual_pose[1] + (1 - self.vis_alpha) * self.current_state[1]
                self.current_state[2] += self.vis_alpha * dyaw
                self.current_state[2] = (self.current_state[2] + np.pi) % (2 * np.pi) - np.pi
            self.last_vis_pose = visual_pose.copy()
        else:
            self.update_dead_reckoning(linear_vel, angular_vel)
            self.last_vis_pose = None

    def update_dead_reckoning(self, linear_vel, angular_vel):
        self.current_state[0] += linear_vel * np.cos(self.current_state[2]) * self.dt
        self.current_state[1] += linear_vel * np.sin(self.current_state[2]) * self.dt
        self.current_state[2] += angular_vel * self.dt
        self.current_state[2] = (self.current_state[2] + np.pi) % (2 * np.pi) - np.pi

    def broadcast_tf(self):
        current_time = self.get_clock().now()
        t = TransformStamped()
        t.header.stamp = current_time.to_msg()
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame
        t.transform.translation.x = self.current_state[0]
        t.transform.translation.y = self.current_state[1]
        t.transform.translation.z = 0.0
        qx, qy, qz, qw = self.euler_to_quaternion(0, 0, self.current_state[2])
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(t)

    def euler_to_quaternion(self, roll, pitch, yaw):
        cy = cos(yaw * 0.5); sy = sin(yaw * 0.5)
        cp = cos(pitch * 0.5); sp = sin(pitch * 0.5)
        cr = cos(roll * 0.5); sr = sin(roll * 0.5)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        return qx, qy, qz, qw

    def should_drive_backwards(self, current_wp):
        delta_x = current_wp[0] - self.current_state[0]
        delta_y = current_wp[1] - self.current_state[1]
        angle_to_target = np.arctan2(delta_y, delta_x)
        angle_diff_to_pos = angle_to_target - self.current_state[2]
        angle_diff_to_pos = (angle_diff_to_pos + np.pi) % (2 * np.pi) - np.pi
        forward_rotation = abs(angle_diff_to_pos) + abs(current_wp[2] - angle_to_target)
        backward_angle_to_pos = angle_to_target + np.pi
        backward_angle_to_pos = (backward_angle_to_pos + np.pi) % (2 * np.pi) - np.pi
        backward_initial_rotation = backward_angle_to_pos - self.current_state[2]
        backward_initial_rotation = (backward_initial_rotation + np.pi) % (2 * np.pi) - np.pi
        backward_rotation = abs(backward_initial_rotation) + abs(current_wp[2] - backward_angle_to_pos)
        return backward_rotation < forward_rotation

    def get_desired_heading_to_goal(self, current_wp, drive_backwards):
        delta_x = current_wp[0] - self.current_state[0]
        delta_y = current_wp[1] - self.current_state[1]
        angle_to_target = np.arctan2(delta_y, delta_x)
        if drive_backwards:
            desired_heading = angle_to_target + np.pi
            desired_heading = (desired_heading + np.pi) % (2 * np.pi) - np.pi
        else:
            desired_heading = angle_to_target
        return desired_heading

    def get_rotation_direction(self, heading_error):
        if abs(heading_error) < self.angle_tolerance:
            return 0.0
        if heading_error > 0:
            return self.left_rotation_vel
        else:
            return -self.right_rotation_vel

    def control_loop(self):
        if self.current_waypoint_idx >= len(self.waypoints):
            self.get_logger().info('All waypoints reached! Stopping robot.')
            self.stop_robot()
            self.broadcast_tf()
            return

        current_wp = self.waypoints[self.current_waypoint_idx]

        if not self.waypoint_reached:
            self.pid.setTarget(current_wp)
            self.drive_backwards = self.should_drive_backwards(current_wp)
            self.waypoint_reached = True
            self.stage = 'rotate_to_goal'
            self.stage_pid.setTarget(current_wp)

        delta_x = current_wp[0] - self.current_state[0]
        delta_y = current_wp[1] - self.current_state[1]
        position_error = np.sqrt(delta_x**2 + delta_y**2)
        twist_msg = Twist()

        if self.stage == 'rotate_to_goal':
            desired_heading = self.get_desired_heading_to_goal(current_wp, self.drive_backwards)
            heading_error = desired_heading - self.current_state[2]
            heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
            if abs(heading_error) < self.angle_tolerance:
                self.stage = 'drive'
                twist_msg.angular.z = 0.0
            else:
                twist_msg.angular.z = float(self.get_rotation_direction(heading_error))

        elif self.stage == 'drive':
            position_error = np.sqrt(delta_x**2 + delta_y**2)
            if position_error < self.tolerance:
                self.stage = 'rotate_to_orient'
                twist_msg.linear.x = 0.0
            else:
                update_value = self.pid.update(self.current_state, self.drive_backwards)
                twist_msg.linear.x = float(update_value[0])

        elif self.stage == 'rotate_to_orient':
            heading_error = current_wp[2] - self.current_state[2]
            heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
            if abs(heading_error) < self.angle_tolerance:
                # record waypoint error before advancing
                pos_err = np.linalg.norm(self.current_state[:2] - current_wp[:2])
                yaw_err = (self.current_state[2] - current_wp[2] + np.pi) % (2 * np.pi) - np.pi
                self.waypoint_errors.append((self.current_waypoint_idx, pos_err, yaw_err))
                self.get_logger().info(
                    f'Waypoint {self.current_waypoint_idx} reached: '
                    f'pos_err={pos_err:.3f} m, yaw_err={yaw_err:.3f} rad')
                self.current_waypoint_idx += 1
                self.waypoint_reached = False
                twist_msg.angular.z = 0.0
            else:
                twist_msg.angular.z = float(self.get_rotation_direction(heading_error))

        # update pose (visual-first, dead-reckoning fallback)
        self.update_pose(twist_msg.linear.x, twist_msg.angular.z)

        # accumulate distance / rotation
        self.total_distance += abs(twist_msg.linear.x) * self.dt
        self.total_rotation += abs(twist_msg.angular.z) * self.dt

        # log trajectory every ~0.5 s
        if int(self.get_clock().now().nanoseconds / 1e8) % 5 == 0:
            self.trajectory.append((self.get_clock().now().nanoseconds / 1e9,
                                    self.current_state[0], self.current_state[1],
                                    self.current_state[2]))

        # print pose
        self.get_logger().info(
            f'Pose: x={self.current_state[0]:.3f}, y={self.current_state[1]:.3f}, '
            f'yaw={self.current_state[2]:.3f} | tag_visible={self.last_vis_pose is not None}')

        self.broadcast_tf()
        self.cmd_vel_pub.publish(twist_msg)

    def stop_robot(self):
        twist_msg = Twist()
        self.cmd_vel_pub.publish(twist_msg)
        self.get_logger().info(f'Total distance: {self.total_distance:.3f} m')
        self.get_logger().info(f'Total rotation: {self.total_rotation:.3f} rad')
        if self.trajectory:
            self.get_logger().info(f'Trajectory samples: {len(self.trajectory)}')


def main(args=None):
    rclpy.init(args=args)
    node = Hw1SolutionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Stopped by keyboard interrupt')
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()