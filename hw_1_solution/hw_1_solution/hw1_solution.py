#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import numpy as np
from math import sin, cos

"""
The class of the pid controller for differential drive robot.
"""
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
        """
        set the target pose.
        """
        self.I = np.array([0.0, 0.0]) 
        self.lastError = np.array([0.0, 0.0])
        self.target = np.array(state)

    def getError(self, currentState, targetState, drive_backwards):
        """
        return the error between current and target state
        for differential drive: distance error and heading error
        """
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
        """
        set maximum velocity for stability.
        """
        self.maximumValue = mv

    def update(self, currentState, drive_backwards):
        """
        calculate the update value based on PID control
        Returns: [linear_velocity, angular_velocity]
        """
        e = self.getError(currentState, self.target, drive_backwards)

        P = self.Kp * e
        self.I = self.I + self.Ki * e * self.timestep 
        I = self.I
        D = self.Kd * (e - self.lastError)
        result = P + I + D

        self.lastError = e

        if abs(result[0]) > self.maximumValue:
            result[0] = np.sign(result[0]) * self.maximumValue
            
        max_angular = 1.5 # rad/s
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
        
        self.odom_frame = 'odom'
        self.base_frame = 'base_link'
        
        self.waypoints = np.array([
            [0.0, 0.0, 0.0], 
            [1.0, 0.0, 0.0],
            [1.0, 1.0, np.pi/2],
            [2.0, 1.0, 0.0],
            [2.0, 2.0, -np.pi/2],
            [1.0, 1.0, -np.pi/4],
            [0.0, 0.0, 0.0]
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
        self.fixed_rotation_vel = 0.785
        
    def update_dead_reckoning(self, linear_vel, angular_vel):
        """
        Update robot pose using dead reckoning
        """
        self.current_state[0] += linear_vel * np.cos(self.current_state[2]) * self.dt
        self.current_state[1] += linear_vel * np.sin(self.current_state[2]) * self.dt
        self.current_state[2] += angular_vel * self.dt
        self.current_state[2] = (self.current_state[2] + np.pi) % (2 * np.pi) - np.pi
        
    def broadcast_tf(self):
        """
        Broadcast TF transform from odom to base_link
        """
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
        """
        Convert Euler angles to quaternion
        """
        cy = cos(yaw * 0.5)
        sy = sin(yaw * 0.5)
        cp = cos(pitch * 0.5)
        sp = sin(pitch * 0.5)
        cr = cos(roll * 0.5)
        sr = sin(roll * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return qx, qy, qz, qw
    
    def should_drive_backwards(self, current_wp):
        """
        Determine if robot should drive backwards considering both:
        1. Angle to target position
        2. Target orientation at waypoint
        
        Returns True if backwards is better, False for forward
        """
        delta_x = current_wp[0] - self.current_state[0]
        delta_y = current_wp[1] - self.current_state[1]
        angle_to_target = np.arctan2(delta_y, delta_x)
        
        angle_diff_to_pos = angle_to_target - self.current_state[2]
        angle_diff_to_pos = (angle_diff_to_pos + np.pi) % (2 * np.pi) - np.pi
        
        angle_diff_to_orient = current_wp[2] - self.current_state[2]
        angle_diff_to_orient = (angle_diff_to_orient + np.pi) % (2 * np.pi) - np.pi
        
        forward_rotation = abs(angle_diff_to_pos) + abs(current_wp[2] - angle_to_target)
        
        backward_angle_to_pos = angle_to_target + np.pi
        backward_angle_to_pos = (backward_angle_to_pos + np.pi) % (2 * np.pi) - np.pi
        backward_initial_rotation = backward_angle_to_pos - self.current_state[2]
        backward_initial_rotation = (backward_initial_rotation + np.pi) % (2 * np.pi) - np.pi
        
        backward_rotation = abs(backward_initial_rotation) + abs(current_wp[2] - backward_angle_to_pos)
        
        drive_backwards = backward_rotation < forward_rotation
        return drive_backwards
    
    def get_desired_heading_to_goal(self, current_wp, drive_backwards):
        """
        Get the desired heading to face towards (or away from) the goal
        """
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
        """
        Determine rotation direction based on heading error.
        Returns: angular velocity with fixed magnitude but correct direction
        """
        if heading_error > 0:
            return self.fixed_rotation_vel
        else:
            return -self.fixed_rotation_vel
        
    def control_loop(self):
        """
        Main control loop with three stages: rotate to goal, drive, rotate to orientation
        """
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
        
        # Stage 1: Rotate to face the goal (or away if driving backwards)
        if self.stage == 'rotate_to_goal':
            desired_heading = self.get_desired_heading_to_goal(current_wp, self.drive_backwards)
            heading_error = desired_heading - self.current_state[2]
            heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
            
            if abs(heading_error) < 0.05:
                self.stage = 'drive'
                twist_msg.angular.z = 0.0
            else:
                twist_msg.angular.z = float(self.get_rotation_direction(heading_error))
        
        # Stage 2: Drive towards/away from the goal
        elif self.stage == 'drive':
            position_error = np.sqrt(delta_x**2 + delta_y**2)

            if position_error < self.tolerance:
                self.stage = 'rotate_to_orient'
                twist_msg.linear.x = 0.0
            else:
                update_value = self.pid.update(self.current_state, self.drive_backwards)
                twist_msg.linear.x = float(update_value[0])
        
        # Stage 3: Rotate to target orientation
        elif self.stage == 'rotate_to_orient':
            heading_error = current_wp[2] - self.current_state[2]
            heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
            if abs(heading_error) < self.angle_tolerance:
                self.current_waypoint_idx += 1
                self.waypoint_reached = False
                twist_msg.angular.z = 0.0
            else:
                twist_msg.angular.z = float(self.get_rotation_direction(heading_error))
        
        self.update_dead_reckoning(twist_msg.linear.x, twist_msg.angular.z)
        self.broadcast_tf()
        self.cmd_vel_pub.publish(twist_msg)

    
    def stop_robot(self):
        """
        Stop the robot by publishing zero velocities
        """
        twist_msg = Twist()
        self.cmd_vel_pub.publish(twist_msg)

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