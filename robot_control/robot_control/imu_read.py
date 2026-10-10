#!/usr/bin/env python3
"""
Live IMU yaw monitor. Subscribes to /imu_data and prints yaw at ~5 Hz.
Requires: motor_control.py running (it publishes /imu_data).

Usage:
    ros2 run robot_control imu_monitor
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import math


def quat_to_yaw(qw, qx, qy, qz):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class ImuMonitor(Node):
    def __init__(self):
        super().__init__('imu_monitor')
        self.sub = self.create_subscription(Imu, 'imu_data', self.cb, 10)
        self.last_print = 0.0
        self.samples = 0
        self.get_logger().info("Listening on /imu_data ...")

    def cb(self, msg: Imu):
        yaw_deg = math.degrees(quat_to_yaw(
            msg.orientation.w, msg.orientation.x,
            msg.orientation.y, msg.orientation.z,
        ))
        self.samples += 1
        # print at ~5 Hz (throttle)
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_print > 0.2:
            self.last_print = now
            print(f"yaw = {yaw_deg:+8.2f}°   (samples: {self.samples})")


def main(args=None):
    rclpy.init(args=args)
    node = ImuMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()