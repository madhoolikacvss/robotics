"""Print and record HW5 robot pose estimates."""

import csv
import math
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node


class RobotPositionLogger(Node):
    def __init__(self):
        super().__init__('hw5_robot_position_logger')
        path = Path(self.declare_parameter(
            'csv_path', 'hw5_pose_stream.csv'
        ).value).expanduser()
        self.file = path.open('w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            'timestamp_sec', 'frame', 'x', 'y', 'yaw_rad',
            'logged_at_sec', 'pose_age_sec',
        ])
        self.file.flush()
        self.latest_pose = None
        self.sequence = 0
        self.logged_sequence = 0
        self.create_subscription(
            PoseStamped, '/robot_pose_estimate', self.pose_callback, 10
        )
        self.create_timer(0.5, self.log_pose)
        self.get_logger().info(f'Writing HW5 pose estimates to {path.resolve()}')

    def pose_callback(self, message):
        self.latest_pose = message
        self.sequence += 1

    def log_pose(self):
        if self.latest_pose is None:
            self.get_logger().info('Waiting for /robot_pose_estimate...')
            return
        message = self.latest_pose
        now = self.get_clock().now().nanoseconds * 1e-9
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        age = now - stamp
        position = message.pose.position
        q = message.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        text = (
            f'Estimated pose [{message.header.frame_id}]: '
            f'x={position.x:.3f}m, y={position.y:.3f}m, '
            f'yaw={math.degrees(yaw):.1f}deg, age={age:.2f}s'
        )
        if age > 0.5 or age < -0.1:
            self.get_logger().warning(
                f'Stale or invalid pose timestamp: {text}'
            )
        else:
            self.get_logger().info(text)
        if self.sequence != self.logged_sequence:
            self.writer.writerow([
                f'{stamp:.9f}', message.header.frame_id,
                f'{position.x:.6f}', f'{position.y:.6f}', f'{yaw:.6f}',
                f'{now:.9f}', f'{age:.6f}',
            ])
            self.file.flush()
            self.logged_sequence = self.sequence

    def destroy_node(self):
        self.file.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RobotPositionLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
