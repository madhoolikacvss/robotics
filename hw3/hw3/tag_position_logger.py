#!/usr/bin/env python3
"""Write fresh AprilTag positions in the robot frame to a CSV file."""

import csv
from pathlib import Path

import rclpy
import yaml
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


class TagPositionLogger(Node):
    """Log each new AprilTag transform expressed in the robot base frame."""

    def __init__(self):
        super().__init__('tag_position_logger')

        self.base_frame = str(
            self.declare_parameter('base_frame', 'base_link').value
        )
        self.tag_frame_prefix = str(
            self.declare_parameter('tag_frame_prefix', 'tag_').value
        )
        self.tag_max_age = float(
            self.declare_parameter('tag_max_age', 0.60).value
        )
        if self.tag_max_age <= 0.0:
            raise ValueError('tag_max_age must be positive.')

        self.csv_path = Path(
            self.declare_parameter(
                'csv_path', 'apriltag_positions.csv'
            ).value
        ).expanduser()
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_file = self.csv_path.open(
            'a', newline='', encoding='utf-8'
        )
        self.csv_writer = csv.writer(self.csv_file)
        if self.csv_path.stat().st_size == 0:
            self.csv_writer.writerow([
                'timestamp_sec', 'tag_frame', 'robot_x_m', 'robot_y_m',
            ])
            self.csv_file.flush()

        self.last_written_stamp: dict[str, float] = {}
        self.warned_frames: set[str] = set()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(0.01, self.log_tag_positions)

        self.get_logger().info(
            f'Listening to TF for {self.tag_frame_prefix}* transforms; '
            f'writing positions in {self.base_frame} to {self.csv_path}.'
        )

    def log_tag_positions(self):
        """Write each fresh tag transform once, using its measurement time."""
        frames = yaml.safe_load(self.tf_buffer.all_frames_as_yaml()) or {}
        if not isinstance(frames, dict):
            raise ValueError('TF buffer returned an invalid frame graph.')

        now = self.get_clock().now().nanoseconds * 1e-9
        for tag_frame in sorted(frames):
            if not tag_frame.startswith(self.tag_frame_prefix):
                continue

            try:
                transform = self.tf_buffer.lookup_transform(
                    self.base_frame,
                    tag_frame,
                    Time(),
                    timeout=Duration(seconds=0.0),
                )
            except TransformException as exc:
                if tag_frame not in self.warned_frames:
                    self.get_logger().warning(
                        f'Cannot transform {tag_frame} to {self.base_frame}: '
                        f'{exc}'
                    )
                    self.warned_frames.add(tag_frame)
                continue

            stamp = transform.header.stamp
            timestamp = stamp.sec + stamp.nanosec * 1e-9
            age = now - timestamp
            if (
                timestamp <= self.last_written_stamp.get(tag_frame, 0.0)
                or timestamp <= 0.0
                or age < -0.10
                or age > self.tag_max_age
            ):
                continue

            translation = transform.transform.translation
            self.csv_writer.writerow([
                f'{timestamp:.9f}',
                tag_frame,
                f'{translation.x:.6f}',
                f'{translation.y:.6f}',
            ])
            self.csv_file.flush()
            self.last_written_stamp[tag_frame] = timestamp
            self.warned_frames.discard(tag_frame)

    def destroy_node(self):
        """Close the CSV file before shutting down the ROS node."""
        if not self.csv_file.closed:
            self.csv_file.close()
        super().destroy_node()


def main(args=None):
    """Run the AprilTag position logger."""
    rclpy.init(args=args)
    node = TagPositionLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
