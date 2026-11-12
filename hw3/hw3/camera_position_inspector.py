#!/usr/bin/env python3
"""Print AprilTag positions transformed into the robot base frame."""

import rclpy
from apriltag_msgs.msg import AprilTagDetectionArray
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from .camera_transform import camera_to_base_translation


class CameraPositionInspector(Node):
    """Display EKF-equivalent camera-to-base tag positions without motion."""

    def __init__(self):
        super().__init__('camera_position_inspector')
        self.camera_frame = str(
            self.declare_parameter('camera_frame', 'camera_frame').value
        )
        self.tag_frame_prefix = str(
            self.declare_parameter('tag_frame_prefix', 'tag_').value
        )
        self.tag_max_age = float(
            self.declare_parameter('tag_max_age', 1.0).value
        )
        if self.tag_max_age <= 0.0:
            raise ValueError('tag_max_age must be positive.')

        self.pending: dict[int, set[str]] = {}
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(
            AprilTagDetectionArray,
            '/detections',
            self.detections_callback,
            10,
        )
        self.timer = self.create_timer(0.02, self.print_positions)
        self.get_logger().info(
            'Printing EKF-equivalent tag positions in the robot base frame; '
            'this node does not publish velocity commands.'
        )

    def detections_callback(self, msg: AprilTagDetectionArray):
        """Queue detections until their matching timestamped TF is available."""
        stamp = msg.header.stamp
        stamp_nanoseconds = stamp.sec * 1_000_000_000 + stamp.nanosec
        tag_frames = self.pending.setdefault(stamp_nanoseconds, set())
        tag_frames.update(
            f'{self.tag_frame_prefix}{detection.id}'
            for detection in msg.detections
        )

    def print_positions(self):
        """Print each pending tag's transformed position once."""
        now = self.get_clock().now().nanoseconds * 1e-9
        for stamp_nanoseconds in sorted(tuple(self.pending)):
            stamp_seconds = stamp_nanoseconds * 1e-9
            age = now - stamp_seconds
            tag_frames = self.pending[stamp_nanoseconds]
            if (
                stamp_seconds <= 0.0
                or age < -0.10
                or age > self.tag_max_age
            ):
                self.get_logger().warning(
                    f'Dropping AprilTag frame at {stamp_seconds:.9f}: '
                    f'age={age:.3f}s is outside the valid range '
                    f'[-0.10, {self.tag_max_age:.3f}]s or timestamp is '
                    'invalid.'
                )
                del self.pending[stamp_nanoseconds]
                continue

            for tag_frame in sorted(tuple(tag_frames)):
                try:
                    transform = self.tf_buffer.lookup_transform(
                        self.camera_frame,
                        tag_frame,
                        Time(
                            nanoseconds=stamp_nanoseconds,
                            clock_type=self.get_clock().clock_type,
                        ),
                        timeout=Duration(seconds=0.0),
                    )
                except TransformException:
                    continue

                translation = transform.transform.translation
                camera_corrected, base_position = camera_to_base_translation(
                    (translation.x, translation.y, translation.z)
                )
                self.get_logger().info(
                    f'{tag_frame} at {stamp_seconds:.9f}: '
                    f'camera_raw=({translation.x:.3f}, '
                    f'{translation.y:.3f}, {translation.z:.3f})m; '
                    f'camera_corrected=({camera_corrected[0]:.3f}, '
                    f'{camera_corrected[1]:.3f}, '
                    f'{camera_corrected[2]:.3f})m; '
                    f'tag_in_robot_frame=({base_position[0]:.3f}, '
                    f'{base_position[1]:.3f}, '
                    f'{base_position[2]:.3f})m.'
                )
                tag_frames.remove(tag_frame)

            if not tag_frames:
                del self.pending[stamp_nanoseconds]


def main(args=None):
    """Run the read-only camera position inspector."""
    rclpy.init(args=args)
    node = CameraPositionInspector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
