"""HW5 fixed-world-tag localization, reusing the HW3 AprilTag EKF."""

import numpy as np
import rclpy
from hw3.ekf_localizer import AprilTagEKFLocalizer

from .tag_map import TAG_POSES_WORLD, planar_tag_pose


def initialize_known_map(ekf, tag_poses):
    """Add fixed landmarks while preserving the EKF's initial robot estimate."""
    robot_state = ekf.state[:3].copy()
    robot_covariance = ekf.covariance[:3, :3].copy()
    landmarks = []
    ekf.landmark_indices = {}
    for name, pose in sorted(tag_poses.items()):
        ekf.landmark_indices[name] = 3 + len(landmarks)
        landmarks.extend(planar_tag_pose(pose))
    ekf.state = np.concatenate((robot_state, landmarks))
    ekf.covariance = np.zeros((ekf.state.size, ekf.state.size))
    ekf.covariance[:3, :3] = robot_covariance


class KnownTagLocalizer(AprilTagEKFLocalizer):
    def __init__(self):
        super().__init__()
        initialize_known_map(self.ekf, TAG_POSES_WORLD)
        self._save_history_state(self.history[0], prior=True)
        self._save_history_state(self.history[0])
        self.get_logger().info(
            f'Loaded {len(TAG_POSES_WORLD)} fixed workspace tags from '
            'hw5/tag_map.py. Replace copied placeholders with measured poses.'
        )

    def _apply_entry_observations(self, entry):
        if entry.recovery:
            self.ekf.inflate_pose(
                self.recovery_position_std, self.recovery_yaw_std
            )
        observations = [obs for obs in entry.observations
                        if obs.tag in self.ekf.landmark_indices]
        entry.diagnostics = self.ekf.update_batch(observations) if observations else []
        entry.accepted_tags = [
            result.tag for result in entry.diagnostics
            if result.position_accepted or result.orientation_accepted
        ]


def main(args=None):
    rclpy.init(args=args)
    node = KnownTagLocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
