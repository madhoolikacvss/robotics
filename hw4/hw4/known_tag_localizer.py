import numpy as np
import rclpy
from hw3.ekf_localizer import AprilTagEKFLocalizer

from .tag_map import TAG_POSES_WORLD, planar_tag_pose


def initialize_known_map(ekf, tag_poses):
    """Append fixed landmarks with zero covariance and zero cross-covariance.

    With these zero blocks, landmark Kalman gains remain zero: observations
    correct the robot pose, while measured world landmark poses stay fixed.
    """
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
        # Delayed observations replay this initial snapshot too.
        self._save_history_state(self.history[0], prior=True)
        self._save_history_state(self.history[0])
        self.get_logger().info(
            f'Loaded {len(TAG_POSES_WORLD)} fixed landmarks from hw4/tag_map.py. '
            'These coordinates must match the current workspace; the initial '
            'values are example placeholders, not measurements.'
        )

    def _apply_entry_observations(self, entry):
        """Correct robot pose using known tags; never learn unknown tags."""
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
