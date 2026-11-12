import numpy as np

from hw3.camera_transform import camera_to_base_translation


def test_camera_to_base_translation_matches_ekf_transform():
    """Keep the inspector transform numerically aligned with the EKF."""
    camera_corrected, base_position = camera_to_base_translation(
        (0.12, 0.03, 0.75)
    )

    np.testing.assert_allclose(camera_corrected, (0.10, 0.03, 0.75))
    np.testing.assert_allclose(base_position, (0.875, -0.10, 0.06))
