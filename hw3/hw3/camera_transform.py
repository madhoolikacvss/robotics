"""Shared AprilTag camera-to-base translation used by the EKF and inspector."""

import numpy as np

CAMERA_X_BIAS = 0.02
CAMERA_TRANSLATION_BASE = (0.125, 0.0, 0.09)
ROTATION_BASE_CAMERA = (
    (0.0, 0.0, 1.0),
    (-1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
)


def camera_to_base_translation(
    camera_translation: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return bias-corrected camera and transformed base-frame translations."""
    camera_corrected = np.asarray(camera_translation, dtype=float).copy()
    if camera_corrected.shape != (3,):
        raise ValueError('camera_translation must contain exactly 3 values.')
    camera_corrected[0] -= CAMERA_X_BIAS
    base_translation = (
        np.asarray(CAMERA_TRANSLATION_BASE)
        + np.asarray(ROTATION_BASE_CAMERA) @ camera_corrected
    )
    return camera_corrected, base_translation
