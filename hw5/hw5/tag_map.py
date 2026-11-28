"""Known workspace AprilTag poses, copied from the HW4 workspace map.

Pose format is (x, y, z, roll, pitch, yaw), in meters and radians. The
coordinates are the existing HW4 measurements/placeholders; update them after
measuring the HW5 room. Three tags are listed on each wall. The four added
poses are placeholders on their respective wall planes. This file contains
workspace tags only; there is no obstacle tag group in HW5.
"""

import math


WORKSPACE_TAGS = {
    'tag_8': {'side': 'right', 'pose': (2.24, 0.33, 0.15, math.pi / 2, 0.0, -math.pi / 2)},
    'tag_9': {'side': 'right', 'pose': (2.17, 2.06, 0.15, math.pi / 2, 0.0, -math.pi / 2)},
    'tag_10': {'side': 'right', 'pose': (2.17, 1.10, 0.15, math.pi / 2, 0.0, -math.pi / 2)},
    'tag_4': {'side': 'bottom', 'pose': (1.99, 0.0, 0.15, math.pi / 2, 0.0, math.pi)},
    'tag_5': {'side': 'bottom', 'pose': (1.22, 0.0, 0.15, math.pi / 2, 0.0, math.pi)},
    'tag_6': {'side': 'bottom', 'pose': (0.33, 0.0, 0.15, math.pi / 2, 0.0, math.pi)},
    'tag_0': {'side': 'top', 'pose': (0.29, 2.56, 0.15, math.pi / 2, 0.0, 0.0)},
    'tag_1': {'side': 'top', 'pose': (1.69, 2.56, 0.15, math.pi / 2, 0.0, 0.0)},
    'tag_2': {'side': 'top', 'pose': (0.94, 2.56, 0.15, math.pi / 2, 0.0, 0.0)},
    'tag_7': {'side': 'left', 'pose': (0.0, 1.53, 0.15, math.pi / 2, 0.0, math.pi / 2)},
    'tag_12': {'side': 'left', 'pose': (0.0, 0.6, 0.15, math.pi / 2, 0.0, math.pi / 2)},
    'tag_11': {'side': 'left', 'pose': (0.0, 2.21, 0.15, math.pi / 2, 0.0, math.pi / 2)},
}

TAG_POSES_WORLD = {
    tag_id: details['pose'] for tag_id, details in WORKSPACE_TAGS.items()
}


def planar_tag_pose(pose):
    """Project the tag's +z normal into the world xy plane for HW3's EKF."""
    x, y, z, roll, pitch, yaw = pose
    normal_x = (math.cos(yaw) * math.sin(pitch) * math.cos(roll)
                + math.sin(yaw) * math.sin(roll))
    normal_y = (math.sin(yaw) * math.sin(pitch) * math.cos(roll)
                - math.cos(yaw) * math.sin(roll))
    if math.hypot(normal_x, normal_y) < 1e-6:
        raise ValueError('The planar localizer needs upright, side-facing tags.')
    return x, y, math.atan2(normal_y, normal_x)
