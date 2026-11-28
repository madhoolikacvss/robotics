"""Known workspace AprilTag poses, copied from the HW4 workspace map.

Pose format is (x, y, z, roll, pitch, yaw), in meters and radians. The
coordinates are the existing HW4 measurements/placeholders; update them after
measuring the HW5 room. Three tags are listed on each wall. The four added
poses are placeholders on their respective wall planes. This file contains
workspace tags only; there is no obstacle tag group in HW5.
"""

import math


WORKSPACE_TAGS = {
    'tag_3': {'side': 'right', 'pose': (2.44, 0.895, 0.15, math.pi / 2, 0.0, -math.pi / 2)},
    'tag_2': {'side': 'right', 'pose': (2.44, 1.99, 0.15, math.pi / 2, 0.0, -math.pi / 2)},
    'tag_0': {'side': 'bottom', 'pose': (0.755, 0.0, 0.15, math.pi / 2, 0.0, math.pi)},
    'tag_1': {'side': 'bottom', 'pose': (1.635, 0.0, 0.15, math.pi / 2, 0.0, math.pi)},
    'tag_4': {'side': 'top', 'pose': (1.8, 2.44, 0.15, math.pi / 2, 0.0, 0.0)},
    'tag_8': {'side': 'top', 'pose': (0.21, 2.44, 0.15, math.pi / 2, 0.0, 0.0)},
    'tag_7': {'side': 'left', 'pose': (0.0, 1.3, 0.15, math.pi / 2, 0.0, math.pi / 2)},
    # Placeholder measurement copied from HW4; replace with the real pose.
    'tag_9': {'side': 'left', 'pose': (0.0, 2.02, 0.15, math.pi / 2, 0.0, math.pi / 2)},
    # New placeholder tags: replace their along-wall coordinates if needed.
    'tag_10': {'side': 'left', 'pose': (0.0, 0.65, 0.15, math.pi / 2, 0.0, math.pi / 2)},
    'tag_11': {'side': 'right', 'pose': (2.44, 1.45, 0.15, math.pi / 2, 0.0, -math.pi / 2)},
    'tag_12': {'side': 'bottom', 'pose': (1.20, 0.0, 0.15, math.pi / 2, 0.0, math.pi)},
    'tag_13': {'side': 'top', 'pose': (1.20, 2.44, 0.15, math.pi / 2, 0.0, 0.0)},
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
