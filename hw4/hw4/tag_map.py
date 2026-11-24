"""Known world tag poses: (x, y, z, roll, pitch, yaw), meters/radians"""

import math

WORKSPACE_TAGS = {
    'tag_3': {'side': 'right', 'pose': (2.44, 0.895, 0.15, math.pi / 2, 0.0, -math.pi / 2)},
    'tag_2': {'side': 'right', 'pose': (2.44, 1.99, 0.15, math.pi / 2, 0.0, -math.pi / 2)},
    'tag_0': {'side': 'bottom', 'pose': (0.755, 0.0, 0.15, math.pi / 2, 0.0, math.pi)},
    'tag_1': {'side': 'bottom', 'pose': (1.635, 0.0, 0.15, math.pi / 2, 0.0, math.pi)},
    'tag_4': {'side': 'top', 'pose': (1.8, 2.44, 0.15, math.pi / 2, 0.0, 0.0)},
    'tag_8': {'side': 'top', 'pose': (0.21, 2.44, 0.15, math.pi / 2, 0.0, 0.0)},
    'tag_7': {'side': 'left', 'pose': (0.0, 1.3, 0.15, math.pi / 2, 0.0, math.pi / 2)},
    # i dont have these tgas rn, so just place holders
    'tag_9': {'side': 'left', 'pose': (0.0, 2.02, 0.15, math.pi / 2, 0.0, math.pi / 2)},
}

OBSTACLE_TAGS = {
    'tag_5': {'side': 'right', 'pose': (1.26, 1.36, 0.15, math.pi / 2, 0.0, math.pi / 2)},
    'tag_6': {'side': 'bottom', 'pose': (1.04, 1.19, 0.15, math.pi / 2, 0.0, 0.0)},
    # i dont have these tgas rn, so just place holders
    'tag_10': {'side': 'left', 'pose': (0.82, 1.36, 0.15, math.pi / 2, 0.0, -math.pi / 2)},
    'tag_11': {'side': 'top', 'pose': (1.04, 1.49, 0.15, math.pi / 2, 0.0, math.pi)},
}

if WORKSPACE_TAGS.keys() & OBSTACLE_TAGS.keys():
    raise ValueError('Workspace and obstacle tags must have distinct IDs.')

# Localization uses exactly the same world poses as the geometry.
TAG_POSES_WORLD = {
    name: tag['pose']
    for group in (WORKSPACE_TAGS, OBSTACLE_TAGS)
    for name, tag in group.items()
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
