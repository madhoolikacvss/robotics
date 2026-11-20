"""Known world tag poses: (x, y, z, roll, pitch, yaw), meters/radians"""

import math

WORKSPACE_TAGS = {
    'tag_3': {'side': 'right', 'pose': (2.20, 0.84, 0.15, math.pi / 2, 0.0, -math.pi / 2)},
    'tag_2': {'side': 'right', 'pose': (2.20, 2.05, 0.15, math.pi / 2, 0.0, -math.pi / 2)},
    'tag_0': {'side': 'bottom', 'pose': (0.37, 0.0, 0.15, math.pi / 2, 0.0, math.pi)},
    'tag_1': {'side': 'bottom', 'pose': (1.33, 0.0, 0.15, math.pi / 2, 0.0, math.pi)},
    'tag_4': {'side': 'top', 'pose': (1.8, 2.79, 0.15, math.pi / 2, 0.0, 0.0)},
    'tag_8': {'side': 'top', 'pose': (0.21, 2.57, 0.15, math.pi / 2, 0.0, 0.0)},
    'tag_7': {'side': 'left', 'pose': (0.0, 0.75, 0.15, math.pi / 2, 0.0, math.pi / 2)},
    # i dont have these tgas rn, so just place holders
    'tag_9': {'side': 'left', 'pose': (0.0, 1.5240, 0.15, math.pi / 2, 0.0, math.pi / 2)},
}

OBSTACLE_TAGS = {
    'tag_5': {'side': 'right', 'pose': (1.24, 1.26, 0.15, math.pi / 2, 0.0, math.pi / 2)},
    'tag_6': {'side': 'bottom', 'pose': (0.99, 0.99, 0.15, math.pi / 2, 0.0, 0.0)},
    # i dont have these tgas rn, so just place holders
    'tag_10': {'side': 'left', 'pose': (0.69, 1.26, 0.15, math.pi / 2, 0.0, -math.pi / 2)},
    'tag_11': {'side': 'top', 'pose': (0.99, 1.34, 0.15, math.pi / 2, 0.0, math.pi)},
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
