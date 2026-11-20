"""
PS: The origin is the physical workspace's bottom-left corner. Positive x points
right and positive y points up in a top-down map. Robot poses refer to its
center
"""

from dataclasses import dataclass
import math

from .tag_map import WORKSPACE_TAGS, OBSTACLE_TAGS


FT_TO_M = 0.3048
ROBOT_SIDE = 1.0 * FT_TO_M
ROBOT_TURN_RADIUS = ROBOT_SIDE / math.sqrt(2.0)
REQUESTED_MARGIN = 0.25 * FT_TO_M
# Extra clearance is added beyond the footprint, not measured from its center.
CENTER_CLEARANCE = ROBOT_TURN_RADIUS + REQUESTED_MARGIN


@dataclass(frozen=True)
class Rectangle:

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self):
        if not all(math.isfinite(value) for value in (
            self.x_min, self.y_min, self.x_max, self.y_max
        )):
            raise ValueError('Rectangle bounds must be finite.')
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError('Rectangle must have positive width and height.')

    @property
    def corners(self):
        """Return bottom-left, bottom-right, top-right, top-left."""
        return (
            (self.x_min, self.y_min),
            (self.x_max, self.y_min),
            (self.x_max, self.y_max),
            (self.x_min, self.y_max),
        )

    def expanded(self, distance):
        """Expand all sides; a negative distance insets the rectangle."""
        return Rectangle(
            self.x_min - distance, self.y_min - distance,
            self.x_max + distance, self.y_max + distance,
        )


def rectangle_from_tags(tags, name):
    sides = {'left': [], 'right': [], 'bottom': [], 'top': []}
    for tag_id, tag in tags.items():
        side = tag['side']
        if side not in sides:
            raise ValueError(
                f'{name} {tag_id}: set side to left/right/bottom/top in tag_map.py.'
            )
        pose = tag['pose']
        if len(pose) != 6 or not all(math.isfinite(value) for value in pose):
            raise ValueError(f'{name} {tag_id}: pose needs six finite values.')
        sides[side].append(pose[0] if side in ('left', 'right') else pose[1])
    missing = [side for side, values in sides.items() if not values]
    if missing:
        raise ValueError(f'{name}: missing tag measurements for {", ".join(missing)}.')
    bounds = {side: sum(values) / len(values) for side, values in sides.items()}
    return Rectangle(bounds['left'], bounds['bottom'], bounds['right'], bounds['top'])


WORKSPACE = rectangle_from_tags(WORKSPACE_TAGS, 'Workspace')
OBSTACLE = rectangle_from_tags(OBSTACLE_TAGS, 'Obstacle')
if not (
    WORKSPACE.x_min < OBSTACLE.x_min < OBSTACLE.x_max < WORKSPACE.x_max
    and WORKSPACE.y_min < OBSTACLE.y_min < OBSTACLE.y_max < WORKSPACE.y_max
):
    raise ValueError('Measured obstacle must lie inside the measured workspace.')


def planning_bounds(center_clearance):
    """Return inset workspace and expanded obstacle for a robot center.

    center_clearance is the TOTAL center-to-boundary distance in meters,
    including the footprint allowance and extra buffer.
    """
    if not math.isfinite(center_clearance) or center_clearance < 0.0:
        raise ValueError('Center clearance must be finite and non-negative.')
    return (
        WORKSPACE.expanded(-center_clearance),
        OBSTACLE.expanded(center_clearance),
    )


USABLE_WORKSPACE, EXPANDED_OBSTACLE = planning_bounds(CENTER_CLEARANCE)
# These are robot-center positions in the physical workspace frame.
START = USABLE_WORKSPACE.corners[0]
GOAL = USABLE_WORKSPACE.corners[2]
