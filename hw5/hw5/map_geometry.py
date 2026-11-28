"""Workspace-only geometry for HW5; no obstacle is modeled."""

from dataclasses import dataclass
import math

from .tag_map import WORKSPACE_TAGS


FT_TO_M = 0.3048
ROBOT_SIDE = 1.0 * FT_TO_M
ROBOT_TURN_RADIUS = ROBOT_SIDE / math.sqrt(2.0)
EXTRA_WALL_CLEARANCE = 0.05
WALL_CENTER_CLEARANCE = ROBOT_TURN_RADIUS + EXTRA_WALL_CLEARANCE
ENDPOINT_INSET = 0.05


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

    def inset(self, distance):
        if not math.isfinite(distance) or distance < 0.0:
            raise ValueError('Inset must be finite and non-negative.')
        return Rectangle(
            self.x_min + distance, self.y_min + distance,
            self.x_max - distance, self.y_max - distance,
        )


def rectangle_from_tags(tags):
    """Infer rectangle bounds by averaging measured tag coordinates per side."""
    sides = {'left': [], 'right': [], 'bottom': [], 'top': []}
    for tag_id, tag in tags.items():
        side = tag['side']
        if side not in sides:
            raise ValueError(f'{tag_id}: side must be left/right/bottom/top.')
        pose = tag['pose']
        if len(pose) != 6 or not all(math.isfinite(value) for value in pose):
            raise ValueError(f'{tag_id}: pose needs six finite values.')
        sides[side].append(pose[0] if side in ('left', 'right') else pose[1])
    missing = [side for side, values in sides.items() if not values]
    if missing:
        raise ValueError(f'Missing workspace tags on: {", ".join(missing)}.')
    bounds = {side: sum(values) / len(values) for side, values in sides.items()}
    return Rectangle(bounds['left'], bounds['bottom'],
                     bounds['right'], bounds['top'])


WORKSPACE = rectangle_from_tags(WORKSPACE_TAGS)
USABLE_WORKSPACE = WORKSPACE.inset(WALL_CENTER_CLEARANCE)

# Initial localization pose, retained from the HW4 setup. The generated route
# itself uses the stricter 26.55 cm wall clearance above.
_START_INSET = ROBOT_TURN_RADIUS + 0.25 * FT_TO_M + ENDPOINT_INSET
START = (WORKSPACE.x_min + _START_INSET, WORKSPACE.y_min + _START_INSET)
