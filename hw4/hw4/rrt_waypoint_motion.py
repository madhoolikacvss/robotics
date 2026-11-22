"""Position-only waypoint motion for RRT routes."""

import math
from typing import Sequence, Tuple

from hw3.velocity_mapping import K_OMEGA, K_V
from hw3.waypoint_motion import (
    HEADING_TOLERANCE,
    K_DISTANCE,
    WaypointMotion,
    wrap_angle,
)


class RRTWaypointMotion(WaypointMotion):
    """Track RRT x/y coordinates without aligning to waypoint yaw."""

    def __init__(self, path: Sequence[Sequence[float]]):
        super().__init__()
        if not path:
            raise ValueError('RRT path must contain at least one waypoint.')

        waypoints = []
        for point in path:
            if len(point) < 2:
                raise ValueError('RRT waypoints must contain x and y.')
            x, y = point[0], point[1]
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError('RRT waypoint coordinates must be finite.')
            waypoints.append((x, y, 0.0))

        self.waypoints: Tuple[Tuple[float, float, float], ...] = tuple(waypoints)

    def command_for_pose(
        self, robot_x: float, robot_y: float, robot_yaw: float
    ) -> tuple[float, float, str, float, float, float, float]:
        waypoint = self.waypoints[self.current_waypoint_idx]
        dx = waypoint[0] - robot_x
        dy = waypoint[1] - robot_y
        distance_error = math.hypot(dx, dy)

        if distance_error <= self.position_tolerance:
            return 0.0, 0.0, 'reached', dx, dy, 0.0, distance_error

        heading_error = wrap_angle(
            math.atan2(dy, dx) - robot_yaw
        )
        if self.turn_complete or abs(heading_error) <= HEADING_TOLERANCE:
            linear = min(K_V, max(0.0, K_DISTANCE * distance_error))
            return linear, 0.0, 'drive', dx, dy, heading_error, distance_error

        angular = math.copysign(K_OMEGA, heading_error)
        return 0.0, angular, 'turn', dx, dy, heading_error, distance_error
