"""Shared waypoint-following behavior for localization and dead reckoning."""

import math

from .velocity_mapping import K_OMEGA, K_V

# # Square
# WAYPOINTS = (
#     (0.00, 0.00, 0.0),
#     (1.000, 0.00, math.pi / 2),
#     (1.00, 1.00, math.pi),
#     (0.00, 1.00, -math.pi/2),
#     (0.00, 0.00, 0.0),
# )

# WAYPOINTS = (
#     (0.0, 0.0, 0.0),
#     (1.0, 0.0, 0.0),  # forward only
#     #(0.0, 0.0, 0.0),         # Turn right back to start
# )
# Repeated Square
WAYPOINTS = (
    (0.00, 0.00, 0.0),
    (1.00, 0.00, math.pi / 2),
    (1.00, 1.00, math.pi),
    (0.00, 1.00, -math.pi/2),
    (0.00, 0.00, 0.0),
    (1.000, 0.00, math.pi / 2),
    (1.00, 1.00, math.pi),
    (0.00, 1.00, -math.pi/2),
    (0.00, 0.00, 0.0),
)


# # Octogon
# WAYPOINTS = (
#     (0.00, 0.00, 0.0),
#     (0.292893, 0.000000, 0.0),
#     (0.707107, 0.000000, 0.0),
#     (1.000000, 0.292893, math.pi / 4),
#     (1.000000, 0.707107, math.pi / 2),
#     (0.707107, 1.000000, 3 * math.pi / 4),
#     (0.292893, 1.000000, math.pi),
#     (0.000000, 0.707107, -3 * math.pi / 4),
#     (0.000000, 0.292893, -math.pi / 2),
#     (0.292893, 0.000000, -math.pi / 4),
# )

CONTROL_PERIOD = 0.01
K_DISTANCE = 0.8
POSITION_TOLERANCE = 0.05
HEADING_TOLERANCE = 0.10
# Direction-specific measured turn rates at the controller's K_OMEGA.
# Low battery constants
# ANGULAR_RATE_CCW = 1.6 # 1.86
# ANGULAR_RATE_CW = 1.9 #2.16
# ANGULAR_ANGLE_OFFSET_CCW = -0.35
# ANGULAR_ANGLE_OFFSET_CW = -0.32

# High battery constants
ANGULAR_RATE_CCW = 2.37 # 2.07
ANGULAR_RATE_CW = 2.75 #2.16
ANGULAR_ANGLE_OFFSET_CCW = -0.39 # -0.59
ANGULAR_ANGLE_OFFSET_CW = -0.56

# Testing
# ANGULAR_RATE_CCW = 2.28 #3.2 # 2.07
# ANGULAR_RATE_CW = 2.31 #3.4 #2.16
# ANGULAR_ANGLE_OFFSET_CCW = -0.59
# ANGULAR_ANGLE_OFFSET_CW = -0.56


def wrap_angle(angle: float) -> float:
    """Normalize an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def estimate_angular_velocity(command: float) -> float:
    """Estimate turn rate using the calibration for the command direction."""
    if abs(command) <= 1e-9:
        return 0.0
    rate = ANGULAR_RATE_CCW if command > 0.0 else ANGULAR_RATE_CW
    return math.copysign(rate, command)


def estimate_angular_displacement(
    command: float, elapsed: float, dt: float
) -> tuple[float, float]:
    """Estimate turn angle over a timestep, including calibrated startup lag."""
    if dt <= 0.0:
        return 0.0, elapsed
    if abs(command) <= 1e-9:
        return 0.0, 0.0

    rate = estimate_angular_velocity(command)
    angle_offset = (
        ANGULAR_ANGLE_OFFSET_CCW
        if command > 0.0 else ANGULAR_ANGLE_OFFSET_CW
    )
    startup_delay = max(0.0, -angle_offset / abs(rate))
    next_elapsed = elapsed + dt
    active_duration = max(0.0, next_elapsed - startup_delay) - max(
        0.0, elapsed - startup_delay
    )
    return rate * active_duration, next_elapsed


class WaypointMotion:
    """Common waypoint state machine and turn shaping for both controllers."""

    def __init__(self):
        self.waypoints = WAYPOINTS
        self.current_waypoint_idx = 0
        self.phase = 'position'
        self.finished = False
        self.turn_started = False
        self.turn_complete = False
        self.turn_direction = 0.0

    def command_for_pose(
        self, robot_x: float, robot_y: float, robot_yaw: float
    ) -> tuple[float, float, str, float, float, float, float]:
        """Return the requested twist and waypoint errors for a pose."""
        waypoint = self.waypoints[self.current_waypoint_idx]
        dx = waypoint[0] - robot_x
        dy = waypoint[1] - robot_y
        distance_error = math.hypot(dx, dy)

        if self.phase == 'position' and distance_error <= POSITION_TOLERANCE:
            self.phase = 'heading'
            self.reset_turn()
        elif self.phase == 'heading' and distance_error > POSITION_TOLERANCE:
            self.phase = 'position'
            self.reset_turn()

        if self.phase == 'position':
            desired_heading = math.atan2(dy, dx)
            heading_error = wrap_angle(desired_heading - robot_yaw)
            if abs(heading_error) > HEADING_TOLERANCE:
                linear = 0.0
                angular = math.copysign(K_OMEGA, heading_error)
                action = 'turn'
            else:
                linear = min(
                    K_V,
                    max(0.0, K_DISTANCE * distance_error),
                )
                angular = 0.0
                action = 'drive'
        else:
            heading_error = wrap_angle(waypoint[2] - robot_yaw)
            linear = 0.0
            if abs(heading_error) <= HEADING_TOLERANCE:
                angular = 0.0
                action = 'reached'
            else:
                angular = math.copysign(K_OMEGA, heading_error)
                action = 'final_heading_turn'

        return (
            linear, angular, action, dx, dy, heading_error, distance_error
        )

    def shape_turn(
        self,
        linear: float,
        angular: float,
        heading_error: float,
    ) -> tuple[float, float]:
        """Turn toward the current error, allowing correction after overshoot."""
        if abs(linear) >= 1e-6 or abs(angular) <= 1e-6:
            self.reset_turn()
            return linear, 0.0

        if abs(heading_error) <= HEADING_TOLERANCE:
            self.reset_turn()
            self.turn_complete = True
            return 0.0, 0.0

        self.turn_started = True
        self.turn_complete = False
        self.turn_direction = math.copysign(1.0, heading_error)
        return 0.0, self.turn_direction * K_OMEGA

    def advance_waypoint(self):
        """Move to the next waypoint."""
        self.current_waypoint_idx += 1
        self.reset_turn()
        if self.current_waypoint_idx >= len(self.waypoints):
            self.finished = True
        else:
            self.phase = 'position'

    def reset_turn(self):
        self.turn_started = False
        self.turn_complete = False
        self.turn_direction = 0.0
