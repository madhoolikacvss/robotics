"""Shared waypoint-following behavior for localization and dead reckoning."""

import math

from .velocity_mapping import K_OMEGA, K_V

WAYPOINTS = (
    (0.00, 0.00, 0.0),
    (0.80, 0.00, 0.0),
    (0.80, 1.60, math.pi),
    (0.00, 0.00, 0.0),
)

CONTROL_PERIOD = 0.02
K_HEADING = 0.10
K_DISTANCE = 0.8
POSITION_TOLERANCE = 0.08
HEADING_TOLERANCE = 0.11
COARSE_TURN_RATE = 2.2
COARSE_TURN_CAPTURE_ANGLE = math.radians(15.0)
FINE_TURN_PULSE_DURATION = 0.05
FINE_TURN_PAUSE_DURATION = 0.20


def wrap_angle(angle: float) -> float:
    """Normalize an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class WaypointMotion:
    """Common waypoint state machine and turn shaping for both controllers."""

    def __init__(self):
        self.waypoints = WAYPOINTS
        self.current_waypoint_idx = 0
        self.phase = 'position'
        self.finished = False
        self.turn_mode = 'idle'
        self.turn_direction = 0.0
        self.turn_until = 0.0
        self.turn_pause_until = 0.0

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

        if self.phase == 'position':
            desired_heading = math.atan2(dy, dx)
            heading_error = wrap_angle(desired_heading - robot_yaw)
            angular = max(
                -K_OMEGA,
                min(
                    K_HEADING * heading_error,
                    K_OMEGA,
                ),
            )
            if abs(heading_error) > HEADING_TOLERANCE:
                angular = math.copysign(
                    K_OMEGA, heading_error
                )
                action = 'turn'
                linear = 0.0
            else:
                linear = min(
                    K_V,
                    max(0.0, K_DISTANCE * distance_error),
                )
                action = 'drive'
        else:
            heading_error = wrap_angle(waypoint[2] - robot_yaw)
            angular = max(
                -K_OMEGA,
                min(
                    K_HEADING * heading_error,
                    K_OMEGA,
                ),
            )
            linear = 0.0
            action = (
                'final_heading_turn'
                if abs(heading_error) > HEADING_TOLERANCE
                else 'reached'
            )

        return (
            linear, angular, action, dx, dy, heading_error, distance_error
        )

    def shape_turn(
        self,
        now: float,
        linear: float,
        angular: float,
        heading_error: float,
        allow_new_turn: bool = True,
    ) -> tuple[float, float]:
        """Apply the shared coarse-turn and fine-pulse behavior."""
        if abs(linear) >= 1e-6 or abs(angular) <= 1e-6:
            self.reset_turn()
            return linear, angular

        direction = math.copysign(1.0, angular)
        if direction != self.turn_direction:
            self.turn_mode = 'idle'
            self.turn_direction = direction

        output_angular = 0.0
        if self.turn_mode == 'coarse':
            turn_complete = (
                now >= self.turn_until
                or abs(heading_error) <= COARSE_TURN_CAPTURE_ANGLE
                or heading_error * direction <= 0.0
            )
            if turn_complete:
                self.turn_mode = 'fine_pause'
                self.turn_pause_until = now + FINE_TURN_PAUSE_DURATION
            else:
                output_angular = direction * K_OMEGA
        elif self.turn_mode == 'fine_pulse':
            if now < self.turn_until:
                output_angular = direction * K_OMEGA
            else:
                self.turn_mode = 'fine_pause'
                self.turn_pause_until = now + FINE_TURN_PAUSE_DURATION
        elif self.turn_mode == 'fine_pause':
            if now >= self.turn_pause_until:
                self.turn_mode = 'idle'

        if self.turn_mode == 'idle' and allow_new_turn:
            if abs(heading_error) > COARSE_TURN_CAPTURE_ANGLE:
                self.turn_mode = 'coarse'
                self.turn_until = now + (
                    abs(heading_error) / COARSE_TURN_RATE
                )
                output_angular = direction * K_OMEGA
            else:
                self.turn_mode = 'fine_pulse'
                self.turn_until = now + FINE_TURN_PULSE_DURATION
                output_angular = direction * K_OMEGA

        return 0.0, output_angular

    def advance_waypoint(self):
        """Move to the next waypoint and reset turn state."""
        self.current_waypoint_idx += 1
        self.reset_turn()
        if self.current_waypoint_idx >= len(self.waypoints):
            self.finished = True
        else:
            self.phase = 'position'

    def reset_turn(self):
        self.turn_mode = 'idle'
        self.turn_direction = 0.0
        self.turn_until = 0.0
        self.turn_pause_until = 0.0
