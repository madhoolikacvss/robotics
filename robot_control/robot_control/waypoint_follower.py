"""Open-loop differential-drive waypoint follower core.

This module has no ROS dependencies, so its behavior can be unit-tested on a
development machine without a robot or ROS installation.

Conventions:
- Waypoint x/y are metres.
- Waypoint theta is radians, counter-clockwise from +x.
- Motor commands are normalized and clamped to [-max_command, max_command].
- wheel_speed_per_command is the calibrated wheel linear speed (m/s) for a
  command magnitude of 1.0. For example, 0.8 means command 0.5 is estimated
  to produce 0.4 m/s at that wheel.
"""

from dataclasses import dataclass
import math
import os
from typing import List


@dataclass(frozen=True)
class Waypoint:
    x: float
    y: float
    theta: float


@dataclass
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0


@dataclass(frozen=True)
class FollowerConfig:
    wheel_base: float = 0.16
    wheel_speed_per_command: float = 0.8
    max_command: float = 0.5
    max_linear_speed: float = 0.20
    max_angular_speed: float = 1.0
    distance_gain: float = 1.0
    heading_gain: float = 2.0
    position_tolerance: float = 0.05
    heading_tolerance: float = math.radians(5.0)

    def __post_init__(self):
        if self.wheel_base <= 0:
            raise ValueError("wheel_base must be positive")
        if self.wheel_speed_per_command <= 0:
            raise ValueError("wheel_speed_per_command must be positive")
        if not (0 < self.max_command <= 0.5):
            raise ValueError("max_command must be in (0, 0.5]")
        if self.max_linear_speed <= 0 or self.max_angular_speed <= 0:
            raise ValueError("maximum speeds must be positive")
        if self.position_tolerance < 0 or self.heading_tolerance < 0:
            raise ValueError("tolerances cannot be negative")


def normalize_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def load_waypoints(filepath: str) -> List[Waypoint]:
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Waypoint file not found: {filepath}")
    wps: List[Waypoint] = []
    with open(filepath, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                raise ValueError(f"Line {lineno}: expected 3 values, got {len(parts)}")
            try:
                x, y, th = (float(p) for p in parts)
            except ValueError as e:
                raise ValueError(f"Line {lineno}: bad floats: {line!r}") from e
            wps.append(Waypoint(x=x, y=y, theta=th))
    if not wps:
        raise ValueError(f"No waypoints parsed from {filepath}")
    return wps


class WaypointFollower:
    """Open-loop waypoint follower for a two-wheel differential-drive robot.

    `step(dt)` first predicts the pose change from the previously issued motor
    commands, then calculates the next normalized [left, right] command.
    This is a model-based estimate, not measured localization; drift accumulates.
    """

    def __init__(
        self,
        waypoints: List[Waypoint],
        config: FollowerConfig = FollowerConfig(),
        initial_pose: Pose2D = Pose2D(),
    ):
        if not waypoints:
            raise ValueError("waypoints must not be empty")
        self.waypoints = list(waypoints)
        self.config = config
        self.pose = Pose2D(initial_pose.x, initial_pose.y, normalize_angle(initial_pose.theta))
        self.waypoint_index = 0
        self.left_command = 0.0
        self.right_command = 0.0
        self.done = False

    @property
    def current_waypoint(self) -> Waypoint:
        return self.waypoints[min(self.waypoint_index, len(self.waypoints) - 1)]

    def _integrate_previous_command(self, dt: float) -> None:
        if dt <= 0:
            return

        # Approximate command-to-wheel-speed relationship using calibration.
        v_l = self.left_command * self.config.wheel_speed_per_command
        v_r = self.right_command * self.config.wheel_speed_per_command
        v = (v_l + v_r) / 2.0
        omega = (v_r - v_l) / self.config.wheel_base
        theta = self.pose.theta

        # Exact integration for constant v and omega during this time step.
        if abs(omega) < 1e-9:
            self.pose.x += v * math.cos(theta) * dt
            self.pose.y += v * math.sin(theta) * dt
        else:
            theta_next = theta + omega * dt
            radius = v / omega
            self.pose.x += radius * (math.sin(theta_next) - math.sin(theta))
            self.pose.y -= radius * (math.cos(theta_next) - math.cos(theta))
        self.pose.theta = normalize_angle(theta + omega * dt)

    def _wheel_commands(self, v: float, omega: float) -> tuple[float, float]:
        cfg = self.config
        v_l = v - omega * cfg.wheel_base / 2.0
        v_r = v + omega * cfg.wheel_base / 2.0

        left = v_l / cfg.wheel_speed_per_command
        right = v_r / cfg.wheel_speed_per_command

        # Preserve the left/right ratio when either command would exceed limits.
        peak = max(abs(left), abs(right))
        if peak > cfg.max_command:
            scale = cfg.max_command / peak
            left *= scale
            right *= scale
        return (
            max(-cfg.max_command, min(cfg.max_command, left)),
            max(-cfg.max_command, min(cfg.max_command, right)),
        )

    def step(self, dt: float) -> tuple[float, float]:
        """Return the next (left, right) command; call at a steady rate."""
        if dt < 0:
            raise ValueError("dt cannot be negative")

        self._integrate_previous_command(dt)

        if self.done:
            self.left_command = self.right_command = 0.0
            return 0.0, 0.0

        cfg = self.config
        wp = self.current_waypoint
        dx = wp.x - self.pose.x
        dy = wp.y - self.pose.y
        distance = math.hypot(dx, dy)

        if distance > cfg.position_tolerance:
            desired_heading = math.atan2(dy, dx)
            heading_error = normalize_angle(desired_heading - self.pose.theta)

            # Turn toward the target first when it is behind the robot.
            if abs(heading_error) > math.pi / 2.0:
                v = 0.0
            else:
                v = min(cfg.max_linear_speed, cfg.distance_gain * distance)
                # Reduce forward speed as heading error grows.
                v *= max(0.0, math.cos(heading_error))

            omega = max(
                -cfg.max_angular_speed,
                min(cfg.max_angular_speed, cfg.heading_gain * heading_error),
            )
        else:
            # At the waypoint position: rotate to its requested final heading.
            heading_error = normalize_angle(wp.theta - self.pose.theta)
            if abs(heading_error) <= cfg.heading_tolerance:
                self.waypoint_index += 1
                if self.waypoint_index >= len(self.waypoints):
                    self.done = True
                    self.left_command = self.right_command = 0.0
                    return 0.0, 0.0
                return self.step(0.0)
            v = 0.0
            omega = max(
                -cfg.max_angular_speed,
                min(cfg.max_angular_speed, cfg.heading_gain * heading_error),
            )

        self.left_command, self.right_command = self._wheel_commands(v, omega)
        return self.left_command, self.right_command
