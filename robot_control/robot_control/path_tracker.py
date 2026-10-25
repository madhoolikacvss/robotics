#!/usr/bin/env python3
"""
Open-loop, timed waypoint tracker.

Assumptions:
- waypoints.txt contains x,y,theta rows; x/y are metres and theta is radians.
- calibration.json contains k_v and k_omega.
- k_v: linear speed when both wheel commands are +0.5 (m/s).
- k_omega: angular speed when wheel commands are (-0.5, +0.5) (rad/s).
- The robot starts at waypoint 0, facing world +x (theta = 0).
- No IMU or localization is used. Position and heading are estimated from time.

This is an open-loop controller: wheel slip, acceleration, battery level, and
motor delays will cause accumulated error. Test at low speed / with wheels lifted
or in a clear area before running a full path.
"""

import json
import math
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


@dataclass
class Waypoint:
    x: float
    y: float
    theta: float


def default_path(filename: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


def load_waypoints(filepath: str) -> List[Waypoint]:
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Waypoint file not found: {filepath}")

    waypoints = []
    with open(filepath) as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 3:
                raise ValueError(f"Line {line_no}: expected x,y,theta")
            try:
                x, y, theta = map(float, parts)
            except ValueError as exc:
                raise ValueError(f"Line {line_no}: invalid numbers: {line!r}") from exc
            waypoints.append(Waypoint(x, y, theta))

    if not waypoints:
        raise ValueError(f"No waypoints found in {filepath}")
    return waypoints


def load_calibration(filepath: str) -> tuple[float, float]:
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"No calibration file at {filepath}")
    with open(filepath) as f:
        calibration = json.load(f)

    try:
        k_v = float(calibration["k_v"])
        k_omega_right = float(calibration["k_omega_right"])
        k_omega_left = float(calibration["k_omega_left"])

    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("calibration.json must contain numeric k_v and k_omega") from exc

    if k_v <= 0 or k_omega_right <= 0 or k_omega_left <= 0:
        raise ValueError("k_v and k_omega must both be positive")
    return k_v, k_omega_right, k_omega_left


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class TimedPathTracker(Node):
    WHEEL_COMMAND = 0.5
    CONTROL_PERIOD = 0.02  # seconds; used to check when each timed action ends

    def __init__(self):
        super().__init__("path_tracker_node")

        self.waypoints = load_waypoints(default_path("waypoints.txt"))
        self.k_v, self.k_omega_right, self.k_omega_left = load_calibration(default_path("calibration.json"))

        # Open-loop pose estimate. The robot is assumed to start at waypoint 0
        # and facing +x. The first waypoint's theta is treated as its desired
        # initial heading; subsequent waypoints are reached in list order.
        self.x = self.waypoints[0].x
        self.y = self.waypoints[0].y
        self.theta = 0.0

        self.target_index = 0
        self.stage = "TURN_TO_WAYPOINT_HEADING"
        self.action_end_time: Optional[float] = None
        self.action_start_time: Optional[float] = None
        self.action_duration = 0.0
        self.action_v = 0.0
        self.action_omega = 0.0
        self.done = False

        self.cmd_pub = self.create_publisher(
            Float32MultiArray, "motor_commands", 10
        )
        self.timer = self.create_timer(self.CONTROL_PERIOD, self.control_loop)

        self.get_logger().info(
            f"Loaded {len(self.waypoints)} waypoints; "
            f"k_v={self.k_v:.4f}, k_omega_right={self.k_omega_right:.4f}, k_omega_left={self.k_omega_left:.4f}. "
            "Open-loop mode: no IMU."
        )
        self.get_logger().warn(
            "Assuming initial heading is world +x. Open-loop errors accumulate."
        )

    def publish_wheels(self, left: float, right: float) -> None:
        msg = Float32MultiArray()
        msg.data = [float(left), float(right)]
        self.cmd_pub.publish(msg)

    def stop(self) -> None:
        self.publish_wheels(0.0, 0.0)

    def begin_action(self, v: float, omega: float, duration: float, stage: str) -> None:
        """Start an action and hold its fixed wheel commands for duration seconds."""
        self.stage = stage
        self.action_v = v
        self.action_omega = omega
        self.action_duration = max(0.0, duration)
        self.action_start_time = time.monotonic()
        self.action_end_time = self.action_start_time + self.action_duration

        # Update the open-loop pose estimate using the same calibrated model.
        # For turning actions v=0; for driving actions omega=0.
        if self.action_duration > 0:
            if abs(omega) > 1e-9:
                self.theta = wrap_angle(self.theta + omega * self.action_duration)
            elif abs(v) > 1e-9:
                self.x += v * math.cos(self.theta) * self.action_duration
                self.y += v * math.sin(self.theta) * self.action_duration

        left, right = self.wheel_commands(v, omega)
        self.publish_wheels(left, right)
        self.get_logger().info(
            f"{stage}: u=({left:+.2f}, {right:+.2f}) "
            f"for {self.action_duration:.2f}s; "
            f"estimated pose=({self.x:.2f}, {self.y:.2f}, "
            f"{math.degrees(self.theta):.1f} deg)"
        )

    def wheel_commands(self, v: float, omega: float) -> tuple[float, float]:
        # k_v = v / (u_L + u_R); k_omega = omega / (u_R - u_L)
        u_sum = v / self.k_v
        if omega < 0:
            u_diff = omega / self.k_omega_right
        else:
            u_diff = omega / self.k_omega_left
        left = (u_sum - u_diff) / 2.0
        right = (u_sum + u_diff) / 2.0

        # This controller only requests straight or in-place motions, so
        # each wheel should be -0.5, 0, or +0.5 (allowing numerical tolerance).
        return (
            max(-self.WHEEL_COMMAND, min(self.WHEEL_COMMAND, left)),
            max(-self.WHEEL_COMMAND, min(self.WHEEL_COMMAND, right)),
        )

    def control_loop(self) -> None:
        if self.done:
            return

        now = time.monotonic()
        if self.action_end_time is not None:
            if now < self.action_end_time:
                # Re-publish so the motor_control node (that expects periodic commands)
                # continues receiving the fixed command.
                left, right = self.wheel_commands(self.action_v, self.action_omega)
                self.publish_wheels(left, right)
                return

            self.stop()
            self.action_end_time = None
            self.action_start_time = None
            self.advance_stage()
            return

        self.plan_next_action()

    def advance_stage(self) -> None:
        if self.stage == "TURN_TO_WAYPOINT_HEADING":
            self.stage = "AT_WAYPOINT_HEADING"
        elif self.stage == "DRIVE_TO_WAYPOINT":
            self.stage = "TURN_TO_FINAL_HEADING"
        elif self.stage == "TURN_TO_FINAL_HEADING":
            self.target_index += 1
            if self.target_index >= len(self.waypoints):
                self.done = True
                self.stop()
                self.get_logger().info("Finished all waypoints.")
                self.timer.cancel()
            else:
                self.stage = "TURN_TO_NEXT_WAYPOINT"
        elif self.stage == "TURN_TO_NEXT_WAYPOINT":
            self.stage = "DRIVE_TO_WAYPOINT"

    def plan_next_action(self) -> None:
        wp = self.waypoints[self.target_index]

        if self.stage == "TURN_TO_WAYPOINT_HEADING":
            # At waypoint 0: rotate to its requested heading.
            target_heading = wp.theta
            delta = wrap_angle(target_heading - self.theta)
            self.start_turn(delta, "TURN_TO_WAYPOINT_HEADING")
            return

        if self.stage == "AT_WAYPOINT_HEADING":
            # Waypoint 0 is the starting location; now process the next point.
            self.target_index += 1
            if self.target_index >= len(self.waypoints):
                self.done = True
                self.stop()
                self.get_logger().info("Finished all waypoints.")
                self.timer.cancel()
                return
            self.stage = "TURN_TO_NEXT_WAYPOINT"
            return

        if self.stage == "TURN_TO_NEXT_WAYPOINT":
            wp = self.waypoints[self.target_index]
            dx = wp.x - self.x
            dy = wp.y - self.y
            target_heading = math.atan2(dy, dx)
            delta = wrap_angle(target_heading - self.theta)
            self.start_turn(delta, "TURN_TO_NEXT_WAYPOINT")
            return

        if self.stage == "DRIVE_TO_WAYPOINT":
            wp = self.waypoints[self.target_index]
            distance = math.hypot(wp.x - self.x, wp.y - self.y)
            # Both wheels +0.5 gives predicted v = k_v.
            duration = distance / self.k_v
            self.begin_action(
                v=self.k_v, omega=0.0, duration=duration, stage="DRIVE_TO_WAYPOINT"
            )
            return

        if self.stage == "TURN_TO_FINAL_HEADING":
            wp = self.waypoints[self.target_index]
            delta = wrap_angle(wp.theta - self.theta)
            self.start_turn(delta, "TURN_TO_FINAL_HEADING")
            return

    def start_turn(self, delta: float, stage: str) -> None:
        # Opposite wheel commands +/-0.5 give predicted |omega| = k_omega.
        if delta >= 0:
            duration = abs(delta) / self.k_omega_left
            omega = self.k_omega_left
        else:
            duration = abs(delta) / self.k_omega_right
            omega = -self.k_omega_right
        self.begin_action(v=0.0, omega=omega, duration=duration, stage=stage)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = TimedPathTracker()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.stop()
            except Exception:
                pass
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
