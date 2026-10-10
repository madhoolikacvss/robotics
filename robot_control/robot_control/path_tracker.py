#!/usr/bin/env python3
"""
Pose-to-pose path tracker: drives to each waypoint (x,y), then rotates
in place to the waypoint's yaw. Advances when both tolerances are met.

Subparts:
  2a — State estimator: yaw from IMU, x/y dead-reckoned from achievable v.
  2b — Two-phase controller: Pure Pursuit (drive) + P-control (align).
  2c — Inverse kinematics using calibrated k_v, k_omega.

Run:
    ros2 run robot_control path_tracker
"""

import os
import math
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Imu


# ============================================================
# Data types
# ============================================================

@dataclass
class Waypoint:
    x: float
    y: float
    theta: float


@dataclass
class RobotState:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    v: float = 0.0
    omega: float = 0.0


class Phase(Enum):
    DRIVE = 1
    ALIGN = 2
    DONE = 3


# ============================================================
# File I/O
# ============================================================

def load_waypoints(filepath: str) -> List[Waypoint]:
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Waypoint file not found: {filepath}")
    wps: List[Waypoint] = []
    with open(filepath) as f:
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


def default_waypoint_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "waypoints.txt")


def default_calibration_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")


def load_calibration(filepath: Optional[str] = None) -> dict:
    path = filepath or default_calibration_path()
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No calibration file at {path}. Run calibration first.")
    with open(path) as f:
        cal = json.load(f)
    for key in ("k_v", "k_omega"):
        if key not in cal:
            raise ValueError(f"Calibration file missing '{key}'.")
    return cal


# ============================================================
# Math helpers
# ============================================================

def world_to_body(state: RobotState, wx: float, wy: float) -> Tuple[float, float]:
    dx = wx - state.x
    dy = wy - state.y
    c, s = math.cos(state.theta), math.sin(state.theta)
    return c * dx + s * dy, -s * dx + c * dy


def quat_to_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def angle_diff(a: float, b: float) -> float:
    """Shortest signed a - b in (-pi, pi]."""
    d = a - b
    return math.atan2(math.sin(d), math.cos(d))


# ============================================================
# Subpart 2c — Inverse kinematics
# ============================================================

def inverse_kinematics(v: float, omega: float,
                       k_v: float, k_omega: float) -> Tuple[float, float]:
    u_sum = v / k_v
    u_diff = omega / k_omega
    u_L = (u_sum - u_diff) / 2.0
    u_R = (u_sum + u_diff) / 2.0
    u_L = max(-0.5, min(0.5, u_L))
    u_R = max(-0.5, min(0.5, u_R))
    return u_L, u_R


# ============================================================
# Two-phase pose-to-pose tracker (Subpart 2b)
# ============================================================

class PoseTracker:
    """
    Drives to each waypoint's (x,y) using Pure Pursuit-like pursuit of the
    current waypoint only, then rotates in place to match the waypoint's yaw.
    Advances when both position and yaw tolerances are met.
    """

    def __init__(self,
                 waypoints: List[Waypoint],
                 pos_tol: float = 0.12,
                 yaw_tol: float = 0.12,
                 v_max: float = 0.25,
                 omega_max: float = 0.9,
                 align_kp: float = 1.5):
        self.wps = waypoints
        self.pos_tol = pos_tol
        self.yaw_tol = yaw_tol
        self.v_max = v_max
        self.omega_max = omega_max
        self.align_kp = align_kp

        self.idx = 0                 # current target waypoint index
        self.phase = Phase.DRIVE
        self.done = False

        # once a waypoint is reached, we "snap" our belief of (x, y) to it
        # so per-segment drift doesn't accumulate indefinitely
        self.snap_on_arrival = True

    # -------- accessor --------
    @property
    def target(self) -> Waypoint:
        return self.wps[self.idx]

    # -------- main entry point --------
    def compute(self, state: RobotState) -> Tuple[float, float]:
        """Returns (v_cmd, omega_cmd). Modifies `state` on snap. Sets done/phase."""
        if self.done:
            return 0.0, 0.0

        if self.phase == Phase.DRIVE:
            return self._drive(state)
        elif self.phase == Phase.ALIGN:
            return self._align(state)
        else:
            return 0.0, 0.0

    # -------- DRIVE phase --------
    def _drive(self, state: RobotState) -> Tuple[float, float]:
        wp = self.target

        dx = wp.x - state.x
        dy = wp.y - state.y
        d = math.hypot(dx, dy)

        if d < self.pos_tol:
            # arrived at (x,y). Transition to ALIGN for this waypoint's theta.
            self.phase = Phase.ALIGN
            state.x = wp.x   # snap estimate
            state.y = wp.y
            return 0.0, 0.0

        # target in body frame (THE matrix from subpart 1)
        x_b, y_b = world_to_body(state, wp.x, wp.y)

        # heading error = angle to target
        heading_err = math.atan2(y_b, x_b)

        # slow down when close or when badly misaligned
        v = self.v_max
        if d < 0.5:
            v *= 0.5
        if abs(heading_err) > math.radians(60):
            v = 0.0                    # rotate toward target if way off
        elif abs(heading_err) > math.radians(30):
            v *= 0.4

        # curvature toward target. Pure pursuit toward a single point:
        #   kappa = 2*y_b / d^2  (using d as look-ahead)
        d2 = max(d * d, 1e-6)
        kappa = 2.0 * y_b / d2
        v_ref = v if v > 1e-3 else self.v_max * 0.5
        omega = v_ref * kappa
        omega = max(-self.omega_max, min(self.omega_max, omega))

        return v, omega

    # -------- ALIGN phase --------
    def _align(self, state: RobotState) -> Tuple[float, float]:
        wp = self.target
        err = angle_diff(wp.theta, state.theta)

        if abs(err) < self.yaw_tol:
            # waypoint fully achieved. Advance.
            self.idx += 1
            if self.idx >= len(self.wps):
                self.done = True
                return 0.0, 0.0
            self.phase = Phase.DRIVE
            return 0.0, 0.0

        # P-control on yaw. Sign check: err > 0 means target is CCW of us
        # -> we want +omega (CCW spin) -> no sign flip.
        omega = self.align_kp * err
        omega = max(-self.omega_max, min(self.omega_max, omega))
        return 0.0, omega     # v = 0 during alignment


# ============================================================
# ROS2 node
# ============================================================

class PathTrackerNode(Node):
    def __init__(self):
        super().__init__('path_tracker_node')

        self.wps = load_waypoints(default_waypoint_path())
        cal = load_calibration()
        self.k_v = float(cal['k_v'])
        self.k_omega = float(cal['k_omega'])
        self.get_logger().info(
            f"Loaded {len(self.wps)} waypoints; "
            f"k_v={self.k_v:.4f}, k_omega={self.k_omega:.4f}"
        )

        self.state = RobotState()
        self.have_imu = False
        self.last_update_t: Optional[float] = None
        self.last_achievable_v = 0.0

        self.tracker = PoseTracker(
            waypoints=self.wps,
            pos_tol=0.12,
            yaw_tol=0.12,
            v_max=0.25,
            omega_max=0.9,
            align_kp=1.5,
        )

        self.cmd_pub = self.create_publisher(Float32MultiArray, 'motor_commands', 10)
        self.imu_sub = self.create_subscription(Imu, 'imu_data', self.imu_cb, 10)
        self.timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info("Path tracker started. Waiting for IMU...")

    def imu_cb(self, msg: Imu):
        self.state.theta = quat_to_yaw(
            msg.orientation.w, msg.orientation.x,
            msg.orientation.y, msg.orientation.z,
        )
        if not self.have_imu:
            self.have_imu = True
            wp0 = self.wps[0]
            self.state.x = wp0.x
            self.state.y = wp0.y
            self.last_update_t = time.time()
            self.get_logger().info(
                f"IMU online. Anchored to waypoint 0 = "
                f"({wp0.x:.2f}, {wp0.y:.2f}), yaw = {math.degrees(self.state.theta):+.2f}°"
            )

    def control_loop(self):
        if not self.have_imu:
            return

        now = time.time()
        if self.last_update_t is None:
            self.last_update_t = now
            return
        dt = now - self.last_update_t
        self.last_update_t = now

        # Subpart 2a — dead-reckon from last achievable v
        self.state.x += self.last_achievable_v * math.cos(self.state.theta) * dt
        self.state.y += self.last_achievable_v * math.sin(self.state.theta) * dt
        self.state.v = self.last_achievable_v

        # Subpart 2b — pose tracker
        v_cmd, omega_cmd = self.tracker.compute(self.state)
        self.state.omega = omega_cmd

        # Subpart 2c — IK
        u_L, u_R = inverse_kinematics(v_cmd, omega_cmd, self.k_v, self.k_omega)
        self.last_achievable_v = self.k_v * (u_L + u_R)

        msg = Float32MultiArray()
        msg.data = [float(u_L), float(u_R)]
        self.cmd_pub.publish(msg)

        # 1 Hz log
        if not hasattr(self, '_last_log') or now - self._last_log > 1.0:
            self._last_log = now
            tgt = self.tracker.target
            self.get_logger().info(
                f"wp={self.tracker.idx}/{len(self.wps)-1} "
                f"phase={self.tracker.phase.name:5s} "
                f"x={self.state.x:+.2f} y={self.state.y:+.2f} "
                f"th={math.degrees(self.state.theta):+6.1f}° | "
                f"tgt=({tgt.x:+.2f},{tgt.y:+.2f},{math.degrees(tgt.theta):+6.1f}°) "
                f"v={v_cmd:.2f} w={omega_cmd:+.2f} "
                f"u=({u_L:+.2f},{u_R:+.2f})"
            )

        if self.tracker.done:
            self.get_logger().info("All waypoints achieved. Stopping.")
            for _ in range(5):
                stop = Float32MultiArray()
                stop.data = [0.0, 0.0]
                self.cmd_pub.publish(stop)
            self.timer.cancel()


# ============================================================
# Entry point
# ============================================================

def main(args=None):
    rclpy.init(args=args)
    node = PathTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            stop = Float32MultiArray()
            stop.data = [0.0, 0.0]
            node.cmd_pub.publish(stop)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()