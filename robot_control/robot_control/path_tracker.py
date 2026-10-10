#!/usr/bin/env python3
"""
Pose-to-pose path tracker: drives to each waypoint (x,y), then rotates
in place to the waypoint's yaw. Advances when both tolerances are met.

Subparts:
  2a — State estimator: yaw from IMU (via FrameManager), x/y dead-reckoned.
  2b — Two-phase controller: Pure Pursuit (drive) + P-control (align).
  2c — Inverse kinematics using calibrated k_v, k_omega.

Frames: see frames.py. {I} -> {B} -> {W}. All conversions go through FrameManager.

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

# FIX 1: import FrameManager so the node can own one.
from robot_control.frames import FrameManager


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
# Math helpers (kept for reference / non-frame uses)
# ============================================================

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
    Pure-math pose tracker. Frame-agnostic — the node is responsible for
    converting target coordinates into the body frame BEFORE calling compute().

    compute(state, x_b, y_b):
        state   : RobotState (only .theta is used for the ALIGN phase; position
                  used only for the "did we arrive?" check)
        x_b, y_b: target waypoint expressed in BODY frame
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

        self.idx = 0
        self.phase = Phase.DRIVE
        self.done = False
        self.snap_on_arrival = True

    @property
    def target(self) -> Waypoint:
        return self.wps[self.idx]

    # ------------------------------------------------------------
    # FIX 2: accept (state, x_b, y_b). One call per control loop.
    # ------------------------------------------------------------
    def compute(self, state: RobotState,
                x_b: float, y_b: float) -> Tuple[float, float]:
        """
        Returns (v_cmd, omega_cmd).

        The target (current waypoint) is provided in BODY frame as (x_b, y_b),
        already transformed by the node's FrameManager. This keeps the tracker
        purely geometric.
        """
        if self.done:
            return 0.0, 0.0

        if self.phase == Phase.DRIVE:
            return self._drive(state, x_b, y_b)
        elif self.phase == Phase.ALIGN:
            return self._align(state)
        else:
            return 0.0, 0.0

    def _drive(self, state: RobotState,
               x_b: float, y_b: float) -> Tuple[float, float]:
        wp = self.target

        # distance to target (from body-frame components)
        d = math.hypot(x_b, y_b)

        if d < self.pos_tol:
            # arrived at (x,y). Snap estimate to waypoint, transition to ALIGN.
            self.phase = Phase.ALIGN
            if self.snap_on_arrival:
                state.x = wp.x
                state.y = wp.y
            return 0.0, 0.0

        # angle from body +x axis to target (in body frame)
        heading_err = math.atan2(y_b, x_b)

        # speed scheduling
        v = self.v_max
        if d < 0.5:
            v *= 0.5
        if abs(heading_err) > math.radians(60):
            v = 0.0
        elif abs(heading_err) > math.radians(30):
            v *= 0.4

        # Pure Pursuit toward a single point (use d as effective look-ahead)
        d2 = max(d * d, 1e-6)
        kappa = 2.0 * y_b / d2
        v_ref = v if v > 1e-3 else self.v_max * 0.5
        omega = v_ref * kappa
        omega = max(-self.omega_max, min(self.omega_max, omega))

        return v, omega

    def _align(self, state: RobotState) -> Tuple[float, float]:
        wp = self.target
        err = angle_diff(wp.theta, state.theta)

        if abs(err) < self.yaw_tol:
            self.idx += 1
            if self.idx >= len(self.wps):
                self.done = True
                return 0.0, 0.0
            self.phase = Phase.DRIVE
            return 0.0, 0.0

        omega = self.align_kp * err
        omega = max(-self.omega_max, min(self.omega_max, omega))
        return 0.0, omega


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

        # FIX 3: instantiate the frame manager.
        # Set imu_to_body_alpha if the IMU is not axis-aligned with the body.
        # For yaw-only control alpha cancels, but keep the hook for the report.
        self.frames = FrameManager(imu_to_body_alpha=0.0)

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

        self.get_logger().info(
            "Path tracker started. Align the robot with +x BEFORE launch. "
            "Waiting for IMU..."
        )

    # ------------------------------------------------------------
    # FIX 4: run the IMU reading through the FrameManager
    # ------------------------------------------------------------
    def imu_cb(self, msg: Imu):
        raw_yaw_imu = quat_to_yaw(
            msg.orientation.w, msg.orientation.x,
            msg.orientation.y, msg.orientation.z,
        )

        if not self.have_imu:
            # Establish the world-frame reference on the very first sample.
            self.frames.set_imu_zero_reference(raw_yaw_imu)
            self.have_imu = True
            wp0 = self.wps[0]
            self.state.x = wp0.x
            self.state.y = wp0.y
            self.last_update_t = time.time()
            self.get_logger().info(
                f"IMU online. Raw yaw at startup = {math.degrees(raw_yaw_imu):+.2f}° "
                f"(this defines world yaw 0°). Anchored to waypoint 0 "
                f"= ({wp0.x:.2f}, {wp0.y:.2f})."
            )

        # Full chain {I} -> {B} -> {W}
        self.state.theta = self.frames.imu_yaw_to_world(raw_yaw_imu)

    def control_loop(self):
        if not self.have_imu:
            return

        now = time.time()
        if self.last_update_t is None:
            self.last_update_t = now
            return
        dt = now - self.last_update_t
        self.last_update_t = now

        # ---- Subpart 2a: dead-reckon x,y from last achievable v ----
        self.state.x += self.last_achievable_v * math.cos(self.state.theta) * dt
        self.state.y += self.last_achievable_v * math.sin(self.state.theta) * dt
        self.state.v = self.last_achievable_v

        # ---- Update the Body -> World transform with the current pose ----
        self.frames.update_world_pose(self.state.x, self.state.y, self.state.theta)

        # ---- Express the target waypoint in BODY frame ----
        wp = self.tracker.target
        x_b, y_b = self.frames.world_point_to_body(wp.x, wp.y)

        # ---- Subpart 2b: pose tracker (called ONCE per loop) ----
        v_cmd, omega_cmd = self.tracker.compute(self.state, x_b, y_b)
        self.state.omega = omega_cmd

        # ---- Subpart 2c: IK ----
        u_L, u_R = inverse_kinematics(v_cmd, omega_cmd, self.k_v, self.k_omega)

        # what v we can actually achieve (post-saturation) — used next loop
        self.last_achievable_v = self.k_v * (u_L + u_R)

        # ---- publish ----
        msg = Float32MultiArray()
        msg.data = [float(u_L), float(u_R)]
        self.cmd_pub.publish(msg)

        # ---- 1 Hz log ----
        if not hasattr(self, '_last_log') or now - self._last_log > 1.0:
            self._last_log = now
            tgt = self.tracker.target
            self.get_logger().info(
                f"wp={self.tracker.idx}/{len(self.wps)-1} "
                f"phase={self.tracker.phase.name:5s} "
                f"x={self.state.x:+.2f} y={self.state.y:+.2f} "
                f"th={math.degrees(self.state.theta):+6.1f}° | "
                f"tgt=({tgt.x:+.2f},{tgt.y:+.2f},{math.degrees(tgt.theta):+6.1f}°) "
                f"tgt_b=({x_b:+.2f},{y_b:+.2f}) "
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