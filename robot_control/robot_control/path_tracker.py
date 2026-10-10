#!/usr/bin/env python3
"""
Path tracker for waypoint following — Pure Pursuit + Differential-Drive IK.

Subparts combined:
  2a — State estimator: yaw from IMU (absolute), x/y dead-reckoned from
       achievable commanded velocity.
  2b — Pure Pursuit controller over the waypoint polyline.
  2c — Inverse kinematic matrix using calibrated k_v, k_omega.

Run:
    ros2 run robot_control path_tracker

Requires:
    motor_control.py running (provides /imu_data, consumes /motor_commands)
    waypoints.txt and calibration.json alongside this file.
"""

import os
import math
import json
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Imu


# ============================================================
# Data structures
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
    theta: float = 0.0      # radians, +CCW
    v: float = 0.0          # last achievable linear velocity (m/s)
    omega: float = 0.0      # last commanded angular velocity (rad/s)


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
        raise FileNotFoundError(
            f"No calibration file at {path}. Run the calibrate node first."
        )
    with open(path) as f:
        cal = json.load(f)
    for key in ("k_v", "k_omega"):
        if key not in cal:
            raise ValueError(f"Calibration file missing '{key}'.")
    return cal


# ============================================================
# Frame transforms (the "matrix calibration" from Subpart 1)
# ============================================================

def world_to_body(state: RobotState, wx: float, wy: float) -> Tuple[float, float]:
    """World -> Body. R(theta)^T applied to (target - robot)."""
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


# ============================================================
# Subpart 2c — Inverse kinematics (v, omega) -> (u_L, u_R)
# ============================================================

def inverse_kinematics(v: float, omega: float,
                       k_v: float, k_omega: float) -> Tuple[float, float]:
    """
    Invert:  v = k_v (u_L + u_R),  omega = k_omega (u_R - u_L)
    Returns clipped (u_L, u_R) in [-0.5, 0.5].
    """
    u_sum = v / k_v
    u_diff = omega / k_omega
    u_L = (u_sum - u_diff) / 2.0
    u_R = (u_sum + u_diff) / 2.0
    # motor saturation
    u_L = max(-0.5, min(0.5, u_L))
    u_R = max(-0.5, min(0.5, u_R))
    return u_L, u_R


# ============================================================
# Subpart 2b — Pure Pursuit controller
# ============================================================

class PurePursuit:
    """
    Pure Pursuit over a polyline. Tracks the current segment index and
    always chases a point L_d meters ahead of the robot's projection.
    """

    def __init__(self,
                 waypoints: List[Waypoint],
                 lookahead: float = 0.4,
                 goal_tol: float = 0.15,
                 v_max: float = 0.25,
                 omega_max: float = 0.8):
        self.wps = waypoints
        self.L_d = lookahead
        self.goal_tol = goal_tol
        self.v_max = v_max
        self.omega_max = omega_max

        self.seg_idx = 0
        self.done = False

    def _seg(self, i: int) -> Tuple[Waypoint, Waypoint]:
        return self.wps[i], self.wps[i + 1]

    def _project_on_segment(self, px, py, i):
        a, b = self._seg(i)
        dx, dy = b.x - a.x, b.y - a.y
        L2 = dx * dx + dy * dy
        if L2 < 1e-9:
            return a.x, a.y, 0.0
        t = ((px - a.x) * dx + (py - a.y) * dy) / L2
        t = max(0.0, min(1.0, t))
        return a.x + t * dx, a.y + t * dy, t

    def _advance_segment_if_needed(self, state: RobotState):
        while self.seg_idx < len(self.wps) - 1:
            _, b = self._seg(self.seg_idx)
            if math.hypot(b.x - state.x, b.y - state.y) < self.goal_tol:
                self.seg_idx += 1
            else:
                break

    def _lookahead_point(self, state: RobotState) -> Optional[Tuple[float, float]]:
        remaining = self.L_d
        i = self.seg_idx
        while i < len(self.wps) - 1:
            a, b = self._seg(i)
            if i == self.seg_idx:
                proj_x, proj_y, _ = self._project_on_segment(state.x, state.y, i)
            else:
                proj_x, proj_y = a.x, a.y
            seg_len = math.hypot(b.x - proj_x, b.y - proj_y)
            if seg_len >= remaining:
                if seg_len < 1e-9:
                    return b.x, b.y
                ux = (b.x - proj_x) / seg_len
                uy = (b.y - proj_y) / seg_len
                return proj_x + ux * remaining, proj_y + uy * remaining
            remaining -= seg_len
            i += 1
        last = self.wps[-1]
        return last.x, last.y

    def compute(self, state: RobotState) -> Tuple[float, float]:
        """Returns (v, omega). Sets self.done=True on arrival at goal."""
        if self.done:
            return 0.0, 0.0

        goal = self.wps[-1]
        d_goal = math.hypot(goal.x - state.x, goal.y - state.y)
        if d_goal < self.goal_tol:
            self.done = True
            return 0.0, 0.0

        self._advance_segment_if_needed(state)
        target = self._lookahead_point(state)
        if target is None:
            self.done = True
            return 0.0, 0.0

        # Subpart 1's calibration matrix in action:
        x_b, y_b = world_to_body(state, target[0], target[1])

        # choose forward speed based on alignment
        heading_err = math.atan2(y_b, x_b) if x_b > 0 else math.pi
        v = self.v_max
        if x_b <= 0.0:
            v = 0.0                                     # target behind -> rotate in place
        elif abs(heading_err) > math.radians(45):
            v = 0.0                                     # turn first, drive later
        elif abs(heading_err) > math.radians(20):
            v = 0.4 * self.v_max
        if d_goal < 0.5:
            v *= 0.5                                    # ease off near goal

        # curvature and yaw rate
        L2 = max(self.L_d * self.L_d, 1e-6)
        kappa = 2.0 * y_b / L2
        v_ref = v if v > 1e-3 else self.v_max * 0.5     # for in-place turns
        omega = v_ref * kappa
        omega = max(-self.omega_max, min(self.omega_max, omega))

        return v, omega


# ============================================================
# The ROS2 node — Subparts 2a + 2b + 2c together
# ============================================================

class PathTrackerNode(Node):
    def __init__(self):
        super().__init__('path_tracker_node')

        # ---- load config ----
        self.wps = load_waypoints(default_waypoint_path())
        cal = load_calibration()
        self.k_v = float(cal['k_v'])
        self.k_omega = float(cal['k_omega'])
        self.get_logger().info(
            f"Loaded {len(self.wps)} waypoints; "
            f"k_v={self.k_v:.4f} m/s/unit, k_omega={self.k_omega:.4f} rad/s/unit"
        )

        # ---- state ----
        self.state = RobotState()
        self.have_imu = False
        self.last_update_t: Optional[float] = None
        self.last_achievable_v = 0.0

        # ---- controller ----
        self.controller = PurePursuit(
            waypoints=self.wps,
            lookahead=0.4,          # tuned for ~2 m course
            goal_tol=0.15,
            v_max=0.25,             # ~37% of max achievable 0.683; safe headroom
            omega_max=0.8,
        )

        # ---- ROS I/O ----
        self.cmd_pub = self.create_publisher(Float32MultiArray, 'motor_commands', 10)
        self.imu_sub = self.create_subscription(Imu, 'imu_data', self.imu_cb, 10)

        # ---- 20 Hz control loop ----
        self.timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info("Path tracker started. Waiting for IMU...")

    # ------------------------------------------------------------
    # Subpart 2a — sensor callback
    # ------------------------------------------------------------
    def imu_cb(self, msg: Imu):
        """Update absolute yaw from IMU. First message also anchors the state to waypoint 0."""
        self.state.theta = quat_to_yaw(
            msg.orientation.w,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
        )
        if not self.have_imu:
            self.have_imu = True
            wp0 = self.wps[0]
            self.state.x = wp0.x
            self.state.y = wp0.y
            self.last_update_t = time.time()
            self.get_logger().info(
                f"First IMU. Anchoring to waypoint 0 = "
                f"({wp0.x:.2f}, {wp0.y:.2f}), yaw = {math.degrees(self.state.theta):+.2f}°"
            )

    # ------------------------------------------------------------
    # Control loop — runs at 20 Hz
    # ------------------------------------------------------------
    def control_loop(self):
        if not self.have_imu:
            return

        now = time.time()
        if self.last_update_t is None:
            self.last_update_t = now
            return
        dt = now - self.last_update_t
        self.last_update_t = now

        # --- Subpart 2a: dead-reckon x,y from last achievable v ---
        self.state.x += self.last_achievable_v * math.cos(self.state.theta) * dt
        self.state.y += self.last_achievable_v * math.sin(self.state.theta) * dt
        self.state.v = self.last_achievable_v

        # --- Subpart 2b: Pure Pursuit -> (v, omega) ---
        v_cmd, omega_cmd = self.controller.compute(self.state)
        self.state.omega = omega_cmd

        # --- Subpart 2c: IK -> (u_L, u_R), respecting saturation ---
        u_L, u_R = inverse_kinematics(v_cmd, omega_cmd, self.k_v, self.k_omega)

        # remember what we actually asked the wheels to do (post-clip)
        self.last_achievable_v = self.k_v * (u_L + u_R)

        # --- publish ---
        msg = Float32MultiArray()
        msg.data = [float(u_L), float(u_R)]
        self.cmd_pub.publish(msg)

        # --- 1 Hz progress log ---
        if not hasattr(self, '_last_log') or now - self._last_log > 1.0:
            self._last_log = now
            self.get_logger().info(
                f"x={self.state.x:+.2f} y={self.state.y:+.2f} "
                f"th={math.degrees(self.state.theta):+6.1f}° "
                f"seg={self.controller.seg_idx} "
                f"v_cmd={v_cmd:.2f} v_ach={self.last_achievable_v:.2f} "
                f"w_cmd={omega_cmd:+.2f} "
                f"u=({u_L:+.2f},{u_R:+.2f})"
            )

        # --- arrival ---
        if self.controller.done:
            self.get_logger().info("Goal reached. Stopping.")
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