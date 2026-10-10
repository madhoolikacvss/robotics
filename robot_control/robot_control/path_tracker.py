#!/usr/bin/env python3
"""
Path tracker node for waypoint following.
Subpart 1: Waypoint loading & state representation.
"""

import os
import math
import numpy as np
from dataclasses import dataclass
from typing import List


# ---------- Data structures ----------

@dataclass
class Waypoint:
    """Single target pose in the WORLD frame."""
    x: float        # meters
    y: float        # meters
    theta: float    # radians (desired heading at this waypoint)


@dataclass
class RobotState:
    """Robot's current pose and velocity in the WORLD frame."""
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0      # yaw, radians, +CCW
    v: float = 0.0          # linear velocity (m/s)
    omega: float = 0.0      # angular velocity (rad/s)


# ---------- Waypoint file I/O ----------

def load_waypoints(filepath: str) -> List[Waypoint]:
    """
    Parse a waypoint file with lines of the form: x, y, theta
    Skips blank lines and lines starting with '#'.
    Raises ValueError on malformed lines.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Waypoint file not found: {filepath}")

    waypoints: List[Waypoint] = []
    with open(filepath, "r") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                raise ValueError(
                    f"Line {lineno}: expected 3 comma-separated values, got {len(parts)}: {line!r}"
                )
            try:
                x, y, th = (float(p) for p in parts)
            except ValueError as e:
                raise ValueError(f"Line {lineno}: could not parse floats from {line!r}") from e
            waypoints.append(Waypoint(x=x, y=y, theta=th))

    if not waypoints:
        raise ValueError(f"No valid waypoints parsed from {filepath}")

    return waypoints


def default_waypoint_path() -> str:
    """Location of waypoints.txt shipped alongside this module."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "waypoints.txt")


# ---------- Frame transforms (the "matrix calibration" part) ----------

def world_to_body(state: RobotState, wx: float, wy: float) -> tuple[float, float]:
    """
    Transform a point from WORLD frame to the robot's BODY frame.

    Body frame convention (ROS REP-103):
        +x_body = forward
        +y_body = left
        +theta  = CCW (right-hand rule about +z)

    Rotation matrix R(theta)^T applied to (target - robot):
        [ x_b ]   [  cos(th)   sin(th) ] [ dx ]
        [ y_b ] = [ -sin(th)   cos(th) ] [ dy ]
    """
    dx = wx - state.x
    dy = wy - state.y
    c, s = math.cos(state.theta), math.sin(state.theta)
    x_b = c * dx + s * dy
    y_b = -s * dx + c * dy
    return x_b, y_b


def body_to_world_delta(state: RobotState, vx_b: float, vy_b: float) -> tuple[float, float]:
    """
    Transform a velocity/vector from BODY frame back to WORLD frame.
    Rotation matrix R(theta):
        [ vx_w ]   [ cos(th)  -sin(th) ] [ vx_b ]
        [ vy_w ] = [ sin(th)   cos(th) ] [ vy_b ]
    """
    c, s = math.cos(state.theta), math.sin(state.theta)
    vx_w = c * vx_b - s * vy_b
    vy_w = s * vx_b + c * vy_b
    return vx_w, vy_w


# ---------- Kinematic model (differential drive) ----------

def integrate_kinematics(state: RobotState, v: float, omega: float, dt: float) -> RobotState:
    """
    Forward-integrate a differential-drive kinematic model one step.

    Exact integration for constant (v, omega) over dt:
        x'     = x + (v/omega) * ( sin(th + omega*dt) - sin(th) )     if omega != 0
        y'     = y - (v/omega) * ( cos(th + omega*dt) - cos(th) )
        th'    = th + omega * dt

    Falls back to straight-line integration when omega ~ 0.
    """
    new = RobotState(x=state.x, y=state.y, theta=state.theta, v=v, omega=omega)

    if abs(omega) < 1e-6:
        new.x = state.x + v * math.cos(state.theta) * dt
        new.y = state.y + v * math.sin(state.theta) * dt
        new.theta = state.theta
    else:
        new.x = state.x + (v / omega) * (math.sin(state.theta + omega * dt) - math.sin(state.theta))
        new.y = state.y - (v / omega) * (math.cos(state.theta + omega * dt) - math.cos(state.theta))
        new.theta = state.theta + omega * dt

    # wrap theta to (-pi, pi]
    new.theta = math.atan2(math.sin(new.theta), math.cos(new.theta))
    return new


# ---------- Sanity check (run as script) ----------
if __name__ == "__main__":
    wps = load_waypoints(default_waypoint_path())
    print(f"Loaded {len(wps)} waypoints:")
    for i, wp in enumerate(wps):
        print(f"  [{i}] x={wp.x:+.3f}  y={wp.y:+.3f}  theta={wp.theta:+.3f}  ({math.degrees(wp.theta):+.1f} deg)")

    # Verify the transform round-trips and that a target straight ahead maps to +x
    st = RobotState(x=1.0, y=0.0, theta=math.pi / 2)   # at (1,0), facing +y
    xb, yb = world_to_body(st, wx=1.0, wy=1.0)         # target (1,1) is one unit forward
    print(f"\nSanity: robot@(1,0,90deg) -> target(1,1) in body frame = ({xb:+.3f}, {yb:+.3f})   (expect ~(+1, 0))")

    # Kinematic integration test: drive forward 0.5 m
    st2 = RobotState(x=0.0, y=0.0, theta=0.0)
    st2 = integrate_kinematics(st2, v=1.0, omega=0.0, dt=0.5)
    print(f"Sanity: forward 1 m/s * 0.5 s from origin -> x={st2.x:.3f}, y={st2.y:.3f}   (expect 0.5, 0)")