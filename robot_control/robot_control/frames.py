#!/usr/bin/env python3
"""
Coordinate frame transforms for the path tracker.

Frames used:
  {I}  IMU frame       - wherever the chip's axes point; fixed to the robot
  {B}  Body frame      - +x forward, +y left, +z up (ROS REP-103)
  {W}  World frame     - +x east of the course, defined at robot launch

All transforms are represented as 3x3 homogeneous matrices so they compose
cleanly:  T_{A->C} = T_{B->C} @ T_{A->B}

2D poses are represented as (x, y, theta). Homogeneous point vectors are
[x, y, 1]^T. Rotations-only transforms have translation entries of zero.
"""

import math
import numpy as np
from typing import Tuple


# ============================================================
# Basic building blocks
# ============================================================

def rot2d(theta: float) -> np.ndarray:
    """2D rotation matrix R(theta)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])


def se2(theta: float, tx: float = 0.0, ty: float = 0.0) -> np.ndarray:
    """
    Homogeneous SE(2) transform:
        [ cos(t)  -sin(t)  tx ]
        [ sin(t)   cos(t)  ty ]
        [   0        0      1 ]
    """
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, tx],
                     [s,  c, ty],
                     [0,  0,  1]])


def apply_se2(T: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    """Apply a 3x3 homogeneous transform to a 2D point."""
    v = T @ np.array([x, y, 1.0])
    return float(v[0]), float(v[1])


def invert_se2(T: np.ndarray) -> np.ndarray:
    """Analytic inverse of an SE(2) transform (faster and exact)."""
    R = T[:2, :2]
    t = T[:2, 2]
    R_inv = R.T
    t_inv = -R_inv @ t
    out = np.eye(3)
    out[:2, :2] = R_inv
    out[:2, 2] = t_inv
    return out


# ============================================================
# Transform 1: IMU -> Body
# ============================================================

class ImuToBody:
    """
    Constant transform from IMU frame to Body frame.

    Parameters
    ----------
    alpha : float
        Fixed yaw (radians) of the IMU frame relative to the body frame.
        Depends on the chip's mounting. Set to 0 if the IMU axes are
        aligned with the robot's forward/left axes (typical).
    tx, ty : float
        Fixed translation from IMU to body origin, meters. Usually small
        compared to our tolerances; set to 0 unless you know them.
    """

    def __init__(self, alpha: float = 0.0, tx: float = 0.0, ty: float = 0.0):
        self.alpha = alpha
        self.tx = tx
        self.ty = ty
        # T_{I->B}: rotates IMU-frame quantities into body-frame.
        self.T = se2(alpha, tx, ty)
        self.T_inv = invert_se2(self.T)

    def yaw_imu_to_body(self, yaw_imu: float) -> float:
        """Convert a yaw angle from {I} to {B} (adds alpha)."""
        return yaw_imu + self.alpha

    def yaw_body_to_imu(self, yaw_body: float) -> float:
        """Convert a yaw angle from {B} to {I} (subtracts alpha)."""
        return yaw_body - self.alpha

    def vec_imu_to_body(self, vx_imu: float, vy_imu: float) -> Tuple[float, float]:
        """Convert a 2D vector from {I} to {B}."""
        return apply_se2(self.T, vx_imu, vy_imu)

    def vec_body_to_imu(self, vx_body: float, vy_body: float) -> Tuple[float, float]:
        """Convert a 2D vector from {B} to {I}."""
        return apply_se2(self.T_inv, vx_body, vy_body)


# ============================================================
# Transform 2: Body -> World
# ============================================================

class BodyToWorld:
    """
    Time-varying transform from Body frame to World frame.

    Parameters
    ----------
    xw, yw, theta_w : float
        Robot's current pose in the world frame. Must be updated
        whenever the robot's position or heading changes.
    """

    def __init__(self, xw: float = 0.0, yw: float = 0.0, theta_w: float = 0.0):
        self.xw = xw
        self.yw = yw
        self.theta_w = theta_w
        self.T = se2(theta_w, xw, yw)
        self.T_inv = invert_se2(self.T)

    def update(self, xw: float, yw: float, theta_w: float) -> None:
        self.xw, self.yw, self.theta_w = xw, yw, theta_w
        self.T = se2(theta_w, xw, yw)
        self.T_inv = invert_se2(self.T)

    def point_body_to_world(self, xb: float, yb: float) -> Tuple[float, float]:
        """Point in {B} -> point in {W}."""
        return apply_se2(self.T, xb, yb)

    def point_world_to_body(self, xw: float, yw: float) -> Tuple[float, float]:
        """Point in {W} -> point in {B}."""
        return apply_se2(self.T_inv, xw, yw)

    def vec_body_to_world(self, vx_b: float, vy_b: float) -> Tuple[float, float]:
        """Vector in {B} -> vector in {W} (rotation only)."""
        R = self.T[:2, :2]
        v = R @ np.array([vx_b, vy_b])
        return float(v[0]), float(v[1])

    def vec_world_to_body(self, vx_w: float, vy_w: float) -> Tuple[float, float]:
        """Vector in {W} -> vector in {B} (rotation only)."""
        R = self.T_inv[:2, :2]
        v = R @ np.array([vx_w, vy_w])
        return float(v[0]), float(v[1])


# ============================================================
# Frame manager: composes everything
# ============================================================

class FrameManager:
    """
    Owns both transforms and exposes the composed chain.

    Usage:
        fm = FrameManager(imu_to_body_alpha=0.0)
        fm.update_world_pose(x, y, theta)   # whenever state changes
        theta_w = fm.imu_yaw_to_world(raw_imu_yaw, imu_zero_ref)
        x_b, y_b = fm.world_point_to_body(wx, wy)
    """

    def __init__(self, imu_to_body_alpha: float = 0.0,
                 imu_to_body_tx: float = 0.0,
                 imu_to_body_ty: float = 0.0):
        self.I_to_B = ImuToBody(imu_to_body_alpha, imu_to_body_tx, imu_to_body_ty)
        self.B_to_W = BodyToWorld()
        self._imu_zero_ref = None   # captured at startup

    # ----- IMU -> World (yaw-only, the practical case) -----
    def set_imu_zero_reference(self, raw_yaw_at_startup: float) -> None:
        """
        Establish the world-frame heading reference.

        We declare: the robot's body-frame heading at launch == world heading 0.
        Therefore any future IMU yaw maps to world yaw by subtracting the
        startup reference.
        """
        self._imu_zero_ref = raw_yaw_at_startup

    def imu_yaw_to_world(self, raw_imu_yaw: float) -> float:
        """
        Full chain:  theta_I -> theta_B -> theta_W.

        theta_B = theta_I + alpha          (IMU->Body rotation)
        theta_W = theta_B - theta_B_init   (Body->World rotation, def. of world)
                = theta_I + alpha - (theta_I_init + alpha)
                = theta_I - theta_I_init

        Note: alpha cancels for yaw. We keep the explicit alpha step for
        symmetry with the vector case, in case the report needs it.
        """
        if self._imu_zero_ref is None:
            raise RuntimeError("set_imu_zero_reference() must be called first")
        theta_B = self.I_to_B.yaw_imu_to_body(raw_imu_yaw)
        theta_B_init = self.I_to_B.yaw_imu_to_body(self._imu_zero_ref)
        return math.atan2(math.sin(theta_B - theta_B_init),
                          math.cos(theta_B - theta_B_init))

    # ----- World pose update -----
    def update_world_pose(self, xw: float, yw: float, theta_w: float) -> None:
        self.B_to_W.update(xw, yw, theta_w)

    # ----- World <-> Body (for target points) -----
    def world_point_to_body(self, wx: float, wy: float) -> Tuple[float, float]:
        return self.B_to_W.point_world_to_body(wx, wy)

    def body_point_to_world(self, xb: float, yb: float) -> Tuple[float, float]:
        return self.B_to_W.point_body_to_world(xb, yb)

    # ----- Full chain for vectors (for completeness) -----
    def imu_vector_to_world(self, vx_imu: float, vy_imu: float) -> Tuple[float, float]:
        """{I} vector -> {B} vector -> {W} vector (full chain)."""
        vx_b, vy_b = self.I_to_B.vec_imu_to_body(vx_imu, vy_imu)
        return self.B_to_W.vec_body_to_world(vx_b, vy_b)

    def world_vector_to_imu(self, vx_w: float, vy_w: float) -> Tuple[float, float]:
        """{W} vector -> {B} vector -> {I} vector."""
        vx_b, vy_b = self.B_to_W.vec_world_to_body(vx_w, vy_w)
        return self.I_to_B.vec_body_to_imu(vx_b, vy_b)


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    print("Frame transform self-test")
    print("=" * 50)

    # Test 1: Body -> World round trip
    fm = FrameManager(imu_to_body_alpha=0.0)
    fm.update_world_pose(xw=1.0, yw=2.0, theta_w=math.pi / 2)

    # A point 1 unit forward of robot should be at (1, 3) in world
    xb, yb = 1.0, 0.0
    xw, yw = fm.body_point_to_world(xb, yb)
    print(f"Body (1, 0) with robot at (1,2,90°) -> World ({xw:.3f}, {yw:.3f})  (expect 1, 3)")

    # Round-trip back
    xb2, yb2 = fm.world_point_to_body(xw, yw)
    print(f"  Round-trip -> Body ({xb2:.3f}, {yb2:.3f})  (expect 1, 0)")

    # Test 2: IMU yaw with a nonzero alpha
    fm2 = FrameManager(imu_to_body_alpha=math.radians(30.0))
    fm2.set_imu_zero_reference(math.radians(124.0))
    theta_w = fm2.imu_yaw_to_world(math.radians(124.0))
    print(f"IMU 124° with zero ref 124°, alpha=30° -> World {math.degrees(theta_w):.2f}° (expect 0)")
    theta_w2 = fm2.imu_yaw_to_world(math.radians(154.0))
    print(f"IMU 154° (30° turn)                     -> World {math.degrees(theta_w2):.2f}° (expect 30)")

    # Test 3: World <-> Body with rotation
    fm3 = FrameManager()
    fm3.update_world_pose(xw=0.0, yw=0.0, theta_w=math.radians(90.0))
    wx, wy = 1.0, 0.0   # target to the east
    xb, yb = fm3.world_point_to_body(wx, wy)
    print(f"Target (1,0) with robot yaw 90° -> Body ({xb:.3f}, {yb:.3f})  (expect 0, -1)")
    # (0, -1) means: to the robot's right. Correct! Robot faces north, target is east, which is to its right.