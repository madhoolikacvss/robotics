#!/usr/bin/env python3
"""
Programmatic calibration for the differential-drive kinematic matrix.

Runs two experiments:
  A) Linear:  u_L = u_R = +V_TEST for T_LIN seconds -> measure distance -> k_v
  B) Angular: u_L = -V_TURN, u_R = +V_TURN for T_ANG seconds -> IMU yaw delta -> k_omega

Results written to calibration.json in this package's directory.

Usage:
    ros2 run robot_control calibrate linear   # Experiment A
    ros2 run robot_control calibrate angular  # Experiment B
    ros2 run robot_control calibrate show     # Print stored results
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Imu
import math
import time
import json
import os
import argparse

# Calibration parameters (edit these if you want different test values)
V_TEST = 0.3        # matches keyboard 'W' value
V_TURN = 0.5        # matches keyboard 'A' value
T_LIN = 3.0         # seconds for linear test
T_ANG = 2.0         # seconds for angular test
PUBLISH_HZ = 20.0   # must match keyboard_control.py timer

CAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")


# ---------- Quaternion to yaw (matches the convention motor_control.py publishes) ----------

def quat_to_yaw(qw, qx, qy, qz):
    """Extract yaw (radians) from quaternion. Assumes ROS REP-103 (+CCW)."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_diff(a, b):
    """Smallest signed difference a - b, wrapped to (-pi, pi]."""
    d = a - b
    return math.atan2(math.sin(d), math.cos(d))


# ---------- Calibration node ----------

class CalibrationNode(Node):
    def __init__(self):
        super().__init__('calibration_node')

        self.pub = self.create_publisher(Float32MultiArray, 'motor_commands', 10)
        self.sub = self.create_subscription(Imu, 'imu_data', self.imu_cb, 10)

        self.latest_yaw = None       # radians, from IMU
        self.yaw_samples = []        # (timestamp, yaw) for filtering

        self.get_logger().info('Calibration node ready.')

    def imu_cb(self, msg: Imu):
        yaw = quat_to_yaw(
            msg.orientation.w,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
        )
        self.latest_yaw = yaw
        self.yaw_samples.append((time.time(), yaw))
        # keep only last 5 s of samples
        cutoff = time.time() - 5.0
        self.yaw_samples = [(t, y) for t, y in self.yaw_samples if t >= cutoff]

    def send_wheels(self, u_l, u_r):
        msg = Float32MultiArray()
        msg.data = [float(u_l), float(u_r)]
        self.pub.publish(msg)

    def stop(self):
        """Send several stop messages to defeat the motor_control timeout."""
        for _ in range(5):
            self.send_wheels(0.0, 0.0)
            time.sleep(0.05)

    def run_linear_experiment(self):
        """
        Drive both wheels forward at V_TEST for T_LIN seconds.
        User measures straight-line distance afterwards and we solve for k_v.
        """
        self.get_logger().info(
            f"=== LINEAR CALIBRATION ===\n"
            f"Commanding u_L=u_R={V_TEST} for {T_LIN:.1f}s.\n"
            f"Mark the starting position NOW. Robot starts in 3 s..."
        )
        time.sleep(3.0)

        # drive
        t0 = time.time()
        rate_period = 1.0 / PUBLISH_HZ
        next_t = t0
        while time.time() - t0 < T_LIN:
            self.send_wheels(V_TEST, V_TEST)
            next_t += rate_period
            sleep = next_t - time.time()
            if sleep > 0:
                time.sleep(sleep)

        self.stop()
        self.get_logger().info("Done. Robot stopped.")

        # prompt for measured distance
        raw = input("Enter measured distance traveled in METERS (e.g. 1.24): ").strip()
        try:
            d = float(raw)
        except ValueError:
            self.get_logger().error(f"Could not parse {raw!r}. Aborting.")
            return

        v_actual = d / T_LIN
        k_v = v_actual / (V_TEST + V_TEST)

        self.get_logger().info(
            f"Measured distance: {d:.3f} m over {T_LIN:.1f} s\n"
            f"  -> v_actual = {v_actual:.4f} m/s\n"
            f"  -> k_v      = {k_v:.4f} m/s per unit command"
        )
        self.save_results({"k_v": k_v, "v_actual": v_actual, "distance": d,
                           "T_lin": T_LIN, "u_test": V_TEST})

    def run_angular_experiment(self):
        """
        Spin in place for T_ANG seconds and read yaw change from IMU.
        Fully automatic — no manual measurement needed.
        """
        self.get_logger().info(
            f"=== ANGULAR CALIBRATION ===\n"
            f"Commanding u_L=-{V_TURN}, u_R=+{V_TURN} for {T_ANG:.1f}s.\n"
            f"Robot starts in 3 s. Keep robot still BEFORE start."
        )
        time.sleep(3.0)

        # give IMU a moment to settle, then snapshot starting yaw
        if self.latest_yaw is None:
            self.get_logger().error("No IMU data received. Is motor_control.py running?")
            return

        # average yaw over a 0.3 s window for a cleaner theta_0
        theta_0 = self._average_yaw(0.3)
        self.get_logger().info(f"theta_0 = {math.degrees(theta_0):+.2f} deg")

        # spin
        t0 = time.time()
        rate_period = 1.0 / PUBLISH_HZ
        next_t = t0
        while time.time() - t0 < T_ANG:
            self.send_wheels(-V_TURN, V_TURN)
            next_t += rate_period
            sleep = next_t - time.time()
            if sleep > 0:
                time.sleep(sleep)

        self.stop()

        # small settle, then average final yaw
        time.sleep(0.3)
        theta_1 = self._average_yaw(0.3)
        self.get_logger().info(f"theta_1 = {math.degrees(theta_1):+.2f} deg")

        dtheta = angle_diff(theta_1, theta_0)
        omega_actual = dtheta / T_ANG
        k_omega = omega_actual / (V_TURN - (-V_TURN))  # = omega / 1.0

        # sign convention report
        sign_note = ("CCW positive (ROS-standard)" if dtheta > 0
                     else "CW positive (NED / inverted)")
        self.get_logger().info(
            f"Delta yaw: {math.degrees(dtheta):+.2f} deg over {T_ANG:.1f} s\n"
            f"  -> omega_actual = {omega_actual:+.4f} rad/s\n"
            f"  -> k_omega      = {k_omega:+.4f} rad/s per unit command\n"
            f"  -> IMU sign     = {sign_note}"
        )
        self.save_results({"k_omega": k_omega, "omega_actual": omega_actual,
                           "dtheta_deg": math.degrees(dtheta), "T_ang": T_ANG,
                           "u_test": V_TURN,
                           "imu_sign_ccw_positive": bool(dtheta > 0)})

    def _average_yaw(self, window_s):
        """Average the last `window_s` seconds of yaw samples, respecting wrapping."""
        cutoff = time.time() - window_s
        ys = [y for t, y in self.yaw_samples if t >= cutoff]
        if not ys:
            return self.latest_yaw
        # average via unit vectors to handle wrap-around correctly
        sx = sum(math.cos(y) for y in ys)
        sy = sum(math.sin(y) for y in ys)
        return math.atan2(sy, sx)

    # ---------- Persistence ----------

    def save_results(self, updates: dict):
        existing = {}
        if os.path.isfile(CAL_FILE):
            try:
                with open(CAL_FILE) as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
        existing.update(updates)
        with open(CAL_FILE, "w") as f:
            json.dump(existing, f, indent=2)
        self.get_logger().info(f"Saved -> {CAL_FILE}")


def load_calibration():
    """Helper for other modules (path_tracker) to read the calibration file."""
    if not os.path.isfile(CAL_FILE):
        return None
    with open(CAL_FILE) as f:
        return json.load(f)


def show_calibration():
    if not os.path.isfile(CAL_FILE):
        print(f"No calibration file at {CAL_FILE}")
        return
    with open(CAL_FILE) as f:
        data = json.load(f)
    print(f"Calibration from {CAL_FILE}:")
    for k, v in data.items():
        print(f"  {k:24s} = {v}")


def main(args=None):
    parser = argparse.ArgumentParser(description="Kinematic calibration tool")
    parser.add_argument("mode", choices=["linear", "angular", "show"],
                        help="Which experiment to run, or 'show' stored values")
    cli, _ = parser.parse_known_args()

    if cli.mode == "show":
        show_calibration()
        return

    rclpy.init(args=args)
    node = CalibrationNode()
    try:
        if cli.mode == "linear":
            node.run_linear_experiment()
        elif cli.mode == "angular":
            node.run_angular_experiment()
    finally:
        try:
            node.stop()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()