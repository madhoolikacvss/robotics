#!/usr/bin/env python3
"""
Simple measurement script for calibration regression.

Runs ONE experiment per invocation:
  - Sends a fixed (u_L, u_R) command for T seconds
  - Reports distance (for linear) or yaw change (for angular)

You run this 20-30 times, record the outputs, then fit:
    distance = k_v * u_avg * T
    yaw      = k_omega * u_diff * T

Usage:
    # Linear: drive straight at u=0.3 for 2 s. Measure distance with tape.
    ros2 run robot_control measure linear --u 0.3 --t 2.0

    # Angular: turn left at diff=0.5 for 2 s. Auto-measures yaw from IMU.
    ros2 run robot_control measure angular --diff 0.5 --t 2.0
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Imu
import math
import time
import threading
import argparse


def quat_to_yaw(qw, qx, qy, qz):
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz)
    )


def angle_diff(a, b):
    d = a - b
    return math.atan2(math.sin(d), math.cos(d))


class MeasureNode(Node):
    def __init__(self):
        super().__init__('measure_node')
        self.pub = self.create_publisher(Float32MultiArray, 'motor_commands', 10)
        self.sub = self.create_subscription(Imu, 'imu_data', self.cb, 10)
        self.lock = threading.Lock()
        self.latest_yaw = None
        self.yaw_buf = []       # (t, yaw)

    def cb(self, msg):
        yaw = quat_to_yaw(msg.orientation.w, msg.orientation.x,
                          msg.orientation.y, msg.orientation.z)
        with self.lock:
            self.latest_yaw = yaw
            self.yaw_buf.append((time.time(), yaw))
            cutoff = time.time() - 10.0
            self.yaw_buf = [(t, y) for t, y in self.yaw_buf if t >= cutoff]

    def send(self, uL, uR):
        m = Float32MultiArray()
        m.data = [float(uL), float(uR)]
        self.pub.publish(m)

    def stop(self):
        for _ in range(5):
            self.send(0.0, 0.0)
            time.sleep(0.05)

    def avg_yaw_recent(self, window_s=0.3):
        cutoff = time.time() - window_s
        with self.lock:
            ys = [y for t, y in self.yaw_buf if t >= cutoff]
            latest = self.latest_yaw
        if not ys:
            return latest
        sx = sum(math.cos(y) for y in ys)
        sy = sum(math.sin(y) for y in ys)
        return math.atan2(sy, sx)

    # ---------- experiments ----------

    def run_linear(self, u, duration):
        print(f"LINEAR: u_L=u_R={u} for {duration} s")
        print("Place robot at start line. Starts in 3 s...")
        time.sleep(3.0)

        t0 = time.time()
        next_t = t0
        while time.time() - t0 < duration:
            self.send(u, u)
            next_t += 0.05
            s = next_t - time.time()
            if s > 0:
                time.sleep(s)

        self.stop()
        print("Done. Robot stopped.")
        raw = input("Measured distance in METERS: ").strip()
        try:
            d = float(raw)
        except ValueError:
            print("Bad input, aborting.")
            return

        u_avg = u   # both wheels same, so u_L = u_R = u
        # distance = k_v * u_avg * duration
        k_v = d / (u_avg * duration)
        print(f"\nResult: d = {d:.3f} m, u = {u:.3f}, t = {duration:.2f} s")
        print(f"        k_v = d / (u * t) = {k_v:.4f} m per (unit·s)")
        print(f"        (equivalently: 1 unit command for 1 s moves {k_v:.3f} m)")

    def run_angular(self, u_diff, duration):
        # distribute the diff: u_L = -diff/2, u_R = +diff/2
        uL, uR = -u_diff / 2.0, +u_diff / 2.0
        print(f"ANGULAR: u_L={uL:+.3f}, u_R={uR:+.3f} for {duration} s")
        print(f"u_diff = {u_diff:+.3f}")
        print("Starts in 3 s. Robot free to rotate...")
        time.sleep(3.0)

        if self.latest_yaw is None:
            print("ERROR: no IMU data.")
            return
        theta0 = self.avg_yaw_recent(0.3)

        t0 = time.time()
        next_t = t0
        while time.time() - t0 < duration:
            self.send(uL, uR)
            next_t += 0.05
            s = next_t - time.time()
            if s > 0:
                time.sleep(s)

        self.stop()
        time.sleep(0.3)
        theta1 = self.avg_yaw_recent(0.3)

        dtheta_rad = angle_diff(theta1, theta0)
        dtheta_deg = math.degrees(dtheta_rad)
        # yaw = k_omega * u_diff * duration
        k_omega = dtheta_rad / (u_diff * duration)

        print(f"\nResult: Δyaw = {dtheta_deg:+.2f}°, u_diff = {u_diff:+.3f}, t = {duration:.2f} s")
        print(f"        k_omega = Δyaw / (u_diff * t) = {k_omega:+.4f} rad per (unit·s)")
        print(f"        (equivalently: 1 unit diff for 1 s turns {math.degrees(k_omega):+.2f}°)")


def main(args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["linear", "angular"])
    ap.add_argument("--u", type=float, default=0.3, help="wheel command magnitude (linear)")
    ap.add_argument("--diff", type=float, default=0.5, help="u_R - u_L (angular)")
    ap.add_argument("--t", type=float, default=2.0, help="duration in seconds")
    cli, _ = ap.parse_known_args()

    rclpy.init()
    node = MeasureNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()

    try:
        if cli.mode == "linear":
            node.run_linear(cli.u, cli.t)
        else:
            node.run_angular(cli.diff, cli.t)
    finally:
        try:
            node.stop()
        except Exception:
            pass
        executor.shutdown()
        spin.join()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()