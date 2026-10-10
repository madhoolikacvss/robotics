#!/usr/bin/env python3
"""
Angular calibration with continuous yaw sampling.

Runs two spin tests (CCW then CW), each for T seconds, and computes:
  - Total accumulated yaw (via wrapped increments, robust to encoder wrap)
  - Actual angular velocity = accumulated_angle / measured_duration
  - k_omega (both directions), for symmetry check

Raw yaw samples are logged to angular_calibration_log.csv for inspection.

Requires: motor_control.py running (publishes /imu_data, consumes /motor_commands)

Usage:
    ros2 run robot_control angular_calibrate
    ros2 run robot_control angular_calibrate --t 3.0
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Imu
import math
import time
import json
import os
import csv
import threading
import argparse


V_TURN = 0.5            # magnitude of wheel command
T_TURN_DEFAULT = 3.0    # seconds per spin
PUBLISH_HZ = 20.0

CAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")
CSV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "angular_calibration_log.csv")


# ------------------------------------------------------------
# Yaw extraction
# ------------------------------------------------------------

def quat_to_yaw(qw, qx, qy, qz):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


# ------------------------------------------------------------
# Node
# ------------------------------------------------------------

class AngularCalibrateNode(Node):
    def __init__(self):
        super().__init__('angular_calibrate_node')

        self.pub = self.create_publisher(Float32MultiArray, 'motor_commands', 10)
        self.sub = self.create_subscription(Imu, 'imu_data', self.cb, 10)

        self.lock = threading.Lock()
        self.latest_yaw = None         # radians (raw IMU frame)
        self.sample_count = 0
        self.sample_log = []           # list of (t, yaw_rad) for current test

        self.get_logger().info("Angular calibration node ready.")

    def cb(self, msg: Imu):
        yaw = quat_to_yaw(
            msg.orientation.w, msg.orientation.x,
            msg.orientation.y, msg.orientation.z,
        )
        t = time.time()
        with self.lock:
            self.latest_yaw = yaw
            self.sample_count += 1
            # Always append while a test is active; enabled by run_test()
            if self.recording:
                self.sample_log.append((t, yaw))

    # enable/disable recording flag
    recording = False

    # -------- command helpers --------
    def send(self, uL, uR):
        m = Float32MultiArray()
        m.data = [float(uL), float(uR)]
        self.pub.publish(m)

    def stop_motors(self):
        for _ in range(5):
            self.send(0.0, 0.0)
            time.sleep(0.05)

    def wait_for_imu(self, timeout=3.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                if self.latest_yaw is not None:
                    return True
            time.sleep(0.05)
        return False

    # -------- core test --------
    def run_one(self, uL, uR, duration, label):
        """
        Spins with (uL, uR) for `duration` seconds. Records yaw continuously.
        Returns dict with accumulated angle, omega, k_omega.
        """
        print(f"\n=== {label} ===")
        print(f"Command: u_L = {uL:+.3f}, u_R = {uR:+.3f}   u_diff = {uR-uL:+.3f}")
        print(f"Duration: {duration:.2f} s")
        print(f"Starts in 3 s. Keep robot still and free to rotate.")

        time.sleep(3.0)

        if not self.wait_for_imu():
            print("ERROR: no IMU data. Is motor_control.py running?")
            return None

        # clear log and start recording
        with self.lock:
            self.sample_log = []
            self.recording = True

        # drive
        t_start = time.time()
        period = 1.0 / PUBLISH_HZ
        next_t = t_start
        while time.time() - t_start < duration:
            self.send(uL, uR)
            next_t += period
            s = next_t - time.time()
            if s > 0:
                time.sleep(s)

        self.stop_motors()
        t_stop = time.time()

        # small settle before stopping recording
        time.sleep(0.3)

        with self.lock:
            self.recording = False
            log = list(self.sample_log)
            n_samples = len(log)

        measured_duration = t_stop - t_start

        # -------- accumulate wrapped increments --------
        if len(log) < 2:
            print("ERROR: not enough IMU samples collected.")
            return None

        accumulated = 0.0
        # unwrap and accumulate
        prev = log[0][1]
        for _, y in log[1:]:
            d = wrap(y - prev)
            accumulated += d
            prev = y

        accumulated_deg = math.degrees(accumulated)
        omega = accumulated / measured_duration         # rad/s (signed)
        u_diff = uR - uL
        k_omega = omega / u_diff if abs(u_diff) > 1e-6 else 0.0

        # -------- report --------
        print(f"\nResults ({label}):")
        print(f"  Samples collected      : {n_samples}")
        print(f"  Sample rate (measured) : {n_samples / measured_duration:.1f} Hz")
        print(f"  Measured duration      : {measured_duration:.3f} s")
        print(f"  Accumulated Δyaw       : {accumulated_deg:+.2f}°   ({accumulated:+.4f} rad)")
        print(f"  Angular velocity ω     : {omega:+.4f} rad/s   ({math.degrees(omega):+.2f} °/s)")
        print(f"  u_diff                 : {u_diff:+.3f}")
        print(f"  k_omega (this run)     : {k_omega:+.4f} rad/s per unit")

        return {
            "label": label,
            "u_L": uL, "u_R": uR, "u_diff": u_diff,
            "duration": measured_duration,
            "samples": n_samples,
            "delta_rad": accumulated,
            "delta_deg": accumulated_deg,
            "omega_rad_s": omega,
            "k_omega": k_omega,
            "log": log,
        }

    # -------- run both directions --------
    def run_both(self, duration):
        r_ccw = self.run_one(-V_TURN, +V_TURN, duration, "CCW (left turn)")
        r_cw  = self.run_one(+V_TURN, -V_TURN, duration, "CW (right turn)")

        results = [r for r in (r_ccw, r_cw) if r is not None]
        if not results:
            print("No valid results.")
            return

        # -------- save CSV of all samples --------
        try:
            with open(CSV_FILE, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["run", "t_seconds", "yaw_rad", "yaw_deg"])
                for r in results:
                    if not r["log"]:
                        continue
                    t0 = r["log"][0][0]
                    for t, y in r["log"]:
                        w.writerow([r["label"], f"{t-t0:.4f}", f"{y:.6f}", f"{math.degrees(y):.4f}"])
            print(f"\nRaw samples saved -> {CSV_FILE}")
        except Exception as e:
            print(f"CSV write failed: {e}")

        # -------- summary --------
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for r in results:
            print(f"  {r['label']:18s}  ω={r['omega_rad_s']:+.4f} rad/s  "
                  f"k_omega={r['k_omega']:+.4f}")

        if len(results) == 2:
            k_ccw = abs(results[0]["k_omega"])
            k_cw  = abs(results[1]["k_omega"])
            k_avg = (k_ccw + k_cw) / 2.0
            asym = abs(k_ccw - k_cw) / k_avg * 100.0 if k_avg > 0 else 0.0
            print(f"\n  Symmetry check:")
            print(f"    |k_ccw| = {k_ccw:.4f}")
            print(f"    |k_cw|  = {k_cw:.4f}")
            print(f"    mean    = {k_avg:.4f}")
            print(f"    asym    = {asym:.1f}%  "
                  f"({'OK' if asym < 15 else 'HIGH — check motor wiring/friction'})")

            # sign convention
            sign_ok = (results[0]["omega_rad_s"] > 0 and results[1]["omega_rad_s"] < 0)
            print(f"\n  IMU sign convention: "
                  f"{'CCW positive (ROS-standard) ✅' if sign_ok else 'INVERTED ⚠️'}")

            # propose new k_omega
            print(f"\n  Proposed k_omega to write: {k_avg:.4f} rad/s per unit")
            print(f"  (Current calibration.json: ")
            try:
                with open(CAL_FILE) as f:
                    cur = json.load(f)
                print(f"     k_omega = {cur.get('k_omega', 'MISSING')}")
            except Exception:
                print("     (no calibration.json found)")
            print("   )")
            print("\n  NOT writing automatically. Update calibration.json manually if satisfied.")

    # -------- write helper (kept separate, called only if you enable it) --------
    def save_k_omega(self, k_omega):
        """Overwrite calibration.json's k_omega field. Not called by default."""
        existing = {}
        if os.path.isfile(CAL_FILE):
            try:
                with open(CAL_FILE) as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
        existing["k_omega"] = k_omega
        with open(CAL_FILE, "w") as f:
            json.dump(existing, f, indent=2)


# ------------------------------------------------------------
# Entry
# ------------------------------------------------------------

def main(args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=float, default=T_TURN_DEFAULT,
                    help="duration per spin (seconds)")
    cli, _ = ap.parse_known_args()

    rclpy.init()
    node = AngularCalibrateNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()

    try:
        node.run_both(cli.t)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        try:
            node.stop_motors()
        except Exception:
            pass
        executor.shutdown()
        spin.join()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()