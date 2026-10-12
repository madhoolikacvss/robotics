#!/usr/bin/env python3
"""
Simple measurement script for calibration regression.

Runs ONE experiment per invocation:
  - linear : sends (u, u) for T seconds; you measure the distance with tape.
  - angular: sends (-diff/2, +diff/2) for T seconds; yaw is read from the IMU
             with robust phase unwrapping so that rotations > 360 deg are
             handled correctly.

You run each mode 15-30 times, record the printed scalar, then fit:
    distance = k_v    * u_avg  * T
    yaw      = k_omega * u_diff * T

Usage:
    ros2 run robot_control measure linear  --u 0.3 --t 2.0
    ros2 run robot_control measure angular --diff 0.5 --t 2.0
    ros2 run robot_control measure angular --diff -0.5 --t 2.0
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

import numpy as np


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def quat_to_yaw(qw, qx, qy, qz):
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz)
    )


def angle_diff(a, b):
    """Shortest signed a - b in (-pi, pi]."""
    d = a - b
    return math.atan2(math.sin(d), math.cos(d))


# ------------------------------------------------------------
# Node
# ------------------------------------------------------------

class MeasureNode(Node):
    def __init__(self):
        super().__init__('measure_node')

        self.pub = self.create_publisher(Float32MultiArray, 'motor_commands', 10)
        self.sub = self.create_subscription(Imu, 'imu_data', self.cb, 10)

        self.lock = threading.Lock()
        self.latest_yaw = None
        self.yaw_buf = []           # rolling buffer (t, yaw_rad)
        self.record_log = []        # samples for the current test
        self.recording = False

    # -------- IMU callback --------
    def cb(self, msg: Imu):
        yaw = quat_to_yaw(
            msg.orientation.w, msg.orientation.x,
            msg.orientation.y, msg.orientation.z,
        )
        t = time.time()
        with self.lock:
            self.latest_yaw = yaw
            # rolling buffer for pre-test averaging
            self.yaw_buf.append((t, yaw))
            cutoff = time.time() - 10.0
            self.yaw_buf = [(tt, yy) for tt, yy in self.yaw_buf if tt >= cutoff]
            # per-test recording
            if self.recording:
                self.record_log.append((t, yaw))

    # -------- command helpers --------
    def send(self, uL, uR):
        m = Float32MultiArray()
        m.data = [float(uL), float(uR)]
        self.pub.publish(m)

    def stop_motors(self):
        for _ in range(5):
            self.send(0.0, 0.0)
            time.sleep(0.05)

    def avg_yaw_recent(self, window_s=0.3):
        """Circular mean of the last `window_s` seconds of yaw samples."""
        cutoff = time.time() - window_s
        with self.lock:
            ys = [y for t, y in self.yaw_buf if t >= cutoff]
            latest = self.latest_yaw
        if not ys:
            return latest
        sx = sum(math.cos(y) for y in ys)
        sy = sum(math.sin(y) for y in ys)
        return math.atan2(sy, sx)

    # ============================================================
    # LINEAR test
    # ============================================================
    def run_linear(self, u, duration):
        print(f"\n=== LINEAR MEASUREMENT ===")
        print(f"Command: u_L = u_R = {u:+.3f}   for {duration:.2f} s")
        print(f"Place the robot at a start mark. Starts in 3 s...")
        time.sleep(3.0)

        t0 = time.time()
        next_t = t0
        while time.time() - t0 < duration:
            self.send(u, u)
            next_t += 0.05
            s = next_t - time.time()
            if s > 0:
                time.sleep(s)

        self.stop_motors()
        print("Done. Robot stopped.")

        raw = input("Measured distance in METERS (e.g. 0.52): ").strip()
        try:
            d = float(raw)
        except ValueError:
            print(f"Bad input {raw!r}. Aborting.")
            return

        u_avg = u                         # both wheels equal
        k_v = d / (u_avg * duration)      # m per (unit * s)
        print(f"\nResults:")
        print(f"  distance d         = {d:.3f} m")
        print(f"  u_avg              = {u:+.3f}")
        print(f"  duration           = {duration:.3f} s")
        print(f"  d / (u_avg * t)    = {k_v:.4f}   m per (unit * s)")
        print(f"  speed at u=0.3     = {k_v * 0.3:.4f} m/s")

    # ============================================================
    # ANGULAR test
    # ============================================================
    def run_angular(self, u_diff, duration):
        uL = -u_diff / 2.0
        uR = +u_diff / 2.0

        print(f"\n=== ANGULAR MEASUREMENT ===")
        print(f"Command: u_L = {uL:+.3f}, u_R = {uR:+.3f}   (u_diff = {u_diff:+.3f})")
        print(f"Duration: {duration:.2f} s")
        print(f"Starts in 3 s. Robot free to rotate...")
        time.sleep(3.0)

        if self.latest_yaw is None:
            print("ERROR: no IMU data.")
            return

        # start recording
        with self.lock:
            self.record_log = []
            self.recording = True

        # ---- drive ----
        t_start = time.time()
        next_t = t_start
        while time.time() - t_start < duration:
            self.send(uL, uR)
            next_t += 0.05
            s = next_t - time.time()
            if s > 0:
                time.sleep(s)

        self.stop_motors()
        t_stop = time.time()

        # let one more sample arrive, then stop recording
        time.sleep(0.2)
        with self.lock:
            self.recording = False
            log = list(self.record_log)

        measured_duration = t_stop - t_start

        # ---- need enough samples ----
        if len(log) < 5:
            print(f"ERROR: only {len(log)} IMU samples. Aborting.")
            return

        times = np.array([t for t, _ in log])
        yaws = np.array([y for _, y in log])

        # ---------- diagnostics ----------
        dt = np.diff(times)
        mean_gap = float(np.mean(dt))
        median_gap = float(np.median(dt))
        max_gap = float(np.max(dt))
        n_big_gaps = int(np.sum(dt > 0.1))

        # wrapped step sizes (before unwrap)
        raw_steps = np.diff(yaws)
        wrapped_steps = np.arctan2(np.sin(raw_steps), np.cos(raw_steps))
        max_step = float(np.max(np.abs(wrapped_steps)))

        print(f"\nTiming diagnostics:")
        print(f"  samples               : {len(log)}")
        print(f"  mean dt               : {mean_gap*1000:.2f} ms")
        print(f"  median dt             : {median_gap*1000:.2f} ms")
        print(f"  max dt                : {max_gap*1000:.2f} ms")
        print(f"  gaps > 100 ms         : {n_big_gaps}")
        print(f"  max |step| (wrapped)  : {math.degrees(max_step):.2f}°")
        if max_step > 0.8 * math.pi:
            print(f"  ⚠️  step close to π — unwrap reliability marginal")
        if n_big_gaps > 0:
            print(f"  ⚠️  {n_big_gaps} gap(s) > 100 ms — could hide wraps")

        # ---------- three unwrap methods ----------
        # Method A — naive per-step accumulation (matches old behavior)
        accum_naive = float(np.sum(wrapped_steps))

        # Method B — numpy.unwrap on the full sequence (primary)
        unwrapped = np.unwrap(yaws)
        accum_unwrap = float(unwrapped[-1] - unwrapped[0])

        # Method C — drop samples after big gaps, then unwrap
        keep = np.ones(len(times), dtype=bool)
        for i in range(1, len(times)):
            if times[i] - times[i - 1] > 0.1:
                keep[i] = False
        clean_yaws = yaws[keep]
        if len(clean_yaws) >= 5:
            u_clean = np.unwrap(clean_yaws)
            accum_clean = float(u_clean[-1] - u_clean[0])
        else:
            accum_clean = accum_unwrap
        n_dropped = int(len(yaws) - len(clean_yaws))

        # primary result
        accumulated = accum_unwrap
        accumulated_deg = math.degrees(accumulated)
        omega = accumulated / measured_duration
        k_omega = omega / u_diff if abs(u_diff) > 1e-6 else 0.0

        # ---------- report ----------
        print(f"\nAccumulated Δyaw:")
        print(f"  naive per-step       : {math.degrees(accum_naive):+9.2f}°  "
              f"({accum_naive:+.4f} rad)")
        print(f"  numpy.unwrap (full)  : {math.degrees(accum_unwrap):+9.2f}°  "
              f"({accum_unwrap:+.4f} rad)   ← primary")
        if n_dropped > 0:
            print(f"  gap-filtered ({n_dropped:2d} dropped) : "
                  f"{math.degrees(accum_clean):+9.2f}°  ({accum_clean:+.4f} rad)")

        # agreement check
        dA = math.degrees(abs(accum_naive - accum_unwrap))
        dC = math.degrees(abs(accum_unwrap - accum_clean))
        if dA > 30:
            print(f"  ⚠️  naive vs unwrap differ by {dA:.1f}° — trust unwrap")
        if n_dropped > 0 and dC > 10:
            print(f"  ⚠️  unwrap vs gap-filtered differ by {dC:.1f}° — inspect CSV")

        print(f"\nResults:")
        print(f"  measured duration    = {measured_duration:.3f} s")
        print(f"  Δyaw (used)          = {accumulated_deg:+.2f}°  "
              f"({accumulated:+.4f} rad)")
        print(f"  ω                    = {omega:+.4f} rad/s  "
              f"({math.degrees(omega):+.2f} °/s)")
        print(f"  u_diff               = {u_diff:+.3f}")
        print(f"  k_omega              = ω / u_diff = {k_omega:+.4f} rad per (unit * s)")
        print(f"  sign convention      = "
              f"{'CCW positive (ROS-standard)' if omega > 0 else 'CW positive / inverted'}")

        # full-cycle sanity note
        full_turns = abs(accumulated) / (2.0 * math.pi)
        print(f"  full revolutions     = {full_turns:.3f}")


# ------------------------------------------------------------
# Entry
# ------------------------------------------------------------

def main(args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["linear", "angular"])
    ap.add_argument("--u", type=float, default=0.3,
                    help="wheel command magnitude for linear mode")
    ap.add_argument("--diff", type=float, default=0.5,
                    help="u_R - u_L for angular mode")
    ap.add_argument("--t", type=float, default=2.0,
                    help="duration in seconds")
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