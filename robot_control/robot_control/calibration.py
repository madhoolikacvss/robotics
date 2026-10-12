#!/usr/bin/env python3
"""Manual calibration for the open-loop waypoint follower.

Test 1 (straight): drive both wheels at +command for T seconds, measure distance
    wheel_speed_per_command = distance / (command * T)

Test 2 (spin): left=-command, right=+command for T seconds, measure rotation
    omega = 2 * command * k / wheel_base,  theta = omega * T
    wheel_base = 2 * command * k * T / theta

Usage:
    python3 calibrate.py                      # straight + spin
    python3 calibrate.py --skip-straight --k 0.8   # spin only, with a known k
"""

import argparse
import math
import statistics
import time

import rclpy
from std_msgs.msg import Float32MultiArray


def ask_float(prompt):
    while True:
        try:
            return float(input(prompt).strip())
        except ValueError:
            print("  Please enter a number.")


class Driver:
    def __init__(self, rate_hz=20.0):
        rclpy.init()
        self.node = rclpy.create_node("calibration")
        self.pub = self.node.create_publisher(Float32MultiArray, "motor_commands", 10)
        self.period = 1.0 / rate_hz
        time.sleep(1.0)  # let the publisher connect

    def send(self, left, right):
        msg = Float32MultiArray()
        msg.data = [float(left), float(right)]
        self.pub.publish(msg)

    def run(self, left, right, duration):
        end = time.monotonic() + duration
        while time.monotonic() < end:
            self.send(left, right)
            time.sleep(self.period)
        self.stop()

    def stop(self):
        for _ in range(5):
            self.send(0.0, 0.0)
            time.sleep(0.02)

    def close(self):
        self.stop()
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def summarize(name, values):
    mean = statistics.mean(values)
    spread = statistics.stdev(values) if len(values) > 1 else 0.0
    print(f"  {name}: mean={mean:.4f}  stdev={spread:.4f}  (n={len(values)})")
    return mean


def straight_test(drv, cmd, duration, trials):
    print(f"\n=== STRAIGHT TEST: command={cmd}, duration={duration}s ===")
    print("Mark the robot's starting position (e.g. front of a wheel) on the floor.")
    results = []
    for i in range(trials):
        input(f"\nTrial {i + 1}/{trials}: place robot at the start, press Enter to drive...")
        drv.run(cmd, cmd, duration)
        dist_cm = ask_float("  Distance travelled (cm): ")
        k = (dist_cm / 100.0) / (cmd * duration)
        print(f"  -> wheel_speed_per_command = {k:.4f}")
        results.append(k)
    return summarize("wheel_speed_per_command", results)


def spin_test(drv, cmd, duration, trials, k):
    print(f"\n=== SPIN TEST: command={cmd}, duration={duration}s, k={k:.4f} ===")
    print("Mark the robot's heading (e.g. tape line on the chassis) to count rotation.")
    print("Aim for about 1-2 full turns; adjust --spin-duration if it's far off.")
    results = []
    for i in range(trials):
        input(f"\nTrial {i + 1}/{trials}: align the robot to your mark, press Enter to spin...")
        drv.run(-cmd, cmd, duration)  # positive command-difference = CCW
        deg = ask_float("  Rotation (degrees, CCW positive, e.g. 400 for 1.1 turns): ")
        if deg <= 0:
            print("  Skipping (rotation must be positive).")
            continue
        wb = 2.0 * cmd * k * duration / math.radians(deg)
        print(f"  -> wheel_base = {wb:.4f} m")
        results.append(wb)
    return summarize("wheel_base", results) if results else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--command", type=float, default=0.3, help="motor command for tests (<= 0.5)")
    p.add_argument("--straight-duration", type=float, default=3.0)
    p.add_argument("--spin-duration", type=float, default=4.0)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--skip-straight", action="store_true")
    p.add_argument("--k", type=float, default=0.8, help="wheel_speed_per_command if skipping")
    args = p.parse_args()

    if not 0 < args.command <= 0.5:
        raise SystemExit("--command must be in (0, 0.5]")

    drv = Driver()
    try:
        k = args.k
        if not args.skip_straight:
            k = straight_test(drv, args.command, args.straight_duration, args.trials)
        wb = spin_test(drv, args.command, args.spin_duration, args.trials, k)

        print("\n=== RESULTS ===")
        print(f"wheel_speed_per_command: {k:.4f}")
        if wb:
            print(f"wheel_base:              {wb:.4f}")
        print("\nExample launch:")
        print(f"  ros2 run <pkg> waypoint_follower --ros-args "
              f"-p wheel_speed_per_command:={k:.4f}" + (f" -p wheel_base:={wb:.4f}" if wb else ""))
    except KeyboardInterrupt:
        pass
    finally:
        drv.close()


if __name__ == "__main__":
    main()
