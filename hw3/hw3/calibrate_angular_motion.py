#!/usr/bin/env python3
"""Calibrate measured turn angle against command duration."""

import math
import threading
import time
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

from .velocity_mapping import K_OMEGA


def fit_angle_time_model(
    samples: list[tuple[float, float]],
) -> tuple[float, float]:
    """Fit measured angle to angular offset + angular scale * duration."""
    if len(samples) < 2:
        raise ValueError('At least two angle-time samples are required.')

    durations = [duration for duration, _ in samples]
    angles = [angle for _, angle in samples]
    duration_mean = sum(durations) / len(durations)
    angle_mean = sum(angles) / len(angles)
    variance = sum(
        (duration - duration_mean) ** 2 for duration in durations
    )
    if variance <= 1e-12:
        raise ValueError('Angle-time samples must use different durations.')

    scale = sum(
        (duration - duration_mean) * (angle - angle_mean)
        for duration, angle in samples
    ) / variance
    if scale <= 0.0:
        raise ValueError('Measured angle must increase with turn duration.')
    offset = angle_mean - scale * duration_mean
    return offset, scale


class AngularMotionCalibrator(Node):
    """Run safe fixed-command turns of varying duration."""

    def __init__(self):
        """Initialize ROS interfaces and calibration trial state."""
        super().__init__('angular_motion_calibrator')

        self.command_topic = str(
            self.declare_parameter('command_topic', '/cmd_vel').value
        )
        self.turn_durations = [
            float(value) for value in self.declare_parameter(
                'turn_durations', [0.25, 0.5, 0.75, 1.0]
            ).value
        ]
        self.settle_duration = float(
            self.declare_parameter('settle_duration', 0.4).value
        )
        self.start_delay = float(
            self.declare_parameter('start_delay', 3.0).value
        )

        if len(set(self.turn_durations)) < 2:
            raise ValueError(
                'Provide at least two different turn durations.'
            )
        if any(value <= 0.0 for value in self.turn_durations):
            raise ValueError('Turn durations must be positive.')
        if self.settle_duration < 0.0 or self.start_delay < 0.0:
            raise ValueError(
                'settle_duration and start_delay cannot be negative.'
            )

        self.cmd_pub = self.create_publisher(Twist, self.command_topic, 10)
        self.trials = [
            (duration, direction)
            for duration in self.turn_durations
            for direction in (1.0, -1.0)
        ]
        self.trial_index = 0
        self.samples: dict[float, list[tuple[float, float]]] = {
            1.0: [],
            -1.0: [],
        }
        self.state = 'countdown'
        self.state_start = time.monotonic()
        self.turn_start_time = 0.0
        self.turn_duration = 0.0
        self.turn_elapsed = 0.0
        self.measurement_ready = threading.Event()
        self.complete = threading.Event()
        self.timer = self.create_timer(0.02, self.control_loop)

        self.get_logger().info(
            f'The robot will run {len(self.trials)} in-place turns at the '
            f'fixed angular command magnitude {K_OMEGA:.2f}, with durations '
            f'{self.turn_durations}s. It starts after {self.start_delay:.1f}s. '
            'After each turn, enter the measured signed angle in degrees '
            '(CCW positive, CW negative). Measure each turn from its own '
            'starting heading. Keep the robot on a clear, level floor and '
            'stop other /cmd_vel publishers.'
        )

    def _publish_command(self, angular: float = 0.0):
        command = Twist()
        command.angular.z = angular
        self.cmd_pub.publish(command)

    def control_loop(self):
        """Publish the active trial command and advance calibration state."""
        now = time.monotonic()
        if self.complete.is_set():
            return

        if self.state in ('countdown', 'settling', 'awaiting_measurement'):
            self._publish_command()
            if (
                self.state == 'countdown'
                and now - self.state_start >= self.start_delay
            ):
                self._start_trial(now)
            elif (
                self.state == 'settling'
                and now - self.state_start >= self.settle_duration
            ):
                self._start_trial(now)
            return

        if self.state == 'turning':
            _, direction = self.trials[self.trial_index]
            elapsed = now - self.turn_start_time
            if elapsed >= self.turn_duration:
                self._publish_command()
                self.turn_elapsed = elapsed
                self.get_logger().info(
                    f'Trial {self.trial_index + 1}/{len(self.trials)} '
                    f'finished: command={direction * K_OMEGA:+.3f} for '
                    f'{elapsed:.3f}s. Enter the signed measured angle '
                    'in degrees.'
                )
                self.state = 'awaiting_measurement'
                self.measurement_ready.set()
            else:
                self._publish_command(direction * K_OMEGA)

    def _start_trial(self, now: float):
        duration, direction = self.trials[self.trial_index]
        self.turn_duration = duration
        self.turn_start_time = now
        self.state = 'turning'
        self.get_logger().info(
            f'Starting trial {self.trial_index + 1}/{len(self.trials)}: '
            f'angular command={direction * K_OMEGA:+.3f} for '
            f'{duration:.2f}s '
            f'({"CCW" if direction > 0 else "CW"}).'
        )

    def record_measurement(self, angle_degrees: float):
        """Store a manually measured signed turn angle and continue."""
        if self.state != 'awaiting_measurement':
            raise RuntimeError('No completed turn is awaiting measurement.')
        _, direction = self.trials[self.trial_index]
        measured_angle = direction * math.radians(angle_degrees)
        if measured_angle <= 0.0:
            raise ValueError(
                'Measured angle must have the same sign as the commanded '
                'turn (CCW positive, CW negative).'
            )
        self.samples[direction].append(
            (self.turn_elapsed, measured_angle)
        )
        self.get_logger().info(
            f'Recorded measured angle={measured_angle:.3f} rad after '
            f'{self.turn_elapsed:.3f}s.'
        )
        self.trial_index += 1
        self.measurement_ready.clear()
        if self.trial_index >= len(self.trials):
            self._finish()
        else:
            self.state = 'settling'
            self.state_start = time.monotonic()

    def _finish(self):
        self._publish_command()
        ccw_offset, ccw_rate = fit_angle_time_model(self.samples[1.0])
        cw_offset, cw_rate = fit_angle_time_model(self.samples[-1.0])
        self.get_logger().info(
            'Calibration complete. Separate angle-time fits at the fixed '
            'controller command: actual_angle = '
            'angle_offset + angular_rate * duration.'
        )
        self.get_logger().info(
            f'CCW: angle_offset={ccw_offset:.6f} rad, '
            f'ANGULAR_RATE_CCW={ccw_rate:.6f} rad/s; '
            f'CW: angle_offset={cw_offset:.6f} rad, '
            f'ANGULAR_RATE_CW={cw_rate:.6f} rad/s. '
            f'Use the two rates in waypoint_motion.py (calibrated at '
            f'|angular command|={K_OMEGA:.2f}).'
        )
        self.complete.set()

    def destroy_node(self):
        """Stop the robot before releasing the ROS node."""
        if rclpy.ok():
            self._publish_command()
        super().destroy_node()


def main(args=None):
    """Run the angular-motion calibration node."""
    rclpy.init(args=args)
    node: Optional[AngularMotionCalibrator] = None
    spin_thread: Optional[threading.Thread] = None
    try:
        node = AngularMotionCalibrator()
        spin_thread = threading.Thread(target=rclpy.spin, args=(node,))
        spin_thread.start()
        while rclpy.ok() and not node.complete.is_set():
            if not node.measurement_ready.wait(timeout=0.1):
                continue
            duration, direction = node.trials[node.trial_index]
            while True:
                try:
                    angle = float(input(
                        f'Measured signed angle after command '
                        f'{direction * K_OMEGA:+.3f} for {duration:.2f}s '
                        f'({"CCW" if direction > 0 else "CW"}), degrees '
                        f'(CCW +, CW -): '
                    ))
                    node.record_measurement(angle)
                    break
                except ValueError as error:
                    print(f'Invalid measurement: {error}')
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info(
                'Calibration interrupted; stopping the robot.'
            )
    finally:
        if node is not None and rclpy.ok():
            node._publish_command()
        if rclpy.ok():
            rclpy.shutdown()
        if spin_thread is not None and spin_thread.is_alive():
            spin_thread.join()
        if node is not None:
            node.destroy_node()


if __name__ == '__main__':
    main()
