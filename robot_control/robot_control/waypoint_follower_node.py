"""ROS 2 wrapper for the open-loop waypoint follower."""

import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from waypoint_follower import (
    FollowerConfig,
    Pose2D,
    WaypointFollower,
    load_waypoints,
)


class WaypointFollowerNode(Node):
    def __init__(self):
        super().__init__("waypoint_follower")

        self.declare_parameter("waypoint_file", "")
        self.declare_parameter("wheel_base", 0.16)
        self.declare_parameter("wheel_speed_per_command", 0.8)
        self.declare_parameter("max_command", 0.5)
        self.declare_parameter("max_linear_speed", 0.20)
        self.declare_parameter("max_angular_speed", 1.0)
        self.declare_parameter("distance_gain", 1.0)
        self.declare_parameter("heading_gain", 2.0)
        self.declare_parameter("position_tolerance", 0.05)
        self.declare_parameter("heading_tolerance_deg", 5.0)
        self.declare_parameter("initial_x", 0.0)
        self.declare_parameter("initial_y", 0.0)
        self.declare_parameter("initial_theta_deg", 0.0)
        self.declare_parameter("control_rate_hz", 20.0)

        filepath = self.get_parameter("waypoint_file").value
        if not filepath:
            raise ValueError("Set the waypoint_file parameter")

        waypoints = load_waypoints(filepath)
        get = lambda name: self.get_parameter(name).value
        cfg = FollowerConfig(
            wheel_base=float(get("wheel_base")),
            wheel_speed_per_command=float(get("wheel_speed_per_command")),
            max_command=float(get("max_command")),
            max_linear_speed=float(get("max_linear_speed")),
            max_angular_speed=float(get("max_angular_speed")),
            distance_gain=float(get("distance_gain")),
            heading_gain=float(get("heading_gain")),
            position_tolerance=float(get("position_tolerance")),
            heading_tolerance=math.radians(float(get("heading_tolerance_deg"))),
        )
        initial_pose = Pose2D(
            x=float(get("initial_x")),
            y=float(get("initial_y")),
            theta=math.radians(float(get("initial_theta_deg"))),
        )
        self.follower = WaypointFollower(waypoints, cfg, initial_pose)

        self.publisher = self.create_publisher(Float32MultiArray, "motor_commands", 10)
        self.period = 1.0 / float(get("control_rate_hz"))
        if self.period <= 0:
            raise ValueError("control_rate_hz must be positive")
        self.timer = self.create_timer(self.period, self.control_step)
        self.last_time = self.get_clock().now()

        self.get_logger().info(
            f"Loaded {len(waypoints)} waypoints from {filepath}; "
            "publishing [left, right] to motor_commands"
        )
        self.get_logger().warn(
            "Open-loop pose is only an estimate. Wheel slip and calibration errors accumulate."
        )

    def control_step(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now
        # Avoid integrating a huge jump after a pause/debugger stop.
        dt = max(0.0, min(dt, 0.2))

        left, right = self.follower.step(dt)
        msg = Float32MultiArray()
        msg.data = [float(left), float(right)]
        self.publisher.publish(msg)

        if self.follower.done:
            self.get_logger().info("All waypoints completed; publishing zero motor commands")
            self.timer.cancel()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = WaypointFollowerNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ValueError, FileNotFoundError) as exc:
        if node is not None:
            node.get_logger().error(str(exc))
        else:
            print(f"Unable to start waypoint follower: {exc}")
    finally:
        if node is not None:
            # Send an explicit stop command during shutdown.
            stop = Float32MultiArray()
            stop.data = [0.0, 0.0]
            node.publisher.publish(stop)
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
