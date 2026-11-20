"""Follow the visibility-graph route using the existing HW3 controller."""

import math

import rclpy
from geometry_msgs.msg import Twist
from hw3.waypoint_motion import HEADING_TOLERANCE, POSITION_TOLERANCE, wrap_angle
from hw3.velocity_mapping import K_V
from .map_geometry import USABLE_WORKSPACE
from .visibility_graph import contains_point
from hw3.ekf_waypoint_controller import EKFWaypointController
from .dijkstra import shortest_path
from .path_waypoints import path_to_waypoints
from .visibility_graph import build_visibility_graph


class VisibilityWaypointController(EKFWaypointController):
    def __init__(self):
        # Compute the route before creating the controller's ROS interfaces.
        nodes, graph = build_visibility_graph()
        path, distance = shortest_path(graph)
        waypoints = path_to_waypoints(nodes, path)
        super().__init__()
        # Replace HW3's default route before spinning any control callbacks.
        self.motion.waypoints = waypoints
        self.drive_interval = float(self.declare_parameter('drive_interval', 1.0).value)
        self.heading_check_pause = float(self.declare_parameter('heading_check_pause', 0.5).value)
        if (not math.isfinite(self.drive_interval) or self.drive_interval <= 0
                or not math.isfinite(self.heading_check_pause) or self.heading_check_pause <= 0):
            raise ValueError('Driving interval and pause must be positive and finite.')
        self.drive_started = None
        self.pause_until = None
        self.boundary_stopped = False
        self.get_logger().info(
            f'HW4 route: {" -> ".join(path)}, distance={distance:.3f}m; '
            f'waypoints={waypoints}'
        )


    def boundary_stop(self, reason):
        if not self.boundary_stopped:
            self.get_logger().error(f'Workspace stop: {reason}. Restart required.')
        self.boundary_stopped = True
        self.publish_stop()

    def control_loop(self):
        now = self._now_seconds()
        if self.boundary_stopped:
            self.publish_stop()
            return
        pose = self.latest_estimated_position
        if pose is not None:
            x, y, yaw = pose
            if not all(math.isfinite(v) for v in pose) or not contains_point(USABLE_WORKSPACE, (x, y)):
                self.boundary_stop('estimated robot center outside usable workspace')
                return

        if (self.motion.finished or pose is None or self.last_pose_received_time is None
                or now - self.last_pose_received_time > self.pose_timeout):
            self.drive_started = None
            self.pause_until = None
            super().control_loop()
            return

        if self.pause_until is not None:
            self.desired_command = Twist()
            self.cmd_pub.publish(Twist())
            # Require a pose received after the settling pause.
            if now < self.pause_until or self.last_pose_received_time < self.pause_until:
                return
            target = self.motion.waypoints[self.motion.current_waypoint_idx]
            error = wrap_angle(math.atan2(target[1] - y, target[0] - x) - yaw)
            self.get_logger().info(f'Periodic heading check: error={math.degrees(error):.1f}deg')
            self.motion.reset_turn()
            self.motion.turn_complete = abs(error) <= HEADING_TOLERANCE
            self.pause_until = None
            self.drive_started = None

        if self.drive_started is not None and now - self.drive_started >= self.drive_interval:
            self.desired_command = Twist()
            self.cmd_pub.publish(Twist())
            self.pause_until = now + self.heading_check_pause
            self.drive_started = None
            self.get_logger().info('Pausing driving for heading check.')
            return

        # Check a 0.25-second forward projection before the inherited loop can
        # publish a drive command. Use mapped full speed, even near a waypoint.
        target = self.motion.waypoints[self.motion.current_waypoint_idx]
        error = wrap_angle(math.atan2(target[1] - y, target[0] - x) - yaw)
        would_drive = (self.motion.phase == 'position'
                       and math.hypot(target[0] - x, target[1] - y) > POSITION_TOLERANCE
                       and (self.motion.turn_complete or abs(error) <= HEADING_TOLERANCE))
        if would_drive:
            projected = (x + K_V * 0.25 * math.cos(yaw), y + K_V * 0.25 * math.sin(yaw))
            if not contains_point(USABLE_WORKSPACE, projected):
                self.boundary_stop('forward projection leaves usable workspace')
                return

        super().control_loop()
        if self.desired_command.linear.x > 0:
            # Hold straight driving until the scheduled check, preventing
            # repeated heading corrections between pauses.
            self.motion.turn_complete = True
            if self.drive_started is None:
                self.drive_started = now
        else:
            self.drive_started = None


def main(args=None):
    rclpy.init(args=args)
    node = VisibilityWaypointController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally: 
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()
