"""Follow an RRT route using only its x/y waypoint coordinates."""

import math

import rclpy
from hw3.velocity_mapping import K_V
from hw3.waypoint_motion import (
    HEADING_TOLERANCE,
    wrap_angle,
)
from hw3.ekf_waypoint_controller import EKFWaypointController

from .map_geometry import USABLE_WORKSPACE
from .rrt import path_length, plan_rrt
from .rrt_waypoint_motion import RRTWaypointMotion
from .visibility_graph import contains_point


class RRTWaypointController(EKFWaypointController):
    def __init__(self):
        path = plan_rrt(seed=0, max_iterations=500)
        super().__init__()
        self.motion = RRTWaypointMotion(path)
        self.boundary_stopped = False
        self.get_logger().info(
            f'RRT route: {len(path)} points, '
            f'distance={path_length(path):.3f}m; '
            f'waypoints={self.motion.waypoints}'
        )

    def boundary_stop(self, reason):
        if not self.boundary_stopped:
            self.get_logger().error(f'Workspace stop: {reason}. Restart required.')
        self.boundary_stopped = True
        self.publish_stop()

    def control_loop(self):
        if self.boundary_stopped:
            self.publish_stop()
            return

        now = self._now_seconds()
        pose = self.latest_estimated_position
        if pose is not None:
            x, y, yaw = pose
            if (not all(math.isfinite(value) for value in pose)
                    or not contains_point(USABLE_WORKSPACE, (x, y))):
                self.boundary_stop(
                    f'estimated center ({x:.3f}, {y:.3f}) outside '
                    f'x=[{USABLE_WORKSPACE.x_min:.3f}, '
                    f'{USABLE_WORKSPACE.x_max:.3f}], '
                    f'y=[{USABLE_WORKSPACE.y_min:.3f}, '
                    f'{USABLE_WORKSPACE.y_max:.3f}]'
                )
                return

            pose_is_fresh = (
                self.last_pose_received_time is not None
                and now - self.last_pose_received_time <= self.pose_timeout
            )
            if pose_is_fresh and not self.motion.finished:
                target = self.motion.waypoints[
                    self.motion.current_waypoint_idx
                ]
                error = wrap_angle(
                    math.atan2(target[1] - y, target[0] - x) - yaw
                )
                would_drive = (
                    math.hypot(target[0] - x, target[1] - y)
                    > self.motion.position_tolerance
                    and (
                        self.motion.turn_complete
                        or abs(error) <= HEADING_TOLERANCE
                    )
                )
                if would_drive:
                    projected = (
                        x + K_V * 0.25 * math.cos(yaw),
                        y + K_V * 0.25 * math.sin(yaw),
                    )
                    if not contains_point(USABLE_WORKSPACE, projected):
                        self.boundary_stop(
                            'forward projection leaves usable workspace'
                        )
                        return

        super().control_loop()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RRTWaypointController()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.publish_stop()
            node.destroy_node()
        rclpy.shutdown()
