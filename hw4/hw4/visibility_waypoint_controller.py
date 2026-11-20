"""Follow the visibility-graph route using the existing HW3 controller."""

import rclpy
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
        self.get_logger().info(
            f'HW4 route: {" -> ".join(path)}, distance={distance:.3f}m; '
            f'waypoints={waypoints}'
        )


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
