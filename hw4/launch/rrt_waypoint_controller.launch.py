"""Launch localization and RRT-based waypoint tracking."""

from launch import LaunchDescription
from launch_ros.actions import Node

from hw4.map_geometry import START


def generate_launch_description():
    frame = 'hw4_map'
    return LaunchDescription([
        Node(
            package='hw4', executable='robot_position_logger',
            name='robot_position_logger', output='screen',
        ),
        Node(
            package='hw4', executable='known_tag_localizer',
            name='ekf_localizer', output='screen',
            parameters=[{
                'world_frame': frame,
                'initial_x': START[0],
                'initial_y': START[1],
                'initial_yaw': 0.0,
            }],
        ),
        Node(
            package='hw4', executable='rrt_waypoint_controller',
            name='rrt_waypoint_controller', output='screen',
            parameters=[{
                'world_frame': frame,
                'position_log_path': 'rrt_robot_position.csv',
                'waypoint_log_path': 'rrt_waypoint_results.csv',
                'tag_position_log_path': 'rrt_tag_positions.csv',
            }],
        ),
    ])
