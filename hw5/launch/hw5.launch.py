"""Launch HW5 fixed-tag localization, coverage control, and pose logging."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from hw5.map_geometry import START


def generate_launch_description():
    frame = 'hw5_map'
    return LaunchDescription([
        DeclareLaunchArgument(
            'visibility', default_value='false',
            description='Show the coverage path, tags, and safe boundary.',
        ),
        Node(
            package='hw5', executable='robot_position_logger',
            name='hw5_robot_position_logger', output='screen',
        ),
        Node(
            package='hw5', executable='known_tag_localizer',
            name='hw5_known_tag_localizer', output='screen',
            parameters=[{
                'world_frame': frame,
                'initial_x': START[0],
                'initial_y': START[1],
                'initial_yaw': 0.0,
            }],
        ),
        Node(
            package='hw5', executable='coverage_waypoint_controller',
            name='coverage_waypoint_controller', output='screen',
            parameters=[{
                'world_frame': frame,
                'position_log_path': 'hw5_robot_position.csv',
                'waypoint_log_path': 'hw5_waypoint_results.csv',
                'tag_position_log_path': 'hw5_tag_positions.csv',
                'visibility': LaunchConfiguration('visibility'),
            }],
        ),
    ])
