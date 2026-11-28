"""Launch HW5 fixed-tag localization and position logging.

This launch does not move the robot; the coverage controller will be added
separately after the coverage path is designed.
"""

from launch import LaunchDescription
from launch_ros.actions import Node

from hw5.map_geometry import START


def generate_launch_description():
    frame = 'hw5_map'
    return LaunchDescription([
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
    ])
