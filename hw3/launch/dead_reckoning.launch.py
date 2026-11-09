from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Run the hardware stack with the open-loop waypoint controller."""
    dead_reckoning_controller = Node(
        package='hw3',
        executable='dead_reckoning_waypoint_controller',
        name='dead_reckoning_waypoint_controller',
        output='screen',
        emulate_tty=True,
    )
    return LaunchDescription([
        dead_reckoning_controller,
    ])
