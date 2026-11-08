from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Run the hardware stack with the open-loop waypoint controller."""
    motor_controller = Node(
        package='hw3',
        executable='motor_control',
        name='motor_control',
        output='screen',
        emulate_tty=True,
    )

    velocity_mapping = Node(
        package='hw3',
        executable='velocity_mapping',
        name='velocity_mapping',
        output='screen',
        emulate_tty=True,
    )

    dead_reckoning_controller = Node(
        package='hw3',
        executable='dead_reckoning_waypoint_controller',
        name='dead_reckoning_waypoint_controller',
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([
        motor_controller,
        velocity_mapping,
        dead_reckoning_controller,
    ])
