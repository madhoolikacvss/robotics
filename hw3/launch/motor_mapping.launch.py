from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
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

    return LaunchDescription([
        motor_controller,
        velocity_mapping,
    ])
