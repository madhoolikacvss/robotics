from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """
    Launch the EKF localization and waypoint-controller nodes.
    Start motor_control and velocity_mapping separately with
    motor_mapping.launch.py.
    """
    ekf_localizer = Node(
        package='hw3',
        executable='ekf_localizer',
        name='ekf_localizer',
        output='screen',
        emulate_tty=True,
        arguments=['--ros-args', '--log-level', 'ekf_localizer:=DEBUG'],
    )

    waypoint_controller = Node(
        package='hw3',
        executable='ekf_waypoint_controller',
        name='ekf_waypoint_controller',
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([
        ekf_localizer,
        waypoint_controller,
    ])