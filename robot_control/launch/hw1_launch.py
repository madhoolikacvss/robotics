from launch import LaunchDescription
from launch.actions import RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node


def generate_launch_description():
    motor_node = Node(
        package='robot_control',
        executable='motor_control',
        name='motor_controller_node',
        output='screen',
        emulate_tty=True,
        parameters=[],
    )
    waypoint_follower_node = Node(
        package='robot_control',
        executable='waypoint_follower_node',
        name='waypoint_follower_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'waypoint_file': 'robotics/robot_control/robot_control/waypoints.txt',
            'wheel_base': 0.16,
            'wheel_speed_per_command': 0.8,
            'max_command': 0.5,
            'max_linear_speed': 0.2,
            'max_angular_speed': 1.0,
            'distance_gain': 1.0,
            'heading_gain': 2.0,
            'position_tolerance': 0.05,
            'heading_tolerance_deg': 5.0,
            'initial_x': 0.0,
            'initial_y': 0.0,
            'initial_theta_deg': 0.0,
            'control_rate_hz': 20.0,
        }],
    )

    # If the motor controller exits for any reason, also bring everything down.
    shutdown_when_motor_exits = RegisterEventHandler(
        OnProcessExit(
            target_action=motor_node,
            on_exit=[Shutdown(reason='Motor controller exited')]
        )
    )

    return LaunchDescription([
        motor_node,
        waypoint_follower_node,
        shutdown_when_motor_exits,
    ])
