from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """
    Launch file for the hw2 package.
    Starts three nodes in sequence:
    1. motor_control - communicates with robot hardware via serial
    2. velocity_mapping - converts Twist commands to motor commands
    3. apriltag_waypoint_controller - AprilTag-based waypoint navigation
    """

    # Node 1: Motor Controller (communicates with robot hardware)
    motor_controller = Node(
        package='hw2',
        executable='motor_control',
        name='motor_control',
        output='screen',
        emulate_tty=True,
    )

    # Node 2: Velocity to Motor Mapping
    velocity_mapping = Node(
        package='hw2',
        executable='velocity_mapping',
        name='velocity_mapping',
        output='screen',
        emulate_tty=True,
    )

    # Node 3: AprilTag Waypoint Navigation
    apriltag_waypoint_controller = Node(
        package='hw2',
        executable='apriltag_waypoint_controller',
        name='apriltag_waypoint_controller',
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([
        motor_controller,
        velocity_mapping,
        apriltag_waypoint_controller,
    ])