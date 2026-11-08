from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """
    Launch file for the hw3 package.
    Starts four nodes:
    1. motor_control - communicates with robot hardware via serial
    2. velocity_mapping - converts Twist commands to motor commands
    3. ekf_localizer - AprilTag EKF localization
    4. ekf_waypoint_controller - follows waypoints using the EKF pose
    """

    # Node 1: Motor Controller (communicates with robot hardware)
    motor_controller = Node(
        package='hw3',
        executable='motor_control',
        name='motor_control',
        output='screen',
        emulate_tty=True,
    )

    # Node 2: Velocity to Motor Mapping
    velocity_mapping = Node(
        package='hw3',
        executable='velocity_mapping',
        name='velocity_mapping',
        output='screen',
        emulate_tty=True,
    )

    # Node 3: AprilTag EKF Localization
    ekf_localizer = Node(
        package='hw3',
        executable='ekf_localizer',
        name='ekf_localizer',
        output='screen',
        emulate_tty=True,
    )

    waypoint_controller = Node(
        package='hw3',
        executable='ekf_waypoint_controller',
        name='ekf_waypoint_controller',
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([
        motor_controller,
        velocity_mapping,
        ekf_localizer,
        waypoint_controller,
    ])