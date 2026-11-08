from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """
    Launch file for the hw3 package.
    Starts three nodes:
    1. motor_control - communicates with robot hardware via serial
    2. velocity_mapping - converts Twist commands to motor commands
    3. hw3 - AprilTag EKF localization
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
    hw3 = Node(
        package='hw3',
        executable='hw3',
        name='apriltag_ekf_localizer',
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([
        motor_controller,
        velocity_mapping,
        hw3,
    ])