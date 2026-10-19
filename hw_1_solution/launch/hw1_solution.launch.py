from launch import LaunchDescription
from launch_ros.actions import Node
import time


def generate_launch_description():
    """
    Launch file for hw_1_solution package.
    Starts three nodes in sequence:
    1. motor_control - communicates with robot hardware via serial
    2. velocity_mapping - converts Twist commands to motor commands
    3. hw1_solution - PID-based waypoint navigation
    """
    
    # Node 1: Motor Controller (communicates with robot hardware)
    motor_controller = Node(
        package='hw_1_solution',
        executable='motor_control',
        name='motor_control',
        output='screen',
        emulate_tty=True,
    )
    
    # Node 2: Velocity to Motor Mapping
    velocity_mapping = Node(
        package='hw_1_solution',
        executable='velocity_mapping',
        name='velocity_mapping',
        output='screen',
        emulate_tty=True,
    )
    
    # Node 3: HW1 Solution (Waypoint Navigation with PID)
    hw1_solution = Node(
        package='hw_1_solution',
        executable='hw1_solution',
        name='hw1_solution',
        output='screen',
        emulate_tty=True,
    )
    
    return LaunchDescription([
        motor_controller,
        velocity_mapping,
        hw1_solution,
    ])