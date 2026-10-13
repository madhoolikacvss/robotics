#!/bin/bash

set -e

source /opt/ros/jazzy/setup.bash
colcon build --packages-select robot_control --symlink-install
source install/setup.bash
ros2 launch robot_control hw1_launch.py
