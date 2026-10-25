#!/bin/bash

set -e

source /opt/ros/jazzy/setup.bash
colcon build --packages-select hw_1_solution --symlink-install
source install/setup.bash
ros2 launch hw_1_solution hw1_solution.launch.py
