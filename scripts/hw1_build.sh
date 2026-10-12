#!/bin/bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select robot_control --symlink-install
source install/setup.bash
