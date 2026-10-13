#!/bin/bash

set -e

source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch robot_control hw1_launch.py
