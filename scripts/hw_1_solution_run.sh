#!/bin/bash

set -e

source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch hw_1_solution hw1_solution.launch.py
