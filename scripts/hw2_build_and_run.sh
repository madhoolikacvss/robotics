#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROS_SETUP="/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"

if [ ! -f "$ROS_SETUP" ]; then
  echo "ROS 2 setup file not found: $ROS_SETUP" >&2
  exit 1
fi

cd "$WORKSPACE_DIR"
set +u
source "$ROS_SETUP"
set -u
colcon build --packages-up-to hw2 --symlink-install
set +u
source install/setup.bash
set -u
ros2 launch hw2 hw2.launch.py
