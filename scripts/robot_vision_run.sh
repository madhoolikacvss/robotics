#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROS_SETUP="/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"

if [ ! -f "$ROS_SETUP" ]; then
  echo "ROS 2 setup file not found: $ROS_SETUP" >&2
  exit 1
fi

cd "$WORKSPACE_DIR"
source "$ROS_SETUP"

if [ ! -f install/setup.bash ]; then
  echo "Workspace is not built yet; run scripts/robot_vision_build_and_run.sh first." >&2
  exit 1
fi

source install/setup.bash
ros2 launch robot_vision_camera robot_vision_camera.launch.py
