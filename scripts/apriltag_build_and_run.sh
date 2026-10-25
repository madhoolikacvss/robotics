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
source "$ROS_SETUP"
colcon build --packages-up-to robot_vision_camera apriltag_ros --symlink-install
source install/setup.bash

camera_pid=""
cleanup() {
  if [ -n "$camera_pid" ] && kill -0 "$camera_pid" 2>/dev/null; then
    kill -INT "$camera_pid" 2>/dev/null || true
    wait "$camera_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

ros2 launch robot_vision_camera robot_vision_camera.launch.py \
  image_rectify:=true \
  camera_parameter_path:="$WORKSPACE_DIR/robot_vision/config/camera_parameter.yaml" \
  topic_name:=camera/image_rectified &
camera_pid=$!

ros2 launch apriltag_ros apriltag_launch.py \
  image_topic:=/camera/image_rectified \
  camera_info_topic:=/camera/camera_info
