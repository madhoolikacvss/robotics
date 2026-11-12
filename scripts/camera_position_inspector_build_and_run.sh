#!/bin/bash

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROS_SETUP="/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"

if [ ! -f "$ROS_SETUP" ]; then
  echo "ROS 2 setup file not found: $ROS_SETUP" >&2
  exit 1
fi

cd "$WORKSPACE_DIR"
source "$ROS_SETUP"
colcon build --packages-up-to \
  hw3 robot_vision_camera apriltag_ros --symlink-install
source install/setup.bash

pids=()
cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -INT "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  exit "$exit_code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

ros2 launch robot_vision_camera robot_vision_camera.launch.py \
  image_rectify:=true \
  camera_parameter_path:="$WORKSPACE_DIR/robot_vision/config/camera_parameter.yaml" \
  topic_name:=camera/image_rectified &
pids+=("$!")

ros2 launch apriltag_ros apriltag_launch.py \
  image_topic:=/camera/image_rectified \
  camera_info_topic:=/camera/camera_info &
pids+=("$!")

ros2 run hw3 camera_position_inspector &
pids+=("$!")

wait -n "${pids[@]}"
