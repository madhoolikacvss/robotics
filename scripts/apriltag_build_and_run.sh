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
colcon build --packages-up-to robot_vision_camera apriltag_ros --symlink-install
source install/setup.bash

camera_pid=""
tf_pid=""
cleanup() {
  for pid in "$camera_pid" "$tf_pid"; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill -INT "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

ros2 run tf2_ros static_transform_publisher \
  --x 0.125 --y 0.0 --z 0.09 \
  --yaw -1.5708 --pitch 0.0 --roll -1.5708 \
  --frame-id base_link \
  --child-frame-id camera_frame &
tf_pid=$!

ros2 launch robot_vision_camera robot_vision_camera.launch.py \
  image_rectify:=true \
  camera_parameter_path:="$WORKSPACE_DIR/robot_vision/config/camera_parameter.yaml" \
  topic_name:=camera/image_rectified &
camera_pid=$!

ros2 launch apriltag_ros apriltag_launch.py \
  image_topic:=/camera/image_rectified \
  camera_info_topic:=/camera/camera_info
