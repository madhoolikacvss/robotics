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
colcon build --packages-up-to hw3 --symlink-install
set +u
source install/setup.bash
set -u

base_frame="${BASE_FRAME:-base_link}"
camera_frame="${CAMERA_FRAME:-camera_frame}"

tf_pid=""
cleanup() {
  if [ -n "$tf_pid" ] && kill -0 "$tf_pid" 2>/dev/null; then
    kill -INT "$tf_pid" 2>/dev/null || true
    wait "$tf_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

ros2 run tf2_ros static_transform_publisher \
  --x "${CAMERA_X:-0.04}" \
  --y "${CAMERA_Y:-0.0}" \
  --z "${CAMERA_Z:-0.125}" \
  --yaw "${CAMERA_YAW:--1.5708}" \
  --pitch "${CAMERA_PITCH:-0.0}" \
  --roll "${CAMERA_ROLL:--1.5708}" \
  --frame-id "$base_frame" \
  --child-frame-id "$camera_frame" &
tf_pid=$!

ros2 run hw3 tag_position_logger --ros-args \
  -p "base_frame:=$base_frame" "$@"
