# ASIGNMENT 1 TODO
## Three matrix caliberations 
### Wheel speed (linear/angular vel) matrix
Need to use keyboard_control.py to control robot 
v = [r/2 r/2][angular_l angular_r].T
w = [r/L_w -r/L_w][angular_l angular_r].T

r= wheel radius in m 
L_w = trackwidth 
angular vel in rad/s

since the keyboard sends unitless values u_l, u_r in [-0.5, 0.5] we need to get k_v, k_w where:
v = k_v(u_l, u_r) and w = k_w(u_r-u_l)

Command W for a measured number of seconds, measure distance traveled, solve for k_v
​
Command A for a measured time, measure yaw change with the IMU, solve for k_w. Do this once on the floor with a tape measure and a stopwatch.

### IMU yaw sign and offset

The IMU outputs yaw in degrees (motor_control.py line: yaw = float(imu_data.get('y', 0))). Two things to verify:
1. Sign convention: Does +yaw correspond to CCW (ROS REP-103) or CW?
2. Zero offset: Does the robot's "forward" direction (per motor command W) read as 0 deg on the IMU?

Need to place the robot at a known heading, record IMU yaw. Command A (left turn), watch yaw. If yaw goes negative, IMU uses NED convention (CW positive) and you need to negate it in subscriber.

### integration
x_k+1 = x_k + v_k**cos(theta_k)**time
y_k+1 = y_k + v_k**sin(theta_k)**time

thtata needs to be taken from IMU

## Robot vision camera

On the Rubik Pi, build and launch the camera node with:

```sh
./scripts/robot_vision_build_and_run.sh
```

To launch it again without rebuilding:

```sh
./scripts/robot_vision_run.sh
```

Both scripts use ROS 2 Jazzy by default; set `ROS_DISTRO` if the installation
uses another ROS 2 distribution. The launch file starts the `robot_vision_camera`
node and Foxglove bridge. The camera node uses GStreamer's `qtiqmmfsrc`, so it
requires the Rubik Pi camera source plugin and is not a generic USB-camera driver.


# Assignment 2 TODO:

### Step 1: Define and Verify the Transform Chain
ros2 run tf2_ros static_transform_publisher \
  --x 0.05 --y 0.0 --z 0.10 \
  --yaw -1.5708 --pitch 0.0 --roll -1.5708 \
  --frame-id base_link --child-frame-id camera_optical_frame

verify with: ros2 run tf2_ros tf2_echo base_link camera_optical_frame

another verify: start aptriltag_ros and run ros2 run tf2_ros tf2_echo base_link tag_0
```
world → tag_X (constant, hardcoded from measurement)
camera_optical_frame → tag_X (dynamic, from apriltag_ros TF)
base_link → camera_optical_frame (constant, measured from mounting)
```


### Step 2: Implement the Localization Function (Tag → World Pose)

`lookup_transform` returns the **last cached** transform even when no new detection has occurred. You must check the timestamp of the returned `TransformStamped` against the current time. If it is older than ~0.5 s, treat it as "no tag visible" and fall back to dead reckoning. This is the single most common bug when using `apriltag_ros` for localization.

**check:** Place the robot at a known position relative to a tag, run the node, and print the computed `(x, y, yaw)`. Compare with the ruler-measured truth. If it is off by a constant rotation (e.g., 90°), `T_base_cam` orientation is wrong.

Stuff added to HW solution:
TF buffer/listener — added Buffer and TransformListener so the node can read apriltag_ros's TF output.

2. get_pose_from_tags() — loops over TAG_WORLD, tries lookup_transform('camera_optical_frame', 'tag_X'), checks freshness (0.5 s), then computes T_world_base = T_world_tag · inv(T_cam_tag) · T_base_cam.

3. T_BASE_CAM — hardcoded from your measurement (4 cm forward, 12.5 cm high) plus the forward-facing optical-frame rotation. Verify this with a physical test before running the full trajectory.

4. update_pose() — visual pose is smoothed with vis_alpha; when no fresh tag, falls back to update_dead_reckoning(). last_vis_pose is reset to None when no tag is seen, so the first frame after re-acquisition overwrites directly instead of blending with a stale estimate.

5. Waypoint error logging — when each waypoint is declared reached, the position and yaw error are recorded and printed.

6. Total distance/rotation — accumulated from commanded twist every tick.

7. Trajectory sampling — appends (time, x, y, yaw) roughly every 0.5 s.

8. Pose printing — every control tick prints the current pose and whether a tag was visible this frame.

### Step 3: Integrate Visual Pose with Dead Reckoning (Noise Handling) - When No Landmark Is Detected

Fallback is **dead reckoning using the commands you send**. Your `hw1_solution.py` already has `update_dead_reckoning(linear_vel, angular_vel)`, which integrates the commanded twist. Keep it as the fallback.

**Important caveat:** this is open-loop dead reckoning (using commanded velocity, not wheel encoders). It will drift more than encoder-based odometry. Since you have no encoders exposed, this is the only option. Mention in the report that the drift during tag-less periods is bounded by the short duration between waypoints (your trajectory segments are 1–2 m, and at typical speeds the robot crosses them in a few seconds), so the drift is acceptable before the next tag resets the pose.

**Optional refinement:** If you want smoother behavior during tag-less periods, use the **last visual pose + dead reckoning** rather than pure dead reckoning from the start. That is already what the `else` branch does if you keep `self.current_state` from the last visual update.

### Step 4: Record Localization Error Dynamically

You need **ground truth** to compute error. Since you have no external tracking system, use the waypoints themselves as ground-truth checkpoints:

**At each waypoint:**
- When the controller declares the waypoint reached, record:
  - `t = now`
  - `pose_est = self.current_state` (the visual/dead-reckoned estimate)
  - `pose_true = self.waypoints[current_waypoint_idx]` (the known waypoint coordinate)
  - `pos_error = ||pose_est[:2] - pose_true[:2]||`
  - `yaw_error = wrap(pose_est[2] - pose_true[2])`

Write these to a CSV file or print them. This directly satisfies requirement (h) "location error (both rotation and translation) on each waypoint."

**If you also want a continuous error trace** for the trajectory plot, you would need a denser ground truth. Without an external tracker, the best you can do is:
- Interpolate ground truth along the straight-line segments between waypoints (since the robot is supposed to move in straight lines between waypoints).
- At each control tick, find the nearest point on the ideal path and compute perpendicular distance.

This is approximate but defensible: the assignment expects you to estimate error from your algorithm, not from an external motion-capture system. A simpler alternative that is fully honest: **record the estimated pose every 0.5 s, plot it as the trajectory (requirement i), and report the waypoint errors (requirement h) separately.** The trajectory plot does not require ground truth—it just shows where your algorithm thinks the robot went.

### Step 5: Total Moving Distance (Including Rotation)

Accumulate two quantities at every control tick (`dt = 0.1 s`):

```python
self.total_distance += abs(linear_vel) * dt
self.total_rotation += abs(angular_vel) * dt
```

If you want to follow the "including rotation" requirement more literally, you can convert rotation to an equivalent distance using the wheel radius (e.g., `|ω| * dt * wheel_radius`), but the assignment likely just wants both numbers reported. A ROS message `pr2_mechanism_controllers/Odometer` defines exactly `distance` and `angle` as separate fields , and open-source odometry code also tracks `total_distance_traveled` and `total_angle_traveled` separately . Report both.

**Important:** use the **commanded** velocity (the twist you publish), not the visual pose difference, because the visual pose is intermittent. This also matches what the robot actually attempted to do.

### Step 6: Path Tracking with the Corrected Pose

Your existing `hw1_solution.py` already has a three-stage controller (rotate to goal → drive → rotate to orient). You do **not** need to rewrite it. Just replace the pose source:

- In `control_loop`, before the stage logic, call `self.update_pose(twist_msg.linear.x, twist_msg.angular.z)` instead of `self.update_dead_reckoning(...)`.
- Keep everything else (PID, waypoint tolerance, stage transitions) unchanged.

**One adjustment:** if the visual pose is noisy, the controller may oscillate when close to a waypoint. Increase the position tolerance slightly (e.g., from 0.05 to 0.08 m) or add a small deadband on the heading error. Also consider **not updating the pose during the `rotate_to_orient` stage** if the robot is stationary and no tag is visible—dead reckoning of zero velocity adds nothing but noise.

### Step 7: Report Items

| Report requirement | How to fulfill |
|---|---|
| (a) Controller graph | Draw the chain: `AprilTag TF → Pose Estimator → PID/Stage Controller → /cmd_vel → Robot`. Label edges with `T_world_base`, `Twist`. |
| (b) Landmark placement | One tag per waypoint, laid flat on the ground, pattern facing up. Rationale: each waypoint provides a correction opportunity. Optional intermediate tags shorten drift periods. |
| (c) Pose estimation & noise | Describe `T_world_base = T_world_tag · inv(T_cam_tag) · T_base_cam`. Noise handled by PnP + bias calibration + low-pass filter + outlier rejection. |
| (d) No landmark in FOV | Fall back to dead reckoning from commanded velocity; bounded drift because segments are short; pose resets when next tag is seen. |
| (f) Smoothness | Plot commanded `linear.x` and `angular.z` over time. Discuss whether the three-stage controller causes stop-and-go (it does, at each waypoint) and whether that is acceptable. |
| (g) Total distance | Report `total_distance` and `total_rotation` accumulated from commanded velocities. |
| (h) Waypoint errors | Table of `(x_err, y_err, yaw_err)` at each of the 4 waypoints. |
| (i) 2D trajectory | Log `self.current_state[:2]` every 0.5 s during the run, plot with matplotlib. |
| (j) Reflection | Discuss the bias calibration impact, the tag-less drift, and any oscillation near waypoints. |

### Execution Order (Fastest Path)

1. Verify static TF `base_link → camera_optical_frame` (Step 1).
2. Write and test `get_pose_from_tags()` in isolation: print the computed pose while manually moving the robot near a tag (Step 2).
3. Wire it into `hw1_solution.py` with the visual-overwrite / dead-reckoning fallback (Steps 3–4).
4. Add logging: waypoint errors, total distance/rotation, trajectory samples (Steps 5–6).
5. Run the 4-waypoint trajectory, record data, generate plots.
6. Tune only if the robot oscillates or misses waypoints badly: adjust `alpha` in the low-pass filter, or increase tolerances slightly.

The only thing that could derail this plan is an incorrect `T_base_cam` orientation. Test that first before touching the controller.