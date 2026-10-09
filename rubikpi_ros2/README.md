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

