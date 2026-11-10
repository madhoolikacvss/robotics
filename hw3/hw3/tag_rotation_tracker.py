"""Diagnostic-only rotation from successive observations of a fixed tag.

No world tag orientation or EKF state is used to calculate visual rotation.
Each tag has its own continuous segment; segments must not be summed together.
"""

import math
import numpy as np


FIELDS = [
    'timestamp_sec', 'observation_timestamp_sec', 'age_sec', 'tag', 'status',
    'segment', 'segment_start_sec', 'dt_sec', 'qx', 'qy', 'qz', 'qw',
    'camera_delta_yaw_rad', 'camera_total_yaw_rad', 'camera_yaw_rate_rad_s',
    'relative_rotation_rad', 'tilt_change_rad',
    'model_yaw_at_capture_rad', 'model_delta_yaw_rad', 'model_total_yaw_rad',
    'ekf_yaw_at_processing_rad', 'angular_command_at_processing',
]


class TagRotationTracker:
    """Track same-tag planar rotation, with explicit gap/jump resets."""

    def __init__(self, rotation_base_camera, max_gap=0.6,
                 max_rate=5.0, max_step=math.radians(45),
                 max_tilt=math.radians(20)):
        self.mount = np.array(rotation_base_camera, dtype=float)
        self.max_gap = max_gap
        self.max_rate = max_rate
        self.max_step = max_step
        self.max_tilt = max_tilt
        self.previous = {}
        self.segments = {}

    def reset(self):
        self.previous.clear()

    def observe(self, tag, stamp, quaternion, model_yaw=None):
        """Return one CSV record; invalid samples never produce visual motion."""
        q = np.asarray(quaternion, dtype=float)
        row = dict(zip(('qx', 'qy', 'qz', 'qw'), q))
        row['model_yaw_at_capture_rad'] = model_yaw
        old = self.previous.get(tag)
        if old is not None and stamp <= old['stamp']:
            row['status'] = 'duplicate_or_out_of_order'
            return row
        if q.shape != (4,) or not np.all(np.isfinite(q)) or np.linalg.norm(q) < 1e-9:
            self.previous.pop(tag, None)
            row['status'] = 'invalid_orientation'
            return row
        x, y, z, w = q / np.linalg.norm(q)
        camera_tag = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
        ])
        current = self.mount @ camera_tag
        status = 'reference' if old is None else 'gap_reset'
        if old is not None and old['model_start'] is None and model_yaw is not None:
            # The first image can predate EKF startup. Re-anchor both totals
            # once prediction history covers capture time, rather than compare
            # visual and model totals with different reference timestamps.
            old = None
            status = 'model_reference_reset'
        total = 0.0
        start = stamp
        model_start = model_yaw
        if old is not None:
            dt = stamp - old['stamp']
            row['dt_sec'] = dt
            if dt <= self.max_gap:
                # R_Bprev_Tag * R_Bcurrent_Tag.T = R_Bprev_Bcurrent.
                relative = old['rotation'] @ current.T
                yaw = math.atan2(relative[1, 0], relative[0, 0])
                angle = math.acos(float(np.clip((np.trace(relative)-1)/2, -1, 1)))
                tilt = math.acos(float(np.clip(relative[2, 2], -1, 1)))
                row.update(relative_rotation_rad=angle, tilt_change_rad=tilt)
                if (angle > self.max_step or angle > self.max_rate*dt + math.radians(3)
                        or tilt > self.max_tilt):
                    status = 'orientation_jump_reset'
                else:
                    status = 'tracked'
                    total = old['total'] + yaw
                    start = old['start']
                    model_start = old['model_start']
                    row.update(camera_delta_yaw_rad=yaw,
                               camera_yaw_rate_rad_s=yaw/dt)
                    if model_yaw is not None and old['model'] is not None:
                        row['model_delta_yaw_rad'] = model_yaw - old['model']
        if status != 'tracked':
            self.segments[tag] = self.segments.get(tag, 0) + 1
        row.update(status=status, segment=self.segments[tag],
                   segment_start_sec=start, camera_total_yaw_rad=total)
        if model_yaw is not None and model_start is not None:
            row['model_total_yaw_rad'] = model_yaw - model_start
        self.previous[tag] = dict(stamp=stamp, rotation=current, total=total,
                                  start=start, model=model_yaw, model_start=model_start)
        return row
