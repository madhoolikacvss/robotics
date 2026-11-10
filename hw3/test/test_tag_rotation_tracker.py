"""Synthetic fixed-tag observations; no ROS or robot required."""
import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location(
    'tag_rotation_tracker',
    Path(__file__).parents[1] / 'hw3' / 'tag_rotation_tracker.py',
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RotationTrackerTests(unittest.TestCase):
    def test_left_right_and_quaternion_sign(self):
        tracker = module.TagRotationTracker(np.eye(3))
        # For a fixed tag, camera-to-tag yaw decreases during a left turn.
        def observation(angle, sign=1):
            return [0, 0, sign*math.sin(-angle/2), sign*math.cos(angle/2)]
        tracker.observe('a', 1, observation(0), 0)
        row = tracker.observe('a', 1.1, observation(.2, -1), .25)
        self.assertAlmostEqual(row['camera_delta_yaw_rad'], .2)
        self.assertAlmostEqual(row['model_delta_yaw_rad'], .25)
        row = tracker.observe('a', 1.2, observation(.1), .15)
        self.assertAlmostEqual(row['camera_delta_yaw_rad'], -.1)
        self.assertAlmostEqual(row['camera_total_yaw_rad'], .1)

    def test_mount_and_unknown_tag_orientation(self):
        # Mount maps camera z onto robot x; robot yaw is camera -y rotation.
        mount = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        tracker = module.TagRotationTracker(mount)
        # Arbitrary fixed tag orientation: 90 degrees about camera x at start.
        q0 = np.array([math.sqrt(.5), 0, 0, math.sqrt(.5)])
        tracker.observe('a', 1, q0)
        # q_y(+angle) * q_x(pi/2) represents the new camera-tag rotation.
        a = .2; c = math.cos(a/2); s = math.sin(a/2); h = math.sqrt(.5)
        row = tracker.observe('a', 1.1, [c*h, s*h, -s*h, c*h])
        self.assertAlmostEqual(row['camera_delta_yaw_rad'], a)

    def test_resets_and_independent_tags(self):
        tracker = module.TagRotationTracker(np.eye(3))
        tracker.observe('a', 1, [0, 0, 0, 1])
        self.assertEqual(tracker.observe('a', 1, [0, 0, 0, 1])['status'],
                         'duplicate_or_out_of_order')
        self.assertEqual(tracker.observe('b', 1.1, [0, 0, 1, 0])['status'], 'reference')
        row = tracker.observe('a', 1.1, [0, 0, 1, 0])
        self.assertEqual(row['status'], 'orientation_jump_reset')
        self.assertNotIn('camera_delta_yaw_rad', row)
        self.assertEqual(tracker.observe('a', 2, [0, 0, 1, 0])['status'], 'gap_reset')
        self.assertEqual(tracker.observe('a', 2.1, [0, 0, 0, 0])['status'], 'invalid_orientation')
        self.assertEqual(tracker.observe('a', 2.2, [0, 0, 0, 1])['status'], 'reference')


if __name__ == '__main__':
    unittest.main()
