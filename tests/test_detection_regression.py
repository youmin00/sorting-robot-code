from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROGRAM_PATH = PROJECT_ROOT / "통합_자동분류_실행.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_program():
    spec = importlib.util.spec_from_file_location("detection_regression", PROGRAM_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"프로그램을 불러올 수 없습니다: {PROGRAM_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


program = load_program()


class SyntheticDetectionRegressionTests(unittest.TestCase):
    def test_pipeline_prefetch_positions_require_complete_and_stable_xy(self):
        positions = program.scene_robot_xy_positions
        stable = program.scene_positions_stable

        reference = positions([
            {"robot_x_mm": -20.0, "robot_y_mm": 40.0},
            {"robot_x_mm": 25.0, "robot_y_mm": 55.0},
        ])
        reordered_close = positions([
            {"robot_x_mm": 25.8, "robot_y_mm": 54.4},
            {"robot_x_mm": -19.2, "robot_y_mm": 40.6},
        ])
        moved = positions([
            {"robot_x_mm": -16.0, "robot_y_mm": 40.0},
            {"robot_x_mm": 25.0, "robot_y_mm": 55.0},
        ])

        self.assertIsNotNone(reference)
        self.assertIsNotNone(reordered_close)
        self.assertTrue(stable(reference, reordered_close, 2.0))
        self.assertFalse(stable(reference, moved, 2.0))
        self.assertIsNone(positions([{"robot_x_mm": 1.0, "robot_y_mm": None}]))

    def test_stable_pick_scene_requires_complete_xyz(self):
        positions = program.scene_robot_ready_xy_positions

        self.assertEqual(
            [(-20.0, 40.0)],
            positions([
                {
                    "robot_x_mm": -20.0,
                    "robot_y_mm": 40.0,
                    "robot_z_mm": 50.0,
                }
            ]),
        )
        self.assertIsNone(
            positions([
                {
                    "robot_x_mm": -20.0,
                    "robot_y_mm": 40.0,
                    "robot_z_mm": None,
                }
            ])
        )

    def test_stable_scene_waits_until_z_is_available(self):
        incomplete = {
            "id": 1,
            "robot_x_mm": -20.0,
            "robot_y_mm": 40.0,
            "robot_z_mm": None,
        }
        complete = dict(incomplete, robot_z_mm=50.0)

        class FakeCamera:
            def __init__(self):
                self.scenes = [[incomplete], [incomplete], [complete], [complete]]
                self.frame_count = 0
                self.last_tracked_objects = []
                self.last_selected_candidates = [{}]

            def get_frames(self):
                return None, None

            def _compose_display(self, _color, _depth):
                index = min(self.frame_count, len(self.scenes) - 1)
                self.last_tracked_objects = self.scenes[index]
                self.frame_count += 1
                return np.zeros((1, 1, 3), dtype=np.uint8)

            def _draw_runtime_mode_banner(self, _display):
                return None

            def _show_preview_and_statistics(self, _display):
                return None

            def _reset_tracking_state(self):
                self.last_tracked_objects = []
                self.last_selected_candidates = []

        camera = FakeCamera()
        with patch.object(program.cv2, "waitKey", return_value=-1):
            result = program.D435Camera._wait_for_stable_scene(
                camera,
                max_wait_sec=0.001,
                required_stable_frames=2,
            )

        self.assertGreaterEqual(camera.frame_count, 4)
        self.assertEqual(50.0, result[0]["robot_z_mm"])

    def test_empty_scene_waits_after_brief_object_evidence(self):
        ready = program.empty_scene_ready_after_grace

        self.assertFalse(ready(False, 1.0, 0.0, None, 1.25))
        self.assertTrue(ready(False, 1.25, 0.0, None, 1.25))
        self.assertFalse(ready(False, 2.0, 0.0, 1.0, 1.25))
        self.assertTrue(ready(False, 2.25, 0.0, 1.0, 1.25))
        self.assertTrue(ready(True, 0.1, 0.0, 0.1, 1.25))

    def test_depth_percentile_fallback_is_saved_for_robot_z(self):
        config = program.CameraConfig(center_roi_only=False, show_depth_panel=False)
        camera = program.D435Camera(config)
        camera.depth_scale = 0.001
        color = np.full((480, 848, 3), 100, dtype=np.uint8)
        depth = np.full((480, 848), 341, dtype=np.uint16)
        depth[180:271, 300:391] = 286

        with patch.object(camera, "_estimate_support_depth_m", return_value=None):
            camera._get_candidate_contours(color, depth)

        self.assertIsNotNone(camera.support_depth_m)
        self.assertAlmostEqual(0.341, camera.support_depth_m, places=3)

    def test_statistics_reset_hides_current_area_until_next_check(self):
        camera = program.D435Camera(program.CameraConfig())
        camera.last_tracked_objects = [{"id": 1}]
        camera.sorting_statistics.attempts = 4

        camera._reset_statistics_for_next_area("c")
        with patch.object(program.cv2, "imshow"):
            camera._show_preview_and_statistics(
                np.zeros((2, 2, 3), dtype=np.uint8)
            )

        self.assertEqual(0, camera.sorting_statistics.current_detected)
        self.assertEqual(0, camera.sorting_statistics.attempts)
        self.assertTrue(camera._start_statistics_for_next_area())
        self.assertFalse(camera._start_statistics_for_next_area())
        with patch.object(program.cv2, "imshow"):
            camera._show_preview_and_statistics(
                np.zeros((2, 2, 3), dtype=np.uint8)
            )
        self.assertEqual(1, camera.sorting_statistics.current_detected)

    def test_statistics_reset_during_work_is_deferred_until_next_area(self):
        camera = program.D435Camera(program.CameraConfig())
        camera.sorting_statistics.attempts = 4
        camera.sorting_statistics.successful = 2

        camera._queue_statistics_reset_for_next_area()
        self.assertEqual(4, camera.sorting_statistics.attempts)
        self.assertEqual(2, camera.sorting_statistics.successful)
        self.assertTrue(camera.statistics_reset_pending_for_next_area)

        self.assertTrue(camera._apply_statistics_reset_for_arriving_area())
        self.assertFalse(camera._apply_statistics_reset_for_arriving_area())
        self.assertEqual(0, camera.sorting_statistics.attempts)
        self.assertEqual(0, camera.sorting_statistics.successful)
        self.assertFalse(camera.statistics_reset_pending_for_next_area)

    def test_statistics_panel_uses_current_detected_size_counts(self):
        camera = program.D435Camera(program.CameraConfig())
        camera.slot_size_labels[0] = "3cm"
        camera.slot_size_labels[1] = "5cm"
        camera.last_tracked_objects = [
            {"id": 1, "robot_z_mm": 31.0},
            {"id": 2, "robot_z_mm": 48.0},
            {"id": 3, "robot_z_mm": 51.0},
        ]

        with patch.object(program.cv2, "imshow"):
            camera._show_preview_and_statistics(
                np.zeros((2, 2, 3), dtype=np.uint8)
            )

        self.assertEqual(3, camera.sorting_statistics.current_detected)
        self.assertEqual(
            {30: 1, 50: 2},
            camera.sorting_statistics.current_detected_by_size,
        )

    def test_two_object_scene_is_unchanged(self):
        config = program.CameraConfig(center_roi_only=False, show_depth_panel=False)
        camera = program.D435Camera(config)
        camera.depth_scale = 0.001
        camera.detection_overlay_enabled = True

        color = np.full((480, 848, 3), 100, dtype=np.uint8)
        depth = np.full((480, 848), 500, dtype=np.uint16)
        cv2.rectangle(color, (300, 180), (390, 270), (230, 230, 230), -1)
        depth[180:271, 300:391] = 450
        cv2.rectangle(color, (500, 200), (570, 270), (20, 20, 20), -1)
        depth[200:271, 500:571] = 470

        display = camera._compose_display(color, depth)

        self.assertEqual(
            [
                {"id": 1, "x": 345, "y": 225, "z": 0.45},
                {"id": 2, "x": 535, "y": 235, "z": 0.47000000000000003},
            ],
            camera.last_tracked_objects,
        )
        self.assertEqual(
            "depth-top-h3cm:1+depth-top-h5cm:1+"
            "depth-top(zbg=0.500,h=0.010~0.090,bw=0.010)+"
            "depth-band(z=0.500,bw=0.035)+"
            "depth-plane(dbg=0.500,dh=0.004)+depth-edge(th=0.0012)",
            camera.last_contour_mode,
        )
        self.assertEqual(
            "181843e3105f72f9526beaaf339433f6dcd5e7f7cf73cd166e5fd39f961bd2ff",
            hashlib.sha256(display.tobytes()).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
