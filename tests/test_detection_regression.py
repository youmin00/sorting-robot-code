from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest

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
            "9d499532a4624f8629c1ec7f32aa3e3465b6231e4591fd70dfb02476eb4b8ac5",
            hashlib.sha256(display.tobytes()).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
