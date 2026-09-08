from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROGRAM_PATH = PROJECT_ROOT / "통합_자동분류_실행.py"
CAMERA_TEST_PATH = PROJECT_ROOT / "카메라_공압_시험.py"
RIGHT_ARM_TEST_PATH = PROJECT_ROOT / "오른팔_동작_시험.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_program(path=PROGRAM_PATH, module_name="integrated_sorting"):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"프로그램을 불러올 수 없습니다: {PROGRAM_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


program = load_program()
camera_test = load_program(CAMERA_TEST_PATH, "camera_pneumatic_test")
right_arm_test = load_program(RIGHT_ARM_TEST_PATH, "right_arm_motion_test")


def scalar_reference_ik(
    camera_x_mm: float,
    camera_y_mm: float,
    target_z_mm: float,
    compression_mm: float,
    max_tilt_deg: float,
    forward_to_center_mm: float,
):
    """Original loop implementation retained only as a parity oracle."""
    robot_x = float(forward_to_center_mm) - float(camera_y_mm)
    robot_y = float(camera_x_mm)
    target_radius = float(np.hypot(robot_x, robot_y))
    if target_radius < program.ROBOT_TOOL_LEFT_OFFSET_MM:
        return None
    radial = float(np.sqrt(target_radius**2 - program.ROBOT_TOOL_LEFT_OFFSET_MM**2))
    yaw = float(
        np.arctan2(robot_y, robot_x)
        - np.arctan2(program.ROBOT_TOOL_LEFT_OFFSET_MM, radial)
    )
    j1 = 90.0 - float(np.degrees(yaw))
    if not 0.0 <= j1 <= 180.0:
        return None

    tilts = [0.0]
    for half_step in range(1, int(max_tilt_deg * 2.0) + 1):
        tilt = half_step * 0.5
        tilts.extend((tilt, -tilt))

    best = None
    for tilt in tilts:
        best_at_tilt = None
        for j2_half in range(361):
            j2 = j2_half * 0.5
            for j3_half in range(181):
                j3 = j3_half * 0.5
                a2 = j2 - 60.0 - j3
                j4 = tilt - a2
                if not 0.0 <= j4 <= 180.0:
                    continue
                a3 = -90.0 + tilt
                l3 = program.ROBOT_L3_MM - compression_mm
                rr = (
                    program.ROBOT_L1_MM * np.cos(np.radians(j2))
                    + program.ROBOT_L2_MM * np.cos(np.radians(a2))
                    + l3 * np.cos(np.radians(a3))
                )
                zz = (
                    program.ROBOT_J2_HEIGHT_MM
                    + program.ROBOT_L1_MM * np.sin(np.radians(j2))
                    + program.ROBOT_L2_MM * np.sin(np.radians(a2))
                    + l3 * np.sin(np.radians(a3))
                )
                dist = float(np.hypot(rr - radial, zz - target_z_mm))
                candidate = {
                    "j1": round(j1 * 2.0) / 2.0,
                    "j2": j2,
                    "j3": j3,
                    "j4": j4,
                    "tilt": tilt,
                    "error": dist,
                }
                if best_at_tilt is None or dist < best_at_tilt["error"]:
                    best_at_tilt = candidate
                if best is None or dist < best["error"]:
                    best = candidate
        if best_at_tilt is not None and best_at_tilt["error"] <= program.ROBOT_IK_TOLERANCE_MM:
            return best_at_tilt
    return best


class ArmInverseKinematicsTests(unittest.TestCase):
    def test_vectorized_search_matches_original_search(self):
        cases = (
            (0.0, 50.0, 46.0, 0.0, 15.0, 209.5),
            (0.0, 50.0, 30.0, 2.0, 15.0, 209.5),
            (75.0, 80.0, 50.0, 0.0, 15.0, 209.5),
            (-90.0, 35.0, 30.0, 4.0, 45.0, 209.5),
            (40.0, 60.0, 66.0, 8.0, 45.0, 205.5),
        )
        for case in cases:
            with self.subTest(case=case):
                expected = scalar_reference_ik(*case)
                actual = program._arm_ik(*case)
                if expected is None:
                    self.assertIsNone(actual)
                    continue
                self.assertIsNotNone(actual)
                for field in ("j1", "j2", "j3", "j4", "tilt"):
                    self.assertEqual(expected[field], actual[field])
                self.assertTrue(
                    math.isclose(expected["error"], actual["error"], abs_tol=1e-12)
                )

    def test_out_of_range_target_remains_unreachable(self):
        self.assertIsNone(program._arm_ik(0.0, 209.5, 30.0, 0.0, 15.0, 209.5))


class ArmSerialSafetyTests(unittest.TestCase):
    def test_hardware_safe_byte_delay_uses_second_staged_setting(self):
        self.assertEqual(0.012, program.ARM_SERIAL_BYTE_DELAY_SEC)

    def test_dual_arm_command_is_unchanged(self):
        program._ARM_PLAN_CACHE.clear()
        left = program.build_arm_pick_plan(
            {"robot_x_mm": 0.0, "robot_y_mm": 50.0, "robot_z_mm": 30.0},
            "left",
        )
        right = program.build_arm_pick_plan(
            {"robot_x_mm": 0.0, "robot_y_mm": -50.0, "robot_z_mm": 30.0},
            "right",
        )
        self.assertEqual(
            "M,1,30,94,72,36,28.5,68.5,39,35,68,39.5,34.5,70.5,38,27.5,"
            "1,30,90,73.5,38,29,70,41.5,36,70,42,35,72,40,28\n",
            program._build_dual_arm_line(left, right, sequential=False),
        )

    def test_camera_test_uses_simultaneous_firmware_command(self):
        plan = {
            "size": 30,
            "approach": {"j1": 90, "j2": 120, "j3": 56, "j4": 0},
            "contact": {"j2": 119, "j3": 57, "j4": 1},
            "preload": {"j2": 118, "j3": 58, "j4": 2},
            "lift": {"j2": 120, "j3": 56, "j4": 0},
            "arm": "left",
        }
        with patch.object(camera_test, "write_arm_line") as write_line:
            camera_test.send_dual_arm_plans(Mock(), plan, plan)

        command = write_line.call_args.args[1]
        self.assertTrue(command.startswith("M,"))
        self.assertEqual(31, len(command.rstrip("\n").split(",")))

    def test_camera_test_waits_for_emergency_acknowledgement(self):
        arm_ser = Mock()
        arm_ser.readline.side_effect = [b"", b"EMERGENCY_DONE\r\n"]

        with (
            patch.object(camera_test.cv2, "waitKey", return_value=ord("x")),
            patch.object(camera_test, "request_emergency_stop") as stop,
        ):
            result = camera_test.wait_arm_reply(arm_ser, ("DONE",), timeout_sec=1.0)

        stop.assert_called_once_with(arm_ser)
        self.assertEqual("EMERGENCY", result)

    def test_right_arm_test_accepts_emergency_completion(self):
        arm_ser = Mock()
        arm_ser.readline.return_value = b"EMERGENCY_DONE\r\n"

        with patch.object(right_arm_test.msvcrt, "kbhit", return_value=False):
            result = right_arm_test.wait_for_result(arm_ser)

        self.assertEqual("EMERGENCY_DONE", result)


class DisplayPerformanceTests(unittest.TestCase):
    def make_camera(self, show_depth_panel: bool):
        camera = program.D435Camera.__new__(program.D435Camera)
        camera.cfg = SimpleNamespace(show_depth_panel=show_depth_panel)
        camera.rect_detection_enabled = False
        camera.detection_overlay_enabled = False
        camera.last_tracked_objects = []
        camera._depth_to_colormap = Mock(
            return_value=np.zeros((2, 3, 3), dtype=np.uint8)
        )
        return camera

    def test_hidden_depth_panel_skips_unused_colormap(self):
        camera = self.make_camera(show_depth_panel=False)
        color = np.zeros((2, 3, 3), dtype=np.uint8)
        depth = np.ones((2, 3), dtype=np.uint16)

        display = camera._compose_display(color, depth)

        self.assertIs(display, color)
        camera._depth_to_colormap.assert_not_called()

    def test_visible_depth_panel_still_builds_colormap(self):
        camera = self.make_camera(show_depth_panel=True)
        color = np.zeros((2, 3, 3), dtype=np.uint8)
        depth = np.ones((2, 3), dtype=np.uint16)

        display = camera._compose_display(color, depth)

        self.assertEqual((2, 6, 3), display.shape)
        camera._depth_to_colormap.assert_called_once_with(depth)


if __name__ == "__main__":
    unittest.main()
