"""
Intel RealSense D435 camera setup script.

Usage:
  python "카메라_공압_시험.py"
  python "카메라_공압_시험.py" --serial <DEVICE_SERIAL> --width 640 --height 480 --fps 30

Controls:
  q / ESC : quit
  x       : emergency stop
  z       : reset the camera and both robot arms
  p       : start automatic sorting of detected reachable cubes
"""

from __future__ import annotations

import argparse
import os
import time
import serial
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError as exc:
    raise SystemExit("opencv-python is required. Install: pip install opencv-python") from exc

try:
    import pyrealsense2 as rs
except ImportError as exc:
    raise SystemExit("pyrealsense2 is required. Install: pip install pyrealsense2") from exc

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None

MAX_TRACKED_OBJECTS = 8

STM32_PORT = "COM5"
STM32_BAUD = 115200
AUTO_START_CONVEYOR = True
AUTO_REPLY_EMPTY = True

CAMERA_SETTLE_SEC = 2.0
CAMERA_SETTLE_FRAMES = 20

EMPTY_CONFIRM_FRAMES = 12
OBJECT_REMOVED_CONFIRM_FRAMES = 12
MULTI_OBJECT_MISSING_CONFIRM_FRAMES = 24
OBJECT_RECHECK_SEC = 0.3
OBJECT_SLOT_MATCH_PX = 65
OBJECT_REMOVED_MATCH_PX = 65

ROBOT_X_SCALE = 0.907
ROBOT_Y_SCALE = 0.870
ROBOT_X_OFFSET_CM = 0.0
ROBOT_Y_OFFSET_CM = 0.0
ROBOT_XY_REFERENCE_HEIGHT_M = 0.03
ROBOT_TALL_OBJECT_Y_GAIN = 1.06

# Camera, robot-arm, and pneumatic-only settings.
ROBOT_ARM_PORT = "COM3"
ROBOT_ARM_BAUD = 115200
ROBOT_FORWARD_TO_CENTER_MM = 209.5
ROBOT_TOOL_LEFT_OFFSET_MM = 12.0
ROBOT_L1_MM = 130.0
ROBOT_L2_MM = 130.0
ROBOT_L3_MM = 76.0
ROBOT_J2_HEIGHT_MM = 51.3
ROBOT_IK_TOLERANCE_MM = 3.0
ROBOT_APPROACH_J4_OFFSET_DEG = 4.5
ROBOT_PRELOAD_J4_OFFSET_DEG = 3.0
ROBOT_J1_FINE_OFFSET_DEG = -0.5
ROBOT_POSITIVE_X_J1_OFFSET_DEG = -4.0
EDGE_3CM_MIN_CAMERA_X_MM = 90.0
EDGE_3CM_MIN_CAMERA_Y_MM = 75.0
EDGE_3CM_EXTRA_DROP_MM = 4.0


def _arm_ik(camera_x_mm: float, camera_y_mm: float, target_z_mm: float,
            compression_mm: float, max_tilt_deg: float) -> Optional[dict]:
    """Same 0.5-degree IK used by the verified single-cube simulator."""
    robot_x = ROBOT_FORWARD_TO_CENTER_MM - float(camera_y_mm)
    robot_y = float(camera_x_mm)
    target_radius = float(np.hypot(robot_x, robot_y))
    if target_radius < ROBOT_TOOL_LEFT_OFFSET_MM:
        return None
    radial = float(np.sqrt(target_radius ** 2 - ROBOT_TOOL_LEFT_OFFSET_MM ** 2))
    yaw = float(np.arctan2(robot_y, robot_x) - np.arctan2(ROBOT_TOOL_LEFT_OFFSET_MM, radial))
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
                l3 = ROBOT_L3_MM - compression_mm
                rr = (
                    ROBOT_L1_MM * np.cos(np.radians(j2))
                    + ROBOT_L2_MM * np.cos(np.radians(a2))
                    + l3 * np.cos(np.radians(a3))
                )
                zz = (
                    ROBOT_J2_HEIGHT_MM
                    + ROBOT_L1_MM * np.sin(np.radians(j2))
                    + ROBOT_L2_MM * np.sin(np.radians(a2))
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
        if best_at_tilt is not None and best_at_tilt["error"] <= ROBOT_IK_TOLERANCE_MM:
            return best_at_tilt
    return best


def object_in_left_workspace(obj: dict) -> bool:
    if any(obj.get(key) is None for key in ("robot_x_mm", "robot_y_mm", "robot_z_mm")):
        return False
    camera_x = float(obj["robot_x_mm"])
    camera_y = float(obj["robot_y_mm"])
    measured_z = float(obj["robot_z_mm"])
    cube_size = 30 if abs(measured_z - 30.0) <= abs(measured_z - 50.0) else 50
    half = cube_size / 2.0
    return (
        camera_y >= 0.0
        and camera_y + half <= 105.0
        # X는 물체 전체 폭이 아니라 물체 중심 좌표로 작업영역을 구분한다.
        and -120.0 <= camera_x <= 120.0
    )


def object_in_right_workspace(obj: dict) -> bool:
    if any(obj.get(key) is None for key in ("robot_x_mm", "robot_y_mm", "robot_z_mm")):
        return False
    camera_x = float(obj["robot_x_mm"])
    camera_y = float(obj["robot_y_mm"])
    measured_z = float(obj["robot_z_mm"])
    cube_size = 30 if abs(measured_z - 30.0) <= abs(measured_z - 50.0) else 50
    half = cube_size / 2.0
    return (
        camera_y < 0.0
        and camera_y - half >= -105.0
        and -120.0 <= camera_x <= 120.0
    )


def build_arm_pick_plan(obj: dict, arm: str = "left") -> dict:
    global_camera_x = float(obj["robot_x_mm"])
    global_camera_y = float(obj["robot_y_mm"])
    measured_z = float(obj["robot_z_mm"])
    cube_size = 30 if abs(measured_z - 30.0) <= abs(measured_z - 50.0) else 50
    if arm == "right":
        if not object_in_right_workspace(obj):
            raise ValueError("큐브 전체가 오른쪽 로봇 작업영역 안에 있지 않습니다.")
        camera_x = -global_camera_x
        camera_y = -global_camera_y
    else:
        if not object_in_left_workspace(obj):
            raise ValueError("큐브 전체가 왼쪽 로봇 작업영역 안에 있지 않습니다.")
        camera_x = global_camera_x
        camera_y = global_camera_y
    top = float(cube_size)
    edge_extra_drop = (
        EDGE_3CM_EXTRA_DROP_MM
        if cube_size == 30
        and camera_x >= EDGE_3CM_MIN_CAMERA_X_MM
        and camera_y >= EDGE_3CM_MIN_CAMERA_Y_MM
        else 0.0
    )
    contact_z = top - edge_extra_drop
    approach = _arm_ik(camera_x, camera_y, top + 16.0, 0.0, 15.0)
    contact = _arm_ik(camera_x, camera_y, contact_z, 0.0, 15.0)
    preload_compression = 2.0
    preload = _arm_ik(camera_x, camera_y, contact_z, preload_compression, 15.0)
    for _ in range(2):
        if preload is None:
            break
        adjusted = min(5.0, 2.0 + abs(float(preload["tilt"])) * 0.2)
        if abs(adjusted - preload_compression) < 0.01:
            break
        preload_compression = adjusted
        preload = _arm_ik(camera_x, camera_y, contact_z, preload_compression, 15.0)
    lift = _arm_ik(camera_x, camera_y, top + 16.0, 8.0, 45.0)
    poses = (approach, contact, preload, lift)
    if any(p is None or p["error"] > ROBOT_IK_TOLERANCE_MM for p in poses):
        raise ValueError("접근·접촉·예압·상승 자세 중 도달할 수 없는 단계가 있습니다.")

    # 실기 캘리브레이션:
    # - 상부 접근·접촉 J4 +4.5°, 수직이 맞는 예압 J4 +3°
    # - 피킹 J1은 항상 -0.5°, 물체 중심 Camera X가 양수면 추가로 -4°
    for pose in (approach, contact):
        pose["j4"] = float(np.clip(
            pose["j4"] + ROBOT_APPROACH_J4_OFFSET_DEG, 0.0, 180.0
        ))
    preload["j4"] = float(np.clip(
        preload["j4"] + ROBOT_PRELOAD_J4_OFFSET_DEG, 0.0, 180.0
    ))

    j1_offset = ROBOT_J1_FINE_OFFSET_DEG
    if camera_x > 0.0:
        j1_offset += ROBOT_POSITIVE_X_J1_OFFSET_DEG
    for pose in poses:
        pose["j1"] = float(np.clip(pose["j1"] + j1_offset, 0.0, 180.0))

    return {
        "arm": arm,
        "size": cube_size,
        "camera_x": global_camera_x,
        "camera_y": global_camera_y,
        "local_camera_x": camera_x,
        "local_camera_y": camera_y,
        "edge_extra_drop": edge_extra_drop,
        "approach": approach,
        "contact": contact,
        "preload": preload,
        "lift": lift,
    }


def write_arm_line(arm_ser, line: str) -> None:
    arm_ser.reset_input_buffer()
    # The current BSP UART receiver is polling one byte at a time. Pace the
    # USB-VCP bytes so the STM32 RDR is not overrun by one large PC-side burst.
    for byte in line.encode("ascii"):
        arm_ser.write(bytes((byte,)))
        arm_ser.flush()
        time.sleep(0.015)


def send_arm_plan(arm_ser, plan: dict) -> None:
    poses = [plan["approach"], plan["contact"], plan["preload"], plan["lift"]]
    values = [float(plan["size"]), poses[0]["j1"]]
    for pose in poses:
        values.extend((pose["j2"], pose["j3"], pose["j4"]))
    line = "P," + ",".join(f"{value:.1f}" for value in values) + "\n"
    write_arm_line(arm_ser, line)
    log_status(
        f"로봇팔 전송: {plan['size']} mm, Camera X={plan['camera_x']:.1f}, "
        f"Y={plan['camera_y']:.1f} mm, 추가 하강={plan['edge_extra_drop']:.1f} mm",
        PHASE_SORTING,
    )


def _plan_values(plan: Optional[dict]) -> list[float]:
    if plan is None:
        return [0.0] * 14
    values = [float(plan["size"]), float(plan["approach"]["j1"])]
    for pose_name in ("approach", "contact", "preload", "lift"):
        pose = plan[pose_name]
        values.extend((float(pose["j2"]), float(pose["j3"]), float(pose["j4"])))
    return values


def send_dual_arm_plans(arm_ser, left_plan: Optional[dict], right_plan: Optional[dict]) -> None:
    values = [1.0 if left_plan is not None else 0.0]
    values.extend(_plan_values(left_plan))
    values.append(1.0 if right_plan is not None else 0.0)
    values.extend(_plan_values(right_plan))
    # Current firmware reserves D for the one-arm-at-a-time cross pipeline.
    # This test waits for DONE and may send both plans, so it must use M.
    line = "M," + ",".join(f"{value:.1f}" for value in values) + "\n"
    write_arm_line(arm_ser, line)
    sides = []
    if left_plan is not None:
        sides.append(f"왼팔 {left_plan['size']}mm")
    if right_plan is not None:
        sides.append(f"오른팔 {right_plan['size']}mm")
    log_status("양팔 계획 전송: " + ", ".join(sides), PHASE_SORTING)


def request_emergency_stop(arm_ser) -> None:
    log_warning("긴급 정지: 양팔을 즉시 멈추고 공압을 끕니다.")
    if arm_ser is not None:
        write_arm_line(arm_ser, "X\n")


def request_dual_arm_reset(arm_ser) -> None:
    if arm_ser is not None:
        write_arm_line(arm_ser, "H\n")


def wait_arm_reply(
    arm_ser,
    final_replies: tuple[str, ...],
    timeout_sec: float = 120.0,
    allow_emergency_key: bool = True,
) -> Optional[str]:
    deadline = time.time() + timeout_sec
    emergency_requested = False
    while time.time() < deadline:
        if allow_emergency_key and (cv2.waitKey(1) & 0xFF) == ord("x"):
            request_emergency_stop(arm_ser)
            emergency_requested = True
            allow_emergency_key = False
            deadline = time.time() + 60.0
        reply = arm_ser.readline().decode("utf-8", errors="replace").strip()
        if not reply:
            continue
        log_status(f"로봇팔: {reply}", PHASE_SORTING)
        if emergency_requested:
            if reply == "EMERGENCY_DONE":
                return "EMERGENCY"
            continue
        if reply in final_replies:
            return reply
    log_warning("로봇팔 응답 시간이 초과되었습니다.")
    return None


PHASE_SYSTEM = "시스템"
PHASE_BELT_MOVING = "밸트 이동중"
PHASE_DETECTING = "물체 감지중"
PHASE_SORTING = "분류작업중"
PHASE_DONE = "분류완료"

ANSI_RESET = "\033[0m"
ANSI_BRIGHT_CYAN = "\033[96m"
ANSI_BRIGHT_GREEN = "\033[92m"
ANSI_BRIGHT_YELLOW = "\033[93m"
ANSI_BRIGHT_MAGENTA = "\033[95m"
ANSI_BRIGHT_WHITE = "\033[97m"
ANSI_BRIGHT_RED = "\033[91m"

PHASE_COLORS = {
    PHASE_SYSTEM: ANSI_BRIGHT_WHITE,
    PHASE_BELT_MOVING: ANSI_BRIGHT_GREEN,
    PHASE_DETECTING: ANSI_BRIGHT_CYAN,
    PHASE_SORTING: ANSI_BRIGHT_YELLOW,
    PHASE_DONE: ANSI_BRIGHT_MAGENTA,
}


def color_tag(tag: str, color: str) -> str:
    return f"{color}[{tag}]{ANSI_RESET}"


def log_status(message: str, phase: str = PHASE_SYSTEM) -> None:
    print(f"{color_tag(phase, PHASE_COLORS.get(phase, ANSI_BRIGHT_WHITE))} {message}", flush=True)


def log_warning(message: str) -> None:
    print(f"{color_tag('주의', ANSI_BRIGHT_RED)} {message}", flush=True)


def log_save(message: str) -> None:
    print(f"{color_tag('저장', ANSI_BRIGHT_MAGENTA)} {message}", flush=True)


def clean_stm32_message(message: str) -> str:
    cleaned = message.strip()
    while cleaned.startswith("[STM32]"):
        cleaned = cleaned[len("[STM32]"):].strip()
    return cleaned


def korean_stm32_message(message: str) -> Optional[tuple[str, str]]:
    cleaned = clean_stm32_message(message)
    known_messages = {
        "Auto mode started. Belt moving.": (PHASE_BELT_MOVING, "컨베이어 자동 모드가 시작되어 벨트가 움직입니다."),
        "Belt moving.": (PHASE_BELT_MOVING, "컨베이어 벨트가 움직입니다."),
        "Belt stopped.": (PHASE_DETECTING, "컨베이어 벨트가 멈췄습니다."),
        "O received. Object exists. Waiting.": (PHASE_SORTING, "STM32가 물체 감지 신호를 받아 정지 상태를 유지합니다."),
        "E received. Belt restarted.": (PHASE_BELT_MOVING, "STM32가 재시작 신호를 받아 컨베이어를 다시 움직입니다."),
    }
    if cleaned.startswith("Belt stopped after ") and cleaned.endswith(" sec move."):
        return None
    return known_messages.get(cleaned, (PHASE_SYSTEM, f"STM32 메시지: {cleaned}"))


def object_still_visible(original_obj: dict, current_objects: list[dict]) -> bool:
    for current_obj in current_objects:
        dx = float(current_obj["x"] - original_obj["x"])
        dy = float(current_obj["y"] - original_obj["y"])
        if float(np.hypot(dx, dy)) <= OBJECT_REMOVED_MATCH_PX:
            return True
    return False


@dataclass
class CameraConfig:
    width: int = 848
    height: int = 480
    fps: int = 30
    serial: Optional[str] = None
    enable_color: bool = True
    enable_depth: bool = True
    align_to_color: bool = True
    depth_min_m: float = 0.2
    depth_max_m: float = 2.5
    show_depth_panel: bool = False
    detect_rectangle: bool = True
    rect_min_area: int = 500
    rect_min_side_px: int = 45
    depth_top_min_side_ratio: float = 0.82
    rect_max_aspect_ratio: float = 8.0
    rect_min_extent: float = 0.18
    rect_max_area_ratio: float = 0.80
    rect_border_margin_px: int = 4
    center_roi_only: bool = True
    center_roi_ratio: float = 0.60
    target_min_distance_m: float = 0.10
    target_max_distance_m: float = 0.60
    target_min_height_m: float = 0.004
    target_band_half_width_m: float = 0.035
    quad_depth_std_max_m: float = 0.035
    quad_depth_valid_min_ratio: float = 0.45
    quad_depth_lift_min_m: float = 0.006
    quad_depth_ring_margin_px: int = 10
    top_face_min_abs_nz: float = 0.58
    top_face_depth_span_max_m: float = 0.030
    debug_detection: bool = False
    debug_print_interval: int = 15
    corner_only_display: bool = True
    corner_smooth_alpha: float = 0.72
    corner_hold_frames: int = 12
    center_weight_power: float = 2.6
    corner_deadband_px: float = 4.0
    corner_jump_guard_px: float = 22.0
    corner_continuity_weight: float = 0.50
    corner_micro_center_px: float = 1.1
    corner_micro_alpha: float = 0.88
    coord_lock_enabled: bool = False
    coord_lock_enter_px: float = 3.2
    coord_lock_exit_px: float = 10.0
    coord_lock_frames: int = 3
    coord_lock_depth_exit_m: float = 0.022
    coord_lock_static_mode: bool = True
    coord_lock_static_min_frames: int = 18
    coord_lock_sticky_slots: bool = True
    coord_lock_sticky_max_miss: int = 300
    coord_lock_move_release_frames: int = 6
    show_hold_slots: bool = False
    measure_size: bool = True
    max_objects: int = 4
    object_nms_iou: float = 0.22
    object_min_score_ratio: float = 0.30
    object_duplicate_center_ratio: float = 0.75
    object_duplicate_depth_diff_m: float = 0.03
    object_known_band_duplicate_center_ratio: float = 0.52
    object_small_band_duplicate_center_ratio: float = 0.44
    object_cross_band_duplicate_center_ratio: float = 0.24
    object_secondary_score_ratio: float = 0.10
    object_secondary_area_ratio: float = 0.10
    object_area_score_power: float = 0.42
    object_near_score_gain: float = 0.12
    object_small_band_score_gain: float = 0.18
    white_bias_enabled: bool = True
    white_min_ratio: float = 0.18
    white_sat_max: int = 64
    white_value_min: int = 140
    white_score_gain: float = 0.55
    depth_top_priority_enabled: bool = True
    depth_top_min_height_m: float = 0.010
    depth_top_max_height_m: float = 0.090
    depth_top_band_m: float = 0.010
    depth_top_support_percentile: float = 86.0
    depth_top_split_enabled: bool = True
    depth_top_split_aspect_ratio: float = 1.35
    depth_top_peak_split_enabled: bool = False
    depth_top_quad_inset_ratio: float = 0.12
    quad_edge_refine_enabled: bool = False
    quad_edge_refine_normal_ratio: float = 0.14
    quad_edge_refine_tangent_ratio: float = 0.22
    quad_edge_refine_max_shift_ratio: float = 0.24
    quad_edge_refine_min_points: int = 8
    display_quad_contour_refine_enabled: bool = True
    display_quad_contour_blend: float = 0.68
    display_quad_contour_max_center_shift_ratio: float = 0.22
    display_quad_approx_eps_ratios: tuple[float, ...] = (0.014, 0.020, 0.026, 0.032, 0.040)
    display_quad_deadband_px: float = 1.6
    display_quad_smooth_alpha: float = 0.72
    depth_top_known_height_bands_enabled: bool = True
    depth_top_height_targets_m: tuple[float, ...] = (0.03, 0.05)
    depth_top_height_tol_m: float = 0.012
    depth_top_known_band_min_side_factor: float = 0.78
    depth_top_small_band_min_side_factor: float = 0.72
    depth_top_small_object_height_m: float = 0.038
    depth_top_small_split_area_ratio: float = 1.28
    depth_top_known_band_split_area_ratio: float = 1.34
    depth_top_band_balance_enabled: bool = False
    depth_top_band_score_gain: float = 0.0
    support_depth_bin_m: float = 0.004
    support_depth_band_m: float = 0.008
    depth_filter_enabled: bool = True
    depth_temporal_alpha: float = 0.35
    depth_temporal_delta: float = 20.0
    depth_spatial_alpha: float = 0.50
    depth_spatial_delta: float = 20.0
    depth_hole_fill_mode: int = 1
    tune_sensors: bool = False
    frame_wait_timeout_ms: int = 1500
    startup_frame_timeout_ms: int = 1200
    startup_frame_retries: int = 3


class D435Camera:
    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.pipeline = rs.pipeline()
        self.align = rs.align(rs.stream.color) if cfg.align_to_color else None
        self.depth_scale = 0.001  # default fallback
        self.depth_filters = []
        self.measure_intrinsics = None
        self.measure_stream_name = "unknown"
        self._running = False
        self.detection_overlay_enabled = False
        self.rect_detection_enabled = cfg.detect_rectangle
        self.center_roi_enabled = cfg.center_roi_only
        max_objects = max(1, int(cfg.max_objects))
        self.prev_slot_quads: list[Optional[np.ndarray]] = [None for _ in range(max_objects)]
        self.prev_display_slot_quads: list[Optional[np.ndarray]] = [None for _ in range(max_objects)]
        self.slot_miss_counts: list[int] = [0 for _ in range(max_objects)]
        self.lock_slot_quads: list[Optional[np.ndarray]] = [None for _ in range(max_objects)]
        self.lock_slot_depths: list[Optional[float]] = [None for _ in range(max_objects)]
        self.slot_lock_streaks: list[int] = [0 for _ in range(max_objects)]
        self.slot_locked_flags: list[bool] = [False for _ in range(max_objects)]
        self.slot_locked_ages: list[int] = [0 for _ in range(max_objects)]
        self.slot_move_release_streaks: list[int] = [0 for _ in range(max_objects)]
        self.slot_size_labels: list[Optional[str]] = [None for _ in range(max_objects)]
        self.last_candidates = []
        self.last_selected_candidates = []
        self.last_tracked_objects = []
        self.preserve_sort_slots = False
        self.reserved_sort_slot_ids: set[int] = set()
        self.last_contour_mode = ""
        self.last_contour_entries = []
        self.support_depth_m: Optional[float] = None
        self.support_plane_coeffs: Optional[np.ndarray] = None
        self.debug_frame_index = 0
        self.ui_font_path = self._resolve_ui_font_path()
        self.ui_font_cache: dict[int, object] = {}

    def start(self) -> None:
        self._ensure_device_present()
        profile, used_profile = self._start_with_fallback_profiles()
        device = profile.get_device()
        self._print_device_info(device)
        if self.cfg.tune_sensors:
            log_status("카메라 센서 설정을 적용합니다.")
            try:
                self._configure_sensors(device)
            except Exception as exc:
                log_warning(f"센서 설정을 적용하지 못했습니다: {exc}")
        self._cache_measure_intrinsics(profile)
        self._setup_depth_filters()
        self.depth_scale = device.first_depth_sensor().get_depth_scale()
        self._running = True
        log_status("카메라가 정상적으로 시작되었습니다.")
        if ImageFont is None:
            log_warning("Pillow가 없어 프리뷰의 한글 표시가 깨질 수 있습니다.")
        elif self.ui_font_path is None:
            log_warning("한글 폰트를 찾지 못해 프리뷰의 한글 표시가 깨질 수 있습니다.")
        if self.cfg.enable_color and self.cfg.enable_depth and (not self.cfg.align_to_color):
            log_warning("컬러/깊이 영상 정렬이 꺼져 있어 크기 측정 정확도가 낮아질 수 있습니다.")

    def stop(self) -> None:
        if self._running:
            self.pipeline.stop()
            self._running = False
            log_status("카메라를 종료했습니다.")

    def get_frames(self, timeout_ms: Optional[int] = None):
        wait_timeout = int(self.cfg.frame_wait_timeout_ms if timeout_ms is None else timeout_ms)
        try:
            frames = self.pipeline.wait_for_frames(wait_timeout)
        except RuntimeError as exc:
            raise RuntimeError(f"No frames received within {wait_timeout} ms ({exc})") from exc
        if self.align is not None:
            frames = self.align.process(frames)

        depth_frame = frames.get_depth_frame() if self.cfg.enable_depth else None
        color_frame = frames.get_color_frame() if self.cfg.enable_color else None

        if depth_frame and self.depth_filters:
            filtered = depth_frame
            for flt in self.depth_filters:
                filtered = flt.process(filtered)
            depth_frame = filtered.as_depth_frame()

        if self.cfg.enable_depth and not depth_frame:
            raise RuntimeError("Depth frame not available.")
        if self.cfg.enable_color and not color_frame:
            raise RuntimeError("Color frame not available.")

        depth_img = np.asanyarray(depth_frame.get_data()) if depth_frame else None
        color_img = np.asanyarray(color_frame.get_data()) if color_frame else None
        return color_img, depth_img

    def preview(self, save_dir: str = "captures") -> None:
        os.makedirs(save_dir, exist_ok=True)

        # This recovery copy intentionally does not open or control the conveyor.
        ser = None
        arm_ser = None
        try:
            arm_ser = serial.Serial(ROBOT_ARM_PORT, ROBOT_ARM_BAUD, timeout=0.2)
            time.sleep(2.0)
            log_status(f"로봇팔 STM32와 연결되었습니다. ({ROBOT_ARM_PORT})")
        except Exception as exc:
            log_warning(f"로봇팔 STM32 연결 실패: {exc}")
            log_warning("카메라 확인만 가능하며 p를 눌러도 로봇은 움직이지 않습니다.")
        self.detection_overlay_enabled = True
        self.preserve_sort_slots = False
        self.reserved_sort_slot_ids.clear()
        log_status("카메라·로봇팔·공압 전용 모드입니다. 컨베이어는 사용하지 않습니다.")
        log_status("물체를 확인한 뒤 p를 누르면 자동 분류를 시작합니다.")
        log_status("프리뷰 실행 중입니다. 종료하려면 q 또는 ESC를 누르세요.")
        cv2.namedWindow("D435 Preview", cv2.WINDOW_NORMAL)
        wait_fail_count = 0
        wait_h = max(self.cfg.height, 360)
        wait_w = (
            self.cfg.width * 2
            if (self.cfg.enable_color and self.cfg.enable_depth and bool(self.cfg.show_depth_panel))
            else self.cfg.width
        )
        wait_w = max(wait_w, 640)
        wait_screen = np.zeros((wait_h, wait_w, 3), dtype=np.uint8)

        while True:
            try:
                color_img, depth_img = self.get_frames()
                wait_fail_count = 0
                display = self._compose_display(color_img, depth_img)

                if ser is not None and ser.in_waiting > 0:
                    msg = ser.readline().decode(errors="ignore").strip()

                    if msg == "CHECK":
                        self.detection_overlay_enabled = True
                        log_status(f"검사 요청을 받았습니다. {CAMERA_SETTLE_SEC:.1f}초 동안 화면을 안정화합니다.", PHASE_DETECTING)
                    elif msg:
                        translated = korean_stm32_message(msg)
                        if translated:
                            phase, message = translated
                            if phase == PHASE_BELT_MOVING:
                                self.preserve_sort_slots = False
                                self.reserved_sort_slot_ids.clear()
                                self.detection_overlay_enabled = False
                            log_status(message, phase)

                    if msg == "CHECK":

                        # 1) 벨트가 멈춘 직후 카메라 안정화 시간
                        settle_start = time.time()

                        while time.time() - settle_start < CAMERA_SETTLE_SEC:
                            color_img, depth_img = self.get_frames()
                            display = self._compose_display(color_img, depth_img)
                            self._draw_runtime_mode_banner(display)
                            cv2.imshow("D435 Preview", display)
                            cv2.waitKey(1)

                        # 2) 추가 프레임으로 객체 검출 안정화
                        for _ in range(CAMERA_SETTLE_FRAMES):
                            color_img, depth_img = self.get_frames()
                            display = self._compose_display(color_img, depth_img)
                            self._draw_runtime_mode_banner(display)
                            cv2.imshow("D435 Preview", display)
                            cv2.waitKey(1)

                        # 3) 현재 검출된 객체 좌표 수집
                        objects = [dict(obj) for obj in self.last_tracked_objects]

                        # 4) 객체가 있으면 컨베이어 재시작 금지
                        if len(objects) > 0:
                            self.preserve_sort_slots = True
                            self.reserved_sort_slot_ids = {int(obj["id"]) for obj in objects}
                            log_status(f"물체가 {len(objects)}개 감지되었습니다. 컨베이어를 정지 상태로 유지합니다.", PHASE_DETECTING)
                            for obj in objects:
                                if obj["z"] is None:
                                    log_status(f'객체 {obj["id"]}: x={obj["x"]}, y={obj["y"]}, 거리 측정 안 됨', PHASE_DETECTING)
                                else:
                                    log_status(f'객체 {obj["id"]}: x={obj["x"]}, y={obj["y"]}, 거리={obj["z"]:.3f}m', PHASE_DETECTING)

                            # STM32에게 물체가 있다고 알림 → STM32는 정지 상태 유지
                            ser.write(b"O\n")
                            log_status("STM32에 물체 감지 신호를 보냈습니다.", PHASE_SORTING)
                            if arm_ser is None:
                                log_warning("로봇팔 COM 포트가 없어 분류할 수 없습니다. 벨트를 정지 상태로 유지합니다.")
                                continue

                            # 5) 도달 가능한 물체를 하나씩 분류하고 매번 카메라로 다시 확인
                            processed_count = 0
                            while True:
                                current_objects = [dict(obj) for obj in self.last_tracked_objects]
                                valid_objects = [
                                    obj for obj in current_objects
                                    if obj.get("robot_x_mm") is not None
                                    and obj.get("robot_y_mm") is not None
                                    and obj.get("robot_z_mm") is not None
                                ]
                                left_objects = [
                                    obj for obj in valid_objects if object_in_left_workspace(obj)
                                ]
                                right_objects = [
                                    obj for obj in valid_objects if object_in_right_workspace(obj)
                                ]
                                eligible_objects = left_objects + right_objects
                                left_plans = []
                                right_plans = []
                                for obj in left_objects:
                                    try:
                                        left_plans.append(build_arm_pick_plan(obj, "left"))
                                    except ValueError as exc:
                                        log_warning(f"왼팔 객체 {obj.get('id', '?')} 제외: {exc}")
                                for obj in right_objects:
                                    try:
                                        right_plans.append(build_arm_pick_plan(obj, "right"))
                                    except ValueError as exc:
                                        log_warning(f"오른팔 객체 {obj.get('id', '?')} 제외: {exc}")

                                if left_plans or right_plans:
                                    left_plans.sort(
                                        key=lambda plan: float(np.hypot(
                                            ROBOT_FORWARD_TO_CENTER_MM - plan["local_camera_y"],
                                            plan["local_camera_x"],
                                        ))
                                    )
                                    right_plans.sort(
                                        key=lambda plan: float(np.hypot(
                                            ROBOT_FORWARD_TO_CENTER_MM - plan["local_camera_y"],
                                            plan["local_camera_x"],
                                        ))
                                    )
                                    left_plan = left_plans[0] if left_plans else None
                                    right_plan = right_plans[0] if right_plans else None
                                    send_dual_arm_plans(arm_ser, left_plan, right_plan)
                                    log_status("양팔 완료 응답을 기다립니다.", PHASE_SORTING)
                                    reply = wait_arm_reply(arm_ser, ("DONE", "ERROR", "BUSY"))
                                    if reply != "DONE":
                                        log_warning(f"양팔 분류가 완료되지 않았습니다: {reply or '응답 없음'}")
                                        break
                                    processed_count += int(left_plan is not None) + int(right_plan is not None)

                                    # 피킹 전 프레임을 버리고 실제 작업영역을 다시 측정
                                    self._reset_tracking_state()
                                    for _ in range(CAMERA_SETTLE_FRAMES + 10):
                                        color_img, depth_img = self.get_frames()
                                        display = self._compose_display(color_img, depth_img)
                                        self._draw_runtime_mode_banner(display)
                                        cv2.imshow("D435 Preview", display)
                                        cv2.waitKey(1)
                                    continue

                                if current_objects:
                                    missing_coordinates = [
                                        obj for obj in current_objects
                                        if obj.get("robot_x_mm") is None
                                        or obj.get("robot_y_mm") is None
                                        or obj.get("robot_z_mm") is None
                                    ]
                                    outside_objects = [
                                        obj for obj in valid_objects
                                        if not object_in_left_workspace(obj)
                                        and not object_in_right_workspace(obj)
                                    ]

                                    if missing_coordinates or eligible_objects:
                                        log_status(
                                            "물체가 남아 있어 좌표를 다시 측정한 뒤 분류를 계속합니다.",
                                            PHASE_DETECTING,
                                        )
                                        self._reset_tracking_state()
                                        for _ in range(CAMERA_SETTLE_FRAMES + 10):
                                            color_img, depth_img = self.get_frames()
                                            display = self._compose_display(color_img, depth_img)
                                            self._draw_runtime_mode_banner(display)
                                            cv2.imshow("D435 Preview", display)
                                            cv2.waitKey(1)
                                        continue

                                    if outside_objects:
                                        log_warning(
                                            f"작업영역 밖 물체 {len(outside_objects)}개를 건너뛰고 "
                                            "컨베이어를 계속 이동합니다."
                                        )
                                        ser.write(b"E\n")
                                        ser.flush()
                                        self.preserve_sort_slots = False
                                        self.reserved_sort_slot_ids.clear()
                                        self.detection_overlay_enabled = False
                                        log_status(
                                            "컨베이어 재시작 신호 E를 보냈습니다. "
                                            "다음 작업구역 벽에서 다시 정지합니다.",
                                            PHASE_BELT_MOVING,
                                        )
                                        break

                                # 마지막 분류 후 연속 프레임으로 작업영역이 정말 비었는지 재확인
                                log_status("마지막 물체 분류 후 작업영역을 다시 확인합니다.", PHASE_DETECTING)
                                empty_frames = 0
                                for _ in range(EMPTY_CONFIRM_FRAMES + 8):
                                    color_img, depth_img = self.get_frames()
                                    display = self._compose_display(color_img, depth_img)
                                    self._draw_runtime_mode_banner(display)
                                    cv2.imshow("D435 Preview", display)
                                    cv2.waitKey(1)
                                    current = [dict(obj) for obj in self.last_tracked_objects]
                                    empty_frames = empty_frames + 1 if not current else 0

                                if empty_frames < EMPTY_CONFIRM_FRAMES:
                                    log_status("재확인 중 물체가 보여 분류를 계속합니다.", PHASE_SORTING)
                                    continue

                                ser.write(b"E\n")
                                ser.flush()
                                log_status(
                                    f"총 {processed_count}개 분류 후 EMPTY를 확인하여 "
                                    "컨베이어 재시작 신호 E를 보냈습니다.",
                                    PHASE_BELT_MOVING,
                                )
                                self.preserve_sort_slots = False
                                self.reserved_sort_slot_ids.clear()
                                self.detection_overlay_enabled = False
                                break

                        # 6) 처음부터 객체가 없으면 바로 E 전송
                        else:
                            self.preserve_sort_slots = False
                            self.reserved_sort_slot_ids.clear()
                            log_status("작업 영역에 물체가 없습니다.", PHASE_DONE)
                            ser.write(b"E\n")
                            ser.flush()
                            log_status("STM32에 컨베이어 재시작 신호를 보냈습니다.", PHASE_BELT_MOVING)
                            self.detection_overlay_enabled = False
            except RuntimeError as exc:
                wait_fail_count += 1
                if wait_fail_count == 1 or wait_fail_count % 15 == 0:
                    log_warning(f"카메라 영상을 기다리는 중입니다. ({wait_fail_count}회)")
                display = wait_screen.copy()
                cv2.putText(
                    display,
                    "Waiting for camera frames...",
                    (24, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 200, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display,
                    "Check USB3 cable/port and close RealSense Viewer.",
                    (24, 102),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 200, 255),
                    2,
                    cv2.LINE_AA,
                )
            self._draw_runtime_mode_banner(display)
            cv2.imshow("D435 Preview", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("x"):
                request_emergency_stop(arm_ser)
                if arm_ser is not None:
                    wait_arm_reply(
                        arm_ser, ("EMERGENCY_DONE", "ERROR"), 60.0,
                        allow_emergency_key=False
                    )
                log_warning("긴급 정지가 완료되었습니다.")
                continue
            if key == ord("z"):
                request_dual_arm_reset(arm_ser)
                if arm_ser is not None:
                    reply = wait_arm_reply(
                        arm_ser, ("EMERGENCY_DONE", "ERROR"), 60.0,
                        allow_emergency_key=False
                    )
                    if reply != "EMERGENCY_DONE":
                        log_warning("양팔 안전 복귀 완료를 확인하지 못했습니다.")
                        continue
                self._reset_tracking_state()
                log_status("Z 리셋이 완료되었습니다.", PHASE_SYSTEM)
                continue
            if key == ord("p"):
                if arm_ser is None:
                    log_warning("로봇팔 COM 포트가 연결되지 않았습니다.")
                    continue
                log_status("다중 물체 자동 처리를 시작합니다.", PHASE_SORTING)
                processed_count = 0
                while True:
                    objects = [dict(obj) for obj in self.last_tracked_objects]
                    valid_objects = [
                        obj for obj in objects
                        if obj.get("robot_x_mm") is not None
                        and obj.get("robot_y_mm") is not None
                        and obj.get("robot_z_mm") is not None
                    ]
                    left_objects = [obj for obj in valid_objects if object_in_left_workspace(obj)]
                    right_objects = [obj for obj in valid_objects if object_in_right_workspace(obj)]
                    eligible_objects = left_objects + right_objects
                    left_plans = []
                    right_plans = []
                    for obj in left_objects:
                        try:
                            left_plans.append(build_arm_pick_plan(obj, "left"))
                        except ValueError as exc:
                            log_warning(f"왼팔 객체 {obj.get('id', '?')} 제외: {exc}")
                    for obj in right_objects:
                        try:
                            right_plans.append(build_arm_pick_plan(obj, "right"))
                        except ValueError as exc:
                            log_warning(f"오른팔 객체 {obj.get('id', '?')} 제외: {exc}")

                    if not left_plans and not right_plans:
                        if eligible_objects:
                            log_warning("물체는 남아 있지만 현재 로봇이 도달할 수 없습니다.")
                            break
                        log_status("물체 없음 확인을 위해 카메라를 다시 검사합니다.", PHASE_DETECTING)
                        empty_frames = 0
                        for _ in range(EMPTY_CONFIRM_FRAMES + 8):
                            color_img, depth_img = self.get_frames()
                            display = self._compose_display(color_img, depth_img)
                            self._draw_runtime_mode_banner(display)
                            cv2.imshow("D435 Preview", display)
                            cv2.waitKey(1)
                            current = [
                                obj for obj in self.last_tracked_objects
                                if object_in_left_workspace(obj) or object_in_right_workspace(obj)
                            ]
                            empty_frames = empty_frames + 1 if not current else 0
                        if empty_frames < EMPTY_CONFIRM_FRAMES:
                            continue
                        log_status("작업영역에 물체가 없습니다. 종료 명령을 전송합니다.", PHASE_DONE)
                        write_arm_line(arm_ser, "F\n")
                        wait_arm_reply(arm_ser, ("FINISHED", "ERROR"), 60.0)
                        log_status(f"총 {processed_count}개 물체 처리를 종료했습니다.", PHASE_DONE)
                        break

                    left_plans.sort(
                        key=lambda plan: float(np.hypot(
                            ROBOT_FORWARD_TO_CENTER_MM - plan["local_camera_y"],
                            plan["local_camera_x"],
                        ))
                    )
                    right_plans.sort(
                        key=lambda plan: float(np.hypot(
                            ROBOT_FORWARD_TO_CENTER_MM - plan["local_camera_y"],
                            plan["local_camera_x"],
                        ))
                    )
                    left_plan = left_plans[0] if left_plans else None
                    right_plan = right_plans[0] if right_plans else None
                    send_dual_arm_plans(arm_ser, left_plan, right_plan)
                    log_status("양팔 완료 응답을 기다립니다.", PHASE_SORTING)
                    reply = wait_arm_reply(arm_ser, ("DONE", "ERROR", "BUSY"))
                    if reply != "DONE":
                        break
                    processed_count += int(left_plan is not None) + int(right_plan is not None)

                    # Discard buffered pre-pick frames and reacquire the scene.
                    self._reset_tracking_state()
                    for _ in range(CAMERA_SETTLE_FRAMES + 10):
                        color_img, depth_img = self.get_frames()
                        display = self._compose_display(color_img, depth_img)
                        self._draw_runtime_mode_banner(display)
                        cv2.imshow("D435 Preview", display)
                        cv2.waitKey(1)
        if ser is not None:
            ser.close()
            log_status("STM32 연결을 종료했습니다.")
        if arm_ser is not None:
            arm_ser.close()
            log_status("로봇팔 STM32 연결을 종료했습니다.")
            
        cv2.destroyAllWindows()

    def _reset_tracking_state(self) -> None:
        # 로봇 동작 후 남은 물체를 새 장면으로 다시 등록할 수 있도록
        # 이전 물체의 예약 슬롯과 검출 결과를 함께 해제한다.
        self.preserve_sort_slots = False
        self.reserved_sort_slot_ids.clear()
        self.last_candidates = []
        self.last_selected_candidates = []
        self.last_tracked_objects = []
        max_objects = max(1, int(self.cfg.max_objects))
        for slot_idx in range(max_objects):
            self.prev_slot_quads[slot_idx] = None
            self.prev_display_slot_quads[slot_idx] = None
            self.slot_miss_counts[slot_idx] = 0
            self.slot_size_labels[slot_idx] = None
            self._clear_slot_lock(slot_idx)

    def _draw_runtime_mode_banner(self, frame: np.ndarray) -> None:
        return

    def _compose_display(self, color_img, depth_img):
        if color_img is not None and self.rect_detection_enabled and self.detection_overlay_enabled:
            color_img = self._detect_and_draw_rectangle(color_img, depth_img)
        elif not self.detection_overlay_enabled:
            self.last_tracked_objects = []

        if depth_img is not None:
            depth_colormap = self._depth_to_colormap(depth_img)
        else:
            depth_colormap = None

        if color_img is not None and depth_colormap is not None:
            if not bool(self.cfg.show_depth_panel):
                return color_img
            if color_img.shape[:2] != depth_colormap.shape[:2]:
                depth_colormap = cv2.resize(
                    depth_colormap,
                    (color_img.shape[1], color_img.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            return np.hstack((color_img, depth_colormap))
        if color_img is not None:
            return color_img
        if depth_colormap is not None:
            return depth_colormap
        raise RuntimeError("No stream enabled.")

    def _depth_to_colormap(self, depth_raw: np.ndarray) -> np.ndarray:
        depth_m = depth_raw.astype(np.float32) * self.depth_scale
        depth_m = np.clip(depth_m, self.cfg.depth_min_m, self.cfg.depth_max_m)

        scale = max(self.cfg.depth_max_m - self.cfg.depth_min_m, 1e-6)
        depth_norm = (depth_m - self.cfg.depth_min_m) / scale
        depth_u8 = (255.0 * (1.0 - depth_norm)).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)

        invalid = depth_raw == 0
        depth_color[invalid] = (0, 0, 0)
        return depth_color

    def _detect_and_draw_rectangle(self, color_img: np.ndarray, depth_img: Optional[np.ndarray]) -> np.ndarray:
        annotated = color_img.copy()
        work_img = color_img
        work_depth = depth_img

        if self.measure_intrinsics is not None:
            origin_x = int(round(float(self.measure_intrinsics.ppx)))
            origin_y = int(round(float(self.measure_intrinsics.ppy)))
        else:
            origin_x = annotated.shape[1] // 2
            origin_y = annotated.shape[0] // 2

        axis_overlay = annotated.copy()
        axis_color = (80, 255, 80)
        cv2.line(axis_overlay, (0, origin_y), (annotated.shape[1] - 1, origin_y), axis_color, 1, cv2.LINE_AA)
        cv2.line(axis_overlay, (origin_x, 0), (origin_x, annotated.shape[0] - 1), axis_color, 1, cv2.LINE_AA)
        cv2.addWeighted(axis_overlay, 0.32, annotated, 0.68, 0, annotated)
        cv2.circle(annotated, (origin_x, origin_y), 5, axis_color, -1, cv2.LINE_AA)
        cv2.putText(
            annotated,
            "0",
            (origin_x + 12, max(18, origin_y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 255, 80),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(annotated, "+X", (annotated.shape[1] - 46, max(20, origin_y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, axis_color, 2, cv2.LINE_AA)
        cv2.putText(annotated, "-X", (12, max(20, origin_y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, axis_color, 2, cv2.LINE_AA)
        cv2.putText(annotated, "+Y", (origin_x + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, axis_color, 2, cv2.LINE_AA)
        cv2.putText(annotated, "-Y", (origin_x + 10, annotated.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, axis_color, 2, cv2.LINE_AA)

        roi_x0, roi_y0 = 0, 0
        if self.center_roi_enabled:
            work_img, (roi_x0, roi_y0, roi_x1, roi_y1) = self._extract_center_roi(color_img)
            if depth_img is not None:
                work_depth = depth_img[roi_y0:roi_y1, roi_x0:roi_x1]
            cv2.rectangle(annotated, (roi_x0, roi_y0), (roi_x1, roi_y1), (255, 200, 0), 2)
            cv2.putText(
                annotated,
                "Center ROI",
                (roi_x0 + 8, max(18, roi_y0 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 220, 40),
                2,
                cv2.LINE_AA,
            )

        contour_entries, contour_mode, depth_valid_ratio = self._get_candidate_contours(work_img, work_depth)
        self.last_contour_mode = contour_mode
        self.last_contour_entries = [(c.copy(), str(src)) for c, src in contour_entries]
        self.debug_frame_index += 1
        debug_on = bool(self.cfg.debug_detection)
        debug_stats = {
            "input_by_source": {},
            "cand_by_source": {},
            "reject_lift": 0,
            "reject_nz": 0,
            "reject_span": 0,
        }

        candidates = []
        img_h, img_w = work_img.shape[:2]
        img_center = np.array([img_w / 2.0, img_h / 2.0], dtype=np.float32)
        max_center_dist = float(np.linalg.norm(img_center)) + 1e-6
        center_power = self._effective_center_weight_power()
        work_hsv = cv2.cvtColor(work_img, cv2.COLOR_BGR2HSV) if bool(self.cfg.white_bias_enabled) else None
        support_depth_for_candidates = None if self.support_depth_m is None else float(self.support_depth_m)

        for contour, source in contour_entries:
            if debug_on:
                debug_stats["input_by_source"][source] = debug_stats["input_by_source"].get(source, 0) + 1
            area = cv2.contourArea(contour)
            if area < self.cfg.rect_min_area:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            margin = int(self.cfg.rect_border_margin_px)
            if x <= margin or y <= margin or (x + bw) >= (img_w - margin) or (y + bh) >= (img_h - margin):
                # ROI/table boundary-like contour; usually not the target box.
                continue

            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            if width <= 1.0 or height <= 1.0:
                continue
            center_local = np.array([rect[0][0], rect[0][1]], dtype=np.float32)
            center_full = (int(center_local[0] + roi_x0), int(center_local[1] + roi_y0))
            min_side_req = float(self.cfg.rect_min_side_px)
            if source == "depth-top":
                min_side_req *= float(np.clip(self.cfg.depth_top_min_side_ratio, 0.65, 1.0))
                if depth_img is not None:
                    early_depth_m = self._sample_depth_m(depth_img, center_full)
                    support_depth_local = self._predict_support_depth_m(center_local)
                    early_band_info = self._candidate_height_band_info(early_depth_m, support_depth_local)
                    if early_band_info is not None and early_band_info.get("band_target_m") is not None:
                        expected_side_px = self._expected_top_face_side_px(
                            early_depth_m,
                            float(early_band_info["band_target_m"]),
                        )
                        if expected_side_px is not None:
                            side_factor = float(self.cfg.depth_top_known_band_min_side_factor)
                            if float(early_band_info["band_target_m"]) <= float(self.cfg.depth_top_small_object_height_m):
                                side_factor = float(self.cfg.depth_top_small_band_min_side_factor)
                            min_side_req = min(min_side_req, max(24.0, expected_side_px * side_factor))
            elif source == "depth-band":
                min_side_req *= 0.88
            if min(width, height) < min_side_req:
                continue

            box_area = float(width * height)
            if box_area <= 1.0:
                continue

            area_ratio = box_area / float(img_w * img_h)
            if area_ratio > self.cfg.rect_max_area_ratio:
                continue

            extent = float(area / box_area)

            ratio = max(width, height) / max(min(width, height), 1e-6)
            if ratio > self.cfg.rect_max_aspect_ratio:
                continue

            quad_local = cv2.boxPoints(rect).astype(np.float32)
            quad_local = self._order_quad_points(quad_local)
            quad_local = self._refine_quad_from_contour_edges(contour, quad_local)
            if not self._is_rectangular_quad(quad_local, max_cos=0.75):
                continue
            if source == "depth-top":
                quad_local = self._inset_quad(quad_local, self.cfg.depth_top_quad_inset_ratio)

            center_local = np.array([rect[0][0], rect[0][1]], dtype=np.float32)
            center_full = (int(center_local[0] + roi_x0), int(center_local[1] + roi_y0))
            center_dist = float(np.linalg.norm(center_local - img_center))
            center_dist_norm = center_dist / max_center_dist

            depth_m = self._sample_depth_m(depth_img, center_full)
            if depth_m is not None:
                if depth_m < self.cfg.target_min_distance_m or depth_m > self.cfg.target_max_distance_m:
                    continue
            support_depth_local = self._predict_support_depth_m(center_local)
            band_info = self._candidate_height_band_info(depth_m, support_depth_local)

            quad_full = quad_local.copy()
            quad_full[:, 0] += roi_x0
            quad_full[:, 1] += roi_y0
            contour_full = contour.astype(np.float32).copy()
            contour_full[:, 0, 0] += roi_x0
            contour_full[:, 0, 1] += roi_y0

            quad_depth_std = None
            quad_depth_valid = None
            quad_depth_lift = None
            quad_abs_nz = None
            quad_depth_span = None
            no_depth_penalty = 1.0
            if depth_img is not None:
                quad_stats = self._sample_quad_depth_stats(depth_img, quad_full)
                if quad_stats is None:
                    # Color/dark candidates can have sparse depth on low-texture black surfaces.
                    if source not in ("color-edge", "dark-blob"):
                        continue
                    if depth_m is None:
                        if center_dist_norm > 0.58 or area_ratio > 0.22 or ratio > 3.0:
                            continue
                        no_depth_penalty = 0.86 if source == "dark-blob" else 0.80
                else:
                    quad_depth_med, quad_depth_std, quad_depth_valid = quad_stats
                    if source in ("color-edge", "dark-blob"):
                        valid_req = max(0.16, self.cfg.quad_depth_valid_min_ratio * 0.45)
                        std_req = self.cfg.quad_depth_std_max_m * 2.2
                    elif source == "depth-edge":
                        valid_req = max(0.22, self.cfg.quad_depth_valid_min_ratio * 0.70)
                        std_req = self.cfg.quad_depth_std_max_m * 1.6
                    elif source == "depth-top":
                        valid_req = max(0.28, self.cfg.quad_depth_valid_min_ratio * 0.82)
                        std_req = self.cfg.quad_depth_std_max_m * 0.95
                    else:
                        valid_req = self.cfg.quad_depth_valid_min_ratio
                        std_req = self.cfg.quad_depth_std_max_m

                    if quad_depth_valid < valid_req:
                        continue
                    if quad_depth_std > std_req:
                        continue
                    if depth_m is None:
                        depth_m = quad_depth_med

                req_lift = self._required_quad_lift_m(source)
                if req_lift > 0.0:
                    lift_stats = self._sample_quad_depth_lift(depth_img, quad_full)
                    if lift_stats is not None:
                        quad_depth_lift = float(lift_stats["lift_m"])
                        if quad_depth_lift < req_lift:
                            if debug_on:
                                debug_stats["reject_lift"] += 1
                            continue
                    elif source in ("color-edge", "dark-blob") and (
                        quad_depth_valid is not None and quad_depth_valid >= 0.30
                    ):
                        # Flat/texture-only patches often fail ring-lift consistency.
                        if debug_on:
                            debug_stats["reject_lift"] += 1
                        continue

                plane_stats = self._sample_quad_plane_stats(depth_img, quad_full)
                if plane_stats is not None:
                    quad_abs_nz = float(plane_stats["abs_nz"])
                    quad_depth_span = float(plane_stats["depth_span_m"])
                    req_nz, req_span = self._effective_top_face_constraints(source)
                    if quad_abs_nz < req_nz:
                        if debug_on:
                            debug_stats["reject_nz"] += 1
                        continue
                    if req_span > 0.0 and quad_depth_span > req_span:
                        if debug_on:
                            debug_stats["reject_span"] += 1
                        continue
                elif source in ("color-edge", "dark-blob"):
                    # Weak-depth candidates must prove top-face geometry.
                    if debug_on:
                        debug_stats["reject_nz"] += 1
                    continue

            center_weight = max(0.10, 1.0 - (center_dist / max_center_dist))
            center_weight = center_weight ** center_power
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True) if perimeter > 0 else contour
            corners = len(approx)
            if source == "depth-band":
                source_extent_factor = 1.0
            elif source == "depth-top":
                source_extent_factor = 0.92
            elif source == "depth-edge":
                source_extent_factor = 0.9
            elif source == "dark-blob":
                source_extent_factor = 0.75
            elif source == "color-edge":
                source_extent_factor = 0.8
            else:
                source_extent_factor = 1.0

            extent_req = max(0.07, self.cfg.rect_min_extent * source_extent_factor)

            if extent < extent_req:
                continue
            if corners == 4 and cv2.isContourConvex(approx):
                corner_bonus = 1.25
            else:
                corner_bonus = (1.0 / (1.0 + abs(corners - 4))) ** 2
            square_bonus = max(0.55, 1.0 - 0.45 * abs(float(np.log(max(ratio, 1e-6)))))
            extent_bonus = max(0.20, min(1.0, extent)) ** 1.25
            source_weight = self._source_weight(source, depth_valid_ratio, fallback=False)
            white_ratio = None
            white_bonus = 1.0
            if work_hsv is not None:
                white_stats = self._sample_quad_white_stats(work_hsv, quad_local)
                if white_stats is not None:
                    white_ratio = float(white_stats["white_ratio"])
                    white_bonus = self._white_score_bonus(white_stats, source)
            band_bonus = 1.0
            if band_info is not None and band_info.get("band_idx") is not None:
                band_conf = float(band_info.get("band_conf", 0.0))
                band_bonus += float(np.clip(self.cfg.depth_top_band_score_gain, 0.0, 0.5)) * band_conf
                band_target = band_info.get("band_target_m")
                if (
                    band_target is not None
                    and source in ("depth-top", "depth-band")
                    and float(band_target) <= float(self.cfg.depth_top_small_object_height_m)
                ):
                    band_bonus += float(np.clip(self.cfg.object_small_band_score_gain, 0.0, 0.6)) * max(0.40, band_conf)

            area_score = float(np.power(max(box_area, 1.0), np.clip(self.cfg.object_area_score_power, 0.25, 0.60)))
            score = (
                area_score
                * (0.5 + 0.5 * extent)
                * center_weight
                * corner_bonus
                * square_bonus
                * extent_bonus
                * source_weight
                * no_depth_penalty
                * white_bonus
                * band_bonus
            )
            if depth_m is not None:
                near_bonus = max(0.0, self.cfg.target_max_distance_m - depth_m)
                score *= 1.0 + (near_bonus / max(self.cfg.target_max_distance_m, 1e-6)) * float(
                    np.clip(self.cfg.object_near_score_gain, 0.0, 0.35)
                )
            if quad_depth_std is not None:
                planarity_bonus = max(0.55, 1.0 - (quad_depth_std / max(self.cfg.quad_depth_std_max_m, 1e-6)))
                score *= planarity_bonus
            if quad_abs_nz is not None:
                score *= max(0.55, 0.35 + quad_abs_nz)
            if quad_depth_span is not None and self.cfg.top_face_depth_span_max_m > 0.0:
                span_ref = max(0.005, float(self.cfg.top_face_depth_span_max_m))
                score *= max(0.50, 1.0 - (quad_depth_span / (span_ref * 2.2)))

            candidates.append(
                {
                    "score": score,
                    "quad": quad_full,
                    "contour": contour_full,
                    "area": area,
                    "ratio": ratio,
                    "center": center_full,
                    "depth_m": depth_m,
                    "source": source,
                    "quad_depth_std": quad_depth_std,
                    "quad_depth_lift": quad_depth_lift,
                    "quad_abs_nz": quad_abs_nz,
                    "quad_depth_span": quad_depth_span,
                    "white_ratio": white_ratio,
                    "height_m": (None if band_info is None else band_info.get("height_m")),
                    "height_band_idx": (None if band_info is None else band_info.get("band_idx")),
                    "height_band_target_m": (None if band_info is None else band_info.get("band_target_m")),
                    "height_band_error_m": (None if band_info is None else band_info.get("band_error_m")),
                    "height_band_conf": (0.0 if band_info is None else band_info.get("band_conf", 0.0)),
                }
            )
            if debug_on:
                debug_stats["cand_by_source"][source] = debug_stats["cand_by_source"].get(source, 0) + 1

        if not candidates:
            # 2nd pass: loose fallback when edges are fragmented by texture/cables.
            for contour, source in contour_entries:
                if debug_on:
                    key = f"{source}:fb"
                    debug_stats["input_by_source"][key] = debug_stats["input_by_source"].get(key, 0) + 1
                area = cv2.contourArea(contour)
                if area < max(250, int(self.cfg.rect_min_area * 0.40)):
                    continue

                x, y, bw, bh = cv2.boundingRect(contour)
                margin = int(self.cfg.rect_border_margin_px)
                if x <= margin or y <= margin or (x + bw) >= (img_w - margin) or (y + bh) >= (img_h - margin):
                    continue

                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                if width <= 1.0 or height <= 1.0:
                    continue
                center_local = np.array([rect[0][0], rect[0][1]], dtype=np.float32)
                center_full = (int(center_local[0] + roi_x0), int(center_local[1] + roi_y0))
                min_side_req = float(self.cfg.rect_min_side_px * 0.6)
                if source == "depth-top":
                    min_side_req = min_side_req * max(0.9, float(np.clip(self.cfg.depth_top_min_side_ratio, 0.65, 1.0)))
                    if depth_img is not None:
                        early_depth_m = self._sample_depth_m(depth_img, center_full)
                        support_depth_local = self._predict_support_depth_m(center_local)
                        early_band_info = self._candidate_height_band_info(early_depth_m, support_depth_local)
                        if early_band_info is not None and early_band_info.get("band_target_m") is not None:
                            expected_side_px = self._expected_top_face_side_px(
                                early_depth_m,
                                float(early_band_info["band_target_m"]),
                            )
                            if expected_side_px is not None:
                                side_factor = float(self.cfg.depth_top_known_band_min_side_factor)
                                if float(early_band_info["band_target_m"]) <= float(self.cfg.depth_top_small_object_height_m):
                                    side_factor = float(self.cfg.depth_top_small_band_min_side_factor)
                                min_side_req = min(min_side_req, max(22.0, expected_side_px * side_factor))
                if min(width, height) < min_side_req:
                    continue

                box_area = float(width * height)
                if box_area <= 1.0:
                    continue
                extent = float(area / box_area)
                if source == "depth-band":
                    extent_req_fallback = 0.14
                elif source == "dark-blob":
                    extent_req_fallback = 0.10
                else:
                    extent_req_fallback = 0.08
                if extent < extent_req_fallback:
                    continue

                area_ratio = box_area / float(img_w * img_h)
                if area_ratio > 0.95:
                    continue

                ratio = max(width, height) / max(min(width, height), 1e-6)
                if ratio > max(4.5, self.cfg.rect_max_aspect_ratio * 1.25):
                    continue

                quad_local = cv2.boxPoints(rect).astype(np.float32)
                quad_local = self._order_quad_points(quad_local)
                quad_local = self._refine_quad_from_contour_edges(contour, quad_local)
                if source == "depth-top":
                    quad_local = self._inset_quad(quad_local, self.cfg.depth_top_quad_inset_ratio)
                center_dist = float(np.linalg.norm(center_local - img_center))
                center_dist_norm = center_dist / max_center_dist

                depth_m = self._sample_depth_m(depth_img, center_full)
                if depth_m is not None:
                    if depth_m < self.cfg.target_min_distance_m or depth_m > self.cfg.target_max_distance_m:
                        continue
                support_depth_local = self._predict_support_depth_m(center_local)
                band_info = self._candidate_height_band_info(depth_m, support_depth_local)
                quad_full = quad_local.copy()
                quad_full[:, 0] += roi_x0
                quad_full[:, 1] += roi_y0
                contour_full = contour.astype(np.float32).copy()
                contour_full[:, 0, 0] += roi_x0
                contour_full[:, 0, 1] += roi_y0

                quad_depth_std = None
                quad_depth_valid = None
                quad_depth_lift = None
                quad_abs_nz = None
                quad_depth_span = None
                no_depth_penalty = 1.0
                if depth_img is not None:
                    quad_stats = self._sample_quad_depth_stats(depth_img, quad_full)
                    if quad_stats is None:
                        if source not in ("color-edge", "dark-blob"):
                            continue
                        if depth_m is None:
                            if center_dist_norm > 0.58 or area_ratio > 0.22 or ratio > 3.0:
                                continue
                            no_depth_penalty = 0.86 if source == "dark-blob" else 0.80
                    else:
                        quad_depth_med, quad_depth_std, quad_depth_valid = quad_stats
                        if source in ("color-edge", "dark-blob"):
                            valid_req = max(0.14, self.cfg.quad_depth_valid_min_ratio * 0.40)
                            std_req = self.cfg.quad_depth_std_max_m * 2.4
                        elif source == "depth-top":
                            valid_req = max(0.28, self.cfg.quad_depth_valid_min_ratio * 0.82)
                            std_req = self.cfg.quad_depth_std_max_m * 1.00
                        else:
                            valid_req = max(0.25, self.cfg.quad_depth_valid_min_ratio * 0.7)
                            std_req = self.cfg.quad_depth_std_max_m * 1.6

                        if quad_depth_valid < valid_req:
                            continue
                        if quad_depth_std > std_req:
                            continue
                        if depth_m is None:
                            depth_m = quad_depth_med

                    req_lift = self._required_quad_lift_m(source)
                    if req_lift > 0.0:
                        lift_stats = self._sample_quad_depth_lift(depth_img, quad_full)
                        if lift_stats is not None:
                            quad_depth_lift = float(lift_stats["lift_m"])
                            if quad_depth_lift < req_lift:
                                if debug_on:
                                    debug_stats["reject_lift"] += 1
                                continue
                        elif source in ("color-edge", "dark-blob") and (
                            quad_depth_valid is not None and quad_depth_valid >= 0.30
                        ):
                            if debug_on:
                                debug_stats["reject_lift"] += 1
                            continue

                    plane_stats = self._sample_quad_plane_stats(depth_img, quad_full)
                    if plane_stats is not None:
                        quad_abs_nz = float(plane_stats["abs_nz"])
                        quad_depth_span = float(plane_stats["depth_span_m"])
                        req_nz, req_span = self._effective_top_face_constraints(source)
                        if quad_abs_nz < req_nz:
                            if debug_on:
                                debug_stats["reject_nz"] += 1
                            continue
                        if req_span > 0.0 and quad_depth_span > req_span:
                            if debug_on:
                                debug_stats["reject_span"] += 1
                            continue
                    elif source in ("color-edge", "dark-blob"):
                        if debug_on:
                            debug_stats["reject_nz"] += 1
                        continue

                center_weight = max(0.10, 1.0 - (center_dist / max_center_dist))
                center_weight = center_weight ** center_power
                square_bonus = max(0.52, 1.0 - 0.50 * abs(float(np.log(max(ratio, 1e-6)))))
                extent_bonus = max(0.20, min(1.0, extent)) ** 1.2
                source_weight = self._source_weight(source, depth_valid_ratio, fallback=True)
                white_ratio = None
                white_bonus = 1.0
                if work_hsv is not None:
                    white_stats = self._sample_quad_white_stats(work_hsv, quad_local)
                    if white_stats is not None:
                        white_ratio = float(white_stats["white_ratio"])
                        white_bonus = self._white_score_bonus(white_stats, source)
                band_bonus = 1.0
                if band_info is not None and band_info.get("band_idx") is not None:
                    band_conf = float(band_info.get("band_conf", 0.0))
                    band_bonus += float(np.clip(self.cfg.depth_top_band_score_gain, 0.0, 0.5)) * band_conf
                    band_target = band_info.get("band_target_m")
                    if (
                        band_target is not None
                        and source in ("depth-top", "depth-band")
                        and float(band_target) <= float(self.cfg.depth_top_small_object_height_m)
                    ):
                        band_bonus += float(np.clip(self.cfg.object_small_band_score_gain, 0.0, 0.6)) * max(0.40, band_conf)
                area_score = float(
                    np.power(max(box_area, 1.0), np.clip(self.cfg.object_area_score_power, 0.25, 0.60))
                )
                score = (
                    area_score
                    * center_weight
                    * square_bonus
                    * extent_bonus
                    * source_weight
                    * no_depth_penalty
                    * white_bonus
                    * band_bonus
                )
                if depth_m is not None:
                    near_bonus = max(0.0, self.cfg.target_max_distance_m - depth_m)
                    score *= 1.0 + (near_bonus / max(self.cfg.target_max_distance_m, 1e-6)) * float(
                        np.clip(self.cfg.object_near_score_gain, 0.0, 0.35) * 0.85
                    )
                if quad_depth_std is not None:
                    score *= max(0.5, 1.0 - (quad_depth_std / max(self.cfg.quad_depth_std_max_m, 1e-6)))
                if quad_abs_nz is not None:
                    score *= max(0.55, 0.35 + quad_abs_nz)
                if quad_depth_span is not None and self.cfg.top_face_depth_span_max_m > 0.0:
                    span_ref = max(0.005, float(self.cfg.top_face_depth_span_max_m))
                    score *= max(0.50, 1.0 - (quad_depth_span / (span_ref * 2.2)))

                candidates.append(
                    {
                        "score": score,
                        "quad": quad_full,
                        "contour": contour_full,
                        "area": area,
                        "ratio": ratio,
                        "center": center_full,
                        "depth_m": depth_m,
                        "source": source,
                        "quad_depth_std": quad_depth_std,
                        "quad_depth_lift": quad_depth_lift,
                        "quad_abs_nz": quad_abs_nz,
                        "quad_depth_span": quad_depth_span,
                        "white_ratio": white_ratio,
                        "height_m": (None if band_info is None else band_info.get("height_m")),
                        "height_band_idx": (None if band_info is None else band_info.get("band_idx")),
                        "height_band_target_m": (None if band_info is None else band_info.get("band_target_m")),
                        "height_band_error_m": (None if band_info is None else band_info.get("band_error_m")),
                        "height_band_conf": (0.0 if band_info is None else band_info.get("band_conf", 0.0)),
                    }
                )
                if debug_on:
                    key = f"{source}:fb"
                    debug_stats["cand_by_source"][key] = debug_stats["cand_by_source"].get(key, 0) + 1

        max_objects = max(1, int(self.cfg.max_objects))
        if len(self.prev_slot_quads) != max_objects:
            self.prev_slot_quads = [None for _ in range(max_objects)]
            self.prev_display_slot_quads = [None for _ in range(max_objects)]
            self.slot_miss_counts = [0 for _ in range(max_objects)]
            self.lock_slot_quads = [None for _ in range(max_objects)]
            self.lock_slot_depths = [None for _ in range(max_objects)]
            self.slot_lock_streaks = [0 for _ in range(max_objects)]
            self.slot_locked_flags = [False for _ in range(max_objects)]
            self.slot_locked_ages = [0 for _ in range(max_objects)]
            self.slot_move_release_streaks = [0 for _ in range(max_objects)]
            self.slot_size_labels = [None for _ in range(max_objects)]

        # Use only recently matched slots for continuity scoring.
        prev_refs = [
            q.astype(np.float32)
            for q, miss in zip(self.prev_slot_quads, self.slot_miss_counts)
            if q is not None and miss == 0
        ]
        if prev_refs:
            for cand in candidates:
                cand_quad = cand["quad"].astype(np.float32)
                best_move_mean = None
                best_move_max = None
                for ref_quad in prev_refs:
                    aligned = self._align_quad_to_reference(cand_quad, ref_quad)
                    move = np.linalg.norm(aligned - ref_quad, axis=1)
                    move_mean = float(np.mean(move))
                    move_max = float(np.max(move))
                    if best_move_mean is None or move_mean < best_move_mean:
                        best_move_mean = move_mean
                        best_move_max = move_max
                if best_move_mean is not None:
                    continuity = float(np.exp(-best_move_mean / 16.0))
                    continuity_gain = 1.0 + float(self.cfg.corner_continuity_weight) * continuity
                    cand["score"] *= continuity_gain
                    cand["move_mean"] = best_move_mean
                    cand["move_max"] = best_move_max

        selected_candidates = []
        slot_candidates = [None for _ in range(max_objects)]
        if candidates:
            selected_candidates = self._select_top_candidates(
                candidates,
                limit=max_objects,
                iou_threshold=float(self.cfg.object_nms_iou),
            )
            slot_candidates = self._assign_candidates_to_slots(selected_candidates, max_objects)
        self.last_candidates = [dict(c) for c in candidates]
        self.last_selected_candidates = [dict(c) for c in selected_candidates]
        if debug_on:
            interval = max(1, int(self.cfg.debug_print_interval))
            if self.debug_frame_index % interval == 0:
                in_by = ",".join(f"{k}:{v}" for k, v in sorted(debug_stats["input_by_source"].items()))
                can_by = ",".join(f"{k}:{v}" for k, v in sorted(debug_stats["cand_by_source"].items()))
                print(
                    f"[DBG] f={self.debug_frame_index} mode={contour_mode} depth_fg={depth_valid_ratio:.2f} "
                    f"contours={len(contour_entries)} ({in_by}) cand={len(candidates)} ({can_by}) "
                    f"sel={len(selected_candidates)} rej(lift/nz/span)="
                    f"{debug_stats['reject_lift']}/{debug_stats['reject_nz']}/{debug_stats['reject_span']}"
                )
                if selected_candidates:
                    parts = []
                    for idx, cand in enumerate(selected_candidates[:max_objects], start=1):
                        cxy = np.mean(cand["quad"].astype(np.float32), axis=0)
                        parts.append(
                            "O{idx}:{src}@({x:.0f},{y:.0f}) z={z} lift={lift} nz={nz} span={span}".format(
                                idx=idx,
                                src=cand.get("source", "na"),
                                x=float(cxy[0]),
                                y=float(cxy[1]),
                                z=("na" if cand.get("depth_m") is None else f"{float(cand['depth_m']):.3f}"),
                                lift=("na" if cand.get("quad_depth_lift") is None else f"{float(cand['quad_depth_lift']):.3f}"),
                                nz=("na" if cand.get("quad_abs_nz") is None else f"{float(cand['quad_abs_nz']):.2f}"),
                                span=("na" if cand.get("quad_depth_span") is None else f"{float(cand['quad_depth_span']):.3f}"),
                            )
                        )
                    print("[DBG] " + " | ".join(parts))

        active_items = []
        show_hold = bool(self.cfg.show_hold_slots)
        for slot_idx in range(max_objects):
            cand = slot_candidates[slot_idx] if slot_idx < len(slot_candidates) else None
            if cand is not None:
                quad = cand["quad"].astype(np.float32)
                prev_quad = self.prev_slot_quads[slot_idx]
                prev_miss = self.slot_miss_counts[slot_idx] if slot_idx < len(self.slot_miss_counts) else 0
                # Snap immediately on the first live frame after acquire/reacquire.
                # Once the slot is continuously tracked again, restore smoothing.
                if prev_quad is not None and prev_miss == 0:
                    quad = self._stabilize_quad(quad, prev_quad)
                quad, locked_depth = self._apply_slot_coordinate_lock(slot_idx, quad, cand.get("depth_m"))
                cand["quad"] = quad
                if locked_depth is not None:
                    cand["depth_m"] = float(locked_depth)
                self.prev_slot_quads[slot_idx] = quad.copy()
                self.slot_miss_counts[slot_idx] = 0
                active_items.append((slot_idx, cand, False))
            else:
                prev_quad = self.prev_slot_quads[slot_idx]
                if prev_quad is None:
                    self._clear_slot_lock(slot_idx)
                    continue
                self.slot_miss_counts[slot_idx] += 1
                sticky_locked = (
                    slot_idx < len(self.slot_locked_flags)
                    and bool(self.slot_locked_flags[slot_idx])
                    and bool(self.cfg.coord_lock_static_mode)
                    and bool(self.cfg.coord_lock_sticky_slots)
                )
                if sticky_locked:
                    locked_quad = self.lock_slot_quads[slot_idx]
                    if locked_quad is not None:
                        self.prev_slot_quads[slot_idx] = locked_quad.copy()
                        hold_cand = {
                            "quad": locked_quad.copy(),
                            "source": "hold-lock",
                            "depth_m": self.lock_slot_depths[slot_idx],
                            "quad_depth_std": None,
                            "move_mean": None,
                        }
                        active_items.append((slot_idx, hold_cand, True))

                    sticky_limit = max(1, int(self.cfg.coord_lock_sticky_max_miss))
                    if self.slot_miss_counts[slot_idx] > sticky_limit:
                        self.prev_slot_quads[slot_idx] = None
                        if slot_idx < len(self.prev_display_slot_quads):
                            self.prev_display_slot_quads[slot_idx] = None
                        self.slot_miss_counts[slot_idx] = 0
                        if slot_idx < len(self.slot_size_labels):
                            self.slot_size_labels[slot_idx] = None
                        self._clear_slot_lock(slot_idx)
                    continue

                if show_hold and self.slot_miss_counts[slot_idx] <= self.cfg.corner_hold_frames:
                    hold_cand = {
                        "quad": prev_quad.copy(),
                        "source": "hold",
                        "depth_m": None,
                        "quad_depth_std": None,
                        "move_mean": None,
                    }
                    active_items.append((slot_idx, hold_cand, True))
                if (not show_hold) or self.slot_miss_counts[slot_idx] > self.cfg.corner_hold_frames:
                    self.prev_slot_quads[slot_idx] = None
                    if slot_idx < len(self.prev_display_slot_quads):
                        self.prev_display_slot_quads[slot_idx] = None
                    self.slot_miss_counts[slot_idx] = 0
                    if slot_idx < len(self.slot_size_labels):
                        self.slot_size_labels[slot_idx] = None
                    self._clear_slot_lock(slot_idx)

        if not active_items:
            self.last_tracked_objects = []
            not_found_msg = (
                f"Rectangle: not found (target <= {self.cfg.target_max_distance_m:.2f} m)"
                if depth_img is not None
                else "Rectangle: not found"
            )
            cv2.putText(
                annotated,
                not_found_msg,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            return annotated

        palette = [
            (0, 0, 255),
            (0, 255, 255),
            (255, 180, 0),
            (255, 0, 255),
            (120, 255, 120),
        ]
        detected_count = 0
        hold_count = 0
        status_chunks = []
        size_counts: dict[str, int] = {}
        legend_rows = []
        center_rows = []
        tracked_objects = []

        for slot_idx, cand, is_hold in active_items:
            quad = cand["quad"].astype(np.float32)
            draw_quad = quad
            if not is_hold:
                draw_quad = self._refine_display_quad_from_contour(cand.get("contour"), quad)
            draw_quad = self._stabilize_display_quad(slot_idx, draw_quad)
            if slot_idx < len(self.prev_display_slot_quads):
                self.prev_display_slot_quads[slot_idx] = draw_quad.copy()
            quad_i = draw_quad.astype(np.int32).reshape((-1, 1, 2))
            color = palette[slot_idx % len(palette)]
            if not bool(self.cfg.corner_only_display):
                cv2.drawContours(annotated, [quad_i], -1, (255, 0, 0), 3 if not is_hold else 2)

            point_color = (0, 165, 255) if is_hold else color
            for pt in draw_quad.astype(np.int32):
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(annotated, (x, y), 5, point_color, -1)

            center_xy = np.mean(draw_quad, axis=0).astype(np.int32)
            legend_size = None
            depth_val = cand.get("depth_m")
            if depth_val is None:
                depth_val = self._sample_depth_m(depth_img, center_xy)

            # Draw center point marker.
            cv2.circle(annotated, (int(center_xy[0]), int(center_xy[1])), 5, (255, 255, 255), -1)
            cv2.circle(annotated, (int(center_xy[0]), int(center_xy[1])), 8, color, 2)

            if self.cfg.measure_size and (not is_hold):
                size_info = self._estimate_quad_size_cm(quad, depth_img)
                if size_info is not None:
                    w_cm = size_info["w_cm"]
                    h_cm = size_info["h_cm"]
                    legend_size = (w_cm, h_cm)
            size_label = self._classify_candidate_size_label(cand, legend_size)
            if size_label is None and slot_idx < len(self.slot_size_labels):
                size_label = self.slot_size_labels[slot_idx]
            elif size_label is not None and slot_idx < len(self.slot_size_labels):
                self.slot_size_labels[slot_idx] = size_label
            if size_label:
                size_counts[size_label] = size_counts.get(size_label, 0) + 1
            object_title = f"객체 {slot_idx + 1}" + (f" ({size_label})" if size_label else "")

            lock_tag = "L" if (slot_idx < len(self.slot_locked_flags) and self.slot_locked_flags[slot_idx]) else "-"
            status_chunks.append(f"O{slot_idx + 1}:{cand.get('source', 'na')}:{lock_tag}")
            if legend_size is not None:
                legend_rows.append((object_title, color))
                legend_rows.append((f"  가로 : {legend_size[0]:.1f} cm", color))
                legend_rows.append((f"  세로 : {legend_size[1]:.1f} cm", color))
            elif is_hold:
                legend_rows.append((object_title, (0, 165, 255)))
                legend_rows.append(("  유지(이전값)", (0, 165, 255)))
            elif not self.cfg.measure_size:
                legend_rows.append((object_title, color))
                legend_rows.append(("  크기 측정 끔", color))
            else:
                legend_rows.append((object_title, color))
                legend_rows.append(("  크기 측정 불가", color))
            legend_rows.append(("", (180, 180, 180)))

            center_rows.append((object_title, color))
            robot_coords = self._estimate_robot_coords_mm(center_xy, depth_val, cand.get("height_m"))
            if robot_coords is not None:
                center_rows.append((f"  로봇 X: {robot_coords['x_mm']:+.0f} mm", color))
                center_rows.append((f"  로봇 Y: {robot_coords['y_mm']:+.0f} mm", color))
                if robot_coords["z_mm"] is None:
                    center_rows.append(("  로봇 Z: 계산 불가", color))
                else:
                    center_rows.append((f"  로봇 Z: {robot_coords['z_mm']:.0f} mm", color))
            else:
                center_rows.append(("  로봇 X: 계산 불가", color))
                center_rows.append(("  로봇 Y: 계산 불가", color))
                center_rows.append(("  로봇 Z: 계산 불가", color))
            center_rows.append(("", (180, 180, 180)))
            if not is_hold:
                tracked_obj = {
                            "id": slot_idx + 1,
                            "x": int(center_xy[0]),
                            "y": int(center_xy[1]),
                            "z": None if depth_val is None else float(depth_val),
                        }
                if robot_coords is not None:
                    tracked_obj.update(
                        {
                            "robot_x_mm": float(robot_coords["x_mm"]),
                            "robot_y_mm": float(robot_coords["y_mm"]),
                            "robot_z_mm": None if robot_coords["z_mm"] is None else float(robot_coords["z_mm"]),
                        }
                    )
                tracked_objects.append(tracked_obj)
            if is_hold:
                hold_count += 1
            else:
                detected_count += 1

        status = f"Objects: {detected_count}/{max_objects}"
        target_labels = []
        for target_h in self.cfg.depth_top_height_targets_m:
            label = self._cube_size_label_from_target_height(target_h)
            if label and label not in target_labels:
                target_labels.append(label)
        if target_labels:
            status += " | " + " | ".join(f"{label}: {size_counts.get(label, 0)}" for label in target_labels)

        cv2.putText(
            annotated,
            status,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (80, 255, 80),
            2,
            cv2.LINE_AA,
        )
        self.last_tracked_objects = tracked_objects
        self._draw_center_legend_bottom_right(annotated, center_rows)
        return annotated

    def _get_candidate_contours(self, work_img: np.ndarray, work_depth: Optional[np.ndarray]):
        def _find_color_contours():
            gray_local = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
            gray_local = cv2.GaussianBlur(gray_local, (5, 5), 0)
            # Contrast boost helps black-object edges on mixed bright backgrounds.
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_local = clahe.apply(gray_local)
            med_local = float(np.median(gray_local))
            canny_low_local = int(max(10, 0.45 * med_local))
            canny_high_local = int(min(255, max(canny_low_local + 16, 1.15 * med_local)))
            edges_a = cv2.Canny(gray_local, canny_low_local, canny_high_local)
            edges_b = cv2.Canny(
                gray_local,
                max(6, canny_low_local // 2),
                min(255, max(canny_high_local, canny_low_local + 26)),
            )
            edges_local = cv2.bitwise_or(edges_a, edges_b)
            k_local = np.ones((3, 3), dtype=np.uint8)
            edges_local = cv2.morphologyEx(edges_local, cv2.MORPH_CLOSE, k_local, iterations=1)
            return cv2.findContours(edges_local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        def _find_dark_contours():
            gray_local = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
            gray_local = cv2.GaussianBlur(gray_local, (5, 5), 0)
            p20 = float(np.percentile(gray_local, 20))
            p45 = float(np.percentile(gray_local, 45))
            dark_thr = int(np.clip(0.65 * p20 + 0.35 * p45, 18, 120))
            dark_mask = np.zeros_like(gray_local, dtype=np.uint8)
            dark_mask[gray_local <= dark_thr] = 255
            k_dark = np.ones((3, 3), dtype=np.uint8)
            dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, k_dark, iterations=1)
            dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, k_dark, iterations=2)
            contours = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            return contours, dark_thr

        if work_depth is not None:
            depth_m = work_depth.astype(np.float32) * self.depth_scale
            valid = (
                (work_depth > 0)
                & (depth_m >= self.cfg.target_min_distance_m)
                & (depth_m <= self.cfg.target_max_distance_m)
            )
            valid_ratio = float(np.count_nonzero(valid)) / float(valid.size + 1e-6)
            all_contours = []
            mode_tags = []
            support_depth_for_candidates = None

            if np.count_nonzero(valid) > 500:
                valid_depth = depth_m[valid]
                depth_priority_contours = []
                support_depth_est = self._estimate_support_depth_m(depth_m, valid)
                support_depth = (
                    float(support_depth_est)
                    if support_depth_est is not None
                    else float(np.percentile(valid_depth, np.clip(self.cfg.depth_top_support_percentile, 70.0, 98.0)))
                )
                support_depth_for_candidates = support_depth
                if self.support_plane_coeffs is not None:
                    yy, xx = np.indices(depth_m.shape, dtype=np.float32)
                    coeffs = self.support_plane_coeffs.astype(np.float32)
                    support_depth_map = coeffs[0] * xx + coeffs[1] * yy + coeffs[2]
                    support_depth = float(np.median(support_depth_map[valid])) if np.any(valid) else support_depth
                else:
                    support_depth_map = np.full(depth_m.shape, support_depth, dtype=np.float32)
                if bool(self.cfg.depth_top_priority_enabled):
                    min_height = max(float(self.cfg.target_min_height_m), float(self.cfg.depth_top_min_height_m))
                    max_height = max(min_height + 0.010, float(self.cfg.depth_top_max_height_m))
                    top_height_m = support_depth_map - depth_m
                    if bool(self.cfg.depth_top_known_height_bands_enabled):
                        h_tol = max(0.006, float(self.cfg.depth_top_height_tol_m))
                        for target_h in self.cfg.depth_top_height_targets_m:
                            target_h = float(target_h)
                            if target_h < min_height - h_tol or target_h > max_height + h_tol:
                                continue
                            height_mask = valid & (np.abs(top_height_m - target_h) <= h_tol)
                            if np.count_nonzero(height_mask) < max(90, int(self.cfg.rect_min_area * 0.18)):
                                continue
                            height_fg = np.zeros(work_depth.shape, dtype=np.uint8)
                            height_fg[height_mask] = 255
                            height_kernel = np.ones((3, 3), dtype=np.uint8)
                            height_fg = cv2.morphologyEx(height_fg, cv2.MORPH_OPEN, height_kernel, iterations=1)
                            height_fg = cv2.morphologyEx(height_fg, cv2.MORPH_CLOSE, height_kernel, iterations=1)
                            comp_count_h, comp_labels_h, comp_stats_h, _ = cv2.connectedComponentsWithStats(
                                height_fg,
                                connectivity=8,
                            )
                            min_area_h = max(90, int(self.cfg.rect_min_area * 0.16))
                            band_added = 0
                            for label_idx in range(1, int(comp_count_h)):
                                comp_area_h = int(comp_stats_h[label_idx, cv2.CC_STAT_AREA])
                                if comp_area_h < min_area_h:
                                    continue
                                comp_mask_h = comp_labels_h == label_idx
                                comp_depth_h = depth_m[comp_mask_h]
                                if comp_depth_h.size < 40:
                                    continue
                                top_z_h = float(np.percentile(comp_depth_h, 25))
                                split_masks_h = self._split_depth_top_component_masks(
                                    comp_mask_h,
                                    top_z_h,
                                    target_height_m=target_h,
                                )
                                for split_mask in split_masks_h:
                                    split_area = int(np.count_nonzero(split_mask))
                                    if split_area < min_area_h:
                                        continue
                                    split_fg = np.zeros(work_depth.shape, dtype=np.uint8)
                                    split_fg[split_mask] = 255
                                    contours_h, _ = cv2.findContours(
                                        split_fg,
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE,
                                    )
                                    if not contours_h:
                                        continue
                                    depth_priority_contours.extend((c, "depth-top") for c in contours_h)
                                    band_added += len(contours_h)
                            if band_added > 0:
                                mode_tags.append(f"depth-top-h{int(round(target_h * 100.0))}cm:{band_added}")

                    top_mask = valid & (top_height_m >= min_height) & (top_height_m <= max_height)
                    if np.count_nonzero(top_mask) >= max(220, int(self.cfg.rect_min_area * 0.45)):
                        top_fg = np.zeros(work_depth.shape, dtype=np.uint8)
                        top_fg[top_mask] = 255
                        top_kernel = np.ones((5, 5), dtype=np.uint8)
                        top_fg = cv2.morphologyEx(top_fg, cv2.MORPH_OPEN, top_kernel, iterations=1)
                        top_fg = cv2.morphologyEx(top_fg, cv2.MORPH_CLOSE, top_kernel, iterations=1)
                        comp_count, comp_labels, comp_stats, _ = cv2.connectedComponentsWithStats(top_fg, connectivity=8)
                        band_hw = max(0.004, float(self.cfg.depth_top_band_m))
                        min_comp_area = max(120, int(self.cfg.rect_min_area * 0.32))
                        for label_idx in range(1, int(comp_count)):
                            comp_area = int(comp_stats[label_idx, cv2.CC_STAT_AREA])
                            if comp_area < min_comp_area:
                                continue
                            comp_mask = comp_labels == label_idx
                            comp_depth = depth_m[comp_mask]
                            if comp_depth.size < 80:
                                continue
                            top_z = float(np.percentile(comp_depth, 25))
                            top_band_mask = comp_mask & (np.abs(depth_m - top_z) <= band_hw)
                            if np.count_nonzero(top_band_mask) < max(90, int(comp_area * 0.30)):
                                continue
                            split_masks = self._split_depth_top_component_masks(top_band_mask, top_z)
                            for split_mask in split_masks:
                                top_band_fg = np.zeros(work_depth.shape, dtype=np.uint8)
                                top_band_fg[split_mask] = 255
                                band_kernel = np.ones((3, 3), dtype=np.uint8)
                                top_band_fg = cv2.morphologyEx(top_band_fg, cv2.MORPH_OPEN, band_kernel, iterations=1)
                                top_band_fg = cv2.morphologyEx(top_band_fg, cv2.MORPH_CLOSE, band_kernel, iterations=1)
                                contours, _ = cv2.findContours(top_band_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                if contours:
                                    depth_priority_contours.extend((c, "depth-top") for c in contours)
                        if depth_priority_contours:
                            all_contours.extend(depth_priority_contours)
                            mode_tags.append(
                                f"depth-top(zbg={support_depth:.3f},h={min_height:.3f}~{max_height:.3f},bw={band_hw:.3f})"
                            )

                # 0) Depth-band candidate around nearest stable layer (usually box front face).
                z_near = float(np.percentile(valid_depth, 12))
                band_hw = max(0.01, float(self.cfg.target_band_half_width_m))
                band_mask = valid & (np.abs(depth_m - z_near) <= band_hw)
                band_ratio = float(np.count_nonzero(band_mask)) / float(band_mask.size + 1e-6)
                if band_ratio >= 0.004:
                    band_fg = np.zeros(work_depth.shape, dtype=np.uint8)
                    band_fg[band_mask] = 255
                    band_kernel = np.ones((5, 5), dtype=np.uint8)
                    band_fg = cv2.morphologyEx(band_fg, cv2.MORPH_OPEN, band_kernel, iterations=1)
                    band_fg = cv2.morphologyEx(band_fg, cv2.MORPH_CLOSE, band_kernel, iterations=1)
                    band_contours, _ = cv2.findContours(band_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if band_contours:
                        all_contours.extend((c, "depth-band") for c in band_contours)
                        mode_tags.append(f"depth-band(z={z_near:.3f},bw={band_hw:.3f})")

                bg_depth = support_depth
                bg_depth_map = support_depth_map

                # Segment object as "closer than dominant table/background depth".
                delta = max(float(self.cfg.target_min_height_m), 0.002)
                near_mask = np.zeros_like(valid, dtype=bool)
                near_ratio = 0.0
                for _ in range(4):
                    near_mask = valid & (depth_m <= (bg_depth_map - delta))
                    near_ratio = float(np.count_nonzero(near_mask)) / float(near_mask.size + 1e-6)
                    if near_ratio <= 0.45 or delta >= 0.02:
                        break
                    delta *= 1.5

                if near_ratio >= 0.003:
                    fg = np.zeros(work_depth.shape, dtype=np.uint8)
                    fg[near_mask] = 255
                    kernel = np.ones((5, 5), dtype=np.uint8)
                    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
                    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=1)
                    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        all_contours.extend((c, "depth-plane") for c in contours)
                        mode_tags.append(f"depth-plane(dbg={bg_depth:.3f},dh={delta:.3f})")

                # Fallback: depth discontinuity edges (ignores printed texture).
                depth_f = depth_m.copy()
                med_depth = float(np.median(valid_depth))
                depth_f[~valid] = med_depth
                depth_f = cv2.GaussianBlur(depth_f, (5, 5), 0)
                gx = cv2.Sobel(depth_f, cv2.CV_32F, 1, 0, 3)
                gy = cv2.Sobel(depth_f, cv2.CV_32F, 0, 1, 3)
                grad = cv2.magnitude(gx, gy)
                grad_valid = grad[valid]
                thr = max(0.0012, float(np.percentile(grad_valid, 90))) if grad_valid.size > 0 else 0.0012
                edges = np.zeros(work_depth.shape, dtype=np.uint8)
                edges[(grad >= thr) & valid] = 255
                e_kernel = np.ones((3, 3), dtype=np.uint8)
                edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, e_kernel, iterations=2)
                edges = cv2.dilate(edges, e_kernel, iterations=1)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    all_contours.extend((c, "depth-edge") for c in contours)
                    mode_tags.append(f"depth-edge(th={thr:.4f})")

                if depth_priority_contours:
                    return all_contours, "+".join(mode_tags), valid_ratio

            dark_contours, dark_thr = _find_dark_contours()
            if dark_contours:
                all_contours.extend((c, "dark-blob") for c in dark_contours)
                mode_tags.append(f"dark-blob(t={dark_thr})")

            color_contours = _find_color_contours()
            if color_contours:
                all_contours.extend((c, "color-edge") for c in color_contours)
                mode_tags.append("color-edge")

            if all_contours:
                return all_contours, "+".join(mode_tags), valid_ratio

            # Very sparse depth -> fallback to color edges.
            dark_contours, dark_thr = _find_dark_contours()
            color_contours = _find_color_contours()
            all_fb = []
            if dark_contours:
                all_fb.extend((c, "dark-blob") for c in dark_contours)
            if color_contours:
                all_fb.extend((c, "color-edge") for c in color_contours)
            return all_fb, f"edge-fallback-low-depth+dark(t={dark_thr})", valid_ratio

        dark_contours, dark_thr = _find_dark_contours()
        color_contours = _find_color_contours()
        all_fb = []
        if dark_contours:
            all_fb.extend((c, "dark-blob") for c in dark_contours)
        if color_contours:
            all_fb.extend((c, "color-edge") for c in color_contours)
        return all_fb, f"edge-fallback+dark(t={dark_thr})", 0.0

    def _sample_depth_m(self, depth_img: Optional[np.ndarray], center_xy):
        if depth_img is None:
            return None
        cx, cy = int(center_xy[0]), int(center_xy[1])
        h, w = depth_img.shape[:2]
        x0, x1 = max(cx - 2, 0), min(cx + 3, w)
        y0, y1 = max(cy - 2, 0), min(cy + 3, h)
        patch = depth_img[y0:y1, x0:x1]
        valid = patch[patch > 0]
        if valid.size == 0:
            return None
        return float(np.median(valid) * self.depth_scale)

    def _estimate_robot_coords_mm(self, center_xy, depth_m: Optional[float], object_height_m: Optional[float] = None):
        if depth_m is None or self.measure_intrinsics is None:
            return None

        intr = self.measure_intrinsics
        px = float(center_xy[0])
        py = float(center_xy[1])
        object_depth_m = float(depth_m)
        support_depth_m = self._predict_support_depth_m(center_xy)
        xy_depth_m = float(support_depth_m) if support_depth_m is not None else object_depth_m

        # Camera-centered graph-style coordinates:
        # +X is image right, +Y is image up. X/Y are projected onto the belt top plane,
        # while Z is object height above the belt top.
        x_cm = ((px - float(intr.ppx)) * xy_depth_m / float(intr.fx)) * 100.0
        y_cm = ((float(intr.ppy) - py) * xy_depth_m / float(intr.fy)) * 100.0

        if (
            support_depth_m is not None
            and object_height_m is not None
            and np.isfinite(float(object_height_m))
        ):
            support_m = float(support_depth_m)
            obj_h_m = max(0.0, float(object_height_m))
            ref_h_m = max(0.0, float(ROBOT_XY_REFERENCE_HEIGHT_M))
            if support_m > max(obj_h_m, ref_h_m) + 0.05:
                height_factor = (support_m - obj_h_m) / max(support_m - ref_h_m, 1e-6)
                height_factor = float(np.clip(height_factor, 0.80, 1.08))
                x_cm *= height_factor
                y_cm *= height_factor
                if obj_h_m > ref_h_m + 0.01:
                    y_cm *= float(ROBOT_TALL_OBJECT_Y_GAIN)

        x_cm = x_cm * float(ROBOT_X_SCALE) + float(ROBOT_X_OFFSET_CM)
        y_cm = y_cm * float(ROBOT_Y_SCALE) + float(ROBOT_Y_OFFSET_CM)

        if object_height_m is not None and np.isfinite(float(object_height_m)):
            z_cm = max(0.0, float(object_height_m) * 100.0)
        else:
            if support_depth_m is None:
                z_cm = None
            else:
                z_cm = max(0.0, (float(support_depth_m) - object_depth_m) * 100.0)

        return {
            "x_mm": float(x_cm * 10.0),
            "y_mm": float(y_cm * 10.0),
            "z_mm": None if z_cm is None else float(z_cm * 10.0),
            "depth_m": object_depth_m,
        }

    def _fit_support_plane_model(self, depth_m: np.ndarray, support_mask: np.ndarray) -> Optional[np.ndarray]:
        ys, xs = np.where(support_mask)
        if xs.size < 600:
            return None

        if xs.size > 5000:
            step = int(np.ceil(xs.size / 5000.0))
            xs = xs[::step]
            ys = ys[::step]
        z = depth_m[ys, xs].astype(np.float32)
        if z.size < 300:
            return None

        a = np.stack(
            (
                xs.astype(np.float32),
                ys.astype(np.float32),
                np.ones(xs.shape[0], dtype=np.float32),
            ),
            axis=1,
        )
        try:
            coeffs, _, _, _ = np.linalg.lstsq(a, z, rcond=None)
        except np.linalg.LinAlgError:
            return None

        pred = a @ coeffs
        resid = z - pred
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med)))
        resid_thr = max(0.0035, 2.8 * 1.4826 * mad)
        keep = np.abs(resid - med) <= resid_thr
        if int(np.count_nonzero(keep)) >= 240:
            a_keep = a[keep]
            z_keep = z[keep]
            try:
                coeffs, _, _, _ = np.linalg.lstsq(a_keep, z_keep, rcond=None)
            except np.linalg.LinAlgError:
                pass

        return coeffs.astype(np.float32)

    def _predict_support_depth_m(self, xy) -> Optional[float]:
        if self.support_plane_coeffs is not None:
            x = float(xy[0])
            y = float(xy[1])
            coeffs = self.support_plane_coeffs.astype(np.float32)
            return float(coeffs[0] * x + coeffs[1] * y + coeffs[2])
        if self.support_depth_m is not None:
            return float(self.support_depth_m)
        return None

    def _estimate_support_depth_m(self, depth_m: np.ndarray, valid_mask: np.ndarray) -> Optional[float]:
        valid_count = int(np.count_nonzero(valid_mask))
        if valid_count < 800:
            return None

        valid_depth = depth_m[valid_mask]
        if valid_depth.size < 800:
            return None

        q_lo = float(np.percentile(valid_depth, 55))
        q_hi = float(np.percentile(valid_depth, 99))
        if not np.isfinite(q_lo) or not np.isfinite(q_hi) or q_hi <= q_lo:
            return None

        bin_m = max(0.0015, float(self.cfg.support_depth_bin_m))
        band_m = max(bin_m, float(self.cfg.support_depth_band_m))
        bins = np.arange(q_lo, q_hi + bin_m, bin_m, dtype=np.float32)
        if bins.size < 3:
            return None

        hist, edges = np.histogram(valid_depth, bins=bins)
        if hist.size == 0:
            return None

        top_bin_indices = np.argsort(hist)[::-1][: min(6, hist.size)]
        best_center = None
        best_score = None
        best_support_mask = None
        for bin_idx in top_bin_indices:
            count = int(hist[bin_idx])
            if count < max(180, int(valid_depth.size * 0.015)):
                continue

            center = float(0.5 * (edges[bin_idx] + edges[bin_idx + 1]))
            band_mask = valid_mask & (np.abs(depth_m - center) <= band_m)
            band_count = int(np.count_nonzero(band_mask))
            if band_count < max(280, count):
                continue

            fg = np.zeros(depth_m.shape, dtype=np.uint8)
            fg[band_mask] = 255
            kernel = np.ones((5, 5), dtype=np.uint8)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=1)
            comp_count, comp_labels, comp_stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
            if comp_count <= 1:
                continue

            best_label = 1 + int(np.argmax(comp_stats[1:, cv2.CC_STAT_AREA]))
            largest_area = int(comp_stats[best_label, cv2.CC_STAT_AREA])
            far_bias = 1.0 + 0.25 * ((center - q_lo) / max(q_hi - q_lo, 1e-6))
            score = float(largest_area) * far_bias
            if best_score is None or score > best_score:
                best_score = score
                best_center = center
                best_support_mask = comp_labels == best_label

        if best_center is None:
            return None

        plane_coeffs = None
        if best_support_mask is not None:
            plane_coeffs = self._fit_support_plane_model(depth_m, best_support_mask)

        if self.support_depth_m is None:
            self.support_depth_m = float(best_center)
        else:
            alpha = 0.82
            if abs(float(best_center) - float(self.support_depth_m)) > 0.025:
                alpha = 0.55
            self.support_depth_m = alpha * float(self.support_depth_m) + (1.0 - alpha) * float(best_center)

        if plane_coeffs is not None:
            if self.support_plane_coeffs is None:
                self.support_plane_coeffs = plane_coeffs
            else:
                alpha_plane = 0.82
                if abs(float(best_center) - float(self.support_depth_m)) > 0.025:
                    alpha_plane = 0.55
                self.support_plane_coeffs = (
                    alpha_plane * self.support_plane_coeffs.astype(np.float32)
                    + (1.0 - alpha_plane) * plane_coeffs.astype(np.float32)
                ).astype(np.float32)
        return float(self.support_depth_m)

    def _split_depth_top_component_masks(
        self,
        component_mask: np.ndarray,
        top_z: float,
        target_height_m: Optional[float] = None,
    ) -> list[np.ndarray]:
        mask_u8 = np.zeros(component_mask.shape, dtype=np.uint8)
        mask_u8[component_mask] = 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        main_contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(main_contour))
        if area < max(120.0, float(self.cfg.rect_min_area) * 0.25):
            return [component_mask]

        rect = cv2.minAreaRect(main_contour)
        width, height = rect[1]
        short_side = max(1.0, min(float(width), float(height)))
        long_side = max(float(width), float(height))
        known_band_mode = target_height_m is not None
        small_object_mode = (
            target_height_m is not None
            and float(target_height_m) <= float(self.cfg.depth_top_small_object_height_m)
        )
        expected_side_px = None
        expected_area_px = None
        area_ratio = None
        if (
            target_height_m is not None
            and top_z > 0.05
            and self.measure_intrinsics is not None
        ):
            expected_side_px = float(
                0.5
                * (
                    (float(self.measure_intrinsics.fx) * float(target_height_m) / float(top_z))
                    + (float(self.measure_intrinsics.fy) * float(target_height_m) / float(top_z))
                )
            )
            expected_side_px = max(8.0, expected_side_px)
            expected_area_px = float(expected_side_px * expected_side_px)
        split_count = 1
        if bool(self.cfg.depth_top_split_enabled) and width > 1.0 and height > 1.0:
            aspect = long_side / short_side
            if aspect >= float(self.cfg.depth_top_split_aspect_ratio):
                split_count = int(np.clip(round(long_side / short_side), 1, 3))
            square_equiv = float(area) / max(short_side * short_side, 1.0)
            if square_equiv >= 1.55:
                split_count = max(split_count, int(np.clip(round(square_equiv), 1, 4)))
            if expected_area_px is not None:
                area_ratio = float(area) / max(expected_area_px, 1.0)
                if small_object_mode:
                    ratio_thr = float(self.cfg.depth_top_small_split_area_ratio)
                elif known_band_mode:
                    ratio_thr = float(self.cfg.depth_top_known_band_split_area_ratio)
                else:
                    ratio_thr = 1.55
                if area_ratio >= ratio_thr:
                    split_count = max(split_count, int(np.clip(round(area_ratio), 1, 4)))
                if known_band_mode and expected_side_px is not None:
                    if long_side >= expected_side_px * (1.40 if small_object_mode else 1.48):
                        split_count = max(split_count, 2)
                    if area_ratio >= (2.20 if small_object_mode else 2.30):
                        split_count = max(split_count, 3)

        if split_count <= 1:
            return [component_mask]

        if (
            known_band_mode
            and expected_side_px is not None
            and area_ratio is not None
            and split_count <= 3
        ):
            long_ratio = long_side / max(expected_side_px, 1e-6)
            short_ratio = short_side / max(expected_side_px, 1e-6)
            area_match_tol = 0.65 if small_object_mode else 0.75
            long_match_tol = 0.30 if small_object_mode else 0.36
            short_low = 0.64 if small_object_mode else 0.70
            short_high = 1.28 if small_object_mode else 1.36
            same_size_chain_like = (
                abs(float(area_ratio) - float(split_count)) <= area_match_tol
                and abs(long_ratio - float(split_count)) <= long_match_tol
                and short_low <= short_ratio <= short_high
            )
            if same_size_chain_like:
                direct_min_area = max(
                    70,
                    int(
                        area
                        * (
                            0.12
                            if small_object_mode
                            else 0.14
                        )
                    ),
                )
                direct_masks = self._split_component_mask_by_projection(
                    component_mask,
                    split_count,
                    min_area=direct_min_area,
                    expected_side_px=expected_side_px,
                )
                if len(direct_masks) >= 2:
                    return direct_masks

        # First try distance-transform peaks; this works better when same-height cubes touch.
        dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
        max_dist = float(np.max(dist))
        peak_split_enabled = bool(self.cfg.depth_top_peak_split_enabled) or known_band_mode
        if peak_split_enabled and max_dist >= 3.0:
            if small_object_mode and split_count <= 2:
                peak_ratio = 0.40
            elif known_band_mode and split_count <= 2:
                peak_ratio = 0.44
            else:
                peak_ratio = 0.48 if split_count <= 2 else 0.42
            peak_thresh = max(2.0, max_dist * peak_ratio)
            peak_basis = expected_side_px if expected_side_px is not None else short_side
            peak_kernel_size = int(
                max(
                    5,
                    round(
                        peak_basis
                        * (
                            0.14
                            if small_object_mode
                            else (0.17 if known_band_mode else 0.20)
                        )
                    ),
                )
            )
            if peak_kernel_size % 2 == 0:
                peak_kernel_size += 1
            peak_kernel = np.ones((peak_kernel_size, peak_kernel_size), dtype=np.uint8)
            dist_dilated = cv2.dilate(dist, peak_kernel)
            peak_mask = (dist >= peak_thresh) & (dist >= (dist_dilated - 1e-5)) & component_mask
            peak_u8 = np.zeros(component_mask.shape, dtype=np.uint8)
            peak_u8[peak_mask] = 255
            peak_u8 = cv2.morphologyEx(peak_u8, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)
            peak_count, peak_labels, peak_stats, peak_centroids = cv2.connectedComponentsWithStats(peak_u8, connectivity=8)
            peak_candidates = []
            for label_idx in range(1, int(peak_count)):
                peak_area = int(peak_stats[label_idx, cv2.CC_STAT_AREA])
                if peak_area < 6:
                    continue
                cx, cy = peak_centroids[label_idx]
                px = int(np.clip(round(cx), 0, dist.shape[1] - 1))
                py = int(np.clip(round(cy), 0, dist.shape[0] - 1))
                peak_value = float(dist[py, px])
                peak_candidates.append((peak_value, np.array([float(cx), float(cy)], dtype=np.float32)))

            if peak_candidates:
                peak_candidates.sort(key=lambda item: item[0], reverse=True)
                if small_object_mode:
                    basis = expected_side_px if expected_side_px is not None else short_side
                    min_sep = max(10.0, basis * 0.34)
                elif known_band_mode:
                    basis = expected_side_px if expected_side_px is not None else short_side
                    min_sep = max(12.0, basis * 0.42)
                else:
                    min_sep = max(14.0, short_side * 0.48)
                filtered_peaks = []
                for peak_value, peak_xy in peak_candidates:
                    if any(float(np.linalg.norm(peak_xy - kept_xy)) < min_sep for _, kept_xy in filtered_peaks):
                        continue
                    filtered_peaks.append((peak_value, peak_xy))

                target_count = min(len(filtered_peaks), max(2, split_count))
                if target_count >= 2:
                    seeds = np.stack([peak_xy for _, peak_xy in filtered_peaks[:target_count]], axis=0)
                    ys, xs = np.where(component_mask)
                    if xs.size >= (90 * target_count):
                        pts = np.stack((xs, ys), axis=1).astype(np.float32)
                        d2 = np.sum((pts[:, None, :] - seeds[None, :, :]) ** 2, axis=2)
                        labels = np.argmin(d2, axis=1)
                        split_masks: list[np.ndarray] = []
                        min_area = max(
                            70,
                            int(
                                area
                                * (
                                    0.12
                                    if small_object_mode
                                    else (0.14 if known_band_mode else 0.16)
                                )
                            ),
                        )
                        for cluster_idx in range(target_count):
                            cluster_mask = np.zeros(component_mask.shape, dtype=np.uint8)
                            cluster_pts = pts[labels == cluster_idx]
                            if cluster_pts.shape[0] < min_area:
                                continue
                            cluster_xy = np.round(cluster_pts).astype(np.int32)
                            cluster_mask[cluster_xy[:, 1], cluster_xy[:, 0]] = 255
                            kernel = np.ones((3, 3), dtype=np.uint8)
                            cluster_mask = cv2.morphologyEx(cluster_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
                            cluster_mask = cv2.bitwise_and(cluster_mask, mask_u8)
                            cluster_contours, _ = cv2.findContours(cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if not cluster_contours:
                                continue
                            cluster_contour = max(cluster_contours, key=cv2.contourArea)
                            if cv2.contourArea(cluster_contour) < min_area:
                                continue
                            refined_mask = np.zeros(component_mask.shape, dtype=np.uint8)
                            cv2.drawContours(refined_mask, [cluster_contour], -1, 255, thickness=-1)
                            split_masks.append(refined_mask > 0)
                        if len(split_masks) >= 2:
                            return split_masks

        min_area = max(
            70,
            int(
                area
                * (
                    0.13
                    if small_object_mode
                    else (0.15 if known_band_mode else 0.18)
                )
            ),
        )
        split_masks = self._split_component_mask_by_projection(
            component_mask,
            split_count,
            min_area=min_area,
            expected_side_px=(expected_side_px if known_band_mode else None),
        )
        return split_masks if len(split_masks) >= 2 else [component_mask]

    def _split_component_mask_by_projection(
        self,
        component_mask: np.ndarray,
        split_count: int,
        min_area: int,
        expected_side_px: Optional[float] = None,
    ) -> list[np.ndarray]:
        ys, xs = np.where(component_mask)
        if xs.size < (120 * split_count):
            return []

        pts = np.stack((xs, ys), axis=1).astype(np.float32)
        mean = np.mean(pts, axis=0, keepdims=True)
        centered = pts - mean
        if centered.shape[0] < split_count:
            return []

        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        major_axis = vh[0].astype(np.float32)
        proj = (centered @ major_axis.reshape(2, 1)).reshape(-1).astype(np.float32)
        if expected_side_px is not None:
            p_min = float(np.min(proj))
            p_max = float(np.max(proj))
            if (p_max - p_min) < max(6.0, expected_side_px * 0.90):
                return []
            edges = np.linspace(p_min, p_max, split_count + 1, dtype=np.float32)
            labels = np.digitize(proj, edges[1:-1], right=False).astype(np.int32)
        else:
            order = np.argsort(proj)
            labels = np.empty(proj.shape[0], dtype=np.int32)
            cuts = np.linspace(0, proj.shape[0], split_count + 1, dtype=np.int32)
            for cluster_idx in range(split_count):
                seg = order[cuts[cluster_idx] : cuts[cluster_idx + 1]]
                labels[seg] = cluster_idx

        split_masks: list[np.ndarray] = []
        for cluster_idx in range(split_count):
            cluster_mask = np.zeros(component_mask.shape, dtype=np.uint8)
            cluster_pts = pts[labels == cluster_idx]
            if cluster_pts.shape[0] < min_area:
                continue
            cluster_xy = np.round(cluster_pts).astype(np.int32)
            cluster_mask[cluster_xy[:, 1], cluster_xy[:, 0]] = 255
            kernel = np.ones((3, 3), dtype=np.uint8)
            cluster_mask = cv2.morphologyEx(cluster_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            cluster_mask = cv2.morphologyEx(cluster_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            cluster_contours, _ = cv2.findContours(cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cluster_contours:
                continue
            cluster_contour = max(cluster_contours, key=cv2.contourArea)
            if cv2.contourArea(cluster_contour) < min_area:
                continue
            refined_mask = np.zeros(component_mask.shape, dtype=np.uint8)
            cv2.drawContours(refined_mask, [cluster_contour], -1, 255, thickness=-1)
            split_masks.append(refined_mask > 0)

        return split_masks

    def _inset_quad(self, quad_xy: np.ndarray, inset_ratio: float) -> np.ndarray:
        ratio = float(np.clip(inset_ratio, 0.0, 0.45))
        if ratio <= 1e-6:
            return quad_xy.astype(np.float32)
        quad = quad_xy.astype(np.float32)
        center = np.mean(quad, axis=0, keepdims=True)
        return center + (quad - center) * (1.0 - ratio)

    @staticmethod
    def _intersect_parametric_lines(
        p1: np.ndarray,
        d1: np.ndarray,
        p2: np.ndarray,
        d2: np.ndarray,
    ) -> Optional[np.ndarray]:
        det = float(d1[0] * d2[1] - d1[1] * d2[0])
        if abs(det) < 1e-6:
            return None
        diff = p2 - p1
        t = float((diff[0] * d2[1] - diff[1] * d2[0]) / det)
        return (p1 + t * d1).astype(np.float32)

    def _refine_quad_from_contour_edges(self, contour: np.ndarray, quad_xy: np.ndarray) -> np.ndarray:
        quad = quad_xy.astype(np.float32)
        if not bool(self.cfg.quad_edge_refine_enabled):
            return quad
        if contour is None or len(contour) < 12:
            return quad

        contour_pts = contour.reshape(-1, 2).astype(np.float32)
        if contour_pts.shape[0] < 12:
            return quad

        side_vecs = np.roll(quad, -1, axis=0) - quad
        side_lengths = np.linalg.norm(side_vecs, axis=1)
        mean_side = float(np.mean(side_lengths)) if side_lengths.size else 0.0
        if mean_side < 8.0:
            return quad

        tangent_margin = float(max(6.0, mean_side * float(self.cfg.quad_edge_refine_tangent_ratio)))
        normal_margin = float(max(4.0, mean_side * float(self.cfg.quad_edge_refine_normal_ratio)))
        min_points = int(max(6, self.cfg.quad_edge_refine_min_points))
        fitted_lines = []

        for edge_idx in range(4):
            p0 = quad[edge_idx]
            p1 = quad[(edge_idx + 1) % 4]
            edge = p1 - p0
            edge_len = float(np.linalg.norm(edge))
            if edge_len < 1e-3:
                return quad
            tangent = edge / edge_len
            normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
            rel = contour_pts - p0
            proj = rel @ tangent
            dist = np.abs(rel @ normal)
            use = (proj >= -tangent_margin) & (proj <= (edge_len + tangent_margin)) & (dist <= normal_margin)
            edge_pts = contour_pts[use]
            if edge_pts.shape[0] < min_points:
                return quad
            fit = cv2.fitLine(edge_pts.reshape(-1, 1, 2), cv2.DIST_L2, 0, 0.01, 0.01)
            fit = np.asarray(fit, dtype=np.float32).reshape(-1)
            direction = np.array([float(fit[0]), float(fit[1])], dtype=np.float32)
            norm_dir = float(np.linalg.norm(direction))
            if norm_dir < 1e-6:
                return quad
            direction /= norm_dir
            if float(np.dot(direction, tangent)) < 0.0:
                direction = -direction
            origin = np.array([float(fit[2]), float(fit[3])], dtype=np.float32)
            fitted_lines.append((origin, direction))

        refined = []
        for corner_idx in range(4):
            line_a = fitted_lines[(corner_idx - 1) % 4]
            line_b = fitted_lines[corner_idx]
            corner = self._intersect_parametric_lines(line_a[0], line_a[1], line_b[0], line_b[1])
            if corner is None or not np.all(np.isfinite(corner)):
                return quad
            refined.append(corner)

        refined_quad = np.asarray(refined, dtype=np.float32)
        refined_quad = self._align_quad_to_reference(refined_quad, quad)
        if not self._is_rectangular_quad(refined_quad, max_cos=0.82):
            return quad

        area_ref = float(abs(cv2.contourArea(quad.reshape((-1, 1, 2)))))
        area_new = float(abs(cv2.contourArea(refined_quad.reshape((-1, 1, 2)))))
        if area_ref <= 1.0 or area_new <= 1.0:
            return quad

        shift = float(np.linalg.norm(np.mean(refined_quad, axis=0) - np.mean(quad, axis=0)))
        max_shift = float(max(6.0, mean_side * float(self.cfg.quad_edge_refine_max_shift_ratio)))
        if shift > max_shift:
            return quad

        new_lengths = np.linalg.norm(np.roll(refined_quad, -1, axis=0) - refined_quad, axis=1)
        if np.min(new_lengths) < (0.55 * np.min(side_lengths)) or np.max(new_lengths) > (1.45 * np.max(side_lengths)):
            return quad
        if area_new < (0.55 * area_ref) or area_new > (1.45 * area_ref):
            return quad
        return refined_quad

    def _refine_display_quad_from_contour(self, contour_xy: Optional[np.ndarray], quad_xy: np.ndarray) -> np.ndarray:
        quad = quad_xy.astype(np.float32)
        if not bool(self.cfg.display_quad_contour_refine_enabled):
            return quad
        if contour_xy is None:
            return quad

        contour = contour_xy.astype(np.float32).reshape((-1, 1, 2))
        if len(contour) < 4:
            return quad

        peri = float(cv2.arcLength(contour, True))
        if peri < 20.0:
            return quad

        ref_area = float(abs(cv2.contourArea(quad.reshape((-1, 1, 2)))))
        if ref_area <= 1.0:
            return quad

        ref_lengths = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
        mean_side = float(np.mean(ref_lengths)) if ref_lengths.size else 0.0
        if mean_side < 6.0:
            return quad

        best_quad = None
        best_err = None
        for eps_ratio in self.cfg.display_quad_approx_eps_ratios:
            approx = cv2.approxPolyDP(contour, float(eps_ratio) * peri, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            cand = approx.reshape(4, 2).astype(np.float32)
            cand = self._order_quad_points(cand)
            cand = self._align_quad_to_reference(cand, quad)
            if not self._is_rectangular_quad(cand, max_cos=0.82):
                continue

            cand_area = float(abs(cv2.contourArea(cand.reshape((-1, 1, 2)))))
            if cand_area < (0.55 * ref_area) or cand_area > (1.45 * ref_area):
                continue

            cand_lengths = np.linalg.norm(np.roll(cand, -1, axis=0) - cand, axis=1)
            if np.min(cand_lengths) < (0.55 * np.min(ref_lengths)) or np.max(cand_lengths) > (1.45 * np.max(ref_lengths)):
                continue

            shift = float(np.linalg.norm(np.mean(cand, axis=0) - np.mean(quad, axis=0)))
            if shift > max(6.0, mean_side * float(self.cfg.display_quad_contour_max_center_shift_ratio)):
                continue

            err = float(np.mean(np.linalg.norm(cand - quad, axis=1)))
            if best_err is None or err < best_err:
                best_err = err
                best_quad = cand

        if best_quad is None:
            return quad

        blend = float(np.clip(self.cfg.display_quad_contour_blend, 0.0, 1.0))
        refined = ((1.0 - blend) * quad + blend * best_quad).astype(np.float32)
        refined = self._align_quad_to_reference(refined, quad)
        if not self._is_rectangular_quad(refined, max_cos=0.82):
            return quad
        return refined

    def _stabilize_display_quad(self, slot_idx: int, quad_xy: np.ndarray) -> np.ndarray:
        quad = quad_xy.astype(np.float32)
        if slot_idx < 0 or slot_idx >= len(self.prev_display_slot_quads):
            return quad

        prev = self.prev_display_slot_quads[slot_idx]
        if prev is None:
            return quad

        ref = prev.astype(np.float32)
        aligned = self._align_quad_to_reference(quad, ref)
        move = np.linalg.norm(aligned - ref, axis=1)
        move_mean = float(np.mean(move))
        move_max = float(np.max(move))
        deadband = float(max(0.0, self.cfg.display_quad_deadband_px))
        if move_mean <= deadband and move_max <= (deadband * 1.9 + 0.1):
            return ref

        alpha = float(np.clip(self.cfg.display_quad_smooth_alpha, 0.0, 0.92))
        return alpha * ref + (1.0 - alpha) * aligned

    def _sample_depth_m_inward(self, depth_img: Optional[np.ndarray], corner_xy: np.ndarray, center_xy: np.ndarray):
        if depth_img is None:
            return None, None

        # Corner pixels are often invalid on dark objects; sample inward toward quad center.
        for t in (0.0, 0.12, 0.24, 0.36):
            sample_xy = (1.0 - t) * corner_xy + t * center_xy
            depth_m = self._sample_depth_m(depth_img, sample_xy)
            if depth_m is not None:
                return depth_m, sample_xy

        return None, None

    def _estimate_quad_size_cm(self, quad_xy: np.ndarray, depth_img: Optional[np.ndarray]):
        if depth_img is None or self.measure_intrinsics is None:
            return None

        quad = quad_xy.astype(np.float32)
        center = np.mean(quad, axis=0).astype(np.float32)
        pts3 = []
        valid_corner_count = 0

        for corner in quad:
            depth_m, sample_xy = self._sample_depth_m_inward(depth_img, corner, center)
            if depth_m is None or sample_xy is None:
                return None

            valid_corner_count += 1
            xyz = rs.rs2_deproject_pixel_to_point(
                self.measure_intrinsics,
                [float(sample_xy[0]), float(sample_xy[1])],
                float(depth_m),
            )
            pts3.append(np.asarray(xyz, dtype=np.float32))

        if valid_corner_count < 4:
            return None

        p = np.asarray(pts3, dtype=np.float32)
        e01 = float(np.linalg.norm(p[1] - p[0]))
        e12 = float(np.linalg.norm(p[2] - p[1]))
        e23 = float(np.linalg.norm(p[3] - p[2]))
        e30 = float(np.linalg.norm(p[0] - p[3]))

        side_a = 0.5 * (e01 + e23)
        side_b = 0.5 * (e12 + e30)
        long_m = max(side_a, side_b)
        short_m = min(side_a, side_b)
        if long_m < 0.005 or short_m < 0.005:
            return None

        # Opposite-edge agreement as a rough reliability signal.
        consistency = max(
            abs(e01 - e23) / max(side_a, 1e-6),
            abs(e12 - e30) / max(side_b, 1e-6),
        )

        return {
            "w_cm": long_m * 100.0,
            "h_cm": short_m * 100.0,
            "consistency": float(consistency),
        }

    def _sample_quad_depth_stats(self, depth_img: Optional[np.ndarray], quad_xy: np.ndarray):
        if depth_img is None:
            return None
        h, w = depth_img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        quad_i = np.round(quad_xy).astype(np.int32)
        quad_i[:, 0] = np.clip(quad_i[:, 0], 0, w - 1)
        quad_i[:, 1] = np.clip(quad_i[:, 1], 0, h - 1)
        cv2.fillConvexPoly(mask, quad_i, 255)

        inside = mask > 0
        total_inside = int(np.count_nonzero(inside))
        if total_inside < 200:
            return None

        valid = inside & (depth_img > 0)
        valid_count = int(np.count_nonzero(valid))
        if valid_count < 80:
            return None

        depth_vals = depth_img[valid].astype(np.float32) * self.depth_scale
        depth_med = float(np.median(depth_vals))
        depth_std = float(np.std(depth_vals))
        valid_ratio = float(valid_count) / float(total_inside + 1e-6)
        return depth_med, depth_std, valid_ratio

    def _sample_quad_depth_lift(self, depth_img: Optional[np.ndarray], quad_xy: np.ndarray):
        if depth_img is None:
            return None

        h, w = depth_img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        quad_i = np.round(quad_xy).astype(np.int32)
        quad_i[:, 0] = np.clip(quad_i[:, 0], 0, w - 1)
        quad_i[:, 1] = np.clip(quad_i[:, 1], 0, h - 1)
        cv2.fillConvexPoly(mask, quad_i, 255)

        inside = mask > 0
        inside_total = int(np.count_nonzero(inside))
        if inside_total < 180:
            return None

        ring_margin_px = int(np.clip(self.cfg.quad_depth_ring_margin_px, 2, 40))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_margin_px + 1, 2 * ring_margin_px + 1))
        outer = cv2.dilate(mask, kernel, iterations=1) > 0
        ring = outer & (~inside)
        ring_total = int(np.count_nonzero(ring))
        if ring_total < 140:
            return None

        valid = depth_img > 0
        inside_valid = inside & valid
        ring_valid = ring & valid
        inside_valid_count = int(np.count_nonzero(inside_valid))
        ring_valid_count = int(np.count_nonzero(ring_valid))
        if inside_valid_count < 80 or ring_valid_count < 80:
            return None

        inside_ratio = float(inside_valid_count) / float(inside_total + 1e-6)
        ring_ratio = float(ring_valid_count) / float(ring_total + 1e-6)
        if inside_ratio < 0.18 or ring_ratio < 0.14:
            return None

        inside_vals = depth_img[inside_valid].astype(np.float32) * self.depth_scale
        ring_vals = depth_img[ring_valid].astype(np.float32) * self.depth_scale
        if inside_vals.size < 40 or ring_vals.size < 40:
            return None

        # Object top should be closer than its immediate surroundings.
        inside_m = float(np.percentile(inside_vals, 45))
        ring_m = float(np.percentile(ring_vals, 62))
        lift_m = ring_m - inside_m
        return {
            "lift_m": lift_m,
            "inside_m": inside_m,
            "ring_m": ring_m,
            "inside_ratio": inside_ratio,
            "ring_ratio": ring_ratio,
        }

    def _sample_quad_plane_stats(self, depth_img: Optional[np.ndarray], quad_xy: np.ndarray):
        if depth_img is None or self.measure_intrinsics is None:
            return None

        h, w = depth_img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        quad_i = np.round(quad_xy).astype(np.int32)
        quad_i[:, 0] = np.clip(quad_i[:, 0], 0, w - 1)
        quad_i[:, 1] = np.clip(quad_i[:, 1], 0, h - 1)
        cv2.fillConvexPoly(mask, quad_i, 255)

        valid = (mask > 0) & (depth_img > 0)
        count = int(np.count_nonzero(valid))
        if count < 120:
            return None

        ys, xs = np.where(valid)
        if count > 320:
            step = int(np.ceil(count / 320.0))
            ys = ys[::step]
            xs = xs[::step]
        if ys.size < 40:
            return None

        z = depth_img[ys, xs].astype(np.float32) * self.depth_scale
        intr = self.measure_intrinsics
        x3 = (xs.astype(np.float32) - float(intr.ppx)) * z / float(intr.fx)
        y3 = (ys.astype(np.float32) - float(intr.ppy)) * z / float(intr.fy)
        pts = np.stack((x3, y3, z), axis=1).astype(np.float32)
        if pts.shape[0] < 40:
            return None

        mean = np.mean(pts, axis=0, keepdims=True)
        d = pts - mean
        cov = (d.T @ d) / max(float(pts.shape[0] - 1), 1.0)
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        order = np.argsort(eig_vals)
        normal = eig_vecs[:, order[0]]
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-8:
            return None
        normal = normal / normal_norm
        abs_nz = abs(float(normal[2]))

        depth_span = float(np.percentile(z, 90) - np.percentile(z, 10))
        return {
            "abs_nz": abs_nz,
            "depth_span_m": depth_span,
        }

    def _sample_quad_white_stats(self, hsv_img: Optional[np.ndarray], quad_xy: np.ndarray):
        if hsv_img is None:
            return None

        h, w = hsv_img.shape[:2]
        quad_i = np.round(quad_xy).astype(np.int32)
        quad_i[:, 0] = np.clip(quad_i[:, 0], 0, w - 1)
        quad_i[:, 1] = np.clip(quad_i[:, 1], 0, h - 1)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, quad_i, 255)

        valid = mask > 0
        count = int(np.count_nonzero(valid))
        if count < 80:
            return None

        sat = hsv_img[:, :, 1][valid].astype(np.float32)
        val = hsv_img[:, :, 2][valid].astype(np.float32)
        sat_thr = float(np.clip(self.cfg.white_sat_max, 0, 255))
        val_thr = float(np.clip(self.cfg.white_value_min, 0, 255))
        white_mask = (sat <= sat_thr) & (val >= val_thr)
        white_ratio = float(np.count_nonzero(white_mask)) / float(count)
        return {
            "white_ratio": white_ratio,
            "sat_mean": float(np.mean(sat)) if sat.size else 255.0,
            "val_mean": float(np.mean(val)) if val.size else 0.0,
        }

    def _candidate_height_band_info(self, depth_m: Optional[float], support_depth_m: Optional[float]):
        if depth_m is None or support_depth_m is None:
            return None

        height_m = float(support_depth_m) - float(depth_m)
        info = {
            "height_m": height_m,
            "band_idx": None,
            "band_target_m": None,
            "band_error_m": None,
            "band_conf": 0.0,
        }
        if not bool(self.cfg.depth_top_known_height_bands_enabled):
            return info

        targets = [float(v) for v in self.cfg.depth_top_height_targets_m]
        if not targets:
            return info

        tol = max(0.008, float(self.cfg.depth_top_height_tol_m) * 1.35)
        best_idx = None
        best_err = None
        for idx, target_h in enumerate(targets):
            err = abs(height_m - target_h)
            if best_err is None or err < best_err:
                best_idx = idx
                best_err = err

        if best_idx is None or best_err is None or best_err > tol:
            return info

        info["band_idx"] = int(best_idx)
        info["band_target_m"] = float(targets[best_idx])
        info["band_error_m"] = float(best_err)
        info["band_conf"] = float(np.clip(1.0 - (best_err / max(tol, 1e-6)), 0.0, 1.0))
        return info

    @staticmethod
    def _cube_size_label_from_target_height(target_height_m: Optional[float]) -> Optional[str]:
        if target_height_m is None:
            return None
        target_cm = float(target_height_m) * 100.0
        if target_cm <= 0.0:
            return None
        return f"{int(round(target_cm))}cm"

    def _classify_candidate_size_label(
        self,
        cand: dict,
        legend_size_cm: Optional[tuple[float, float]] = None,
    ) -> Optional[str]:
        label = self._cube_size_label_from_target_height(cand.get("height_band_target_m"))
        if label is not None:
            return label

        if legend_size_cm is None:
            return None

        targets_cm = [float(v) * 100.0 for v in self.cfg.depth_top_height_targets_m if float(v) > 0.0]
        if not targets_cm:
            return None

        side_cm = 0.5 * (float(legend_size_cm[0]) + float(legend_size_cm[1]))
        best_target = min(targets_cm, key=lambda target_cm: abs(side_cm - target_cm))
        if abs(side_cm - best_target) > 1.4:
            return None
        return f"{int(round(best_target))}cm"

    def _expected_top_face_side_px(self, depth_m: Optional[float], target_height_m: Optional[float]) -> Optional[float]:
        if (
            depth_m is None
            or target_height_m is None
            or self.measure_intrinsics is None
            or depth_m <= 0.05
            or target_height_m <= 0.0
        ):
            return None

        fx = float(self.measure_intrinsics.fx)
        fy = float(self.measure_intrinsics.fy)
        side_px = 0.5 * ((fx * float(target_height_m) / float(depth_m)) + (fy * float(target_height_m) / float(depth_m)))
        return float(max(8.0, side_px))

    def _white_score_bonus(self, white_stats, source: str) -> float:
        if not bool(self.cfg.white_bias_enabled):
            return 1.0

        white_ratio = float(white_stats.get("white_ratio", 0.0))
        sat_mean = float(white_stats.get("sat_mean", 255.0))
        val_mean = float(white_stats.get("val_mean", 0.0))
        min_ratio = float(np.clip(self.cfg.white_min_ratio, 0.05, 0.95))
        if white_ratio <= (min_ratio * 0.45):
            return 1.0

        ratio_conf = float(np.clip((white_ratio - min_ratio) / max(1e-6, 1.0 - min_ratio), 0.0, 1.0))
        sat_conf = float(np.clip((float(self.cfg.white_sat_max) - sat_mean) / max(1.0, float(self.cfg.white_sat_max)), 0.0, 1.0))
        val_conf = float(
            np.clip(
                (val_mean - float(self.cfg.white_value_min)) / max(1.0, 255.0 - float(self.cfg.white_value_min)),
                0.0,
                1.0,
            )
        )
        conf = 0.70 * ratio_conf + 0.15 * sat_conf + 0.15 * val_conf
        gain = float(np.clip(self.cfg.white_score_gain, 0.0, 1.0)) * conf
        if source == "dark-blob":
            gain *= 0.25
        return 1.0 + gain

    def _required_quad_lift_m(self, source: str) -> float:
        base = float(max(0.0, self.cfg.quad_depth_lift_min_m))
        if base <= 0.0:
            return 0.0
        max_objs = max(1, int(self.cfg.max_objects))
        if source in ("depth-band", "depth-plane", "depth-edge", "depth-top"):
            # Keep enough lift in multi-object scenes to avoid flat merged candidates.
            if max_objs >= 4:
                base *= 0.92
            elif max_objs >= 3:
                base *= 0.88
            elif max_objs >= 2:
                base *= 0.84
        if source in ("color-edge", "dark-blob"):
            return base
        if source == "depth-edge":
            return base * 0.92
        if source == "depth-plane":
            return base * 0.82
        if source == "depth-top":
            return base * 0.72
        if source == "depth-band":
            return base * 0.90
        return base * 0.92

    def _effective_top_face_constraints(self, source: str) -> tuple[float, float]:
        req_nz = float(np.clip(self.cfg.top_face_min_abs_nz, 0.0, 1.0))
        req_span = float(max(0.0, self.cfg.top_face_depth_span_max_m))
        max_objs = max(1, int(self.cfg.max_objects))

        # In mixed-height multi-object scenes, don't over-loosen span checks.
        if source in ("depth-band", "depth-plane", "depth-edge", "depth-top"):
            if max_objs >= 4:
                req_nz *= 0.94
                req_span *= 1.05
            elif max_objs >= 3:
                req_nz *= 0.92
                req_span *= 1.12
            elif max_objs >= 2:
                req_nz *= 0.94
                req_span *= 1.18
            if source == "depth-plane":
                req_nz *= 0.98
                req_span *= 1.05
            elif source == "depth-top":
                req_nz = min(1.0, req_nz * 1.06)
                req_span *= 0.82

        req_nz = float(np.clip(req_nz, 0.0, 1.0))
        req_span = float(max(0.0, req_span))
        return req_nz, req_span

    @staticmethod
    def _source_weight(source: str, depth_valid_ratio: float, fallback: bool = False) -> float:
        depth_ok = float(depth_valid_ratio) >= 0.28
        if depth_ok:
            if source == "depth-top":
                return 1.22 if fallback else 1.32
            if source == "depth-band":
                return 1.18 if fallback else 1.22
            if source == "depth-plane":
                return 1.14 if fallback else 1.18
            if source == "depth-edge":
                return 1.06 if fallback else 1.10
            if source == "dark-blob":
                return 0.88 if fallback else 0.84
            return 0.98 if fallback else 0.96

        # Low-depth scenes keep the prior preference for color/texture candidates.
        if fallback:
            if source == "dark-blob":
                return 1.22
            return 1.10 if source in ("color-edge", "depth-edge") else 1.0
        if source == "depth-band":
            return 0.95
        if source == "depth-top":
            return 1.08
        if source == "depth-plane":
            return 1.00
        if source == "depth-edge":
            return 1.10
        if source == "dark-blob":
            return 1.30
        return 1.14

    def _stabilize_quad(self, quad_xy: np.ndarray, ref_xy: Optional[np.ndarray] = None) -> np.ndarray:
        quad = quad_xy.astype(np.float32)
        if ref_xy is None:
            return quad

        ref = ref_xy.astype(np.float32)
        aligned = self._align_quad_to_reference(quad, ref)
        move = np.linalg.norm(aligned - ref, axis=1)
        move_mean = float(np.mean(move))
        move_max = float(np.max(move))
        center_move = float(np.linalg.norm(np.mean(aligned, axis=0) - np.mean(ref, axis=0)))

        # Deadband: if movement is tiny, keep corners fixed to remove visible jitter.
        deadband = float(max(0.0, self.cfg.corner_deadband_px))
        if move_mean <= deadband and move_max <= (deadband * 1.8 + 0.1):
            return ref

        alpha = float(np.clip(self.cfg.corner_smooth_alpha, 0.0, 0.90))
        jump_guard = float(max(deadband + 1.0, self.cfg.corner_jump_guard_px))
        micro_center_px = float(max(0.2, self.cfg.corner_micro_center_px))
        if center_move <= micro_center_px and move_mean <= (deadband * 2.6) and move_max <= (jump_guard * 0.70):
            # Static object with tiny center drift: suppress outline shimmer more aggressively.
            alpha = max(alpha, float(np.clip(self.cfg.corner_micro_alpha, 0.82, 0.94)))
        elif center_move <= (micro_center_px * 1.8) and move_mean <= (deadband * 3.2) and move_max <= jump_guard:
            alpha = max(alpha, 0.84)
        if move_mean > jump_guard or move_max > (jump_guard * 1.7):
            # Suspected candidate jump: damp it, but don't let the overlay lag too far behind.
            alpha = max(alpha, 0.80)
        return alpha * ref + (1.0 - alpha) * aligned

    def _clear_slot_lock(self, slot_idx: int) -> None:
        if slot_idx < 0:
            return
        if slot_idx >= len(self.lock_slot_quads):
            return
        self.lock_slot_quads[slot_idx] = None
        self.lock_slot_depths[slot_idx] = None
        self.slot_lock_streaks[slot_idx] = 0
        self.slot_locked_flags[slot_idx] = False
        self.slot_locked_ages[slot_idx] = 0
        self.slot_move_release_streaks[slot_idx] = 0

    def _apply_slot_coordinate_lock(
        self,
        slot_idx: int,
        quad_xy: np.ndarray,
        depth_m: Optional[float],
    ) -> tuple[np.ndarray, Optional[float]]:
        quad = quad_xy.astype(np.float32)
        if slot_idx < 0 or slot_idx >= len(self.lock_slot_quads):
            return quad, depth_m

        if not bool(self.cfg.coord_lock_enabled):
            self.lock_slot_quads[slot_idx] = quad.copy()
            self.lock_slot_depths[slot_idx] = None if depth_m is None else float(depth_m)
            self.slot_lock_streaks[slot_idx] = 0
            self.slot_locked_flags[slot_idx] = False
            self.slot_locked_ages[slot_idx] = 0
            return quad, depth_m

        lock_quad = self.lock_slot_quads[slot_idx]
        lock_depth = self.lock_slot_depths[slot_idx]
        if lock_quad is None:
            self.lock_slot_quads[slot_idx] = quad.copy()
            self.lock_slot_depths[slot_idx] = None if depth_m is None else float(depth_m)
            self.slot_lock_streaks[slot_idx] = 1
            self.slot_locked_flags[slot_idx] = False
            self.slot_locked_ages[slot_idx] = 0
            return quad, depth_m

        center = np.mean(quad, axis=0)
        lock_center = np.mean(lock_quad, axis=0)
        center_dist = float(np.linalg.norm(center - lock_center))
        depth_diff = None
        if depth_m is not None and lock_depth is not None:
            depth_diff = abs(float(depth_m) - float(lock_depth))

        enter_px = float(max(0.0, self.cfg.coord_lock_enter_px))
        exit_px = float(max(enter_px + 0.5, self.cfg.coord_lock_exit_px))
        frames_req = int(max(1, self.cfg.coord_lock_frames))
        depth_exit_m = float(max(0.0, self.cfg.coord_lock_depth_exit_m))
        depth_ok = (depth_diff is None) or (depth_diff <= depth_exit_m)

        if self.slot_locked_flags[slot_idx]:
            self.slot_locked_ages[slot_idx] += 1
            if bool(self.cfg.coord_lock_static_mode) and bool(self.cfg.coord_lock_sticky_slots):
                locked_depth = self.lock_slot_depths[slot_idx]
                return lock_quad.copy(), (locked_depth if locked_depth is not None else depth_m)
            static_min_frames = int(max(1, self.cfg.coord_lock_static_min_frames))
            if bool(self.cfg.coord_lock_static_mode) and self.slot_locked_ages[slot_idx] >= static_min_frames:
                locked_depth = self.lock_slot_depths[slot_idx]
                return lock_quad.copy(), (locked_depth if locked_depth is not None else depth_m)
            if center_dist <= exit_px and depth_ok:
                locked_depth = self.lock_slot_depths[slot_idx]
                return lock_quad.copy(), (locked_depth if locked_depth is not None else depth_m)
            self.slot_locked_flags[slot_idx] = False
            self.slot_lock_streaks[slot_idx] = 0
            self.slot_locked_ages[slot_idx] = 0
            self.lock_slot_quads[slot_idx] = quad.copy()
            self.lock_slot_depths[slot_idx] = None if depth_m is None else float(depth_m)
            return quad, depth_m

        if center_dist <= enter_px and depth_ok:
            self.slot_lock_streaks[slot_idx] += 1
            blend = 0.92
            blended_quad = blend * lock_quad + (1.0 - blend) * quad
            self.lock_slot_quads[slot_idx] = blended_quad.astype(np.float32)
            if depth_m is not None:
                if lock_depth is None:
                    self.lock_slot_depths[slot_idx] = float(depth_m)
                else:
                    self.lock_slot_depths[slot_idx] = float(0.92 * lock_depth + 0.08 * float(depth_m))

            if self.slot_lock_streaks[slot_idx] >= frames_req:
                self.slot_locked_flags[slot_idx] = True
                self.slot_locked_ages[slot_idx] = 1
                locked_depth = self.lock_slot_depths[slot_idx]
                return self.lock_slot_quads[slot_idx].copy(), (locked_depth if locked_depth is not None else depth_m)

            smooth_depth = self.lock_slot_depths[slot_idx]
            return self.lock_slot_quads[slot_idx].copy(), (smooth_depth if smooth_depth is not None else depth_m)

        self.slot_lock_streaks[slot_idx] = 0
        self.slot_locked_ages[slot_idx] = 0
        self.lock_slot_quads[slot_idx] = quad.copy()
        self.lock_slot_depths[slot_idx] = None if depth_m is None else float(depth_m)
        return quad, depth_m

    @staticmethod
    def _align_quad_to_reference(quad_xy: np.ndarray, ref_xy: np.ndarray) -> np.ndarray:
        q = quad_xy.astype(np.float32)
        r = ref_xy.astype(np.float32)

        candidates = []
        for k in range(4):
            candidates.append(np.roll(q, shift=k, axis=0))

        q_rev = q[::-1].copy()
        for k in range(4):
            candidates.append(np.roll(q_rev, shift=k, axis=0))

        best = candidates[0]
        best_err = float(np.mean(np.sum((best - r) ** 2, axis=1)))
        for cand in candidates[1:]:
            err = float(np.mean(np.sum((cand - r) ** 2, axis=1)))
            if err < best_err:
                best = cand
                best_err = err
        return best

    @staticmethod
    def _quad_iou(quad_a: np.ndarray, quad_b: np.ndarray) -> float:
        qa = quad_a.astype(np.float32).reshape((-1, 1, 2))
        qb = quad_b.astype(np.float32).reshape((-1, 1, 2))
        area_a = float(abs(cv2.contourArea(qa)))
        area_b = float(abs(cv2.contourArea(qb)))
        if area_a <= 1e-6 or area_b <= 1e-6:
            return 0.0
        inter_area, _ = cv2.intersectConvexConvex(qa, qb)
        if inter_area <= 0.0:
            return 0.0
        union = area_a + area_b - float(inter_area)
        if union <= 1e-6:
            return 0.0
        return float(inter_area / union)

    @staticmethod
    def _candidate_slot_key(cand) -> tuple[float, float]:
        quad = cand["quad"].astype(np.float32)
        center = np.mean(quad, axis=0)
        return float(center[0]), float(center[1])

    def _solve_slot_candidate_assignment(self, slot_indices, candidate_indices, cand_centers, max_assign_dist_px: float):
        if not slot_indices or not candidate_indices:
            return {}

        ref_centers = {}
        for slot_idx in slot_indices:
            ref_quad = self.prev_slot_quads[slot_idx]
            if ref_quad is None:
                continue
            ref_centers[slot_idx] = np.mean(ref_quad.astype(np.float32), axis=0).astype(np.float32)
        if not ref_centers:
            return {}

        best_mapping = {}
        best_assigned = -1
        best_cost = float("inf")

        candidate_indices = list(candidate_indices)

        def _dfs(pos: int, used: set[int], mapping: dict[int, int], assigned_count: int, total_cost: float):
            nonlocal best_mapping, best_assigned, best_cost
            if pos >= len(slot_indices):
                if assigned_count > best_assigned or (
                    assigned_count == best_assigned and total_cost < best_cost
                ):
                    best_assigned = assigned_count
                    best_cost = total_cost
                    best_mapping = dict(mapping)
                return

            remaining_slots = len(slot_indices) - pos
            if assigned_count + remaining_slots < best_assigned:
                return

            slot_idx = slot_indices[pos]
            ref_center = ref_centers.get(slot_idx)
            if ref_center is None:
                _dfs(pos + 1, used, mapping, assigned_count, total_cost)
                return

            # Allow leaving a slot empty when no candidate is close enough.
            _dfs(pos + 1, used, mapping, assigned_count, total_cost)

            scored = []
            for cand_idx in candidate_indices:
                if cand_idx in used:
                    continue
                dist = float(np.linalg.norm(cand_centers[cand_idx] - ref_center))
                if dist > max_assign_dist_px:
                    continue
                scored.append((dist, cand_idx))
            scored.sort(key=lambda item: item[0])

            for dist, cand_idx in scored:
                mapping[slot_idx] = cand_idx
                used.add(cand_idx)
                _dfs(pos + 1, used, mapping, assigned_count + 1, total_cost + dist)
                used.remove(cand_idx)
                del mapping[slot_idx]

        _dfs(0, set(), {}, 0, 0.0)
        return best_mapping

    def _assign_candidates_to_slots(self, selected_candidates, max_objects: int):
        slot_candidates = [None for _ in range(max_objects)]
        if not selected_candidates:
            return slot_candidates

        cand_centers = [
            np.mean(c["quad"].astype(np.float32), axis=0).astype(np.float32)
            for c in selected_candidates
        ]
        remaining = set(range(len(selected_candidates)))

        # Pass 1: continuity-first matching to keep slot identity stable.
        max_assign_dist_px = float(OBJECT_SLOT_MATCH_PX)
        continuity_slots = []
        for slot_idx in range(max_objects):
            if slot_idx >= len(self.prev_slot_quads):
                break
            sticky_locked = (
                slot_idx < len(self.slot_locked_flags)
                and bool(self.slot_locked_flags[slot_idx])
                and bool(self.cfg.coord_lock_static_mode)
                and bool(self.cfg.coord_lock_sticky_slots)
            )
            if sticky_locked:
                # Sticky slots should remain stable, but must still react to real object movement.
                lock_quad = self.lock_slot_quads[slot_idx]
                if lock_quad is None:
                    continue
                lock_center = np.mean(lock_quad.astype(np.float32), axis=0).astype(np.float32)
                lock_depth = self.lock_slot_depths[slot_idx] if slot_idx < len(self.lock_slot_depths) else None
                best_idx = None
                best_dist = None
                best_depth = None
                for ci in remaining:
                    dist = float(np.linalg.norm(cand_centers[ci] - lock_center))
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best_idx = ci
                        best_depth = selected_candidates[ci].get("depth_m")

                if best_idx is None or best_dist is None:
                    continue

                move_release_px = max(12.0, float(self.cfg.coord_lock_exit_px) * 1.25)
                depth_release_m = max(0.02, float(self.cfg.coord_lock_depth_exit_m) * 1.8)
                release_frames = max(1, int(self.cfg.coord_lock_move_release_frames))
                depth_diff = None
                if best_depth is not None and lock_depth is not None:
                    depth_diff = abs(float(best_depth) - float(lock_depth))
                moved = (best_dist >= move_release_px) or (depth_diff is not None and depth_diff >= depth_release_m)

                if moved:
                    self.slot_move_release_streaks[slot_idx] += 1
                    # Release only when movement evidence is stable for multiple frames.
                    if self.slot_move_release_streaks[slot_idx] >= release_frames:
                        self.slot_locked_flags[slot_idx] = False
                        self.slot_locked_ages[slot_idx] = 0
                        self.slot_lock_streaks[slot_idx] = 0
                        self.slot_move_release_streaks[slot_idx] = 0
                        slot_candidates[slot_idx] = selected_candidates[best_idx]
                    # Consume nearest candidate while evaluating release to avoid slot stealing.
                    remaining.remove(best_idx)
                else:
                    self.slot_move_release_streaks[slot_idx] = 0
                    # Same static object: consume nearest candidate so other slots won't steal it.
                    remaining.remove(best_idx)
                continue
            ref_quad = self.prev_slot_quads[slot_idx]
            if ref_quad is None:
                continue
            if self.slot_miss_counts[slot_idx] != 0:
                reserved = (
                    bool(self.preserve_sort_slots)
                    and (slot_idx + 1) in self.reserved_sort_slot_ids
                    and self.slot_miss_counts[slot_idx] <= max(1, int(self.cfg.corner_hold_frames))
                )
                if not reserved:
                    continue
            continuity_slots.append(slot_idx)

        if continuity_slots and remaining:
            optimal_map = self._solve_slot_candidate_assignment(
                continuity_slots,
                remaining,
                cand_centers,
                max_assign_dist_px=max_assign_dist_px,
            )
            for slot_idx, cand_idx in optimal_map.items():
                slot_candidates[slot_idx] = selected_candidates[cand_idx]
                if cand_idx in remaining:
                    remaining.remove(cand_idx)

        # Pass 2: fill remaining empty slots from left->right/top->bottom order.
        rem_sorted = sorted(remaining, key=lambda i: self._candidate_slot_key(selected_candidates[i]))
        rem_ptr = 0
        for slot_idx in range(max_objects):
            if slot_candidates[slot_idx] is not None:
                continue
            if bool(self.preserve_sort_slots) and (slot_idx + 1) in self.reserved_sort_slot_ids:
                continue
            # Protect recently-tracked slots from being overwritten by unrelated leftovers.
            protected = False
            if slot_idx < len(self.prev_slot_quads):
                prev_exists = self.prev_slot_quads[slot_idx] is not None
                if prev_exists:
                    recent = self.slot_miss_counts[slot_idx] <= max(1, int(self.cfg.corner_hold_frames))
                    locked = slot_idx < len(self.slot_locked_flags) and bool(self.slot_locked_flags[slot_idx])
                    protected = recent or locked
            if protected:
                continue
            if rem_ptr >= len(rem_sorted):
                break
            slot_candidates[slot_idx] = selected_candidates[rem_sorted[rem_ptr]]
            rem_ptr += 1

        return slot_candidates

    def _select_top_candidates(self, candidates, limit: int, iou_threshold: float):
        selected = []
        sorted_cands = sorted(candidates, key=lambda x: x["score"], reverse=True)
        if not sorted_cands:
            return selected

        top_score = float(sorted_cands[0]["score"])
        top_area = float(sorted_cands[0].get("area", 1.0))
        ratio = float(np.clip(self.cfg.object_min_score_ratio, 0.05, 1.0))
        min_score = top_score * ratio
        relaxed_score = top_score * max(0.15, ratio * 0.55)
        secondary_score = top_score * float(np.clip(self.cfg.object_secondary_score_ratio, 0.05, 0.8))
        secondary_area = max(120.0, top_area * float(np.clip(self.cfg.object_secondary_area_ratio, 0.05, 1.0)))

        # For 4+ objects, keep the weakest object from being dropped too early.
        if int(limit) >= 4:
            min_score = min(min_score, top_score * 0.24)
            relaxed_score = min(relaxed_score, top_score * 0.15)
            secondary_score = min(secondary_score, top_score * 0.08)
            secondary_area = min(secondary_area, max(80.0, top_area * 0.07))

        def _duplicate_check(cand, kept_list):
            overlapped = False
            min_center_ratio = 1e9
            for kept in kept_list:
                if self._quad_iou(cand["quad"], kept["quad"]) >= iou_threshold:
                    overlapped = True
                    break

                c_center = np.mean(cand["quad"].astype(np.float32), axis=0)
                k_center = np.mean(kept["quad"].astype(np.float32), axis=0)
                center_dist = float(np.linalg.norm(c_center - k_center))
                c_diag = float(np.linalg.norm(cand["quad"][0] - cand["quad"][2]))
                k_diag = float(np.linalg.norm(kept["quad"][0] - kept["quad"][2]))
                min_diag = max(1.0, min(c_diag, k_diag))
                center_ratio = center_dist / min_diag
                if center_ratio < min_center_ratio:
                    min_center_ratio = center_ratio

                zc = cand.get("depth_m")
                zk = kept.get("depth_m")
                depth_close = (zc is None or zk is None) or (
                    abs(float(zc) - float(zk)) <= self.cfg.object_duplicate_depth_diff_m
                )
                dup_center_ratio = float(self.cfg.object_duplicate_center_ratio)
                cand_band = cand.get("height_band_idx")
                kept_band = kept.get("height_band_idx")
                cand_src = str(cand.get("source", ""))
                kept_src = str(kept.get("source", ""))
                if (
                    cand_band is not None
                    and kept_band is not None
                    and cand_src in ("depth-top", "depth-band")
                    and kept_src in ("depth-top", "depth-band")
                ):
                    if int(cand_band) == int(kept_band):
                        dup_center_ratio = min(dup_center_ratio, float(self.cfg.object_known_band_duplicate_center_ratio))
                        cand_target = cand.get("height_band_target_m")
                        kept_target = kept.get("height_band_target_m")
                        if (
                            cand_target is not None
                            and kept_target is not None
                            and min(float(cand_target), float(kept_target)) <= float(self.cfg.depth_top_small_object_height_m)
                        ):
                            dup_center_ratio = min(dup_center_ratio, float(self.cfg.object_small_band_duplicate_center_ratio))
                    else:
                        dup_center_ratio = min(
                            dup_center_ratio,
                            float(self.cfg.object_cross_band_duplicate_center_ratio),
                        )

                if center_dist <= (min_diag * dup_center_ratio) and depth_close:
                    overlapped = True
                    break
            return overlapped, min_center_ratio

        selected_indices = set()
        if (
            bool(self.cfg.depth_top_band_balance_enabled)
            and bool(self.cfg.depth_top_known_height_bands_enabled)
            and len(self.cfg.depth_top_height_targets_m) >= 2
            and limit >= len(self.cfg.depth_top_height_targets_m)
        ):
            band_count = len(self.cfg.depth_top_height_targets_m)
            base_quota = max(1, int(limit) // band_count)
            extra_quota = max(0, int(limit) - (base_quota * band_count))
            per_band_quota = [base_quota for _ in range(band_count)]
            for band_idx in range(extra_quota):
                per_band_quota[band_idx % band_count] += 1

            band_members = {band_idx: [] for band_idx in range(band_count)}
            for idx, cand in enumerate(sorted_cands):
                band_idx = cand.get("height_band_idx")
                if band_idx is None:
                    continue
                band_idx = int(band_idx)
                if 0 <= band_idx < band_count:
                    band_members[band_idx].append((idx, cand))

            band_selected_counts = [0 for _ in range(band_count)]
            made_progress = True
            while len(selected) < limit and made_progress:
                made_progress = False
                for band_idx in range(band_count):
                    if band_selected_counts[band_idx] >= per_band_quota[band_idx]:
                        continue
                    for idx, cand in band_members[band_idx]:
                        if idx in selected_indices:
                            continue
                        overlapped, min_center_ratio = _duplicate_check(cand, selected)
                        if overlapped:
                            continue
                        if selected and min_center_ratio < 1.05:
                            continue
                        selected.append(cand)
                        selected_indices.add(idx)
                        band_selected_counts[band_idx] += 1
                        made_progress = True
                        break
                    if len(selected) >= limit:
                        return selected

        # Pass 1: strict selection.
        for idx, cand in enumerate(sorted_cands):
            cand_score = float(cand["score"])
            overlapped, min_center_ratio = _duplicate_check(cand, selected)
            if overlapped:
                continue

            if cand_score < min_score:
                if cand_score < relaxed_score:
                    continue
                if selected and min_center_ratio < 1.20:
                    continue
            selected.append(cand)
            selected_indices.add(idx)
            if len(selected) >= limit:
                return selected

        # Pass 2: secondary fill for multi-object scenes (robust to background changes).
        for idx, cand in enumerate(sorted_cands):
            if idx in selected_indices:
                continue
            cand_score = float(cand["score"])
            cand_area = float(cand.get("area", 0.0))
            if cand_score < secondary_score:
                continue
            if cand_area < secondary_area:
                continue

            overlapped, _ = _duplicate_check(cand, selected)
            if overlapped:
                continue
            selected.append(cand)
            selected_indices.add(idx)
            if len(selected) >= limit:
                break

        # Pass 3: recovery fill for weak-but-separated objects (e.g., darker side object on new background).
        if len(selected) < limit and limit >= 2:
            low_score = top_score * 0.07
            min_area = max(120.0, float(self.cfg.rect_min_area) * 0.60)
            max_ratio = max(2.4, float(self.cfg.rect_max_aspect_ratio) * 0.70)
            for idx, cand in enumerate(sorted_cands):
                if idx in selected_indices:
                    continue

                cand_score = float(cand["score"])
                cand_area = float(cand.get("area", 0.0))
                cand_ratio = float(cand.get("ratio", 99.0))
                cand_target = cand.get("height_band_target_m")
                small_known_band = (
                    cand_target is not None
                    and str(cand.get("source", "")) in ("depth-top", "depth-band")
                    and float(cand_target) <= float(self.cfg.depth_top_small_object_height_m)
                )
                cand_min_area = min_area * (0.72 if small_known_band else 1.0)
                if cand_score < low_score or cand_area < cand_min_area:
                    continue
                if cand_ratio > max_ratio:
                    continue

                overlapped, min_center_ratio = _duplicate_check(cand, selected)
                if overlapped:
                    continue
                rescue_center_ratio = 1.35
                if (
                    cand.get("height_band_idx") is not None
                    and str(cand.get("source", "")) in ("depth-top", "depth-band")
                ):
                    rescue_center_ratio = 0.88
                if small_known_band:
                    rescue_center_ratio = min(rescue_center_ratio, 0.68)
                if selected and min_center_ratio < rescue_center_ratio:
                    continue

                selected.append(cand)
                selected_indices.add(idx)
                if len(selected) >= limit:
                    break

        # Pass 4: force-fill spatially separated object candidates when detection is unstable.
        if len(selected) < limit and limit >= 2:
            min_area_force = max(90.0, float(self.cfg.rect_min_area) * 0.38)
            for idx, cand in enumerate(sorted_cands):
                if idx in selected_indices:
                    continue
                cand_area = float(cand.get("area", 0.0))
                cand_target = cand.get("height_band_target_m")
                small_known_band = (
                    cand_target is not None
                    and str(cand.get("source", "")) in ("depth-top", "depth-band")
                    and float(cand_target) <= float(self.cfg.depth_top_small_object_height_m)
                )
                cand_min_area_force = min_area_force * (0.68 if small_known_band else 1.0)
                if cand_area < cand_min_area_force:
                    continue

                overlapped, min_center_ratio = _duplicate_check(cand, selected)
                if overlapped:
                    continue
                rescue_center_ratio = 1.55
                if (
                    cand.get("height_band_idx") is not None
                    and str(cand.get("source", "")) in ("depth-top", "depth-band")
                ):
                    rescue_center_ratio = 0.82
                if small_known_band:
                    rescue_center_ratio = min(rescue_center_ratio, 0.64)
                if selected and min_center_ratio < rescue_center_ratio:
                    continue

                src = str(cand.get("source", ""))
                if src not in ("depth-top", "depth-band", "depth-plane", "dark-blob", "depth-edge"):
                    continue

                selected.append(cand)
                selected_indices.add(idx)
                if len(selected) >= limit:
                    break
        return selected

    def _effective_center_weight_power(self) -> float:
        p = float(np.clip(self.cfg.center_weight_power, 1.0, 4.0))
        if int(self.cfg.max_objects) >= 4:
            return min(p, 1.02)
        if int(self.cfg.max_objects) >= 3:
            return min(p, 1.12)
        if int(self.cfg.max_objects) >= 2:
            return min(p, 1.35)
        return p

    @staticmethod
    def _resolve_ui_font_path() -> Optional[str]:
        if ImageFont is None:
            return None
        candidates = [
            r"C:\Windows\Fonts\malgun.ttf",
            r"C:\Windows\Fonts\malgunbd.ttf",
            r"C:\Windows\Fonts\gulim.ttc",
            r"C:\Windows\Fonts\batang.ttc",
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def _get_ui_font(self, size_px: int):
        if ImageFont is None or not self.ui_font_path:
            return None
        size_px = int(max(12, size_px))
        cached = self.ui_font_cache.get(size_px)
        if cached is not None:
            return cached
        try:
            font = ImageFont.truetype(self.ui_font_path, size=size_px)
            self.ui_font_cache[size_px] = font
            return font
        except Exception:
            return None

    def _draw_object_legend_bottom_left(self, img: np.ndarray, rows):
        if not rows:
            return

        rows_clean = list(rows)
        while rows_clean and rows_clean[-1][0] == "":
            rows_clean.pop()
        if not rows_clean:
            return

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.62
        thickness = 2
        line_h = 24
        pad_x = 12
        pad_y = 12

        max_w = 0
        for text, _ in rows_clean:
            (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
            max_w = max(max_w, tw)

        header = "감지 객체"
        (hw, _), _ = cv2.getTextSize(header, font, scale, thickness)
        max_w = max(max_w, hw)

        box_w = max_w + 2 * pad_x
        box_h = (len(rows_clean) + 1) * line_h + 2 * pad_y
        x0 = 12
        y1 = img.shape[0] - 12
        y0 = max(0, y1 - box_h)

        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y1), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.40, img, 0.60, 0, img)

        y = y0 + pad_y + line_h - 4
        if Image is not None and ImageDraw is not None and self.ui_font_path:
            font_obj = self._get_ui_font(int(scale * 30))
            if font_obj is not None:
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
                draw.text((x0 + pad_x, y - int(scale * 28)), header, font=font_obj, fill=(220, 220, 220))
                for text, color in rows_clean:
                    y += line_h
                    rgb = (int(color[2]), int(color[1]), int(color[0]))
                    draw.text((x0 + pad_x, y - int(scale * 28)), text, font=font_obj, fill=rgb)
                img[:] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
                return
        cv2.putText(img, header, (x0 + pad_x, y), font, scale, (220, 220, 220), thickness, cv2.LINE_AA)
        for text, color in rows_clean:
            y += line_h
            cv2.putText(img, text, (x0 + pad_x, y), font, scale, color, thickness, cv2.LINE_AA)

    def _draw_center_legend_bottom_right(self, img: np.ndarray, rows):
        if not rows:
            return

        rows_clean = list(rows)
        while rows_clean and rows_clean[-1][0] == "":
            rows_clean.pop()
        if not rows_clean:
            return

        header = "객체 중심 좌표"
        font = cv2.FONT_HERSHEY_SIMPLEX
        pad_x = 10
        pad_y = 10
        available_h = max(120, img.shape[0] - 24)
        scale = 0.50
        thickness = 1
        line_h = 20
        box_w = 0
        box_h = 0

        for cand_scale in (0.50, 0.48, 0.46, 0.44, 0.42):
            cand_thickness = 2 if cand_scale >= 0.56 else 1
            sample_h = cv2.getTextSize("Ag", font, cand_scale, cand_thickness)[0][1]
            cand_line_h = max(sample_h + 8, int(round(cand_scale * 30)))
            max_w = 0
            for text, _ in rows_clean:
                (tw, _), _ = cv2.getTextSize(text, font, cand_scale, cand_thickness)
                max_w = max(max_w, tw)
            (hw, _), _ = cv2.getTextSize(header, font, cand_scale, cand_thickness)
            max_w = max(max_w, hw)
            cand_box_w = max_w + 2 * pad_x
            cand_box_h = (len(rows_clean) + 1) * cand_line_h + 2 * pad_y
            scale = cand_scale
            thickness = cand_thickness
            line_h = cand_line_h
            box_w = cand_box_w
            box_h = cand_box_h
            if cand_box_h <= available_h:
                break

        x1 = img.shape[1] - 12
        x0 = max(0, x1 - box_w)
        y1 = img.shape[0] - 12
        y0 = max(0, y1 - box_h)

        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.40, img, 0.60, 0, img)

        y = y0 + pad_y + line_h - 4
        if Image is not None and ImageDraw is not None and self.ui_font_path:
            font_obj = self._get_ui_font(int(scale * 30))
            if font_obj is not None:
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
                draw.text((x0 + pad_x, y - int(scale * 28)), header, font=font_obj, fill=(220, 220, 220))
                for text, color in rows_clean:
                    y += line_h
                    rgb = (int(color[2]), int(color[1]), int(color[0]))
                    draw.text((x0 + pad_x, y - int(scale * 28)), text, font=font_obj, fill=rgb)
                img[:] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
                return
        cv2.putText(img, header, (x0 + pad_x, y), font, scale, (220, 220, 220), thickness, cv2.LINE_AA)
        for text, color in rows_clean:
            y += line_h
            cv2.putText(img, text, (x0 + pad_x, y), font, scale, color, thickness, cv2.LINE_AA)

    def _extract_center_roi(self, img: np.ndarray):
        h, w = img.shape[:2]
        ratio = float(np.clip(self.cfg.center_roi_ratio, 0.2, 1.0))
        roi_w = int(w * ratio)
        roi_h = int(h * ratio)
        x0 = max((w - roi_w) // 2, 0)
        y0 = max((h - roi_h) // 2, 0)
        x1 = min(x0 + roi_w, w)
        y1 = min(y0 + roi_h, h)
        return img[y0:y1, x0:x1], (x0, y0, x1, y1)

    @staticmethod
    def _order_quad_points(pts: np.ndarray) -> np.ndarray:
        center = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        order = np.argsort(angles)
        return pts[order]

    @staticmethod
    def _is_rectangular_quad(quad: np.ndarray, max_cos: float = 0.35) -> bool:
        for i in range(4):
            p_prev = quad[(i - 1) % 4]
            p_curr = quad[i]
            p_next = quad[(i + 1) % 4]

            v1 = p_prev - p_curr
            v2 = p_next - p_curr
            denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
            cos_angle = abs(float(np.dot(v1, v2) / denom))
            if cos_angle > max_cos:
                return False
        return True

    def _save_snapshot(self, color_img, depth_img, save_dir: str) -> None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        if color_img is not None:
            color_path = os.path.join(save_dir, f"{stamp}_color.png")
            cv2.imwrite(color_path, color_img)
            log_save(f"컬러 이미지: {color_path}")
        if depth_img is not None:
            depth_png_path = os.path.join(save_dir, f"{stamp}_depth_raw.png")
            depth_npy_path = os.path.join(save_dir, f"{stamp}_depth_raw.npy")
            depth_vis_path = os.path.join(save_dir, f"{stamp}_depth_vis.png")

            # Keep raw 16-bit depth for metric post-processing.
            cv2.imwrite(depth_png_path, depth_img)
            np.save(depth_npy_path, depth_img)
            cv2.imwrite(depth_vis_path, self._depth_to_colormap(depth_img))
            log_save(f"깊이 원본 이미지: {depth_png_path}")
            log_save(f"깊이 원본 데이터: {depth_npy_path}")
            log_save(f"깊이 확인용 이미지: {depth_vis_path}")

    def _cache_measure_intrinsics(self, profile) -> None:
        self.measure_intrinsics = None
        self.measure_stream_name = "unknown"
        try:
            if self.cfg.enable_color:
                vsp = profile.get_stream(rs.stream.color).as_video_stream_profile()
                self.measure_intrinsics = vsp.get_intrinsics()
                self.measure_stream_name = "color"
            elif self.cfg.enable_depth:
                vsp = profile.get_stream(rs.stream.depth).as_video_stream_profile()
                self.measure_intrinsics = vsp.get_intrinsics()
                self.measure_stream_name = "depth"
            else:
                return

        except Exception as exc:
            log_warning(f"크기 측정 준비 중 문제가 생겼습니다: {exc}")
            self.measure_intrinsics = None
            self.measure_stream_name = "unknown"

    def _setup_depth_filters(self) -> None:
        self.depth_filters = []
        if not (self.cfg.enable_depth and self.cfg.depth_filter_enabled):
            return
        try:
            spatial = rs.spatial_filter()
            temporal = rs.temporal_filter()
            hole = rs.hole_filling_filter()

            # Smooth depth noise while preserving edges enough for corner detection.
            spatial.set_option(rs.option.filter_smooth_alpha, float(np.clip(self.cfg.depth_spatial_alpha, 0.25, 1.0)))
            spatial.set_option(rs.option.filter_smooth_delta, float(np.clip(self.cfg.depth_spatial_delta, 1.0, 50.0)))
            temporal.set_option(rs.option.filter_smooth_alpha, float(np.clip(self.cfg.depth_temporal_alpha, 0.25, 1.0)))
            temporal.set_option(rs.option.filter_smooth_delta, float(np.clip(self.cfg.depth_temporal_delta, 1.0, 80.0)))
            hole.set_option(rs.option.holes_fill, float(np.clip(self.cfg.depth_hole_fill_mode, 0, 2)))

            # Order matters: spatial -> temporal -> hole-fill.
            self.depth_filters = [spatial, temporal, hole]
        except Exception as exc:
            log_warning(f"깊이 보정 필터를 켜지 못했습니다. 원본 깊이값을 사용합니다: {exc}")
            self.depth_filters = []

    def _configure_sensors(self, device) -> None:
        for sensor in device.sensors:
            name = sensor.get_info(rs.camera_info.name)

            if name.lower().startswith("stereo"):
                if sensor.supports(rs.option.enable_auto_exposure):
                    try:
                        sensor.set_option(rs.option.enable_auto_exposure, 1)
                    except RuntimeError as exc:
                        log_warning(f"스테레오 자동 노출 설정에 실패했습니다: {exc}")
                if sensor.supports(rs.option.laser_power):
                    try:
                        laser_min = sensor.get_option_range(rs.option.laser_power).min
                        laser_max = sensor.get_option_range(rs.option.laser_power).max
                        sensor.set_option(rs.option.laser_power, 0.7 * (laser_min + laser_max))
                    except RuntimeError as exc:
                        log_warning(f"레이저 출력 설정에 실패했습니다: {exc}")

            if "rgb" in name.lower() and sensor.supports(rs.option.enable_auto_exposure):
                try:
                    sensor.set_option(rs.option.enable_auto_exposure, 1)
                except RuntimeError as exc:
                    log_warning(f"RGB 자동 노출 설정에 실패했습니다: {exc}")

    def _start_with_fallback_profiles(self):
        attempts = self._build_profile_attempts()
        last_error = None

        for idx, p in enumerate(attempts, start=1):
            config = rs.config()
            if self.cfg.serial:
                config.enable_device(self.cfg.serial)

            if self.cfg.enable_depth:
                config.enable_stream(rs.stream.depth, p["dw"], p["dh"], rs.format.z16, p["dfps"])
            if self.cfg.enable_color:
                config.enable_stream(rs.stream.color, p["cw"], p["ch"], rs.format.bgr8, p["cfps"])

            try:
                profile = self.pipeline.start(config)
                ok, reason = self._probe_streams_after_start()
                if not ok:
                    self.pipeline.stop()
                    self.pipeline = rs.pipeline()
                    if idx < len(attempts):
                        log_warning(f"카메라 영상 설정을 다시 시도합니다. ({idx}/{len(attempts)})")
                    last_error = RuntimeError(reason)
                    continue
                return profile, f"depth={p['dw']}x{p['dh']}@{p['dfps']}, color={p['cw']}x{p['ch']}@{p['cfps']}"
            except RuntimeError as exc:
                last_error = exc
                # Reset pipeline between attempts to avoid stale state.
                self.pipeline = rs.pipeline()
                if idx < len(attempts):
                    log_warning(f"카메라 영상 설정을 다시 시도합니다. ({idx}/{len(attempts)})")

        raise RuntimeError(
            "Failed to start camera with all known profiles. "
            "Close RealSense Viewer and retry. "
            f"Last error: {last_error}"
        )

    def _probe_streams_after_start(self):
        reason = "unknown stream error"
        retries = max(1, int(self.cfg.startup_frame_retries))
        timeout_ms = max(100, int(self.cfg.startup_frame_timeout_ms))

        for _ in range(retries):
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms)
            except RuntimeError as exc:
                reason = f"frame timeout after start ({exc})"
                continue

            if self.align is not None:
                try:
                    frames = self.align.process(frames)
                except RuntimeError as exc:
                    reason = f"align failed ({exc})"
                    continue

            depth_frame = frames.get_depth_frame() if self.cfg.enable_depth else None
            color_frame = frames.get_color_frame() if self.cfg.enable_color else None

            if self.cfg.enable_depth and not depth_frame:
                reason = "depth stream missing after start"
                continue
            if self.cfg.enable_color and not color_frame:
                reason = "color stream missing after start"
                continue

            return True, "ok"

        return False, reason

    def _build_profile_attempts(self):
        req = {
            "dw": self.cfg.width,
            "dh": self.cfg.height,
            "dfps": self.cfg.fps,
            "cw": self.cfg.width,
            "ch": self.cfg.height,
            "cfps": self.cfg.fps,
        }

        fallback = [
            {"dw": 848, "dh": 480, "dfps": 30, "cw": 640, "ch": 480, "cfps": 30},
            {"dw": 640, "dh": 480, "dfps": 30, "cw": 640, "ch": 480, "cfps": 30},
            {"dw": 640, "dh": 480, "dfps": 15, "cw": 640, "ch": 480, "cfps": 15},
            {"dw": 848, "dh": 480, "dfps": 15, "cw": 640, "ch": 480, "cfps": 15},
        ]

        attempts = [req]
        for item in fallback:
            if item != req:
                attempts.append(item)
        return attempts

    @staticmethod
    def _ensure_device_present() -> None:
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("No Intel RealSense device detected. Check USB cable/power.")

        for dev in devices:
            name = dev.get_info(rs.camera_info.name) if dev.supports(rs.camera_info.name) else ""
            pid = dev.get_info(rs.camera_info.product_id) if dev.supports(rs.camera_info.product_id) else ""
            if "Recovery" in name or pid.upper() == "0ADB":
                raise RuntimeError(
                    "Intel RealSense is in Recovery mode (PID 0ADB). "
                    "Update firmware first (RealSense Viewer -> Install), "
                    "then unplug/replug the camera and run again."
                )

    @staticmethod
    def _print_device_info(device) -> None:
        name = device.get_info(rs.camera_info.name)
        if "D435" not in name:
            log_warning(f"연결된 카메라가 D435로 표시되지 않습니다. ({name})")


def parse_args() -> CameraConfig:
    parser = argparse.ArgumentParser(description="Intel RealSense D435 camera setup")
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--serial", type=str, default=None)
    parser.add_argument("--no-align", action="store_true", help="Disable depth-to-color alignment")
    parser.add_argument("--depth-min", type=float, default=0.2, help="Min depth for visualization (m)")
    parser.add_argument("--depth-max", type=float, default=2.5, help="Max depth for visualization (m)")
    parser.add_argument(
        "--target-min-dist",
        type=float,
        default=0.10,
        help="Ignore too-near noisy depth below this distance (m)",
    )
    parser.add_argument(
        "--target-max-dist",
        type=float,
        default=0.60,
        help="Treat objects farther than this as background (m)",
    )
    parser.add_argument(
        "--target-min-height",
        type=float,
        default=0.004,
        help="Minimum height above dominant background depth to be considered object (m)",
    )
    parser.add_argument(
        "--target-band-half-width",
        type=float,
        default=0.035,
        help="Depth band half width around nearest layer for candidate generation (m)",
    )
    parser.add_argument(
        "--quad-depth-std-max",
        type=float,
        default=0.035,
        help="Max depth std inside candidate quad (m)",
    )
    parser.add_argument(
        "--quad-depth-valid-min",
        type=float,
        default=0.45,
        help="Minimum valid-depth ratio inside candidate quad (0~1)",
    )
    parser.add_argument(
        "--quad-depth-lift-min",
        type=float,
        default=0.006,
        help="Minimum depth lift (surrounding - inside) for quad acceptance (m)",
    )
    parser.add_argument(
        "--quad-depth-ring-margin",
        type=int,
        default=10,
        help="Pixel margin for surrounding-depth ring used in lift check",
    )
    parser.add_argument(
        "--top-face-min-nz",
        type=float,
        default=0.58,
        help="Minimum |Nz| of candidate plane normal (higher=more top-face only, 0~1)",
    )
    parser.add_argument(
        "--top-face-depth-span-max",
        type=float,
        default=0.030,
        help="Maximum P90-P10 depth span inside candidate quad (m)",
    )
    parser.add_argument(
        "--debug-detect",
        action="store_true",
        help="Print periodic detection diagnostics to terminal",
    )
    parser.add_argument(
        "--debug-interval",
        type=int,
        default=15,
        help="Print diagnostics every N frames when --debug-detect is on",
    )
    parser.add_argument("--show-outline", action="store_true", help="Draw rectangle outline in addition to corners")
    parser.add_argument(
        "--corner-smooth-alpha",
        type=float,
        default=0.72,
        help="Corner smoothing factor (0=no smoothing, 0.9=very smooth)",
    )
    parser.add_argument(
        "--corner-hold-frames",
        type=int,
        default=12,
        help="Keep last corner positions for this many frames when detection drops",
    )
    parser.add_argument(
        "--corner-deadband-px",
        type=float,
        default=4.0,
        help="Freeze corner updates when movement is smaller than this threshold (pixels)",
    )
    parser.add_argument(
        "--corner-jump-guard-px",
        type=float,
        default=22.0,
        help="If corner jump is larger than this, apply extra-strong smoothing (pixels)",
    )
    parser.add_argument(
        "--corner-continuity-weight",
        type=float,
        default=0.50,
        help="How strongly previous-frame proximity affects candidate ranking (0~1)",
    )
    parser.add_argument(
        "--coordinate-lock",
        dest="coord_lock_enabled",
        action="store_true",
        help="Enable per-object coordinate lock (default: off)",
    )
    parser.add_argument(
        "--no-coordinate-lock",
        dest="coord_lock_enabled",
        action="store_false",
        help="Disable per-object coordinate lock",
    )
    parser.add_argument(
        "--coord-lock-enter-px",
        type=float,
        default=3.2,
        help="Lock center when movement stays within this threshold (pixels)",
    )
    parser.add_argument(
        "--coord-lock-exit-px",
        type=float,
        default=10.0,
        help="Unlock center when movement exceeds this threshold (pixels)",
    )
    parser.add_argument(
        "--coord-lock-frames",
        type=int,
        default=3,
        help="Frames required to engage coordinate lock",
    )
    parser.add_argument(
        "--coord-lock-depth-exit",
        type=float,
        default=0.022,
        help="Unlock lock-state when depth change exceeds this (m)",
    )
    parser.add_argument(
        "--no-static-coordinates",
        action="store_true",
        help="Disable static coordinate freeze after lock becomes stable",
    )
    parser.add_argument(
        "--coord-static-min-frames",
        type=int,
        default=18,
        help="Frames in lock-state before static coordinate freeze engages",
    )
    parser.add_argument(
        "--no-sticky-locked-slots",
        action="store_true",
        help="Disable sticky hold for static-locked slots when detections are temporarily missing",
    )
    parser.add_argument(
        "--sticky-locked-max-miss",
        type=int,
        default=300,
        help="Max missing frames to keep sticky locked slots before reset",
    )
    parser.add_argument(
        "--coord-move-release-frames",
        type=int,
        default=6,
        help="Consecutive movement frames required to release sticky lock",
    )
    parser.add_argument(
        "--show-hold-slots",
        dest="show_hold_slots",
        action="store_true",
        help="Display missing objects as hold(previous) for a few frames",
    )
    parser.add_argument(
        "--no-hold-slots",
        dest="show_hold_slots",
        action="store_false",
        help="Disable hold(previous) display for temporarily missing objects (default)",
    )
    parser.set_defaults(coord_lock_enabled=False, show_hold_slots=False)
    parser.add_argument(
        "--no-depth-filter",
        action="store_true",
        help="Disable RealSense temporal/spatial depth filtering",
    )
    parser.add_argument(
        "--depth-temporal-alpha",
        type=float,
        default=0.35,
        help="Depth temporal filter alpha (lower=more smoothing)",
    )
    parser.add_argument(
        "--depth-temporal-delta",
        type=float,
        default=20.0,
        help="Depth temporal filter delta",
    )
    parser.add_argument(
        "--depth-spatial-alpha",
        type=float,
        default=0.50,
        help="Depth spatial filter alpha (lower=more smoothing)",
    )
    parser.add_argument(
        "--depth-spatial-delta",
        type=float,
        default=20.0,
        help="Depth spatial filter delta",
    )
    parser.add_argument(
        "--depth-hole-fill",
        type=int,
        default=1,
        help="Depth hole fill mode (0~2)",
    )
    parser.add_argument(
        "--no-size-measure",
        action="store_true",
        help="Disable 3D top-face width/height measurement overlay",
    )
    parser.add_argument(
        "--tune-sensors",
        action="store_true",
        help="Enable optional sensor tuning (laser/auto-exposure) on startup",
    )
    parser.add_argument(
        "--frame-timeout-ms",
        type=int,
        default=1500,
        help="Frame wait timeout in preview loop (ms)",
    )
    parser.add_argument(
        "--startup-frame-timeout-ms",
        type=int,
        default=1200,
        help="Frame wait timeout while validating profile right after start (ms)",
    )
    parser.add_argument(
        "--startup-frame-retries",
        type=int,
        default=3,
        help="Number of frame validation retries right after profile start",
    )
    parser.add_argument(
        "--show-depth-panel",
        dest="show_depth_panel",
        action="store_true",
        help="Show the right-side depth preview panel",
    )
    parser.add_argument(
        "--hide-depth-panel",
        dest="show_depth_panel",
        action="store_false",
        help="Hide the right-side depth preview panel (default)",
    )
    parser.set_defaults(show_depth_panel=False)
    parser.add_argument("--no-rect", action="store_true", help="Disable rectangle-corner overlay")
    parser.add_argument("--rect-min-area", type=int, default=500, help="Minimum rectangle area in pixels")
    parser.add_argument("--rect-min-side", type=int, default=45, help="Minimum side length of rectangle in pixels")
    parser.add_argument("--rect-max-aspect", type=float, default=8.0, help="Maximum rectangle aspect ratio")
    parser.add_argument(
        "--center-weight-power",
        type=float,
        default=2.6,
        help="How strongly center-near candidates are preferred (1~4)",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=4,
        help=f"Maximum number of objects to track/display (1~{MAX_TRACKED_OBJECTS})",
    )
    parser.add_argument(
        "--object-nms-iou",
        type=float,
        default=0.22,
        help="IoU threshold for suppressing overlapping object candidates",
    )
    parser.add_argument(
        "--object-min-score-ratio",
        type=float,
        default=0.30,
        help="Reject weak objects below this score ratio relative to top object (0~1)",
    )
    parser.add_argument(
        "--object-dup-center-ratio",
        type=float,
        default=0.75,
        help="Suppress near-duplicate objects when center distance is below diag*ratio",
    )
    parser.add_argument(
        "--object-dup-depth-diff",
        type=float,
        default=0.03,
        help="Depth closeness threshold (m) used in duplicate suppression",
    )
    parser.add_argument(
        "--object-secondary-score-ratio",
        type=float,
        default=0.10,
        help="Secondary candidate fill score ratio (0~1) for multi-object scenes",
    )
    parser.add_argument(
        "--object-secondary-area-ratio",
        type=float,
        default=0.10,
        help="Secondary candidate fill area ratio (0~1) relative to top object",
    )
    parser.add_argument(
        "--white-bias",
        dest="white_bias",
        action="store_true",
        help="Boost candidate scores when the top face looks white/low-saturation",
    )
    parser.add_argument(
        "--no-white-bias",
        dest="white_bias",
        action="store_false",
        help="Disable white top-face score weighting",
    )
    parser.add_argument(
        "--white-min-ratio",
        type=float,
        default=0.18,
        help="Minimum white-pixel ratio inside candidate quad before score boost ramps in",
    )
    parser.add_argument(
        "--white-sat-max",
        type=int,
        default=64,
        help="Maximum HSV saturation considered white-ish (0~255)",
    )
    parser.add_argument(
        "--white-value-min",
        type=int,
        default=140,
        help="Minimum HSV value considered bright enough for white weighting (0~255)",
    )
    parser.add_argument(
        "--white-score-gain",
        type=float,
        default=0.55,
        help="Maximum extra score gain applied by white weighting (0~1)",
    )
    parser.set_defaults(white_bias=None)
    parser.add_argument(
        "--rect-max-area-ratio",
        type=float,
        default=0.80,
        help="Reject candidates larger than this ratio of ROI area (0~1)",
    )
    parser.add_argument(
        "--rect-border-margin",
        type=int,
        default=4,
        help="Reject candidates touching ROI border within this many pixels",
    )
    parser.add_argument(
        "--rect-min-extent",
        type=float,
        default=0.18,
        help="Minimum contour/box area ratio for rectangle candidacy (0~1)",
    )
    parser.add_argument("--center-roi-only", action="store_true", help="Detect rectangles only in center ROI")
    parser.add_argument("--no-center-roi", action="store_true", help="Disable center ROI mode at startup")
    parser.add_argument("--center-roi-ratio", type=float, default=0.60, help="Center ROI size ratio (0~1)")
    args = parser.parse_args()

    max_objects = max(1, min(int(args.max_objects), MAX_TRACKED_OBJECTS))
    if args.center_roi_only:
        center_roi_only = True
    elif args.no_center_roi:
        center_roi_only = False
    else:
        # For 3+ objects, full-frame search works better than center-only ROI.
        center_roi_only = max_objects <= 2

    return CameraConfig(
        width=args.width,
        height=args.height,
        fps=args.fps,
        serial=args.serial,
        align_to_color=not args.no_align,
        depth_min_m=args.depth_min,
        depth_max_m=args.depth_max,
        show_depth_panel=args.show_depth_panel,
        detect_rectangle=not args.no_rect,
        rect_min_area=args.rect_min_area,
        rect_min_side_px=args.rect_min_side,
        rect_max_aspect_ratio=args.rect_max_aspect,
        center_weight_power=args.center_weight_power,
        max_objects=max_objects,
        object_nms_iou=args.object_nms_iou,
        object_min_score_ratio=args.object_min_score_ratio,
        object_duplicate_center_ratio=args.object_dup_center_ratio,
        object_duplicate_depth_diff_m=args.object_dup_depth_diff,
        object_secondary_score_ratio=args.object_secondary_score_ratio,
        object_secondary_area_ratio=args.object_secondary_area_ratio,
        white_bias_enabled=True if args.white_bias is None else args.white_bias,
        white_min_ratio=args.white_min_ratio,
        white_sat_max=args.white_sat_max,
        white_value_min=args.white_value_min,
        white_score_gain=args.white_score_gain,
        rect_max_area_ratio=args.rect_max_area_ratio,
        rect_border_margin_px=args.rect_border_margin,
        rect_min_extent=args.rect_min_extent,
        center_roi_only=center_roi_only,
        center_roi_ratio=args.center_roi_ratio,
        target_min_distance_m=args.target_min_dist,
        target_max_distance_m=args.target_max_dist,
        target_min_height_m=args.target_min_height,
        target_band_half_width_m=args.target_band_half_width,
        quad_depth_std_max_m=args.quad_depth_std_max,
        quad_depth_valid_min_ratio=args.quad_depth_valid_min,
        quad_depth_lift_min_m=args.quad_depth_lift_min,
        quad_depth_ring_margin_px=args.quad_depth_ring_margin,
        top_face_min_abs_nz=args.top_face_min_nz,
        top_face_depth_span_max_m=args.top_face_depth_span_max,
        debug_detection=args.debug_detect,
        debug_print_interval=max(1, int(args.debug_interval)),
        corner_only_display=not args.show_outline,
        corner_smooth_alpha=args.corner_smooth_alpha,
        corner_hold_frames=args.corner_hold_frames,
        corner_deadband_px=args.corner_deadband_px,
        corner_jump_guard_px=args.corner_jump_guard_px,
        corner_continuity_weight=args.corner_continuity_weight,
        coord_lock_enabled=args.coord_lock_enabled,
        coord_lock_enter_px=args.coord_lock_enter_px,
        coord_lock_exit_px=args.coord_lock_exit_px,
        coord_lock_frames=args.coord_lock_frames,
        coord_lock_depth_exit_m=args.coord_lock_depth_exit,
        coord_lock_static_mode=not args.no_static_coordinates,
        coord_lock_static_min_frames=max(1, int(args.coord_static_min_frames)),
        coord_lock_sticky_slots=not args.no_sticky_locked_slots,
        coord_lock_sticky_max_miss=max(1, int(args.sticky_locked_max_miss)),
        coord_lock_move_release_frames=max(1, int(args.coord_move_release_frames)),
        show_hold_slots=args.show_hold_slots,
        measure_size=not args.no_size_measure,
        depth_filter_enabled=not args.no_depth_filter,
        depth_temporal_alpha=args.depth_temporal_alpha,
        depth_temporal_delta=args.depth_temporal_delta,
        depth_spatial_alpha=args.depth_spatial_alpha,
        depth_spatial_delta=args.depth_spatial_delta,
        depth_hole_fill_mode=args.depth_hole_fill,
        tune_sensors=args.tune_sensors,
        frame_wait_timeout_ms=args.frame_timeout_ms,
        startup_frame_timeout_ms=args.startup_frame_timeout_ms,
        startup_frame_retries=args.startup_frame_retries,
    )


def main() -> None:
    cfg = parse_args()
    cam = D435Camera(cfg)

    try:
        cam.start()
        cam.preview(save_dir="captures")
    finally:
        cam.stop()


if __name__ == "__main__":
    main()
