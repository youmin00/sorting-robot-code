"""로봇팔 작업영역 판단, 역기구학 계산, 양팔 계획 선택."""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np

ARM_PLAN_CACHE_GRID_MM = 2.0
ARM_PLAN_CACHE_MAX_ENTRIES = 256

ROBOT_FORWARD_TO_CENTER_MM = 209.5
RIGHT_ARM_FORWARD_TO_CENTER_MM = 205.5
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
RIGHT_ARM_MOUNT_ROTATION_CORRECTION_DEG = 0.0
RIGHT_ARM_CAMERA_X_OFFSET_MM = 0.0
RIGHT_ARM_CAMERA_Y_OFFSET_MM = 0.0
RIGHT_ARM_J1_FINE_OFFSET_DEG = -0.5
RIGHT_ARM_LOCAL_POSITIVE_X_J1_OFFSET_DEG = -4.0
RIGHT_ARM_LOCAL_NEGATIVE_X_J1_OFFSET_DEG = 3.5
RIGHT_ARM_FAR_NEGATIVE_Y_MM = -60.0
RIGHT_ARM_NEAR_NEGATIVE_Y_MM = -20.0
RIGHT_ARM_FAR_NEGATIVE_Y_J1_OFFSET_DEG = 2.0
EDGE_3CM_MIN_CAMERA_X_MM = 90.0
EDGE_3CM_MIN_CAMERA_Y_MM = 75.0
EDGE_3CM_EXTRA_DROP_MM = 4.0
CENTER_RISK_MIN_X_MM = -120.0
CENTER_RISK_MAX_X_MM = 120.0
CENTER_RISK_HALF_Y_MM = 30.0
CENTER_SIMULTANEOUS_MIN_X_GAP_MM = 140.0

_ARM_PLAN_CACHE: dict[tuple, dict] = {}

# IK searches the same fixed 0.5-degree grid for every object.  Precomputing
# the grid and its trigonometric terms avoids repeating tens of thousands of
# Python-level loop iterations for every new camera coordinate.
_ARM_J2_VALUES = np.arange(361, dtype=np.float64) * 0.5
_ARM_J3_VALUES = np.arange(181, dtype=np.float64) * 0.5
_ARM_J2_GRID = _ARM_J2_VALUES[:, None]
_ARM_J3_GRID = _ARM_J3_VALUES[None, :]
_ARM_A2_GRID = _ARM_J2_GRID - 60.0 - _ARM_J3_GRID
_ARM_BASE_RADIUS_GRID = (
    ROBOT_L1_MM * np.cos(np.radians(_ARM_J2_GRID))
    + ROBOT_L2_MM * np.cos(np.radians(_ARM_A2_GRID))
)
_ARM_BASE_HEIGHT_GRID = (
    ROBOT_J2_HEIGHT_MM
    + ROBOT_L1_MM * np.sin(np.radians(_ARM_J2_GRID))
    + ROBOT_L2_MM * np.sin(np.radians(_ARM_A2_GRID))
)

def _arm_ik(camera_x_mm: float, camera_y_mm: float, target_z_mm: float,
            compression_mm: float, max_tilt_deg: float,
            forward_to_center_mm: float = ROBOT_FORWARD_TO_CENTER_MM) -> Optional[dict]:
    """Search the verified 0.5-degree IK grid using vectorized calculations."""
    robot_x = float(forward_to_center_mm) - float(camera_y_mm)
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
    l3 = ROBOT_L3_MM - compression_mm
    for tilt in tilts:
        j4_grid = tilt - _ARM_A2_GRID
        valid = (j4_grid >= 0.0) & (j4_grid <= 180.0)
        a3_rad = np.radians(-90.0 + tilt)
        radius_grid = _ARM_BASE_RADIUS_GRID + l3 * np.cos(a3_rad)
        height_grid = _ARM_BASE_HEIGHT_GRID + l3 * np.sin(a3_rad)
        distance_grid = np.hypot(radius_grid - radial, height_grid - target_z_mm)
        masked_distances = np.where(valid, distance_grid, np.inf)
        flat_index = int(np.argmin(masked_distances))
        dist = float(masked_distances.flat[flat_index])
        if not np.isfinite(dist):
            continue

        j2_index, j3_index = np.unravel_index(flat_index, masked_distances.shape)
        j2 = float(_ARM_J2_VALUES[j2_index])
        j3 = float(_ARM_J3_VALUES[j3_index])
        a2 = j2 - 60.0 - j3
        best_at_tilt = {
            "j1": round(j1 * 2.0) / 2.0,
            "j2": j2,
            "j3": j3,
            "j4": tilt - a2,
            "tilt": tilt,
            "error": dist,
        }
        if best is None or dist < best["error"]:
            best = best_at_tilt
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
    if arm == "right" and not object_in_right_workspace(obj):
        raise ValueError("큐브 전체가 오른쪽 로봇 작업영역 안에 있지 않습니다.")
    if arm != "right" and not object_in_left_workspace(obj):
        raise ValueError("큐브 전체가 왼쪽 로봇 작업영역 안에 있지 않습니다.")

    cache_key = (
        arm,
        cube_size,
        int(round(global_camera_x / ARM_PLAN_CACHE_GRID_MM)),
        int(round(global_camera_y / ARM_PLAN_CACHE_GRID_MM)),
    )
    cached_plan = _ARM_PLAN_CACHE.get(cache_key)
    if cached_plan is not None:
        result = copy.deepcopy(cached_plan)
        result["camera_x"] = global_camera_x
        result["camera_y"] = global_camera_y
        return result

    if arm == "right":
        rotation_rad = np.radians(180.0 + RIGHT_ARM_MOUNT_ROTATION_CORRECTION_DEG)
        camera_x = (
            np.cos(rotation_rad) * global_camera_x
            - np.sin(rotation_rad) * global_camera_y
            + RIGHT_ARM_CAMERA_X_OFFSET_MM
        )
        camera_y = (
            np.sin(rotation_rad) * global_camera_x
            + np.cos(rotation_rad) * global_camera_y
            + RIGHT_ARM_CAMERA_Y_OFFSET_MM
        )
    else:
        camera_x = global_camera_x
        camera_y = global_camera_y
    top = float(cube_size)
    forward_to_center_mm = (
        RIGHT_ARM_FORWARD_TO_CENTER_MM
        if arm == "right"
        else ROBOT_FORWARD_TO_CENTER_MM
    )
    edge_extra_drop = (
        EDGE_3CM_EXTRA_DROP_MM
        if cube_size == 30
        and camera_x >= EDGE_3CM_MIN_CAMERA_X_MM
        and camera_y >= EDGE_3CM_MIN_CAMERA_Y_MM
        else 0.0
    )
    contact_z = top - edge_extra_drop
    approach = _arm_ik(
        camera_x, camera_y, top + 16.0, 0.0, 15.0, forward_to_center_mm
    )
    contact = _arm_ik(
        camera_x, camera_y, contact_z, 0.0, 15.0, forward_to_center_mm
    )
    preload_compression = 2.0
    preload = _arm_ik(
        camera_x, camera_y, contact_z, preload_compression, 15.0, forward_to_center_mm
    )
    for _ in range(2):
        if preload is None:
            break
        adjusted = min(5.0, 2.0 + abs(float(preload["tilt"])) * 0.2)
        if abs(adjusted - preload_compression) < 0.01:
            break
        preload_compression = adjusted
        preload = _arm_ik(
            camera_x, camera_y, contact_z, preload_compression, 15.0,
            forward_to_center_mm
        )
    lift = _arm_ik(
        camera_x, camera_y, top + 16.0, 8.0, 45.0, forward_to_center_mm
    )
    poses = (approach, contact, preload, lift)
    if any(p is None or p["error"] > ROBOT_IK_TOLERANCE_MM for p in poses):
        raise ValueError("접근·접촉·예압·상승 자세 중 도달할 수 없는 단계가 있습니다.")

    # 실기 캘리브레이션:
    # - 상부 접근·접촉 J4 +4.5°, 수직이 맞는 예압 J4 +3°
    # - 양팔 J1 보정은 독립 설정하며, 각 팔의 local +X에서 추가 보정을 적용한다.
    for pose in (approach, contact):
        pose["j4"] = float(np.clip(
            pose["j4"] + ROBOT_APPROACH_J4_OFFSET_DEG, 0.0, 180.0
        ))
    preload["j4"] = float(np.clip(
        preload["j4"] + ROBOT_PRELOAD_J4_OFFSET_DEG, 0.0, 180.0
    ))

    if arm == "right":
        j1_offset = RIGHT_ARM_J1_FINE_OFFSET_DEG
        if camera_x > 0.0:
            j1_offset += RIGHT_ARM_LOCAL_POSITIVE_X_J1_OFFSET_DEG
        elif camera_x < 0.0:
            y_blend = float(np.clip(
                (RIGHT_ARM_NEAR_NEGATIVE_Y_MM - global_camera_y)
                / (RIGHT_ARM_NEAR_NEGATIVE_Y_MM - RIGHT_ARM_FAR_NEGATIVE_Y_MM),
                0.0,
                1.0,
            ))
            j1_offset += (
                RIGHT_ARM_LOCAL_NEGATIVE_X_J1_OFFSET_DEG
                + y_blend
                * (
                    RIGHT_ARM_FAR_NEGATIVE_Y_J1_OFFSET_DEG
                    - RIGHT_ARM_LOCAL_NEGATIVE_X_J1_OFFSET_DEG
                )
            )
    else:
        j1_offset = ROBOT_J1_FINE_OFFSET_DEG
        if camera_x > 0.0:
            j1_offset += ROBOT_POSITIVE_X_J1_OFFSET_DEG
    for pose in poses:
        pose["j1"] = float(np.clip(pose["j1"] + j1_offset, 0.0, 180.0))

    result = {
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
    if len(_ARM_PLAN_CACHE) >= ARM_PLAN_CACHE_MAX_ENTRIES:
        _ARM_PLAN_CACHE.clear()
    _ARM_PLAN_CACHE[cache_key] = copy.deepcopy(result)
    return result

def plan_in_center_risk_zone(plan: Optional[dict]) -> bool:
    if plan is None:
        return False
    return (
        CENTER_RISK_MIN_X_MM <= float(plan["camera_x"]) <= CENTER_RISK_MAX_X_MM
        and -CENTER_RISK_HALF_Y_MM <= float(plan["camera_y"]) <= CENTER_RISK_HALF_Y_MM
    )


def center_priority_side(left_count: int, right_count: int) -> Optional[str]:
    total = left_count + right_count
    if total == 0:
        return None
    if total == 1:
        return "left" if left_count else "right"
    if total == 2:
        return "left" if left_count else "right"
    if total == 3:
        return "left" if left_count > right_count else "right"
    if left_count == right_count:
        return "left"
    return "left" if left_count > right_count else "right"


def choose_pipeline_plans(
    holding_side: Optional[str],
    left_plans: list[dict],
    right_plans: list[dict],
    preferred_side: str,
) -> tuple[Optional[dict], Optional[dict]]:
    """한 번에 중앙 작업영역으로 진입하는 빈 팔은 반드시 하나만 선택한다."""
    if holding_side == "left":
        return None, (right_plans[0] if right_plans else None)
    if holding_side == "right":
        return (left_plans[0] if left_plans else None), None
    if preferred_side == "left" and left_plans:
        return left_plans[0], None
    if preferred_side == "right" and right_plans:
        return None, right_plans[0]
    if left_plans:
        return left_plans[0], None
    if right_plans:
        return None, right_plans[0]
    return None, None

