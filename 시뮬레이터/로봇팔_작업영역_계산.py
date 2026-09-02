from __future__ import annotations

import base64
import io
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import matplotlib
import matplotlib.font_manager as fm

GUI_BACKEND_AVAILABLE = True
GUI_BACKEND_ERROR = ""


def can_open_tk_window() -> bool:

    global GUI_BACKEND_ERROR
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        root.destroy()
        return True
    except Exception as exc:
        GUI_BACKEND_ERROR = str(exc).splitlines()[0]
        return False


if "--no-show" in sys.argv:
    GUI_BACKEND_AVAILABLE = False
    matplotlib.use("Agg")
elif not can_open_tk_window():
    GUI_BACKEND_AVAILABLE = False
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


LINK1_CM = 13.0
LINK2_CM = 13.0
GRIPPER_PHYSICAL_CM = 8.5
GRIPPER_EFFECTIVE_CM = 7.7
J2_HEIGHT_FROM_FLOOR_CM = 10.325
BASE_RADIUS_CM = 6.0
BASE_HEIGHT_CM = 5.83


J1_MIN_DEG, J1_MAX_DEG = 50.0, 130.0
J2_MIN_DEG, J2_MAX_DEG = 40.0, 90.0
J3_MIN_DEG, J3_MAX_DEG = 45.0, 120.0
J4_MIN_DEG, J4_MAX_DEG = 15.0, 55.0


ANGLE_STEP_DEG = 2.0
J1_STEP_DEG = 2.0


TARGET_GRIPPER_ANGLE_DEG = -90.0


J2_SIGN = 1.0
J3_SIGN = 1.0
J4_SIGN = 1.0


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "이미지")


POINT_SIZE_2D = 2
POINT_SIZE_3D = 1


@dataclass
class WorkspaceData:

    all_points_xz: np.ndarray
    all_angles_j234: np.ndarray
    vertical_points_xz: np.ndarray
    vertical_wrist_xz: np.ndarray
    vertical_angles_j234: np.ndarray
    excluded_by_j4_count: int
    workspace_3d_xyz: np.ndarray
    workspace_3d_angles_j1234: np.ndarray


WORKSPACE_DATA: Optional[WorkspaceData] = None


def setup_korean_font() -> None:

    preferred_fonts = ["D2Coding", "Malgun Gothic", "AppleGothic", "NanumGothic"]
    installed_font_names = {fm.FontProperties(fname=path).get_name() for path in fm.findSystemFonts()}

    for font_name in preferred_fonts:
        if font_name in installed_font_names:
            plt.rcParams["font.family"] = font_name
            break


    plt.rcParams["axes.unicode_minus"] = False


def make_angle_samples(min_deg: float, max_deg: float, step_deg: float) -> np.ndarray:

    return np.arange(min_deg, max_deg + step_deg * 0.5, step_deg)


def deg_to_rad(deg: np.ndarray | float) -> np.ndarray | float:
    return np.deg2rad(deg)


def angle_to_unit_vector(angle_deg: np.ndarray | float) -> Tuple[np.ndarray | float, np.ndarray | float]:

    rad = deg_to_rad(angle_deg)
    return np.cos(rad), np.sin(rad)


def set_axes_equal_2d(ax: plt.Axes) -> None:

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("z [cm]")


def set_axes_equal_3d(ax: plt.Axes, xyz: np.ndarray) -> None:

    if xyz.size == 0:
        return

    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    if radius <= 0:
        radius = 1.0

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def save_figure(fig: plt.Figure, filename: str) -> None:
    os.makedirs(IMAGE_DIR, exist_ok=True)
    fig.savefig(os.path.join(IMAGE_DIR, filename), dpi=200, bbox_inches="tight")


def servo_angles_to_absolute_angles(
    j2_deg: np.ndarray | float,
    j3_deg: np.ndarray | float,
    j4_deg: np.ndarray | float,
) -> Tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float]:



    theta1_deg = 90.0 + J2_SIGN * (j2_deg - 90.0)
    theta2_deg = theta1_deg - J3_SIGN * (180.0 - j3_deg)
    theta3_deg = theta2_deg - J4_SIGN * (90.0 - j4_deg)
    return theta1_deg, theta2_deg, theta3_deg


def forward_kinematics_2d(
    j2_deg: np.ndarray | float,
    j3_deg: np.ndarray | float,
    j4_deg: np.ndarray | float,
    gripper_length_cm: float,
) -> Dict[str, np.ndarray | float]:



    theta1_deg, theta2_deg, theta3_deg = servo_angles_to_absolute_angles(j2_deg, j3_deg, j4_deg)

    c1, s1 = angle_to_unit_vector(theta1_deg)
    c2, s2 = angle_to_unit_vector(theta2_deg)
    c3, s3 = angle_to_unit_vector(theta3_deg)

    j3_x = LINK1_CM * c1
    j3_z = LINK1_CM * s1

    j4_x = j3_x + LINK2_CM * c2
    j4_z = j3_z + LINK2_CM * s2

    tip_x = j4_x + gripper_length_cm * c3
    tip_z = j4_z + gripper_length_cm * s3

    return {
        "j3_x": j3_x,
        "j3_z": j3_z,
        "j4_x": j4_x,
        "j4_z": j4_z,
        "tip_x": tip_x,
        "tip_z": tip_z,
        "theta1_deg": theta1_deg,
        "theta2_deg": theta2_deg,
        "theta3_deg": theta3_deg,
    }


def required_j4_for_vertical_gripper(theta2_deg: np.ndarray | float) -> np.ndarray | float:





    return 90.0 - (theta2_deg - TARGET_GRIPPER_ANGLE_DEG) / J4_SIGN


def segment_intersects_base_cylinder_2d(
    x1: np.ndarray,
    z1: np.ndarray,
    x2: np.ndarray,
    z2: np.ndarray,
    sample_count: int = 15,
    start_t: float = 0.0,
    end_t: float = 1.0,
) -> np.ndarray:
    hit = np.zeros_like(np.asarray(x1, dtype=float), dtype=bool)
    for t in np.linspace(start_t, end_t, sample_count):
        x = x1 + (x2 - x1) * t
        z_floor = z1 + (z2 - z1) * t + J2_HEIGHT_FROM_FLOOR_CM
        hit |= (np.abs(x) <= BASE_RADIUS_CM) & (z_floor >= 0.0) & (z_floor <= BASE_HEIGHT_CM)
    return hit


def calculate_all_reachable_workspace() -> Tuple[np.ndarray, np.ndarray]:



    j2_values = make_angle_samples(J2_MIN_DEG, J2_MAX_DEG, ANGLE_STEP_DEG)
    j3_values = make_angle_samples(J3_MIN_DEG, J3_MAX_DEG, ANGLE_STEP_DEG)
    j4_values = make_angle_samples(J4_MIN_DEG, J4_MAX_DEG, ANGLE_STEP_DEG)

    j2_grid, j3_grid, j4_grid = np.meshgrid(j2_values, j3_values, j4_values, indexing="ij")
    fk = forward_kinematics_2d(j2_grid, j3_grid, j4_grid, GRIPPER_PHYSICAL_CM)

    points_xz = np.column_stack((fk["tip_x"].ravel(), fk["tip_z"].ravel()))
    angles_j234 = np.column_stack((j2_grid.ravel(), j3_grid.ravel(), j4_grid.ravel()))
    return points_xz, angles_j234


def calculate_vertical_suction_workspace() -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:



    j2_values = make_angle_samples(J2_MIN_DEG, J2_MAX_DEG, ANGLE_STEP_DEG)
    j3_values = make_angle_samples(J3_MIN_DEG, J3_MAX_DEG, ANGLE_STEP_DEG)

    j2_grid, j3_grid = np.meshgrid(j2_values, j3_values, indexing="ij")


    theta1_deg, theta2_deg, _ = servo_angles_to_absolute_angles(j2_grid, j3_grid, 90.0)
    required_j4_deg = required_j4_for_vertical_gripper(theta2_deg)

    c1, s1 = angle_to_unit_vector(theta1_deg)
    c2, s2 = angle_to_unit_vector(theta2_deg)

    j4_x = LINK1_CM * c1 + LINK2_CM * c2
    j4_z = LINK1_CM * s1 + LINK2_CM * s2


    contact_x = j4_x
    contact_z = j4_z - GRIPPER_EFFECTIVE_CM
    valid_j4 = (required_j4_deg >= J4_MIN_DEG) & (required_j4_deg <= J4_MAX_DEG)
    valid_floor = (contact_z + J2_HEIGHT_FROM_FLOOR_CM) >= 0.0
    j2_x = np.zeros_like(j4_x)
    j2_z = np.zeros_like(j4_z)
    j3_x = LINK1_CM * c1
    j3_z = LINK1_CM * s1
    hit_link1 = segment_intersects_base_cylinder_2d(j2_x, j2_z, j3_x, j3_z, start_t=0.20)
    hit_link2 = segment_intersects_base_cylinder_2d(j3_x, j3_z, j4_x, j4_z)
    hit_gripper = segment_intersects_base_cylinder_2d(j4_x, j4_z, contact_x, contact_z)
    valid_base_collision = ~(hit_link1 | hit_link2 | hit_gripper)
    valid_pose = valid_j4 & valid_floor & valid_base_collision
    excluded_by_j4_count = int(np.size(valid_j4) - np.count_nonzero(valid_j4))

    points_xz = np.column_stack((contact_x[valid_pose].ravel(), contact_z[valid_pose].ravel()))
    wrist_xz = np.column_stack((j4_x[valid_pose].ravel(), j4_z[valid_pose].ravel()))
    angles_j234 = np.column_stack(
        (
            j2_grid[valid_pose].ravel(),
            j3_grid[valid_pose].ravel(),
            required_j4_deg[valid_pose].ravel(),
        )
    )

    return points_xz, wrist_xz, angles_j234, excluded_by_j4_count


def calculate_3d_workspace(vertical_points_xz: np.ndarray, vertical_angles_j234: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:



    if vertical_points_xz.size == 0:
        return np.empty((0, 3)), np.empty((0, 4))

    j1_values = make_angle_samples(J1_MIN_DEG, J1_MAX_DEG, J1_STEP_DEG)

    planar_x = vertical_points_xz[:, 0]
    planar_z = vertical_points_xz[:, 1]

    xyz_list = []
    angle_list = []
    for j1_deg in j1_values:
        yaw_rad = math.radians(j1_deg - 90.0)
        world_x = planar_x * math.cos(yaw_rad)
        world_y = planar_x * math.sin(yaw_rad)
        world_z = planar_z + J2_HEIGHT_FROM_FLOOR_CM

        xyz_list.append(np.column_stack((world_x, world_y, world_z)))
        j1_column = np.full((vertical_angles_j234.shape[0], 1), j1_deg)
        angle_list.append(np.column_stack((j1_column, vertical_angles_j234)))

    return np.vstack(xyz_list), np.vstack(angle_list)


def calculate_workspace_data() -> WorkspaceData:

    all_points_xz, all_angles_j234 = calculate_all_reachable_workspace()
    vertical_points_xz, vertical_wrist_xz, vertical_angles_j234, excluded_by_j4_count = (
        calculate_vertical_suction_workspace()
    )
    workspace_3d_xyz, workspace_3d_angles_j1234 = calculate_3d_workspace(vertical_points_xz, vertical_angles_j234)

    return WorkspaceData(
        all_points_xz=all_points_xz,
        all_angles_j234=all_angles_j234,
        vertical_points_xz=vertical_points_xz,
        vertical_wrist_xz=vertical_wrist_xz,
        vertical_angles_j234=vertical_angles_j234,
        excluded_by_j4_count=excluded_by_j4_count,
        workspace_3d_xyz=workspace_3d_xyz,
        workspace_3d_angles_j1234=workspace_3d_angles_j1234,
    )


def get_workspace_data() -> WorkspaceData:
    global WORKSPACE_DATA
    if WORKSPACE_DATA is None:
        WORKSPACE_DATA = calculate_workspace_data()
    return WORKSPACE_DATA


def check_target_reachable(target_x_cm: float, target_z_cm: float, tolerance_cm: float = 0.5) -> bool:



    data = get_workspace_data()
    if data.vertical_points_xz.size == 0:
        print("수직 흡착 가능한 점이 없습니다. 각도 제한 또는 sign 값을 확인하세요.")
        return False

    target = np.array([target_x_cm, target_z_cm], dtype=float)
    distances = np.linalg.norm(data.vertical_points_xz - target, axis=1)
    best_index = int(np.argmin(distances))
    best_distance = float(distances[best_index])
    best_point = data.vertical_points_xz[best_index]
    best_angles = data.vertical_angles_j234[best_index]

    reachable = best_distance <= tolerance_cm
    print("\n[2D 목표점 확인]")
    print(f"목표점: x={target_x_cm:.2f} cm, z={target_z_cm:.2f} cm")
    print(f"가장 가까운 가능 점: x={best_point[0]:.2f} cm, z={best_point[1]:.2f} cm")
    print(f"거리 오차: {best_distance:.2f} cm (허용 오차 {tolerance_cm:.2f} cm)")
    print(f"각도 후보: J2={best_angles[0]:.1f}°, J3={best_angles[1]:.1f}°, J4={best_angles[2]:.1f}°")

    if reachable:
        print("판정: 도달 가능")
    else:
        print(f"판정: 도달 불가, 허용 오차보다 {best_distance - tolerance_cm:.2f} cm 더 멉니다.")

    return reachable


def check_target_3d_reachable(
    target_x_cm: float,
    target_y_cm: float,
    target_z_cm: float,
    tolerance_cm: float = 0.5,
) -> bool:



    target_radius_cm = math.hypot(target_x_cm, target_y_cm)
    target_yaw_deg = math.degrees(math.atan2(target_y_cm, target_x_cm))
    required_j1_deg = target_yaw_deg + 90.0

    print("\n[3D 목표점 확인]")
    print(f"목표점: X={target_x_cm:.2f} cm, Y={target_y_cm:.2f} cm, Z={target_z_cm:.2f} cm")
    print(f"계산된 J1: {required_j1_deg:.1f}°")

    if required_j1_deg < J1_MIN_DEG or required_j1_deg > J1_MAX_DEG:
        print(f"판정: 도달 불가, J1 각도 제한({J1_MIN_DEG:.0f}°~{J1_MAX_DEG:.0f}°) 밖입니다.")
        return False

    data = get_workspace_data()
    if data.vertical_points_xz.size == 0:
        print("수직 흡착 가능한 2D 점이 없습니다. 각도 제한 또는 sign 값을 확인하세요.")
        return False

    target_2d = np.array([target_radius_cm, target_z_cm - J2_HEIGHT_FROM_FLOOR_CM], dtype=float)
    distances = np.linalg.norm(data.vertical_points_xz - target_2d, axis=1)
    best_index = int(np.argmin(distances))
    best_distance = float(distances[best_index])
    best_point = data.vertical_points_xz[best_index]
    best_angles_j234 = data.vertical_angles_j234[best_index]

    reachable = best_distance <= tolerance_cm
    print(f"수평 거리 r: {target_radius_cm:.2f} cm")
    print(f"가장 가까운 2D 가능 점: r={best_point[0]:.2f} cm, z={best_point[1] + J2_HEIGHT_FROM_FLOOR_CM:.2f} cm")
    print(f"거리 오차: {best_distance:.2f} cm (허용 오차 {tolerance_cm:.2f} cm)")
    print(
        "각도 후보: "
        f"J1={required_j1_deg:.1f}°, "
        f"J2={best_angles_j234[0]:.1f}°, "
        f"J3={best_angles_j234[1]:.1f}°, "
        f"J4={best_angles_j234[2]:.1f}°"
    )

    if reachable:
        print("판정: 도달 가능")
    else:
        print(f"판정: 도달 불가, 허용 오차보다 {best_distance - tolerance_cm:.2f} cm 더 멉니다.")

    return reachable


def choose_representative_pose_indices(data: WorkspaceData, count: int = 5) -> np.ndarray:

    if data.vertical_points_xz.shape[0] == 0:
        return np.array([], dtype=int)

    sorted_indices = np.argsort(data.vertical_points_xz[:, 0])
    if sorted_indices.size <= count:
        return sorted_indices

    pick_positions = np.linspace(0, sorted_indices.size - 1, count).round().astype(int)
    return sorted_indices[pick_positions]


def plot_pose_examples(data: WorkspaceData) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 7))
    indices = choose_representative_pose_indices(data, count=6)

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(indices), 1)))

    for color, idx in zip(colors, indices):
        j2_deg, j3_deg, j4_deg = data.vertical_angles_j234[idx]
        fk = forward_kinematics_2d(j2_deg, j3_deg, j4_deg, GRIPPER_EFFECTIVE_CM)

        j2 = np.array([0.0, 0.0])
        j3 = np.array([fk["j3_x"], fk["j3_z"]], dtype=float)
        j4 = np.array([fk["j4_x"], fk["j4_z"]], dtype=float)
        tip = np.array([j4[0], j4[1] - GRIPPER_EFFECTIVE_CM], dtype=float)


        ax.plot([j2[0], j3[0], j4[0]], [j2[1], j3[1], j4[1]], "-o", color=color, linewidth=2.0, markersize=4)
        ax.plot([j4[0], tip[0]], [j4[1], tip[1]], "-", color=color, linewidth=3.0)
        ax.scatter([tip[0]], [tip[1]], c=[color], s=30, marker="s")

        ax.text(j2[0], j2[1], " J2", fontsize=9)
        ax.text(j3[0], j3[1], " J3", fontsize=9)
        ax.text(j4[0], j4[1], " J4", fontsize=9)

    ax.scatter([0], [0], s=55, c="red", marker="x")
    ax.set_title("수직 흡착 자세 예시")
    set_axes_equal_2d(ax)
    save_figure(fig, "D_vertical_suction_pose_examples.png")
    return fig


def set_projection_axes(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_title(title, pad=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.axhline(0, color="0.30", linewidth=0.8, alpha=0.45)
    ax.axvline(0, color="0.30", linewidth=0.8, alpha=0.45)


def add_floor_line_if_needed(ax: plt.Axes, ylabel: str) -> None:
    if ylabel.startswith("Z"):
        ax.axhline(0, color="black", linewidth=1.4, alpha=0.7)
        xmin, xmax = ax.get_xlim()
        ax.text(xmin + (xmax - xmin) * 0.02, 0, " floor", color="black", fontsize=9, va="bottom")


def draw_projection_view(
    ax: plt.Axes,
    xy: np.ndarray,
    color_values: np.ndarray,
    origin_xy: Tuple[float, float],
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    if xy.size > 0:
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=2.0,
            c=color_values,
            cmap="viridis",
            alpha=0.62,
            edgecolors="none",
        )

    ax.scatter([origin_xy[0]], [origin_xy[1]], s=55, c="red", marker="x")
    ax.text(origin_xy[0], origin_xy[1], " J2", color="red", fontsize=9)
    set_projection_axes(ax, xlabel, ylabel, title)
    add_floor_line_if_needed(ax, ylabel)


def create_four_view_figure(data: WorkspaceData) -> plt.Figure:
    xyz = data.workspace_3d_xyz
    if xyz.size == 0:
        empty = np.empty((0, 2))
        color_values = np.array([])
        side_xy = empty
        top_xy = empty
        front_xy = empty
    else:
        color_values = xyz[:, 2]
        side_xy = xyz[:, [0, 2]]
        top_xy = xyz[:, [0, 1]]
        front_xy = xyz[:, [1, 2]]

    fig = plt.figure(figsize=(15.5, 10.5), constrained_layout=True)
    ax_3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax_side = fig.add_subplot(2, 2, 2)
    ax_top = fig.add_subplot(2, 2, 3)
    ax_front = fig.add_subplot(2, 2, 4)

    if xyz.size > 0:
        ax_3d.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            s=0.8,
            c=xyz[:, 2],
            cmap="viridis",
            alpha=0.42,
        )

    ax_3d.scatter([0], [0], [J2_HEIGHT_FROM_FLOOR_CM], s=45, c="red", marker="x")
    ax_3d.text(0, 0, J2_HEIGHT_FROM_FLOOR_CM, " J2", color="red", fontsize=9)
    ax_3d.set_title("3D 45도 보기", pad=8)
    ax_3d.set_xlabel("X [cm]")
    ax_3d.set_ylabel("Y [cm]")
    ax_3d.set_zlabel("Z [cm]")
    ax_3d.view_init(elev=28, azim=-45)
    set_axes_equal_3d(ax_3d, xyz)

    draw_projection_view(
        ax_side,
        side_xy,
        color_values,
        (0.0, J2_HEIGHT_FROM_FLOOR_CM),
        "X [cm]",
        "Z [cm]",
        "옆면 보기 (X-Z)",
    )
    draw_projection_view(
        ax_top,
        top_xy,
        color_values,
        (0.0, 0.0),
        "X [cm]",
        "Y [cm]",
        "윗면 보기 (X-Y)",
    )
    draw_projection_view(
        ax_front,
        front_xy,
        color_values,
        (0.0, J2_HEIGHT_FROM_FLOOR_CM),
        "Y [cm]",
        "Z [cm]",
        "정면 보기 (Y-Z)",
    )
    fig.suptitle("수직 흡착 가능 작업공간 4분할 보기", fontsize=16)
    return fig


def save_workspace_4view_html(data: WorkspaceData) -> str:
    os.makedirs(IMAGE_DIR, exist_ok=True)
    html_path = os.path.join(IMAGE_DIR, "로봇팔_작업영역_4면도.html")
    fig = create_four_view_figure(data)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("ascii")

    html = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>로봇팔 작업공간 4분할 보기</title>
<style>
html, body {{
    margin: 0;
    min-height: 100%;
    background: #f4f6f8;
    color: #111827;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
main {{
    max-width: 1500px;
    margin: 0 auto;
    padding: 18px;
}}
h1 {{
    margin: 0 0 12px;
    font-size: 22px;
    font-weight: 700;
}}
img {{
    display: block;
    width: 100%;
    height: auto;
    background: white;
    border: 1px solid #d7dde6;
    border-radius: 8px;
}}
</style>
</head>
<body>
<main>
<h1>로봇팔 수직 흡착 가능 작업공간</h1>
<img src="data:image/png;base64,{image_base64}" alt="3D 45도 보기, 옆면, 윗면, 정면 작업공간">
</main>
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    return html_path


def save_joint_controller_html() -> str:
    os.makedirs(IMAGE_DIR, exist_ok=True)
    html_path = os.path.join(IMAGE_DIR, "로봇팔_관절_제어기.html")
    data = get_workspace_data()
    workspace_xyz = data.workspace_3d_xyz
    max_points = 7000
    if workspace_xyz.shape[0] > max_points:
        sample_indices = np.linspace(0, workspace_xyz.shape[0] - 1, max_points).astype(int)
        workspace_xyz = workspace_xyz[sample_indices]
    config = {
        "link1": LINK1_CM,
        "link2": LINK2_CM,
        "gripper": GRIPPER_EFFECTIVE_CM,
        "j2Height": J2_HEIGHT_FROM_FLOOR_CM,
        "baseRadius": BASE_RADIUS_CM,
        "baseHeight": BASE_HEIGHT_CM,
        "targetGripperAngle": TARGET_GRIPPER_ANGLE_DEG,
        "j1Min": J1_MIN_DEG,
        "j1Max": J1_MAX_DEG,
        "j2Min": J2_MIN_DEG,
        "j2Max": J2_MAX_DEG,
        "j3Min": J3_MIN_DEG,
        "j3Max": J3_MAX_DEG,
        "j4Min": J4_MIN_DEG,
        "j4Max": J4_MAX_DEG,
        "j2Sign": J2_SIGN,
        "j3Sign": J3_SIGN,
        "j4Sign": J4_SIGN,
        "workspaceXYZ": np.round(workspace_xyz, 3).tolist(),
    }
    config_json = json.dumps(config, ensure_ascii=False, separators=(",", ":"))
    html = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>로봇팔 조인트 조작</title>
<style>
html, body {
    margin: 0;
    min-height: 100%;
    background: #f4f6f8;
    color: #111827;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
main {
    max-width: 1500px;
    margin: 0 auto;
    padding: 16px;
}
h1 {
    margin: 0 0 12px;
    font-size: 22px;
}
.layout {
    display: grid;
    grid-template-columns: 360px minmax(900px, 1fr);
    gap: 14px;
    align-items: start;
}
.panel {
    background: white;
    border: 1px solid #d7dde6;
    border-radius: 8px;
    padding: 14px;
}
.joint {
    display: grid;
    grid-template-columns: 34px 30px 1fr 30px 64px 72px;
    gap: 8px;
    align-items: center;
    margin: 12px 0;
}
.joint label {
    font-weight: 700;
}
button {
    height: 30px;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    background: #ffffff;
    cursor: pointer;
}
input[type="range"] {
    width: 100%;
}
.value {
    text-align: right;
    font-variant-numeric: tabular-nums;
}
.angle-input {
    width: 58px;
    height: 28px;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 0 6px;
    font-variant-numeric: tabular-nums;
}
.range-label {
    grid-column: 2 / -1;
    color: #6b7280;
    font-size: 12px;
    margin-top: -6px;
}
.status {
    display: grid;
    gap: 7px;
    margin-top: 14px;
    font-size: 14px;
    line-height: 1.35;
}
.ok {
    color: #047857;
    font-weight: 700;
}
.bad {
    color: #b91c1c;
    font-weight: 700;
}
canvas {
    display: block;
    width: 100%;
    height: 430px;
    background: white;
    border: 1px solid #d7dde6;
    border-radius: 8px;
}
@media (max-width: 980px) {
    .layout {
        grid-template-columns: 1fr;
    }
    canvas {
        height: 980px;
    }
}
</style>
</head>
<body>
<main>
<h1>로봇팔 조인트 조작</h1>
<div class="layout">
    <section class="panel">
        <label><input id="autoJ4" type="checkbox" checked> J4 자동 수직 보정</label>
        <br>
        <label><input id="lockValidPose" type="checkbox" checked> 유효 자세 잠금</label>
        <div id="controls"></div>
        <div class="status">
            <div id="poseStatus"></div>
            <div id="anglesStatus"></div>
            <div id="pointStatus"></div>
            <div id="floorDistanceStatus"></div>
            <div id="hintStatus"></div>
        </div>
    </section>
    <canvas id="canvas"></canvas>
</div>
</main>
<script>
const C = __CONFIG__;
const state = {
    J1: 90,
    J2: 45,
    J3: Math.min(90, Math.max(C.j3Min, 60)),
    J4: 90
};
const jointDefs = [
    ["J1", C.j1Min, C.j1Max],
    ["J2", C.j2Min, C.j2Max],
    ["J3", C.j3Min, C.j3Max],
    ["J4", C.j4Min, C.j4Max]
];
const controls = document.getElementById("controls");
const autoJ4 = document.getElementById("autoJ4");
const lockValidPose = document.getElementById("lockValidPose");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
let lastValidState = {...state};

for (const [name, min, max] of jointDefs) {
    const row = document.createElement("div");
    row.className = "joint";
    row.innerHTML = `
        <label>${name}</label>
        <button data-joint="${name}" data-delta="-1">-</button>
        <input id="${name}" type="range" min="${min}" max="${max}" step="1" value="${state[name]}">
        <button data-joint="${name}" data-delta="1">+</button>
        <div class="value" id="${name}Value"></div>
        <input class="angle-input" id="${name}Number" type="number" min="${min}" max="${max}" step="1" value="${state[name]}">
        <div class="range-label">가동각 ${min}°-${max}°</div>
    `;
    controls.appendChild(row);
}

for (const input of controls.querySelectorAll("input")) {
    input.addEventListener("input", () => {
        const joint = input.id.replace("Number", "");
        state[joint] = Number(input.value);
        update(true);
    });
}

for (const button of controls.querySelectorAll("button")) {
    button.addEventListener("click", () => {
        const joint = button.dataset.joint;
        const input = document.getElementById(joint);
        const next = Math.max(Number(input.min), Math.min(Number(input.max), Number(input.value) + Number(button.dataset.delta)));
        input.value = next;
        state[joint] = next;
        update(true);
    });
}

autoJ4.addEventListener("change", () => update(true));
lockValidPose.addEventListener("change", () => update(false));

function rad(deg) {
    return deg * Math.PI / 180;
}

function unit(deg) {
    return [Math.cos(rad(deg)), Math.sin(rad(deg))];
}

function normalizeAngle(deg) {
    let v = ((deg + 180) % 360 + 360) % 360 - 180;
    return v;
}

function clamp(v, min, max) {
    return Math.max(min, Math.min(max, v));
}

function modelAngles(j2, j3, j4) {
    const theta1 = 90 + C.j2Sign * (j2 - 90);
    const theta2 = theta1 - C.j3Sign * (180 - j3);
    const theta3 = theta2 - C.j4Sign * (90 - j4);
    return [theta1, theta2, theta3];
}

function requiredJ4(theta2) {
    return 90 - (theta2 - C.targetGripperAngle) / C.j4Sign;
}

function computePose() {
    let j4 = state.J4;
    const rough = modelAngles(state.J2, state.J3, 90);
    const req = requiredJ4(rough[1]);
    if (autoJ4.checked) {
        j4 = Math.round(clamp(req, C.j4Min, C.j4Max));
        state.J4 = j4;
        document.getElementById("J4").value = j4;
    }
    const [theta1, theta2, theta3] = modelAngles(state.J2, state.J3, j4);
    const u1 = unit(theta1);
    const u2 = unit(theta2);
    const u3 = unit(theta3);
    const p0 = {x: 0, z: C.j2Height};
    const p1 = {x: C.link1 * u1[0], z: C.j2Height + C.link1 * u1[1]};
    const p2 = {x: p1.x + C.link2 * u2[0], z: p1.z + C.link2 * u2[1]};
    const p3 = {x: p2.x + C.gripper * u3[0], z: p2.z + C.gripper * u3[1]};
    const yaw = rad(state.J1 - 90);
    const toWorld = (p) => ({x: p.x * Math.cos(yaw), y: p.x * Math.sin(yaw), z: p.z});
    const w0 = {x: 0, y: 0, z: C.j2Height};
    const w1 = toWorld(p1);
    const w2 = toWorld(p2);
    const w3 = toWorld(p3);
    const verticalError = Math.abs(normalizeAngle(theta3 - C.targetGripperAngle));
    return {j4, req, theta1, theta2, theta3, p0, p1, p2, p3, w0, w1, w2, w3, verticalError};
}

function poseIsValid(pose) {
    const j4InRange = pose.req >= C.j4Min && pose.req <= C.j4Max;
    const floorOk = pose.p3.z >= 0;
    const verticalOk = pose.verticalError <= 0.75;
    const baseOk = !poseHitsBase(pose);
    return j4InRange && floorOk && verticalOk && baseOk;
}

function segmentHitsBase2D(a, b, startT = 0, steps = 16) {
    for (let i = 0; i <= steps; i++) {
        const t = startT + (1 - startT) * i / steps;
        const x = a.x + (b.x - a.x) * t;
        const z = a.z + (b.z - a.z) * t;
        if (Math.abs(x) <= C.baseRadius && z >= 0 && z <= C.baseHeight) return true;
    }
    return false;
}

function poseHitsBase(pose) {
    return (
        segmentHitsBase2D(pose.p0, pose.p1, 0.20) ||
        segmentHitsBase2D(pose.p1, pose.p2) ||
        segmentHitsBase2D(pose.p2, pose.p3)
    );
}

function syncInputs() {
    for (const [name] of jointDefs) {
        const input = document.getElementById(name);
        const numberInput = document.getElementById(name + "Number");
        input.value = state[name];
        numberInput.value = state[name];
        document.getElementById(name + "Value").textContent = `${state[name].toFixed(0)}°`;
    }
    const j4Input = document.getElementById("J4");
    j4Input.disabled = autoJ4.checked;
}

function projectIso(p) {
    const yaw = rad(-45);
    const pitch = rad(25);
    const x1 = p.x * Math.cos(yaw) - p.y * Math.sin(yaw);
    const y1 = p.x * Math.sin(yaw) + p.y * Math.cos(yaw);
    const z1 = p.z;
    const y2 = y1 * Math.cos(pitch) - z1 * Math.sin(pitch);
    return {x: x1, y: y2};
}

function panelTransform(panel, bounds, pad = 34) {
    const minX = bounds.minX;
    const maxX = bounds.maxX;
    const minY = bounds.minY;
    const maxY = bounds.maxY;
    const scale = Math.min((panel.w - pad * 2) / (maxX - minX), (panel.h - pad * 2) / (maxY - minY));
    return (p) => ({
        x: panel.x + pad + (p.x - minX) * scale,
        y: panel.y + panel.h - pad - (p.y - minY) * scale
    });
}

function drawGrid(panel, title, xlabel, ylabel, bounds, floor) {
    ctx.save();
    ctx.fillStyle = "#ffffff";
    ctx.strokeStyle = "#d7dde6";
    ctx.lineWidth = 1;
    ctx.fillRect(panel.x, panel.y, panel.w, panel.h);
    ctx.strokeRect(panel.x, panel.y, panel.w, panel.h);
    ctx.fillStyle = "#111827";
    ctx.font = "16px system-ui";
    ctx.textAlign = "center";
    ctx.fillText(title, panel.x + panel.w / 2, panel.y + 22);
    ctx.font = "12px system-ui";
    ctx.fillText(xlabel, panel.x + panel.w / 2, panel.y + panel.h - 7);
    ctx.save();
    ctx.translate(panel.x + 12, panel.y + panel.h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(ylabel, 0, 0);
    ctx.restore();
    const map = panelTransform(panel, bounds);
    ctx.strokeStyle = "#e5e7eb";
    ctx.lineWidth = 1;
    for (let i = 1; i < 5; i++) {
        const gx = panel.x + panel.w * i / 5;
        const gy = panel.y + panel.h * i / 5;
        ctx.beginPath();
        ctx.moveTo(gx, panel.y + 34);
        ctx.lineTo(gx, panel.y + panel.h - 24);
        ctx.moveTo(panel.x + 28, gy);
        ctx.lineTo(panel.x + panel.w - 16, gy);
        ctx.stroke();
    }
    if (floor) {
        const a = map({x: bounds.minX, y: 0});
        const b = map({x: bounds.maxX, y: 0});
        ctx.strokeStyle = "#111827";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
    }
    ctx.restore();
    return map;
}

function clipPanel(panel) {
    ctx.beginPath();
    ctx.rect(panel.x + 2, panel.y + 2, panel.w - 4, panel.h - 4);
    ctx.clip();
}

function drawFixedBase(map, floor, topView) {
    if (floor) {
        const postBottom = map({x: 0, y: C.baseHeight});
        const postTop = map({x: 0, y: C.j2Height});
        ctx.save();
        ctx.strokeStyle = "#9ca3af";
        ctx.lineWidth = 12;
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.moveTo(postBottom.x, postBottom.y);
        ctx.lineTo(postTop.x, postTop.y);
        ctx.stroke();
        ctx.fillStyle = "#d1d5db";
        ctx.strokeStyle = "#6b7280";
        ctx.lineWidth = 1.5;
        const left = map({x: -C.baseRadius, y: 0});
        const right = map({x: C.baseRadius, y: 0});
        const top = map({x: C.baseRadius, y: C.baseHeight});
        ctx.fillRect(left.x, top.y, right.x - left.x, left.y - top.y);
        ctx.strokeRect(left.x, top.y, right.x - left.x, left.y - top.y);
        ctx.fillStyle = "#6b7280";
        ctx.font = "11px system-ui";
        ctx.fillText("base", left.x + 5, top.y + 14);
        ctx.restore();
    }
    if (topView) {
        const center = map({x: 0, y: 0});
        const edge = map({x: C.baseRadius, y: 0});
        ctx.save();
        ctx.strokeStyle = "#9ca3af";
        ctx.fillStyle = "rgba(209, 213, 219, 0.35)";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(center.x, center.y, Math.abs(edge.x - center.x), 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.restore();
    }
}

function drawWorkspaceBackground(map, viewName) {
    const points = C.workspaceXYZ || [];
    if (!points.length) return;
    ctx.save();
    ctx.fillStyle = "rgba(59, 130, 246, 0.115)";
    for (const p of points) {
        let q;
        if (viewName === "side") q = {x: p[0], y: p[2]};
        else if (viewName === "top") q = {x: p[0], y: p[1]};
        else q = {x: p[1], y: p[2]};
        const m = map(q);
        ctx.fillRect(m.x - 1, m.y - 1, 2, 2);
    }
    ctx.restore();
}

function workspaceBounds(viewName) {
    const points = C.workspaceXYZ || [];
    const projected = [];
    for (const p of points) {
        if (viewName === "side") projected.push({x: p[0], y: p[2]});
        else if (viewName === "top") projected.push({x: p[0], y: p[1]});
        else projected.push({x: p[1], y: p[2]});
    }
    if (viewName === "side") projected.push({x: 0, y: C.j2Height}, {x: 0, y: 0});
    else if (viewName === "top") projected.push({x: 0, y: 0});
    else projected.push({x: 0, y: C.j2Height}, {x: 0, y: 0});
    const xs = projected.map(p => p.x);
    const ys = projected.map(p => p.y);
    const minX0 = Math.min(...xs);
    const maxX0 = Math.max(...xs);
    const minY0 = Math.min(...ys);
    const maxY0 = Math.max(...ys);
    const spanX = Math.max(1, maxX0 - minX0);
    const spanY = Math.max(1, maxY0 - minY0);
    return {
        minX: minX0 - spanX * 0.10,
        maxX: maxX0 + spanX * 0.10,
        minY: minY0 - spanY * 0.16,
        maxY: maxY0 + spanY * 0.16
    };
}

function drawRobot(panel, title, xlabel, ylabel, pts, bounds, floor = false, topView = false, viewName = "side") {
    const map = drawGrid(panel, title, xlabel, ylabel, bounds, floor);
    const mapped = pts.map(map);
    ctx.save();
    clipPanel(panel);
    drawWorkspaceBackground(map, viewName);
    drawFixedBase(map, floor, topView);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#374151";
    ctx.lineWidth = 7;
    ctx.beginPath();
    ctx.moveTo(mapped[0].x, mapped[0].y);
    for (let i = 1; i < mapped.length - 1; i++) ctx.lineTo(mapped[i].x, mapped[i].y);
    ctx.stroke();
    ctx.strokeStyle = "#111827";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(mapped[mapped.length - 2].x, mapped[mapped.length - 2].y);
    ctx.lineTo(mapped[mapped.length - 1].x, mapped[mapped.length - 1].y);
    ctx.stroke();
    for (let i = 0; i < mapped.length; i++) {
        ctx.fillStyle = i === mapped.length - 1 ? "#2563eb" : "#ffffff";
        ctx.strokeStyle = i === 0 ? "#ef4444" : "#111827";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(mapped[i].x, mapped[i].y, i === mapped.length - 1 ? 5 : 7, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
    }
    const labels = ["J2", "J3", "J4", "Tip"];
    ctx.fillStyle = "#111827";
    ctx.font = "12px system-ui";
    for (let i = 0; i < mapped.length; i++) ctx.fillText(labels[i], mapped[i].x + 9, mapped[i].y - 8);
    ctx.restore();
}

function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.floor(rect.width * dpr);
    canvas.height = Math.floor(rect.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function draw() {
    resizeCanvas();
    const pose = computePose();
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    ctx.clearRect(0, 0, w, h);
    const gap = 14;
    const panelW = (w - gap * 2) / 3;
    const panelH = h;
    const panels = [
        {x: 0, y: 0, w: panelW, h: panelH},
        {x: panelW + gap, y: 0, w: panelW, h: panelH},
        {x: (panelW + gap) * 2, y: 0, w: panelW, h: panelH}
    ];
    const side = [
        {x: 0, y: C.j2Height},
        {x: pose.p1.x, y: pose.p1.z},
        {x: pose.p2.x, y: pose.p2.z},
        {x: pose.p3.x, y: pose.p3.z}
    ];
    const top = [
        {x: 0, y: 0},
        {x: pose.w1.x, y: pose.w1.y},
        {x: pose.w2.x, y: pose.w2.y},
        {x: pose.w3.x, y: pose.w3.y}
    ];
    const front = [
        {x: 0, y: C.j2Height},
        {x: pose.w1.y, y: pose.w1.z},
        {x: pose.w2.y, y: pose.w2.z},
        {x: pose.w3.y, y: pose.w3.z}
    ];
    const sideBounds = workspaceBounds("side");
    const topBounds = workspaceBounds("top");
    const frontBounds = workspaceBounds("front");
    drawRobot(panels[0], "옆면 보기 (X-Z)", "X [cm]", "Z [cm]", side, sideBounds, true, false, "side");
    drawRobot(panels[1], "윗면 보기 (X-Y)", "X [cm]", "Y [cm]", top, topBounds, false, true, "top");
    drawRobot(panels[2], "정면 보기 (Y-Z)", "Y [cm]", "Z [cm]", front, frontBounds, true, false, "front");
    for (const [name] of jointDefs) {
        document.getElementById(name + "Value").textContent = `${state[name].toFixed(0)}°`;
    }
    const j4InRange = pose.req >= C.j4Min && pose.req <= C.j4Max;
    const floorOk = pose.p3.z >= 0;
    const verticalOk = pose.verticalError <= 0.75;
    const baseOk = !poseHitsBase(pose);
    document.getElementById("poseStatus").innerHTML = (j4InRange && floorOk && verticalOk && baseOk)
        ? `<span class="ok">현재 자세: 수직 흡착 가능</span>`
        : `<span class="bad">현재 자세: 조건 확인 필요</span>`;
    document.getElementById("anglesStatus").textContent =
        `필요 J4=${pose.req.toFixed(1)}°, 실제 J4=${state.J4.toFixed(0)}°, 수직 오차=${pose.verticalError.toFixed(2)}°`;
    document.getElementById("pointStatus").textContent =
        `흡착점: X=${pose.w3.x.toFixed(2)} cm, Y=${pose.w3.y.toFixed(2)} cm, Z=${pose.w3.z.toFixed(2)} cm`;
    document.getElementById("floorDistanceStatus").textContent =
        `엔드이펙터 끝-바닥 거리: ${Math.max(0, pose.w3.z).toFixed(2)} cm`;
    document.getElementById("hintStatus").textContent =
        !floorOk ? "바닥 조건: 흡착점이 바닥 아래입니다"
        : !baseOk ? "베이스 조건: 원기둥과 충돌합니다"
        : "바닥/베이스 조건: 통과";
}

function update(fromUser) {
    if (lockValidPose.checked) {
        autoJ4.checked = true;
    }
    let pose = computePose();
    if (lockValidPose.checked && !poseIsValid(pose)) {
        Object.assign(state, lastValidState);
        pose = computePose();
    }
    if (poseIsValid(pose)) {
        lastValidState = {...state};
    }
    syncInputs();
    draw();
}

window.addEventListener("resize", draw);
update(false);
</script>
</body>
</html>
""".replace("__CONFIG__", config_json)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    return html_path


def print_range(label: str, values: np.ndarray) -> None:
    if values.size == 0:
        print(f"{label}: 데이터 없음")
    else:
        print(f"{label}: {values.min():.2f} ~ {values.max():.2f} cm")


def print_workspace_summary(data: WorkspaceData) -> None:
    print("\n========== 로봇팔 작업공간 계산 결과 ==========")
    print(f"각도 샘플링 간격: J2/J3/J4={ANGLE_STEP_DEG:.1f}°, J1={J1_STEP_DEG:.1f}°")
    print(f"전체 도달 가능 점 개수: {data.all_points_xz.shape[0]:,}")
    print(f"수직 흡착 가능 점 개수: {data.vertical_points_xz.shape[0]:,}")
    print(f"J4 각도 범위 밖이라 제외된 자세 개수: {data.excluded_by_j4_count:,}")
    print(f"J2 바닥 기준 높이: {J2_HEIGHT_FROM_FLOOR_CM:.3f} cm")
    print(f"베이스 원판 반지름/높이: {BASE_RADIUS_CM:.2f} cm / {BASE_HEIGHT_CM:.2f} cm")

    print_range("수직 흡착 가능 영역 x", data.vertical_points_xz[:, 0] if data.vertical_points_xz.size else np.array([]))
    print_range("수직 흡착 가능 영역 z(J2 기준)", data.vertical_points_xz[:, 1] if data.vertical_points_xz.size else np.array([]))

    if data.workspace_3d_xyz.size > 0:
        print_range("3D 작업공간 X", data.workspace_3d_xyz[:, 0])
        print_range("3D 작업공간 Y", data.workspace_3d_xyz[:, 1])
        print_range("3D 작업공간 Z", data.workspace_3d_xyz[:, 2])
    else:
        print("3D 작업공간: 데이터 없음")

    print(f"결과 저장 폴더: {os.path.abspath(IMAGE_DIR)}")
    print("==============================================\n")


def main() -> None:
    global WORKSPACE_DATA

    setup_korean_font()
    WORKSPACE_DATA = calculate_workspace_data()
    print_workspace_summary(WORKSPACE_DATA)

    html_path = save_workspace_4view_html(WORKSPACE_DATA)
    print(f"4분할 HTML: {html_path}")
    controller_html_path = save_joint_controller_html()
    print(f"조인트 조작 HTML: {controller_html_path}")





    if "--no-show" in sys.argv:
        plt.close("all")
    elif GUI_BACKEND_AVAILABLE:
        plt.show()
    else:
        print("그래프 창을 열 수 없어 PNG 저장만 완료했습니다.")
        if GUI_BACKEND_ERROR:
            print(f"GUI 백엔드 오류: {GUI_BACKEND_ERROR}")
        print("창 표시가 필요하면 Python의 Tk/Tcl 설치를 확인하거나, VSCode에서 Matplotlib 백엔드를 설정하세요.")
        plt.close("all")


if __name__ == "__main__":
    main()
