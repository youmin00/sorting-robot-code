"""카메라 깊이 프레임을 화면 표시용 이미지로 변환한다."""

from __future__ import annotations

import cv2
import numpy as np


def depth_to_colormap(
    depth_raw: np.ndarray,
    depth_scale: float,
    depth_min_m: float,
    depth_max_m: float,
) -> np.ndarray:
    """16비트 깊이값을 기존과 동일한 TURBO 컬러 이미지로 변환한다."""
    depth_m = depth_raw.astype(np.float32) * float(depth_scale)
    depth_m = np.clip(depth_m, float(depth_min_m), float(depth_max_m))

    scale = max(float(depth_max_m) - float(depth_min_m), 1e-6)
    depth_norm = (depth_m - float(depth_min_m)) / scale
    depth_u8 = (255.0 * (1.0 - depth_norm)).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    depth_color[depth_raw == 0] = (0, 0, 0)
    return depth_color
