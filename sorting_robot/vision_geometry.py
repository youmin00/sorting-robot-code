"""물체 사각형의 정렬, 비교, 교차 계산에 쓰는 순수 함수."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def quad_mask_roi(
    image_shape: tuple[int, ...],
    quad_xy: np.ndarray,
    margin: int = 0,
) -> tuple[tuple[slice, slice], np.ndarray]:
    """사각형 주변의 작은 영역과 마스크만 만들어 전체 프레임 할당을 피한다."""
    height, width = int(image_shape[0]), int(image_shape[1])
    quad_i = np.round(quad_xy).astype(np.int32)
    quad_i[:, 0] = np.clip(quad_i[:, 0], 0, width - 1)
    quad_i[:, 1] = np.clip(quad_i[:, 1], 0, height - 1)
    margin = max(0, int(margin))

    x0 = max(int(np.min(quad_i[:, 0])) - margin, 0)
    y0 = max(int(np.min(quad_i[:, 1])) - margin, 0)
    x1 = min(int(np.max(quad_i[:, 0])) + margin + 1, width)
    y1 = min(int(np.max(quad_i[:, 1])) + margin + 1, height)

    local_quad = quad_i - np.array((x0, y0), dtype=np.int32)
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillConvexPoly(mask, local_quad, 255)
    return (slice(y0, y1), slice(x0, x1)), mask


def inset_quad(quad_xy: np.ndarray, inset_ratio: float) -> np.ndarray:
    ratio = float(np.clip(inset_ratio, 0.0, 0.45))
    if ratio <= 1e-6:
        return quad_xy.astype(np.float32)
    quad = quad_xy.astype(np.float32)
    center = np.mean(quad, axis=0, keepdims=True)
    return center + (quad - center) * (1.0 - ratio)


def intersect_parametric_lines(
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


def align_quad_to_reference(quad_xy: np.ndarray, ref_xy: np.ndarray) -> np.ndarray:
    q = quad_xy.astype(np.float32)
    r = ref_xy.astype(np.float32)
    candidates = [np.roll(q, shift=k, axis=0) for k in range(4)]
    q_rev = q[::-1].copy()
    candidates.extend(np.roll(q_rev, shift=k, axis=0) for k in range(4))

    best = candidates[0]
    best_err = float(np.mean(np.sum((best - r) ** 2, axis=1)))
    for candidate in candidates[1:]:
        error = float(np.mean(np.sum((candidate - r) ** 2, axis=1)))
        if error < best_err:
            best = candidate
            best_err = error
    return best


def quad_iou(quad_a: np.ndarray, quad_b: np.ndarray) -> float:
    qa = quad_a.astype(np.float32).reshape((-1, 1, 2))
    qb = quad_b.astype(np.float32).reshape((-1, 1, 2))
    area_a = float(abs(cv2.contourArea(qa)))
    area_b = float(abs(cv2.contourArea(qb)))
    if area_a <= 1e-6 or area_b <= 1e-6:
        return 0.0
    intersection_area, _ = cv2.intersectConvexConvex(qa, qb)
    if intersection_area <= 0.0:
        return 0.0
    union = area_a + area_b - float(intersection_area)
    if union <= 1e-6:
        return 0.0
    return float(intersection_area / union)


def candidate_slot_key(candidate) -> tuple[float, float]:
    quad = candidate["quad"].astype(np.float32)
    center = np.mean(quad, axis=0)
    return float(center[0]), float(center[1])


def order_quad_points(points: np.ndarray) -> np.ndarray:
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return points[np.argsort(angles)]


def is_rectangular_quad(quad: np.ndarray, max_cos: float = 0.35) -> bool:
    for index in range(4):
        previous = quad[(index - 1) % 4]
        current = quad[index]
        following = quad[(index + 1) % 4]
        vector1 = previous - current
        vector2 = following - current
        denominator = (np.linalg.norm(vector1) * np.linalg.norm(vector2)) + 1e-6
        cosine = abs(float(np.dot(vector1, vector2) / denominator))
        if cosine > max_cos:
            return False
    return True
