from __future__ import annotations

from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sorting_robot.vision_display import depth_to_colormap
from sorting_robot.vision_geometry import (
    align_quad_to_reference,
    candidate_slot_key,
    inset_quad,
    intersect_parametric_lines,
    is_rectangular_quad,
    order_quad_points,
    quad_mask_roi,
    quad_iou,
)


def legacy_depth_to_colormap(depth_raw, depth_scale, depth_min_m, depth_max_m):
    depth_m = depth_raw.astype(np.float32) * depth_scale
    depth_m = np.clip(depth_m, depth_min_m, depth_max_m)
    scale = max(depth_max_m - depth_min_m, 1e-6)
    depth_norm = (depth_m - depth_min_m) / scale
    depth_u8 = (255.0 * (1.0 - depth_norm)).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    depth_color[depth_raw == 0] = (0, 0, 0)
    return depth_color


class DepthDisplayTests(unittest.TestCase):
    def test_colormap_is_pixel_identical_to_previous_calculation(self):
        rng = np.random.default_rng(20260903)
        depth = rng.integers(0, 3000, size=(48, 64), dtype=np.uint16)
        depth[::7, ::5] = 0

        expected = legacy_depth_to_colormap(depth, 0.001, 0.2, 2.5)
        actual = depth_to_colormap(depth, 0.001, 0.2, 2.5)

        np.testing.assert_array_equal(expected, actual)


class VisionGeometryTests(unittest.TestCase):
    def setUp(self):
        self.square = np.array(
            [[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]],
            dtype=np.float32,
        )

    def test_quad_order_alignment_and_shape_checks(self):
        shuffled = self.square[[2, 0, 3, 1]]
        ordered = order_quad_points(shuffled)
        aligned = align_quad_to_reference(ordered[::-1], self.square)

        np.testing.assert_array_equal(self.square, aligned)
        self.assertTrue(is_rectangular_quad(ordered))

    def test_quad_inset_iou_and_candidate_center(self):
        inset = inset_quad(self.square, 0.10)
        np.testing.assert_allclose(np.mean(inset, axis=0), (20.0, 20.0))
        self.assertAlmostEqual(1.0, quad_iou(self.square, self.square))
        self.assertEqual((20.0, 20.0), candidate_slot_key({"quad": self.square}))

    def test_line_intersection_and_parallel_case(self):
        origin = np.array([0.0, 0.0], dtype=np.float32)
        horizontal = np.array([1.0, 0.0], dtype=np.float32)
        upper = np.array([2.0, 3.0], dtype=np.float32)
        vertical = np.array([0.0, -1.0], dtype=np.float32)

        np.testing.assert_array_equal(
            np.array([2.0, 0.0], dtype=np.float32),
            intersect_parametric_lines(origin, horizontal, upper, vertical),
        )
        self.assertIsNone(
            intersect_parametric_lines(origin, horizontal, upper, horizontal)
        )

    def test_cropped_quad_mask_matches_full_frame_mask(self):
        image_shape = (80, 120)
        quad = np.array(
            [[-4.0, 12.0], [45.0, 9.0], [48.0, 50.0], [2.0, 55.0]],
            dtype=np.float32,
        )
        full_quad = np.round(quad).astype(np.int32)
        full_quad[:, 0] = np.clip(full_quad[:, 0], 0, image_shape[1] - 1)
        full_quad[:, 1] = np.clip(full_quad[:, 1], 0, image_shape[0] - 1)
        expected = np.zeros(image_shape, dtype=np.uint8)
        cv2.fillConvexPoly(expected, full_quad, 255)

        slices, cropped = quad_mask_roi(image_shape, quad, margin=8)
        actual = np.zeros(image_shape, dtype=np.uint8)
        actual[slices] = cropped

        np.testing.assert_array_equal(expected, actual)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        expected_dilated = cv2.dilate(expected, kernel, iterations=1)
        actual_dilated = np.zeros(image_shape, dtype=np.uint8)
        actual_dilated[slices] = cv2.dilate(cropped, kernel, iterations=1)
        np.testing.assert_array_equal(expected_dilated, actual_dilated)


if __name__ == "__main__":
    unittest.main()
