from __future__ import annotations

import unittest

import numpy as np

from sorting_robot.session_statistics import (
    REMOVED,
    STILL_PRESENT,
    UNCERTAIN,
    PickTarget,
    SceneObject,
    SessionStatistics,
    TargetOutcome,
    VerificationSummary,
    render_statistics_panel,
    verify_pick_targets,
)


class PickVerificationTests(unittest.TestCase):
    def test_absent_target_is_confirmed_removed(self):
        target = PickTarget(0.0, 50.0, 30, "left")
        result = verify_pick_targets(
            [SceneObject(0.0, 50.0, 30)],
            [],
            [target],
        )
        self.assertEqual(REMOVED, result.outcomes[0].status)

    def test_target_still_at_original_position_is_failed_pick(self):
        target = PickTarget(0.0, 50.0, 30, "left")
        result = verify_pick_targets(
            [SceneObject(0.0, 50.0, 30)],
            [SceneObject(1.5, 49.0, 30)],
            [target],
        )
        self.assertEqual(STILL_PRESENT, result.outcomes[0].status)

    def test_moved_same_size_target_is_uncertain_not_success(self):
        target = PickTarget(0.0, 50.0, 30, "left")
        result = verify_pick_targets(
            [SceneObject(0.0, 50.0, 30)],
            [SceneObject(45.0, 50.0, 30)],
            [target],
        )
        self.assertEqual(UNCERTAIN, result.outcomes[0].status)
        self.assertEqual(0, result.newly_visible)

    def test_newly_revealed_object_does_not_replace_success_count(self):
        target = PickTarget(0.0, 50.0, 30, "left")
        result = verify_pick_targets(
            [SceneObject(0.0, 50.0, 30)],
            [SceneObject(100.0, -70.0, 50)],
            [target],
        )
        self.assertEqual(REMOVED, result.outcomes[0].status)
        self.assertEqual(1, result.newly_visible)

    def test_simultaneous_pick_can_report_one_success_and_one_failure(self):
        left = PickTarget(0.0, 55.0, 30, "left")
        right = PickTarget(0.0, -55.0, 50, "right")
        result = verify_pick_targets(
            [SceneObject(0.0, 55.0, 30), SceneObject(0.0, -55.0, 50)],
            [SceneObject(1.0, 54.0, 30)],
            [left, right],
        )
        self.assertEqual(
            [STILL_PRESENT, REMOVED],
            [outcome.status for outcome in result.outcomes],
        )


class SessionStatisticsTests(unittest.TestCase):
    def test_holding_pick_counts_only_after_placement_completion(self):
        stats = SessionStatistics()
        target = PickTarget(0.0, 55.0, 30, "left")
        removed = VerificationSummary((TargetOutcome(target, REMOVED),))

        stats.begin_attempt([target], sequential=True)
        stats.record_holding_pick(removed)
        self.assertEqual(0, stats.successful)
        self.assertIsNotNone(stats.pending_holding)

        stats.complete_holding()
        self.assertEqual(1, stats.successful)
        self.assertEqual(1, stats.successful_by_size[30])
        self.assertEqual(1, stats.successful_by_arm["left"])

    def test_failed_retry_never_inflates_success(self):
        stats = SessionStatistics()
        target = PickTarget(0.0, 55.0, 30, "left")
        failed = VerificationSummary((TargetOutcome(target, STILL_PRESENT),))

        stats.begin_attempt([target], sequential=False)
        stats.record_completed_pick(failed)
        stats.begin_attempt([target], sequential=False)
        stats.record_completed_pick(failed)

        self.assertEqual(0, stats.successful)
        self.assertEqual(2, stats.pickup_failures)
        self.assertEqual(1, stats.retries)

    def test_statistics_panel_is_valid_nonempty_image(self):
        panel = render_statistics_panel(SessionStatistics().snapshot(), font_path=None)
        self.assertEqual((610, 500, 3), panel.shape)
        self.assertEqual(np.uint8, panel.dtype)
        self.assertGreater(int(np.max(panel)), 38)


if __name__ == "__main__":
    unittest.main()
