"""Camera-verified sorting statistics and a lightweight status panel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - the program already supports this fallback
    Image = None
    ImageDraw = None
    ImageFont = None


REMOVED = "removed"
STILL_PRESENT = "still_present"
UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class SceneObject:
    x_mm: float
    y_mm: float
    size_mm: int


@dataclass(frozen=True)
class PickTarget:
    x_mm: float
    y_mm: float
    size_mm: int
    arm: str


@dataclass(frozen=True)
class TargetOutcome:
    target: PickTarget
    status: str


@dataclass(frozen=True)
class VerificationSummary:
    outcomes: tuple[TargetOutcome, ...]
    newly_visible: int = 0


def _nearest_cube_size(height_mm: float) -> int:
    return 30 if abs(float(height_mm) - 30.0) <= abs(float(height_mm) - 50.0) else 50


def scene_objects_from_dicts(objects: Iterable[dict]) -> tuple[list[SceneObject], bool]:
    """Convert camera results and report whether every result was usable."""
    converted: list[SceneObject] = []
    complete = True
    for obj in objects:
        if any(obj.get(key) is None for key in ("robot_x_mm", "robot_y_mm", "robot_z_mm")):
            complete = False
            continue
        converted.append(
            SceneObject(
                float(obj["robot_x_mm"]),
                float(obj["robot_y_mm"]),
                _nearest_cube_size(float(obj["robot_z_mm"])),
            )
        )
    return converted, complete


def targets_from_plans(*plans: Optional[dict]) -> list[PickTarget]:
    targets: list[PickTarget] = []
    for plan in plans:
        if plan is None:
            continue
        targets.append(
            PickTarget(
                float(plan["camera_x"]),
                float(plan["camera_y"]),
                int(plan["size"]),
                str(plan["arm"]),
            )
        )
    return targets


def _distance(a_x: float, a_y: float, b_x: float, b_y: float) -> float:
    return float(np.hypot(a_x - b_x, a_y - b_y))


def _reserve_non_targets(
    before: list[SceneObject],
    after: list[SceneObject],
    target_before_indices: set[int],
) -> set[int]:
    """Reserve post-scene objects belonging to objects that were not picked."""
    pairs: list[tuple[float, int, int]] = []
    for before_index, previous in enumerate(before):
        if before_index in target_before_indices:
            continue
        for after_index, current in enumerate(after):
            if previous.size_mm != current.size_mm:
                continue
            distance = _distance(previous.x_mm, previous.y_mm, current.x_mm, current.y_mm)
            if distance <= 25.0:
                pairs.append((distance, before_index, after_index))

    used_before: set[int] = set()
    used_after: set[int] = set()
    for _, before_index, after_index in sorted(pairs):
        if before_index in used_before or after_index in used_after:
            continue
        used_before.add(before_index)
        used_after.add(after_index)
    return used_after


def verify_pick_targets(
    before: Iterable[SceneObject],
    after: Iterable[SceneObject],
    targets: Iterable[PickTarget],
    *,
    scene_complete: bool = True,
) -> VerificationSummary:
    """Conservatively decide whether each commanded target left the work area.

    A visible target is a failed pick. A nearby unexplained object is uncertain,
    because it may be the original cube pushed by the tool. Only a target absent
    from a complete, stable scene is reported as removed.
    """
    before_list = list(before)
    after_list = list(after)
    target_list = list(targets)

    target_before_indices: set[int] = set()
    for target in target_list:
        candidates = [
            (
                _distance(target.x_mm, target.y_mm, obj.x_mm, obj.y_mm),
                index,
            )
            for index, obj in enumerate(before_list)
            if index not in target_before_indices and obj.size_mm == target.size_mm
        ]
        if candidates:
            distance, index = min(candidates)
            if distance <= max(20.0, target.size_mm * 0.5):
                target_before_indices.add(index)

    used_after = _reserve_non_targets(before_list, after_list, target_before_indices)
    outcomes: list[TargetOutcome] = []

    for target in target_list:
        available = [
            (index, obj) for index, obj in enumerate(after_list)
            if index not in used_after
        ]
        same_size = [
            (
                _distance(target.x_mm, target.y_mm, obj.x_mm, obj.y_mm),
                index,
            )
            for index, obj in available
            if obj.size_mm == target.size_mm
        ]
        strict_radius = max(15.0, target.size_mm * 0.40)
        strict = [item for item in same_size if item[0] <= strict_radius]
        if strict:
            _, index = min(strict)
            used_after.add(index)
            outcomes.append(TargetOutcome(target, STILL_PRESENT))
            continue

        nearby_any = [
            (
                _distance(target.x_mm, target.y_mm, obj.x_mm, obj.y_mm),
                index,
            )
            for index, obj in available
            if _distance(target.x_mm, target.y_mm, obj.x_mm, obj.y_mm)
            <= max(30.0, float(target.size_mm))
        ]
        moved_same_size = [
            item for item in same_size
            if item[0] <= float(target.size_mm + 30)
        ]
        if not scene_complete or nearby_any or moved_same_size:
            ambiguous = nearby_any or moved_same_size
            if ambiguous:
                _, index = min(ambiguous)
                used_after.add(index)
            outcomes.append(TargetOutcome(target, UNCERTAIN))
            continue

        outcomes.append(TargetOutcome(target, REMOVED))

    return VerificationSummary(
        tuple(outcomes),
        newly_visible=max(0, len(after_list) - len(used_after)),
    )


@dataclass
class SessionStatistics:
    current_detected: int = 0
    attempts: int = 0
    successful: int = 0
    pickup_failures: int = 0
    uncertain: int = 0
    retries: int = 0
    motion_errors: int = 0
    successful_by_size: dict[int, int] = field(
        default_factory=lambda: {30: 0, 50: 0}
    )
    successful_by_arm: dict[str, int] = field(
        default_factory=lambda: {"left": 0, "right": 0}
    )
    simultaneous_commands: int = 0
    sequential_commands: int = 0
    completed_cycles: int = 0
    total_cycle_seconds: float = 0.0
    last_result: str = "대기 중"
    last_newly_visible: int = 0
    pending_holding: Optional[TargetOutcome] = None
    _problem_targets: list[PickTarget] = field(default_factory=list, repr=False)

    def set_current_detected(self, count: int) -> None:
        self.current_detected = max(0, int(count))

    @staticmethod
    def _same_logical_target(a: PickTarget, b: PickTarget) -> bool:
        if a.size_mm != b.size_mm:
            return False
        return _distance(a.x_mm, a.y_mm, b.x_mm, b.y_mm) <= max(
            25.0, a.size_mm * 0.75
        )

    def begin_attempt(self, targets: Iterable[PickTarget], *, sequential: bool) -> None:
        target_list = list(targets)
        self.attempts += len(target_list)
        for target in target_list:
            if any(self._same_logical_target(target, old) for old in self._problem_targets):
                self.retries += 1
        if target_list:
            if sequential:
                self.sequential_commands += 1
            else:
                self.simultaneous_commands += 1
        self.last_result = f"집기 명령 전송 ({len(target_list)}개)"

    def _mark_problem(self, target: PickTarget) -> None:
        if not any(self._same_logical_target(target, old) for old in self._problem_targets):
            self._problem_targets.append(target)

    def _clear_problem(self, target: PickTarget) -> None:
        self._problem_targets = [
            old for old in self._problem_targets
            if not self._same_logical_target(target, old)
        ]

    def _commit_success(self, target: PickTarget) -> None:
        self.successful += 1
        self.successful_by_size[target.size_mm] = (
            self.successful_by_size.get(target.size_mm, 0) + 1
        )
        self.successful_by_arm[target.arm] = self.successful_by_arm.get(target.arm, 0) + 1
        self._clear_problem(target)

    def _record_non_success(self, outcome: TargetOutcome) -> None:
        if outcome.status == STILL_PRESENT:
            self.pickup_failures += 1
            self.last_result = "집기 실패: 대상이 작업영역에 남아 있음"
        else:
            self.uncertain += 1
            self.last_result = "확인 불가: 성공 개수에 포함하지 않음"
        self._mark_problem(outcome.target)

    def record_completed_pick(self, summary: VerificationSummary) -> None:
        success_count = 0
        for outcome in summary.outcomes:
            if outcome.status == REMOVED:
                self._commit_success(outcome.target)
                success_count += 1
            else:
                self._record_non_success(outcome)
        self.last_newly_visible = summary.newly_visible
        if success_count:
            self.last_result = f"카메라 확인 완료: {success_count}개 성공"

    def record_holding_pick(self, summary: VerificationSummary) -> None:
        self.pending_holding = None
        self.last_newly_visible = summary.newly_visible
        if not summary.outcomes:
            return
        outcome = summary.outcomes[0]
        if outcome.status == REMOVED:
            self.pending_holding = outcome
            self.last_result = "물체를 든 상태: 놓기 완료 대기"
        else:
            self._record_non_success(outcome)

    def complete_holding(self) -> None:
        if self.pending_holding is not None:
            self._commit_success(self.pending_holding.target)
            self.last_result = "놓기 완료 및 카메라 확인 성공"
        self.pending_holding = None

    def record_motion_error(self) -> None:
        self.motion_errors += 1
        self.last_result = "로봇 응답 오류"

    def finish_cycle(self, elapsed_seconds: float, successful_in_cycle: int) -> None:
        if successful_in_cycle <= 0:
            return
        self.completed_cycles += 1
        self.total_cycle_seconds += max(0.0, float(elapsed_seconds))

    def cancel_pending(self) -> None:
        self.pending_holding = None
        self.last_result = "대기 중"

    def snapshot(self) -> tuple:
        average = (
            self.total_cycle_seconds / self.completed_cycles
            if self.completed_cycles else 0.0
        )
        return (
            self.current_detected,
            self.attempts,
            self.successful,
            self.pickup_failures,
            self.uncertain,
            self.retries,
            self.motion_errors,
            self.successful_by_size.get(30, 0),
            self.successful_by_size.get(50, 0),
            self.successful_by_arm.get("left", 0),
            self.successful_by_arm.get("right", 0),
            self.simultaneous_commands,
            self.sequential_commands,
            1 if self.pending_holding is not None else 0,
            self.last_newly_visible,
            self.completed_cycles,
            round(average, 3),
            self.last_result,
        )


def render_statistics_panel(
    snapshot: tuple,
    font_path: Optional[str] = None,
    *,
    width: int = 500,
    height: int = 610,
) -> np.ndarray:
    """Render the second OpenCV window; the caller caches unchanged snapshots."""
    (
        detected,
        attempts,
        successful,
        failures,
        uncertain_count,
        retries,
        motion_errors,
        size_30,
        size_50,
        left_count,
        right_count,
        simultaneous,
        sequential,
        pending,
        newly_visible,
        completed_cycles,
        average_seconds,
        last_result,
    ) = snapshot

    rows = [
        ("현재 상태", str(last_result), (95, 220, 255)),
        ("현재 카메라 감지", f"{detected}개", (110, 255, 150)),
        ("확인 대기", f"{pending}개", (255, 210, 90)),
        ("작업 시도", f"{attempts}회", (230, 230, 230)),
        ("분류 완료 (카메라 확인)", f"{successful}개", (110, 255, 150)),
        ("집기 실패", f"{failures}회", (255, 120, 120)),
        ("확인 불가", f"{uncertain_count}회", (255, 190, 100)),
        ("재시도", f"{retries}회", (220, 200, 130)),
        ("로봇 응답 오류", f"{motion_errors}회", (255, 120, 120)),
        ("3cm / 5cm 성공", f"{size_30} / {size_50}", (180, 220, 255)),
        ("왼팔 / 오른팔 성공", f"{left_count} / {right_count}", (180, 220, 255)),
        ("동시 / 순차 명령", f"{simultaneous} / {sequential}", (210, 210, 230)),
        ("새로 보임·이동 의심", f"{newly_visible}개", (255, 190, 100)),
        ("완료 작업구역", f"{completed_cycles}회", (210, 210, 230)),
        ("평균 작업구역 시간", f"{average_seconds:.2f}초", (210, 210, 230)),
    ]

    panel = np.full((height, width, 3), (28, 32, 38), dtype=np.uint8)
    if Image is not None and ImageDraw is not None and ImageFont is not None and font_path:
        try:
            title_font = ImageFont.truetype(font_path, 28)
            label_font = ImageFont.truetype(font_path, 18)
            value_font = ImageFont.truetype(font_path, 19)
            pil = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            draw.text((24, 18), "분류 통계", font=title_font, fill=(245, 245, 245))
            draw.line((24, 58, width - 24, 58), fill=(85, 95, 108), width=2)
            y = 76
            for label, value, bgr in rows:
                rgb = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
                draw.text((24, y), label, font=label_font, fill=(175, 185, 198))
                draw.text((width - 24, y), value, font=value_font, fill=rgb, anchor="ra")
                y += 34
            return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        except Exception:
            pass

    cv2.putText(panel, "SORTING STATISTICS", (22, 42), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (240, 240, 240), 2, cv2.LINE_AA)
    fallback_rows = (
        f"Detected now: {detected}", f"Pending: {pending}",
        f"Attempts: {attempts}", f"Camera-confirmed: {successful}",
        f"Pick failures: {failures}", f"Uncertain: {uncertain_count}",
        f"Retries: {retries}", f"3cm / 5cm: {size_30} / {size_50}",
        f"Left / Right: {left_count} / {right_count}",
        f"Average cycle: {average_seconds:.2f}s",
    )
    for index, text in enumerate(fallback_rows):
        cv2.putText(panel, text, (22, 82 + index * 38), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (205, 215, 225), 1, cv2.LINE_AA)
    return panel
