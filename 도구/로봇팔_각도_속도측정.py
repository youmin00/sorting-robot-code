"""하드웨어 없이 로봇팔 각도 계획 계산 속도를 확인한다."""

from __future__ import annotations

from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sorting_robot import arm_planning
from sorting_robot.arm_protocol import build_dual_arm_line


def main() -> None:
    objects = (
        ({"robot_x_mm": 0.0, "robot_y_mm": 50.0, "robot_z_mm": 30.0}, "left"),
        ({"robot_x_mm": 0.0, "robot_y_mm": -50.0, "robot_z_mm": 30.0}, "right"),
    )
    repetitions = 30
    elapsed_samples = []

    for _ in range(repetitions):
        arm_planning._ARM_PLAN_CACHE.clear()
        started = time.perf_counter()
        plans = [arm_planning.build_arm_pick_plan(obj, arm) for obj, arm in objects]
        elapsed_samples.append(time.perf_counter() - started)

    average_ms = sum(elapsed_samples) / len(elapsed_samples) * 1000.0
    worst_ms = max(elapsed_samples) * 1000.0
    command = build_dual_arm_line(plans[0], plans[1], sequential=False)

    print("하드웨어를 사용하지 않는 각도 계산 시험")
    print(f"양팔 계획 평균: {average_ms:.3f} ms")
    print(f"양팔 계획 최댓값: {worst_ms:.3f} ms")
    print(f"생성 명령 크기: {len(command.encode('ascii'))} bytes")


if __name__ == "__main__":
    main()
