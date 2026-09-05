"""PC에서 로봇팔 STM32로 보내는 명령 생성과 안전 전송."""

from __future__ import annotations

import time
from typing import Optional


# 15 ms는 실제 장비의 단일·교차 순차 작업에서 검증된 복귀 기준값이다.
# 현재 값은 두 번째 단계 하드웨어 시험용이며, 오류가 있으면 0.015로 복원한다.
ARM_SERIAL_BYTE_DELAY_SEC = 0.012


def write_arm_line(arm_ser, line: str) -> None:
    arm_ser.reset_input_buffer()
    # 현재 STM32 UART 수신부는 한 바이트씩 폴링한다. PC가 한꺼번에
    # 전송해 수신 레지스터가 넘치지 않도록 기존 간격을 유지한다.
    for byte in line.encode("ascii"):
        arm_ser.write(bytes((byte,)))
        arm_ser.flush()
        time.sleep(ARM_SERIAL_BYTE_DELAY_SEC)


def format_arm_value(value: float) -> str:
    """불필요한 '.0' 없이 기존의 소수점 한 자리 명령 형식을 유지한다."""
    return f"{value:.1f}".rstrip("0").rstrip(".")


def plan_values(plan: Optional[dict]) -> list[float]:
    if plan is None:
        return [0.0] * 14
    values = [float(plan["size"]), float(plan["approach"]["j1"])]
    for pose_name in ("approach", "contact", "preload", "lift"):
        pose = plan[pose_name]
        values.extend((float(pose["j2"]), float(pose["j3"]), float(pose["j4"])))
    return values


def build_single_arm_line(plan: dict) -> str:
    return "P," + ",".join(format_arm_value(value) for value in plan_values(plan)) + "\n"


def build_dual_arm_line(
    left_plan: Optional[dict],
    right_plan: Optional[dict],
    sequential: bool = True,
) -> str:
    values = [1.0 if left_plan is not None else 0.0]
    values.extend(plan_values(left_plan))
    values.append(1.0 if right_plan is not None else 0.0)
    values.extend(plan_values(right_plan))
    return ("D," if sequential else "M,") + ",".join(
        format_arm_value(value) for value in values
    ) + "\n"
