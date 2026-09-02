"""오른팔만 임의의 큐브 좌표로 움직이는 무공압 테스트.

필수 조건:
- STM32_제어코드/로봇팔_제어_메인.c 최신 버전을 로봇팔 STM32의 `main.c`로 적용
- 로봇팔 STM32가 COM3 / 115200으로 연결
- 실제 큐브와 장애물을 작업영역에서 치운 뒤 실행
"""

from __future__ import annotations

import random
import time
import msvcrt

import serial

ROBOT_PORT = "COM3"
ROBOT_BAUD = 115200

# 카메라 없이 사용할 수 있도록 미리 계산해 둔 오른팔 동작 세트다.
# values 순서: size, J1, approach(J2/J3/J4), contact, preload, lift
RIGHT_ARM_TEST_PLANS = (
    {
        "name": "중앙 3cm 가상 큐브",
        "coords": (0.0, -60.0, 30.0),
        "values": (30.0, 94.0, 76.0, 41.0, 29.5, 72.5, 44.5, 36.5,
                   72.0, 45.0, 36.0, 74.5, 43.0, 28.5),
    },
    {
        "name": "오른쪽 5cm 가상 큐브",
        "coords": (60.0, -70.0, 50.0),
        "values": (50.0, 117.5, 78.0, 34.5, 21.0, 76.0, 39.0, 27.5,
                   75.5, 39.5, 27.0, 77.0, 37.0, 20.0),
    },
    {
        "name": "왼쪽 3cm 가상 큐브",
        "coords": (-70.0, -80.0, 30.0),
        "values": (30.0, 62.0, 77.0, 42.5, 30.0, 73.5, 46.0, 37.0,
                   73.0, 46.5, 36.5, 75.5, 44.5, 29.0),
    },
)


def write_line_slow(ser: serial.Serial, line: str) -> None:
    ser.reset_input_buffer()
    for byte in line.encode("ascii"):
        ser.write(bytes((byte,)))
        ser.flush()
        time.sleep(0.015)


def make_right_test_command(plan: dict) -> str:
    values = plan["values"]
    return "T," + ",".join(f"{value:.1f}" for value in values) + "\n"


def wait_for_result(ser: serial.Serial) -> str | None:
    while True:
        if msvcrt.kbhit():
            key = msvcrt.getwch().lower()
            if key == "x":
                write_line_slow(ser, "X\n")
                print("\n긴급 정지 X를 전송했습니다.")
        line = ser.readline().decode("utf-8", errors="replace").strip()
        if not line:
            continue
        print(f"[STM32] {line}")
        if line in ("DONE", "ERROR", "BUSY"):
            return line


def main() -> None:
    print("오른팔 무공압 동작 테스트")
    print("- 왼팔은 로딩 자세에서 대기합니다.")
    print("- 오른팔 공압 CH1은 켜지지 않습니다.")
    print("- Enter: 임의 위치 1개 실행 / q: 종료 / x: 긴급 정지")

    with serial.Serial(ROBOT_PORT, ROBOT_BAUD, timeout=0.2) as ser:
        time.sleep(2.0)
        while True:
            command = input("\n명령> ").strip().lower()
            if command == "q":
                break
            if command == "x":
                write_line_slow(ser, "X\n")
                print("긴급 정지 X를 전송했습니다.")
                continue
            if command:
                print("Enter, q, x 중 하나를 입력하세요.")
                continue

            plan = random.choice(RIGHT_ARM_TEST_PLANS)
            camera_x, camera_y, cube_z = plan["coords"]
            print(
                f"{plan['name']}: "
                f"Camera X={camera_x:+.0f}, Y={camera_y:+.0f}mm"
            )
            write_line_slow(ser, make_right_test_command(plan))
            result = wait_for_result(ser)
            print(f"테스트 결과: {result}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n사용자가 테스트를 종료했습니다.")
