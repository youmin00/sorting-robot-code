"""실제 카메라 없이 합성된 두 물체 장면의 검출 속도를 측정한다."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import time

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROGRAM_PATH = PROJECT_ROOT / "통합_자동분류_실행.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_program():
    spec = importlib.util.spec_from_file_location("detection_benchmark", PROGRAM_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"프로그램을 불러올 수 없습니다: {PROGRAM_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    program = load_program()
    config = program.CameraConfig(center_roi_only=False, show_depth_panel=False)
    camera = program.D435Camera(config)
    camera.depth_scale = 0.001
    camera.detection_overlay_enabled = True

    color = np.full((480, 848, 3), 100, dtype=np.uint8)
    depth = np.full((480, 848), 500, dtype=np.uint16)
    cv2.rectangle(color, (300, 180), (390, 270), (230, 230, 230), -1)
    depth[180:271, 300:391] = 450
    cv2.rectangle(color, (500, 200), (570, 270), (20, 20, 20), -1)
    depth[200:271, 500:571] = 470

    for _ in range(3):
        camera._compose_display(color, depth)

    repetitions = 30
    started = time.perf_counter()
    for _ in range(repetitions):
        camera._compose_display(color, depth)
    average_ms = (time.perf_counter() - started) / repetitions * 1000.0

    print("하드웨어를 사용하지 않는 카메라 검출 시험")
    print(f"프레임당 평균: {average_ms:.3f} ms")
    print(f"검출 물체: {camera.last_tracked_objects}")


if __name__ == "__main__":
    main()
