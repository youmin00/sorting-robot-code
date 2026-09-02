import argparse
import os
import sys
import time

import cv2
import numpy as np


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import 통합_자동분류_실행 as integrated


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure integrated-camera coordinate stability")
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--warmup-frames", type=int, default=60)
    parser.add_argument("--timeout-ms", type=int, default=2500)
    parser.add_argument("--debug-detect", action="store_true")
    parser.add_argument("--save-prefix", default=None)
    parser.add_argument("--verify-runtime-wait", action="store_true")
    args = parser.parse_args()

    original_argv = sys.argv
    try:
        sys.argv = ["통합_자동분류_실행.py"]
        cfg = integrated.parse_args()
    finally:
        sys.argv = original_argv
    cfg.debug_detection = args.debug_detect
    cfg.debug_print_interval = 30

    cam = integrated.D435Camera(cfg)
    cam.detection_overlay_enabled = True
    samples: dict[int, list[tuple[float, float]]] = {}
    seen: dict[int, int] = {}
    scene_samples: list[list[tuple[int, float, float]]] = []
    last_color = None
    last_depth = None
    last_display = None
    runtime_wait_result = None
    runtime_wait_elapsed = None
    cam.start()
    try:
        total_frames = max(0, args.warmup_frames) + max(1, args.frames)
        for frame_index in range(total_frames):
            color_img, depth_img = cam.get_frames(timeout_ms=args.timeout_ms)
            display = cam._compose_display(color_img, depth_img)
            last_color = color_img.copy()
            last_depth = None if depth_img is None else depth_img.copy()
            last_display = display.copy()
            if frame_index < args.warmup_frames:
                continue
            frame_scene: list[tuple[int, float, float]] = []
            for obj in cam.last_tracked_objects:
                object_id = int(obj.get("id", 0))
                seen[object_id] = seen.get(object_id, 0) + 1
                if obj.get("robot_x_mm") is None or obj.get("robot_y_mm") is None:
                    continue
                samples.setdefault(object_id, []).append((
                    float(obj["robot_x_mm"]),
                    float(obj["robot_y_mm"]),
                ))
                frame_scene.append((
                    object_id,
                    float(obj["robot_x_mm"]),
                    float(obj["robot_y_mm"]),
                ))
            scene_samples.append(frame_scene)
        if args.verify_runtime_wait:
            wait_started = time.time()
            runtime_wait_result = cam._wait_for_stable_scene(
                integrated.CAMERA_SETTLE_SEC,
                integrated.CAMERA_SETTLE_FRAMES,
                empty_grace_sec=integrated.CAMERA_INITIAL_EMPTY_GRACE_SEC,
            )
            runtime_wait_elapsed = time.time() - wait_started
    finally:
        cam.stop()

    print(f"DIAGNOSTIC_FRAMES={max(1, args.frames)}")
    print(f"SEEN={seen}")
    print(f"LAST_CONTOUR_MODE={cam.last_contour_mode}")
    print(f"LAST_CANDIDATES={len(cam.last_candidates)}")
    print(f"LAST_SELECTED={len(cam.last_selected_candidates)}")
    if runtime_wait_result is not None and runtime_wait_elapsed is not None:
        print(
            f"RUNTIME_WAIT=PASS objects={len(runtime_wait_result)} "
            f"elapsed={runtime_wait_elapsed:.3f}s"
        )
    old_stable_transitions = 0
    position_stable_transitions = 0
    for before, now in zip(scene_samples, scene_samples[1:]):
        before_by_id = sorted(before)
        now_by_id = sorted(now)
        old_stable = len(before_by_id) == len(now_by_id) and all(
            before_obj[0] == now_obj[0]
            and np.hypot(before_obj[1] - now_obj[1], before_obj[2] - now_obj[2])
            <= integrated.CAMERA_STABLE_POSITION_TOLERANCE_MM
            for before_obj, now_obj in zip(before_by_id, now_by_id)
        )
        old_stable_transitions += int(old_stable)
        position_stable_transitions += int(integrated.scene_positions_stable(
            [(obj[1], obj[2]) for obj in before],
            [(obj[1], obj[2]) for obj in now],
            integrated.CAMERA_STABLE_POSITION_TOLERANCE_MM,
        ))
    transition_count = max(0, len(scene_samples) - 1)
    print(
        f"STABLE_TRANSITIONS={transition_count} "
        f"old_id_based={old_stable_transitions} "
        f"new_position_based={position_stable_transitions}"
    )
    if args.save_prefix and last_color is not None and last_display is not None:
        output_dir = os.path.dirname(os.path.abspath(args.save_prefix))
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(f"{args.save_prefix}_color.png", last_color)
        cv2.imwrite(f"{args.save_prefix}_display.png", last_display)
        if last_depth is not None:
            depth_panel = cam._depth_to_colormap(last_depth)
            cv2.imwrite(f"{args.save_prefix}_depth.png", depth_panel)
        print(f"SAVED_PREFIX={os.path.abspath(args.save_prefix)}")
    if not samples:
        print("NO_VALID_OBJECT_COORDINATES")
        return

    for object_id, values in sorted(samples.items()):
        points = np.asarray(values, dtype=np.float64)
        steps = (
            np.linalg.norm(np.diff(points, axis=0), axis=1)
            if len(points) > 1
            else np.asarray([0.0], dtype=np.float64)
        )
        span = np.ptp(points, axis=0)
        print(
            f"OBJECT {object_id}: n={len(points)} "
            f"step_mean={np.mean(steps):.2f}mm "
            f"step_p95={np.percentile(steps, 95):.2f}mm "
            f"span_x={span[0]:.2f}mm span_y={span[1]:.2f}mm"
        )


if __name__ == "__main__":
    main()
