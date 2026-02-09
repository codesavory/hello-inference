import argparse
from fractions import Fraction
from pathlib import Path

import cv2


def _get_fps(cap: cv2.VideoCapture) -> float:
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    duration_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if duration_ms and total_frames:
        fps = total_frames / (duration_ms / 1000.0)
    elif raw_fps and raw_fps > 0:
        fps = raw_fps
    else:
        fps = 30.0
    return float(Fraction(fps).limit_denominator(1000))


def _resize_to_height(frame, target_height: int):
    height, width = frame.shape[:2]
    if height == target_height:
        return frame
    scale = target_height / float(height)
    target_width = int(round(width * scale))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _put_label(frame, text: str, position=(16, 32)):
    return cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def combine_videos(left_path: Path, right_path: Path, out_path: Path, fps: float | None):
    left_cap = cv2.VideoCapture(str(left_path))
    right_cap = cv2.VideoCapture(str(right_path))
    if not left_cap.isOpened():
        raise RuntimeError(f"Failed to open left video: {left_path}")
    if not right_cap.isOpened():
        raise RuntimeError(f"Failed to open right video: {right_path}")

    left_fps = _get_fps(left_cap)
    right_fps = _get_fps(right_cap)
    output_fps = fps if fps and fps > 0 else min(left_fps, right_fps)

    ok_left, left_frame = left_cap.read()
    ok_right, right_frame = right_cap.read()
    if not ok_left or left_frame is None:
        raise RuntimeError("Failed to read first frame from left video.")
    if not ok_right or right_frame is None:
        raise RuntimeError("Failed to read first frame from right video.")

    target_height = left_frame.shape[0]
    left_frame = _resize_to_height(left_frame, target_height)
    right_frame = _resize_to_height(right_frame, target_height)

    left_frame = _put_label(left_frame, f"left {left_fps:.2f} fps", (16, 32))
    right_frame = _put_label(right_frame, f"right {right_fps:.2f} fps", (16, 32))
    combined = cv2.hconcat([left_frame, right_frame])
    height, width = combined.shape[:2]
    # No output label (only per-side labels).
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, output_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Failed to initialize VideoWriter.")
    writer.write(combined)

    while True:
        ok_left, left_frame = left_cap.read()
        ok_right, right_frame = right_cap.read()
        if not ok_left or not ok_right:
            break
        left_frame = _resize_to_height(left_frame, target_height)
        right_frame = _resize_to_height(right_frame, target_height)
        left_frame = _put_label(left_frame, f"left {left_fps:.2f} fps", (16, 32))
        right_frame = _put_label(right_frame, f"right {right_fps:.2f} fps", (16, 32))
        combined = cv2.hconcat([left_frame, right_frame])
        writer.write(combined)

    left_cap.release()
    right_cap.release()
    writer.release()


def main():
    parser = argparse.ArgumentParser(description="Combine two videos side-by-side.")
    parser.add_argument("--left", required=True, help="Path to the left video.")
    parser.add_argument("--right", required=True, help="Path to the right video.")
    parser.add_argument("--out", required=True, help="Path to output video (mp4).")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override output fps (default: min(left, right)).",
    )
    args = parser.parse_args()
    combine_videos(Path(args.left), Path(args.right), Path(args.out), args.fps)


if __name__ == "__main__":
    main()
