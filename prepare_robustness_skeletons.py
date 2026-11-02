#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path

from process_skeleton_dataset import (
    HandLandmarker,
    build_hand_options,
    create_landmarker,
    ensure_requested_models,
    iter_video_paths,
    process_video,
    skeleton_outputs_exist,
    write_index_maps,
)


DEFAULT_VIDEO_ROOT = Path("Dataset") / "robutsness_test" / "video"
DEFAULT_OUTPUT_ROOT = Path("Dataset") / "robutsness_test" / "skeleton"
DEFAULT_FRAME_SAMPLE_COUNTS = (10, 20, 30)


def parse_int_list(raw_value: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw_value.split(",") if value.strip())
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract hand-only skeleton arrays from generated robustness-test videos "
            "for one or more sampled frame counts."
        )
    )
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--frame-sample-counts",
        default=",".join(str(value) for value in DEFAULT_FRAME_SAMPLE_COUNTS),
        help="Comma-separated frame counts. Default: 10,20,30.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frame_sample_counts = parse_int_list(args.frame_sample_counts)
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive when provided.")
    for frame_sample_count in frame_sample_counts:
        if frame_sample_count <= 0:
            raise ValueError("Frame sample counts must be positive.")

    video_root = args.video_root.resolve()
    output_root = args.output_root.resolve()
    if not video_root.is_dir():
        raise FileNotFoundError(f"Robustness video root not found: {video_root}")

    all_video_paths = list(iter_video_paths(video_root))
    selected_video_paths = all_video_paths[args.start_index:]
    if args.limit is not None:
        selected_video_paths = selected_video_paths[: args.limit]
    if not selected_video_paths:
        raise FileNotFoundError(f"No videos selected from {video_root}")

    planned = 0
    skipped = 0
    for frame_sample_count in frame_sample_counts:
        output_folder = output_root / f"{frame_sample_count}frame"
        write_index_maps(
            output_folder,
            include_hand=True,
            include_body=False,
            include_face=False,
        )
        for video_path in selected_video_paths:
            exists = skeleton_outputs_exist(
                video_path=video_path,
                input_root=video_root,
                output_folder=output_folder,
                include_hand=True,
                include_body=False,
                include_face=False,
            )
            if exists and not args.overwrite:
                skipped += 1
            else:
                planned += 1

    print(
        "Robustness hand skeleton plan: "
        f"videos={len(selected_video_paths)} frame_counts={','.join(map(str, frame_sample_counts))} "
        f"planned={planned} skipped_existing={skipped} "
        f"video_root={video_root} output_root={output_root}"
    )
    if args.dry_run:
        return
    if planned == 0:
        return

    ensure_requested_models(include_hand=True, include_body=False, include_face=False)
    with ExitStack() as stack:
        hand_landmarker, hand_delegate = create_landmarker(
            HandLandmarker,
            build_hand_options,
            "hand landmarker",
        )
        hand_landmarker = stack.enter_context(hand_landmarker)
        print(f"Loaded hand landmarker with delegate={hand_delegate.name}")

        timestamp_offset_ms = 0
        processed = 0
        skipped_existing = 0
        for frame_sample_count in frame_sample_counts:
            output_folder = output_root / f"{frame_sample_count}frame"
            for video_path in selected_video_paths:
                exists = skeleton_outputs_exist(
                    video_path=video_path,
                    input_root=video_root,
                    output_folder=output_folder,
                    include_hand=True,
                    include_body=False,
                    include_face=False,
                )
                if exists and not args.overwrite:
                    skipped_existing += 1
                    continue

                timestamp_offset_ms = process_video(
                    video_path=video_path,
                    input_root=video_root,
                    output_folder=output_folder,
                    hand_landmarker=hand_landmarker,
                    pose_landmarker=None,
                    face_landmarker=None,
                    timestamp_offset_ms=timestamp_offset_ms,
                    include_hand=True,
                    include_body=False,
                    include_face=False,
                    frame_sample_count=frame_sample_count,
                )
                processed += 1
                if processed % 100 == 0:
                    print(
                        "progress "
                        f"processed={processed}/{planned} skipped_existing={skipped_existing}"
                    )

    print(
        "Robustness hand skeleton extraction complete: "
        f"processed={processed} skipped_existing={skipped_existing} output_root={output_root}"
    )


if __name__ == "__main__":
    main()
