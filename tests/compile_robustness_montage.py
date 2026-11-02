#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
from PIL import Image


DEFAULT_MANIFEST = Path("Dataset") / "robutsness_test" / "manifest.csv"
DEFAULT_VIDEO_ROOT = Path("Dataset") / "robutsness_test" / "video"
DEFAULT_OUTPUT = Path("artifacts") / "robustness_reports_old" / "representative_noisy_samples.png"


def as_int(value: str | None, default: int) -> int:
    try:
        return int(float(value or ""))
    except ValueError:
        return default


def read_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if str(row.get("ok", "")).strip().lower() in {"true", "1", "yes"}
        ]
    if not rows:
        raise ValueError(f"No completed robustness rows found in {manifest_path}")
    return rows


def resolve_video_path(row: dict[str, str], video_root: Path) -> Path:
    output_path = Path(row["output_video_path"])
    if output_path.is_absolute() and output_path.exists():
        return output_path
    if output_path.is_absolute():
        return video_root / output_path.name
    return video_root / output_path


def select_rows(
    rows: list[dict[str, str]],
    sample_count: int,
    seed: int,
    *,
    unique_participants: bool,
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    by_variant: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            row.get("effect_variant", "") or row.get("noise_variant", ""),
            row.get("pixel_variant", "") or row.get("noise_variant", ""),
        )
        by_variant.setdefault(key, []).append(row)

    selected: list[dict[str, str]] = []
    selected_participants: set[str] = set()
    variant_keys = sorted(by_variant)
    rng.shuffle(variant_keys)
    while len(selected) < sample_count and variant_keys:
        for key in list(variant_keys):
            candidates = by_variant[key]
            if not candidates:
                variant_keys.remove(key)
                continue
            row_index = rng.randrange(len(candidates))
            row = candidates.pop(row_index)
            participant_id = row.get("Participant ID", "")
            if unique_participants and participant_id in selected_participants:
                continue
            selected.append(row)
            selected_participants.add(participant_id)
            if len(selected) == sample_count:
                break
        if unique_participants and len(selected_participants) >= len({row.get("Participant ID", "") for row in rows}):
            break
    return selected


def read_video_frame(video_path: Path, row: dict[str, str], frame_mode: str) -> Image.Image:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or as_int(row.get("written_frame_count"), 1)
    if frame_mode == "middle":
        frame_index = max(0, frame_count // 2)
    elif frame_mode == "pixel-start":
        frame_index = as_int(row.get("pixel_start_frame"), frame_count // 2)
    else:
        start = as_int(row.get("pixel_start_frame"), frame_count // 2)
        end = as_int(row.get("pixel_end_frame"), start)
        frame_index = (start + end) // 2
    frame_index = max(0, min(frame_count - 1, frame_index))

    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise ValueError(f"Could not read frame {frame_index} from {video_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def build_montage(images: list[Image.Image], columns: int, tile_width: int, border: int) -> Image.Image:
    if not images:
        raise ValueError("No images to compile.")
    resized: list[Image.Image] = []
    for image in images:
        width, height = image.size
        tile_height = round(tile_width * height / width)
        resized.append(image.resize((tile_width, tile_height), Image.Resampling.LANCZOS))

    tile_height = max(image.height for image in resized)
    rows = (len(resized) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (
            columns * tile_width + (columns + 1) * border,
            rows * tile_height + (rows + 1) * border,
        ),
        "white",
    )
    for index, image in enumerate(resized):
        row = index // columns
        column = index % columns
        x = border + column * (tile_width + border)
        y = border + row * (tile_height + border)
        canvas.paste(image, (x, y))
    return canvas


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compile representative noisy robustness frames into one image.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--tile-width", type=int, default=320)
    parser.add_argument("--border", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--unique-participants", action="store_true")
    parser.add_argument(
        "--frame-mode",
        choices=("pixel-middle", "pixel-start", "middle"),
        default="pixel-middle",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = select_rows(
        read_rows(args.manifest),
        args.samples,
        args.seed,
        unique_participants=args.unique_participants,
    )
    images: list[Image.Image] = []
    selected_rows: list[dict[str, str]] = []
    for row in rows:
        video_path = resolve_video_path(row, args.video_root)
        if not video_path.exists():
            continue
        images.append(read_video_frame(video_path, row, args.frame_mode))
        selected_rows.append(row)

    montage = build_montage(images, args.columns, args.tile_width, args.border)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    montage.save(args.output)

    selected_csv = args.output.with_suffix(".selected.csv")
    fieldnames = [
        "Participant ID",
        "Video file",
        "Gloss",
        "output_video_path",
        "effect_variant",
        "pixel_variant",
        "pixel_start_frame",
        "pixel_end_frame",
    ]
    with selected_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected_rows)
    print(f"wrote {args.output}")
    print(f"wrote {selected_csv}")


if __name__ == "__main__":
    main()
