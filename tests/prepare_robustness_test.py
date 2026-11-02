#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


DEFAULT_DATASET_DIR = Path("Dataset")
DEFAULT_INPUT_CSV = DEFAULT_DATASET_DIR / "split" / "robustness_test.csv"
DEFAULT_NOISE_DIR = DEFAULT_DATASET_DIR / "noise"
DEFAULT_OUTPUT_DIR = DEFAULT_DATASET_DIR / "robustness_noise" / "videos"
DEFAULT_MANIFEST = DEFAULT_DATASET_DIR / "split" / "robustness_noise_manifest.csv"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


@dataclass(frozen=True, slots=True)
class NoiseVariant:
    name: str
    path: Path


@dataclass(frozen=True, slots=True)
class NoisePair:
    effect: NoiseVariant
    pixel: NoiseVariant

    @property
    def name(self) -> str:
        return f"{self.effect.name}+{self.pixel.name}"


@dataclass(frozen=True, slots=True)
class PixelPatch:
    x: int
    y: int
    size: int
    center_x: int
    center_y: int
    target_frame: int | str
    target_frame_count: int | str
    target_hand: str


@dataclass(frozen=True, slots=True)
class NoiseWindow:
    start_frame: int
    end_frame: int

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate robustness videos by overlaying pixel-noise or effect PNGs "
            "onto videos listed in Dataset/split/robustness_test.csv."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--noise-dir", type=Path, default=DEFAULT_NOISE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--source-dirs",
        default="raw,testing",
        help="Comma-separated video source directories under --dataset-dir.",
    )
    parser.add_argument(
        "--variants",
        default="all",
        help=(
            "Comma-separated variants from Dataset/noise without extension, "
            "or 'all'. Example: pixel1,effect1,effect3."
        ),
    )
    parser.add_argument(
        "--generation-mode",
        choices=("combined", "single"),
        default="combined",
        help=(
            "combined writes one video with both a full-frame effect and timed "
            "pixel noise. single keeps the legacy one-variant-per-video behavior."
        ),
    )
    parser.add_argument(
        "--assignment",
        choices=("alternate", "random", "all"),
        default="random",
        help=(
            "For combined mode, alternate/random chooses one effect+pixel pair per "
            "CSV row; all writes every effect+pixel pair. For single mode, this "
            "applies to individual variants."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Effect overlay alpha. Pixel noise is always applied with alpha=1.0.",
    )
    parser.add_argument(
        "--pixel-min-scale",
        type=float,
        default=0.30,
        help="Minimum pixel patch size as a fraction of min(video width, height).",
    )
    parser.add_argument(
        "--pixel-max-scale",
        type=float,
        default=0.48,
        help="Maximum pixel patch size as a fraction of min(video width, height).",
    )
    parser.add_argument(
        "--pixel-center-jitter",
        type=float,
        default=0.18,
        help=(
            "Maximum random pixel patch center offset as a fraction of "
            "min(video width, height)."
        ),
    )
    parser.add_argument(
        "--pixel-placement",
        choices=("hand", "center"),
        default="hand",
        help=(
            "Pixel noise placement. 'hand' targets one detected hand location "
            "from Dataset/raw/hand; 'center' uses a random near-center patch."
        ),
    )
    parser.add_argument(
        "--pixel-hand-scale",
        type=float,
        default=1.45,
        help="Patch size multiplier relative to the selected hand bounding box.",
    )
    parser.add_argument(
        "--noise-min-duration-frames",
        type=int,
        default=24,
        help=(
            "Minimum number of consecutive frames where pixel noise is visible. "
            "Effect noise is always applied to the full video."
        ),
    )
    parser.add_argument(
        "--noise-max-duration-frames",
        type=int,
        default=36,
        help=(
            "Maximum number of consecutive frames where pixel noise is visible. "
            "Effect noise is always applied to the full video."
        ),
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=128)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def discover_noise_variants(noise_dir: Path) -> dict[str, NoiseVariant]:
    variants = {}
    for path in sorted(noise_dir.glob("*.png")):
        variants[path.stem] = NoiseVariant(name=path.stem, path=path)
    if not variants:
        raise FileNotFoundError(f"No PNG noise/effect assets found in {noise_dir}")
    return variants


def select_variants(raw_value: str, available: dict[str, NoiseVariant]) -> list[NoiseVariant]:
    if raw_value.strip().lower() == "all":
        return [available[name] for name in sorted(available)]
    names = [value.strip() for value in raw_value.split(",") if value.strip()]
    if not names:
        raise ValueError("--variants cannot be empty.")
    missing = sorted(set(names) - set(available))
    if missing:
        allowed = ", ".join(sorted(available))
        raise ValueError(
            f"Unknown noise variant(s): {', '.join(missing)}. Available: {allowed}"
        )
    return [available[name] for name in names]


def split_effect_pixel_variants(
    variants: list[NoiseVariant],
) -> tuple[list[NoiseVariant], list[NoiseVariant]]:
    effects = [variant for variant in variants if variant.name.startswith("effect")]
    pixels = [variant for variant in variants if variant.name.startswith("pixel")]
    if not effects:
        raise ValueError("Combined mode requires at least one effect* variant.")
    if not pixels:
        raise ValueError("Combined mode requires at least one pixel* variant.")
    return effects, pixels


def resolve_source_dirs(dataset_dir: Path, raw_value: str) -> list[Path]:
    paths = []
    for value in raw_value.split(","):
        value = value.strip()
        if not value:
            continue
        path = Path(value)
        paths.append(path if path.is_absolute() else dataset_dir / path)
    if not paths:
        raise ValueError("--source-dirs cannot be empty.")
    return paths


def find_source_video(video_file: str, source_dirs: list[Path]) -> Path | None:
    for source_dir in source_dirs:
        candidate = source_dir / video_file
        if candidate.exists():
            return candidate
    return None


def safe_stem(video_file: str) -> str:
    stem = Path(video_file).stem
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")


def read_overlay(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read overlay PNG: {path}")
    if image.ndim != 3:
        raise ValueError(f"Expected color overlay image, got shape={image.shape}: {path}")
    if image.shape[2] == 4:
        bgr = image[:, :, :3].astype(np.float32)
        alpha = image[:, :, 3].astype(np.float32) / 255.0
    else:
        bgr = image[:, :, :3].astype(np.float32)
        alpha = np.ones(image.shape[:2], dtype=np.float32)
    return bgr, alpha


def apply_overlay(
    frame: np.ndarray,
    overlay_bgr: np.ndarray,
    overlay_alpha: np.ndarray,
    alpha: float,
) -> np.ndarray:
    height, width = frame.shape[:2]
    resized_bgr = cv2.resize(overlay_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    resized_alpha = cv2.resize(overlay_alpha, (width, height), interpolation=cv2.INTER_LINEAR)
    blend_alpha = np.clip(resized_alpha * alpha, 0.0, 1.0)[:, :, None]
    blended = frame.astype(np.float32) * (1.0 - blend_alpha) + resized_bgr * blend_alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def build_pixel_patch(
    width: int,
    height: int,
    rng: random.Random,
    min_scale: float,
    max_scale: float,
    center_jitter: float,
) -> PixelPatch:
    base_size = min(width, height)
    patch_size = int(round(base_size * rng.uniform(min_scale, max_scale)))
    patch_size = max(1, min(patch_size, width, height))

    jitter_pixels = base_size * center_jitter
    center_x = (width / 2.0) + rng.uniform(-jitter_pixels, jitter_pixels)
    center_y = (height / 2.0) + rng.uniform(-jitter_pixels, jitter_pixels)
    x = int(round(center_x - (patch_size / 2.0)))
    y = int(round(center_y - (patch_size / 2.0)))
    x = max(0, min(x, width - patch_size))
    y = max(0, min(y, height - patch_size))
    return PixelPatch(
        x=x,
        y=y,
        size=patch_size,
        center_x=int(round(center_x)),
        center_y=int(round(center_y)),
        target_frame="center",
        target_frame_count="",
        target_hand="center",
    )


def hand_landmark_path_for_video(dataset_dir: Path, video_file: str) -> Path | None:
    stem = Path(video_file).stem
    candidates = [
        dataset_dir / "raw" / "hand" / f"{stem}.npy",
        dataset_dir / "skeleton" / "30frame" / "hand" / f"{stem}.npy",
        dataset_dir / "skeleton" / "20frame" / "hand" / f"{stem}.npy",
        dataset_dir / "skeleton" / "10frame" / "hand" / f"{stem}.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def raw_hand_path_for_video(dataset_dir: Path, video_file: str) -> Path:
    path = hand_landmark_path_for_video(dataset_dir, video_file)
    if path is not None:
        return path
    return dataset_dir / "raw" / "hand" / f"{Path(video_file).stem}.npy"


def _valid_xy(points: np.ndarray) -> np.ndarray:
    xy = points[:, :2]
    valid = np.isfinite(xy).all(axis=1)
    xy = xy[valid]
    return xy[(xy[:, 0] >= 0.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= 0.0) & (xy[:, 1] <= 1.0)]


def build_hand_pixel_patch(
    hand_path: Path,
    width: int,
    height: int,
    rng: random.Random,
    min_scale: float,
    max_scale: float,
    hand_scale: float,
) -> PixelPatch | None:
    if not hand_path.exists():
        return None
    hand_frames = np.load(hand_path)
    candidates = []
    hand_names = ("left", "right")
    for frame_index, frame_hands in enumerate(hand_frames):
        for hand_index, hand_points in enumerate(frame_hands):
            xy = _valid_xy(hand_points)
            if len(xy) < 4:
                continue
            min_xy = xy.min(axis=0)
            max_xy = xy.max(axis=0)
            center_xy = (min_xy + max_xy) / 2.0
            box_width = max(1.0, (max_xy[0] - min_xy[0]) * width)
            box_height = max(1.0, (max_xy[1] - min_xy[1]) * height)
            candidates.append(
                {
                    "frame_index": frame_index,
                    "hand": hand_names[hand_index] if hand_index < len(hand_names) else str(hand_index),
                    "center_x": float(center_xy[0] * width),
                    "center_y": float(center_xy[1] * height),
                    "box_size": max(box_width, box_height),
                }
            )
    if not candidates:
        return None

    target = rng.choice(candidates)
    base_size = min(width, height)
    random_size = base_size * rng.uniform(min_scale, max_scale)
    hand_size = target["box_size"] * hand_scale
    patch_size = int(round(max(random_size, hand_size)))
    patch_size = max(1, min(patch_size, width, height))
    center_x = int(round(target["center_x"]))
    center_y = int(round(target["center_y"]))
    x = int(round(center_x - (patch_size / 2.0)))
    y = int(round(center_y - (patch_size / 2.0)))
    x = max(0, min(x, width - patch_size))
    y = max(0, min(y, height - patch_size))
    return PixelPatch(
        x=x,
        y=y,
        size=patch_size,
        center_x=center_x,
        center_y=center_y,
        target_frame=int(target["frame_index"]),
        target_frame_count=int(len(hand_frames)),
        target_hand=str(target["hand"]),
    )


def build_noise_window(
    frame_count: int,
    rng: random.Random,
    min_duration_frames: int,
    max_duration_frames: int,
    anchor_frame: int | str | None = None,
    anchor_frame_count: int | str | None = None,
) -> NoiseWindow:
    if frame_count <= 0:
        return NoiseWindow(start_frame=0, end_frame=0)

    max_duration = min(max_duration_frames, frame_count)
    min_duration = min(min_duration_frames, max_duration)
    duration = rng.randint(min_duration, max_duration)

    anchor_video_frame: int | None = None
    if isinstance(anchor_frame, int):
        if isinstance(anchor_frame_count, int) and anchor_frame_count > 1:
            ratio = anchor_frame / float(anchor_frame_count - 1)
            anchor_video_frame = int(round(ratio * (frame_count - 1)))
        else:
            anchor_video_frame = min(anchor_frame, frame_count - 1)

    if anchor_video_frame is None:
        start = rng.randint(0, max(0, frame_count - duration))
    else:
        start = int(round(anchor_video_frame - (duration / 2.0)))
        start = max(0, min(start, frame_count - duration))

    return NoiseWindow(start_frame=start, end_frame=start + duration)


def apply_pixel_patch(
    frame: np.ndarray,
    overlay_bgr: np.ndarray,
    overlay_alpha: np.ndarray,
    alpha: float,
    patch: PixelPatch,
) -> np.ndarray:
    resized_bgr = cv2.resize(
        overlay_bgr,
        (patch.size, patch.size),
        interpolation=cv2.INTER_NEAREST,
    )
    resized_alpha = cv2.resize(
        overlay_alpha,
        (patch.size, patch.size),
        interpolation=cv2.INTER_NEAREST,
    )
    blend_alpha = np.clip(resized_alpha * alpha, 0.0, 1.0)[:, :, None]
    output = frame.copy()
    region = output[patch.y:patch.y + patch.size, patch.x:patch.x + patch.size]
    blended = region.astype(np.float32) * (1.0 - blend_alpha) + resized_bgr * blend_alpha
    output[patch.y:patch.y + patch.size, patch.x:patch.x + patch.size] = np.clip(
        blended,
        0,
        255,
    ).astype(np.uint8)
    return output


def write_augmented_video(
    source_path: Path,
    output_path: Path,
    variant: NoiseVariant,
    alpha: float,
    codec: str,
    seed: int,
    dataset_dir: Path,
    video_file: str,
    pixel_min_scale: float,
    pixel_max_scale: float,
    pixel_center_jitter: float,
    pixel_placement: str,
    pixel_hand_scale: float,
    noise_min_duration_frames: int,
    noise_max_duration_frames: int,
) -> dict[str, object]:
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open source video: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*codec)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output video writer: {output_path}")

    overlay_bgr, overlay_alpha = read_overlay(variant.path)
    is_pixel_noise = variant.name.startswith("pixel")
    pixel_patch = None
    rng = random.Random(seed)
    if is_pixel_noise:
        if pixel_placement == "hand":
            pixel_patch = build_hand_pixel_patch(
                hand_path=raw_hand_path_for_video(dataset_dir, video_file),
                width=width,
                height=height,
                rng=rng,
                min_scale=pixel_min_scale,
                max_scale=pixel_max_scale,
                hand_scale=pixel_hand_scale,
            )
        if pixel_patch is None:
            pixel_patch = build_pixel_patch(
                width=width,
                height=height,
                rng=rng,
                min_scale=pixel_min_scale,
                max_scale=pixel_max_scale,
                center_jitter=pixel_center_jitter,
            )
    if is_pixel_noise:
        noise_window = build_noise_window(
            frame_count=frame_count,
            rng=rng,
            min_duration_frames=noise_min_duration_frames,
            max_duration_frames=noise_max_duration_frames,
            anchor_frame=pixel_patch.target_frame if pixel_patch is not None else None,
            anchor_frame_count=pixel_patch.target_frame_count if pixel_patch is not None else None,
        )
    else:
        noise_window = NoiseWindow(start_frame=0, end_frame=frame_count)
    written_frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            noise_is_visible = noise_window.start_frame <= written_frames < noise_window.end_frame
            if not noise_is_visible:
                output_frame = frame
            elif pixel_patch is None:
                output_frame = apply_overlay(frame, overlay_bgr, overlay_alpha, alpha)
            else:
                output_frame = apply_pixel_patch(
                    frame,
                    overlay_bgr,
                    overlay_alpha,
                    1.0 if is_pixel_noise else alpha,
                    pixel_patch,
                )
            writer.write(output_frame)
            written_frames += 1
    finally:
        cap.release()
        writer.release()

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "source_frame_count": frame_count,
        "written_frame_count": written_frames,
        "noise_mode": "pixel_patch" if is_pixel_noise else "full_frame_effect",
        "effective_alpha": 1.0 if is_pixel_noise else alpha,
        "pixel_patch_x": pixel_patch.x if pixel_patch is not None else "",
        "pixel_patch_y": pixel_patch.y if pixel_patch is not None else "",
        "pixel_patch_size": pixel_patch.size if pixel_patch is not None else "",
        "pixel_patch_center_x": pixel_patch.center_x if pixel_patch is not None else "",
        "pixel_patch_center_y": pixel_patch.center_y if pixel_patch is not None else "",
        "pixel_target_frame": pixel_patch.target_frame if pixel_patch is not None else "",
        "pixel_target_frame_count": pixel_patch.target_frame_count if pixel_patch is not None else "",
        "pixel_target_hand": pixel_patch.target_hand if pixel_patch is not None else "",
        "noise_start_frame": noise_window.start_frame,
        "noise_end_frame": noise_window.end_frame,
        "noise_duration_frames": noise_window.duration_frames,
    }


def write_combined_augmented_video(
    source_path: Path,
    output_path: Path,
    noise_pair: NoisePair,
    alpha: float,
    codec: str,
    seed: int,
    dataset_dir: Path,
    video_file: str,
    pixel_min_scale: float,
    pixel_max_scale: float,
    pixel_center_jitter: float,
    pixel_placement: str,
    pixel_hand_scale: float,
    noise_min_duration_frames: int,
    noise_max_duration_frames: int,
) -> dict[str, object]:
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open source video: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*codec)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output video writer: {output_path}")

    effect_bgr, effect_alpha = read_overlay(noise_pair.effect.path)
    pixel_bgr, pixel_alpha = read_overlay(noise_pair.pixel.path)
    rng = random.Random(seed)

    pixel_patch = None
    if pixel_placement == "hand":
        pixel_patch = build_hand_pixel_patch(
            hand_path=raw_hand_path_for_video(dataset_dir, video_file),
            width=width,
            height=height,
            rng=rng,
            min_scale=pixel_min_scale,
            max_scale=pixel_max_scale,
            hand_scale=pixel_hand_scale,
        )
    if pixel_patch is None:
        pixel_patch = build_pixel_patch(
            width=width,
            height=height,
            rng=rng,
            min_scale=pixel_min_scale,
            max_scale=pixel_max_scale,
            center_jitter=pixel_center_jitter,
        )

    pixel_window = build_noise_window(
        frame_count=frame_count,
        rng=rng,
        min_duration_frames=noise_min_duration_frames,
        max_duration_frames=noise_max_duration_frames,
        anchor_frame=pixel_patch.target_frame,
        anchor_frame_count=pixel_patch.target_frame_count,
    )

    written_frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            output_frame = apply_overlay(frame, effect_bgr, effect_alpha, alpha)
            pixel_is_visible = pixel_window.start_frame <= written_frames < pixel_window.end_frame
            if pixel_is_visible:
                output_frame = apply_pixel_patch(
                    output_frame,
                    pixel_bgr,
                    pixel_alpha,
                    1.0,
                    pixel_patch,
                )
            writer.write(output_frame)
            written_frames += 1
    finally:
        cap.release()
        writer.release()

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "source_frame_count": frame_count,
        "written_frame_count": written_frames,
        "noise_mode": "combined_effect_pixel",
        "effect_variant": noise_pair.effect.name,
        "effect_asset": str(noise_pair.effect.path),
        "effect_start_frame": 0,
        "effect_end_frame": frame_count,
        "effect_duration_frames": frame_count,
        "effect_alpha": alpha,
        "pixel_variant": noise_pair.pixel.name,
        "pixel_asset": str(noise_pair.pixel.path),
        "pixel_alpha": 1.0,
        "pixel_patch_x": pixel_patch.x,
        "pixel_patch_y": pixel_patch.y,
        "pixel_patch_size": pixel_patch.size,
        "pixel_patch_center_x": pixel_patch.center_x,
        "pixel_patch_center_y": pixel_patch.center_y,
        "pixel_target_frame": pixel_patch.target_frame,
        "pixel_target_frame_count": pixel_patch.target_frame_count,
        "pixel_target_hand": pixel_patch.target_hand,
        "pixel_start_frame": pixel_window.start_frame,
        "pixel_end_frame": pixel_window.end_frame,
        "pixel_duration_frames": pixel_window.duration_frames,
    }


def iter_jobs(
    rows: list[dict[str, str]],
    variants: list[NoiseVariant],
    assignment: str,
    seed: int,
):
    rng = random.Random(seed)
    for row_index, row in enumerate(rows):
        if assignment == "all":
            selected = variants
        elif assignment == "random":
            selected = [rng.choice(variants)]
        else:
            selected = [variants[row_index % len(variants)]]
        for variant in selected:
            yield row_index, row, variant


def iter_combined_jobs(
    rows: list[dict[str, str]],
    effects: list[NoiseVariant],
    pixels: list[NoiseVariant],
    assignment: str,
    seed: int,
):
    rng = random.Random(seed)
    pairs = [NoisePair(effect=effect, pixel=pixel) for effect in effects for pixel in pixels]
    for row_index, row in enumerate(rows):
        if assignment == "all":
            selected = pairs
        elif assignment == "random":
            selected = [NoisePair(effect=rng.choice(effects), pixel=rng.choice(pixels))]
        else:
            selected = [pairs[row_index % len(pairs)]]
        for noise_pair in selected:
            yield row_index, row, noise_pair


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0.0 and 1.0.")
    if not 0.0 < args.pixel_min_scale <= args.pixel_max_scale <= 1.0:
        raise ValueError(
            "--pixel-min-scale and --pixel-max-scale must satisfy "
            "0 < min <= max <= 1."
        )
    if args.pixel_center_jitter < 0:
        raise ValueError("--pixel-center-jitter must be non-negative.")
    if args.pixel_hand_scale <= 0:
        raise ValueError("--pixel-hand-scale must be positive.")
    if args.noise_min_duration_frames < 1:
        raise ValueError("--noise-min-duration-frames must be positive.")
    if args.noise_max_duration_frames < args.noise_min_duration_frames:
        raise ValueError(
            "--noise-max-duration-frames must be greater than or equal to "
            "--noise-min-duration-frames."
        )
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive when provided.")
    if len(args.codec) != 4:
        raise ValueError("--codec must be a four-character code, for example mp4v.")

    rows = load_rows(args.input_csv)
    selected_rows = rows[args.start_index:]
    if args.limit is not None:
        selected_rows = selected_rows[:args.limit]

    available_variants = discover_noise_variants(args.noise_dir)
    variants = select_variants(args.variants, available_variants)
    effects: list[NoiseVariant] = []
    pixels: list[NoiseVariant] = []
    if args.generation_mode == "combined":
        effects, pixels = split_effect_pixel_variants(variants)
    source_dirs = resolve_source_dirs(args.dataset_dir, args.source_dirs)

    manifest_rows: list[dict[str, object]] = []
    errors = 0
    planned = 0
    written = 0

    if args.generation_mode == "combined":
        job_iter = iter_combined_jobs(
            selected_rows,
            effects,
            pixels,
            args.assignment,
            args.seed,
        )
    else:
        job_iter = iter_jobs(
            selected_rows,
            variants,
            args.assignment,
            args.seed,
        )

    for csv_row_index, row, noise_spec in job_iter:
        planned += 1
        video_file = row["Video file"]
        source_path = find_source_video(video_file, source_dirs)
        if isinstance(noise_spec, NoisePair):
            noise_name = noise_spec.name
            output_subdir = f"{noise_spec.effect.name}_{noise_spec.pixel.name}"
        else:
            noise_name = noise_spec.name
            output_subdir = noise_spec.name
        output_name = f"{safe_stem(video_file)}__{output_subdir}__a{args.alpha:.2f}.mp4"
        output_path = args.output_dir / output_subdir / output_name
        payload: dict[str, object] = {
            **row,
            "csv_row_index": args.start_index + csv_row_index,
            "generation_mode": args.generation_mode,
            "noise_variant": noise_name,
            "noise_asset": "" if isinstance(noise_spec, NoisePair) else str(noise_spec.path),
            "noise_alpha": args.alpha,
            "source_video_path": str(source_path) if source_path else "",
            "output_video_path": str(output_path),
            "ok": False,
            "error": "",
        }

        if source_path is None:
            payload["error"] = "source video not found"
            errors += 1
            manifest_rows.append(payload)
            print(f"missing source: {video_file}")
            continue

        if output_path.exists() and not args.overwrite:
            payload["ok"] = True
            payload["skipped_existing"] = True
            manifest_rows.append(payload)
            print(f"exists: {output_path}")
            continue

        if args.dry_run:
            payload["ok"] = True
            payload["dry_run"] = True
            manifest_rows.append(payload)
            print(f"would write: {output_path}")
            continue

        try:
            if isinstance(noise_spec, NoisePair):
                video_payload = write_combined_augmented_video(
                    source_path=source_path,
                    output_path=output_path,
                    noise_pair=noise_spec,
                    alpha=args.alpha,
                    codec=args.codec,
                    seed=args.seed + args.start_index + csv_row_index,
                    dataset_dir=args.dataset_dir,
                    video_file=video_file,
                    pixel_min_scale=args.pixel_min_scale,
                    pixel_max_scale=args.pixel_max_scale,
                    pixel_center_jitter=args.pixel_center_jitter,
                    pixel_placement=args.pixel_placement,
                    pixel_hand_scale=args.pixel_hand_scale,
                    noise_min_duration_frames=args.noise_min_duration_frames,
                    noise_max_duration_frames=args.noise_max_duration_frames,
                )
            else:
                video_payload = write_augmented_video(
                    source_path=source_path,
                    output_path=output_path,
                    variant=noise_spec,
                    alpha=args.alpha,
                    codec=args.codec,
                    seed=args.seed + args.start_index + csv_row_index,
                    dataset_dir=args.dataset_dir,
                    video_file=video_file,
                    pixel_min_scale=args.pixel_min_scale,
                    pixel_max_scale=args.pixel_max_scale,
                    pixel_center_jitter=args.pixel_center_jitter,
                    pixel_placement=args.pixel_placement,
                    pixel_hand_scale=args.pixel_hand_scale,
                    noise_min_duration_frames=args.noise_min_duration_frames,
                    noise_max_duration_frames=args.noise_max_duration_frames,
                )
        except Exception as exc:  # noqa: BLE001 - keep manifest useful across long runs.
            payload["error"] = str(exc)
            errors += 1
            print(f"failed: {video_file} variant={noise_name}: {exc}")
        else:
            payload.update(video_payload)
            payload["ok"] = True
            written += 1
            print(f"wrote: {output_path}")
        manifest_rows.append(payload)

    if manifest_rows:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(dict.fromkeys(key for row in manifest_rows for key in row))
        with args.manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)

    print(
        "Robustness noise generation complete: "
        f"planned={planned} written={written} errors={errors} "
        f"manifest={args.manifest}"
    )


if __name__ == "__main__":
    main()
