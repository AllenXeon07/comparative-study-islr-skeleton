#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
import torch

from training.config import ExperimentConfig
from training.preprocessing import FEATURE_DTYPE, load_preprocessed_component
from training.testing_pipeline import load_label_map, load_model_from_config, resolve_weights_path


DEFAULT_MANIFEST = Path("Dataset") / "robutsness_test" / "manifest.csv"
DEFAULT_VIDEO_ROOT = Path("Dataset") / "robutsness_test" / "video"
DEFAULT_SKELETON_ROOT = Path("Dataset") / "robutsness_test" / "skeleton"
DEFAULT_ARTIFACTS_DIR = Path("artifacts")
DEFAULT_OUTPUT_DIR = DEFAULT_ARTIFACTS_DIR / "robustness_reports_multiseed"
DEFAULT_SCENARIOS = ("sd", "si")
DEFAULT_MODELS = ("cnn1d", "gcn", "lstm", "transformer")
DEFAULT_FRAME_SAMPLE_COUNTS = (10, 20, 30)
DEFAULT_SEEDS = (7, 42, 128)
DEFAULT_OLD_RUN_BEFORE = "20261001T000000Z"
NEW_GRID_RUN_PATTERN = re.compile(r"_f\d+_s\d+$")


@dataclass(frozen=True, slots=True)
class RunRecord:
    scenario: str
    model: str
    frame_sample_count: int
    seed: int
    run_dir: Path
    config: dict[str, Any]
    summary: dict[str, Any]

    @property
    def key(self) -> tuple[str, str, int, int]:
        return (self.scenario, self.model, self.frame_sample_count, self.seed)


@dataclass(frozen=True, slots=True)
class RobustnessSample:
    participant_id: str
    video_file: str
    gloss: str
    asl_lex_code: str
    output_video_path: Path
    effect_variant: str
    pixel_variant: str


def parse_csv_list(raw_value: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw_value.split(",") if value.strip())
    if not values:
        raise ValueError("Expected at least one comma-separated value.")
    return values


def parse_int_list(raw_value: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw_value.split(",") if value.strip())
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_manifest(path: Path) -> list[RobustnessSample]:
    if not path.exists():
        raise FileNotFoundError(f"Robustness manifest not found: {path}")
    samples: list[RobustnessSample] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            ok_value = str(row.get("ok", "")).strip().lower()
            if ok_value not in {"true", "1", "yes"}:
                continue
            samples.append(
                RobustnessSample(
                    participant_id=(row.get("Participant ID") or row.get("participant_id") or "").strip(),
                    video_file=(row.get("Video file") or row.get("video_file") or "").strip(),
                    gloss=(row.get("Gloss") or row.get("gloss") or "").strip(),
                    asl_lex_code=(row.get("ASL-LEX Code") or row.get("asl_lex_code") or "").strip(),
                    output_video_path=Path(str(row["output_video_path"])),
                    effect_variant=str(row.get("effect_variant") or "").strip(),
                    pixel_variant=str(row.get("pixel_variant") or "").strip(),
                )
            )
    if not samples:
        raise ValueError(f"No completed samples found in robustness manifest: {path}")
    return samples


def as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def is_completed_run(run_dir: Path, summary: dict[str, Any]) -> bool:
    if not (run_dir / "best.weights.h5").exists() and not (run_dir / "final.weights.h5").exists():
        return False
    metrics = summary.get("metrics", {})
    return all(metric in metrics for metric in ("accuracy", "f1_score", "loss"))


def matches_clean_protocol(config: dict[str, Any]) -> bool:
    """Accept only runs comparable with the final three-seed grid."""
    defaults = ExperimentConfig()
    expected = {
        "modality": "hand",
        "epochs": defaults.epochs,
        "batch_size": defaults.batch_size,
        "num_workers": defaults.num_workers,
        "cache_device": defaults.cache_device,
        "train_partitions": defaults.train_partitions,
        "partition_cycles": defaults.partition_cycles,
        "validate_during_training": defaults.validate_during_training,
        "validation_fraction": defaults.validation_fraction,
        "evaluation_batch_size": defaults.evaluation_batch_size,
        "test_partitions": defaults.test_partitions,
        "optimizer": defaults.optimizer,
        "learning_rate": defaults.learning_rate,
        "partition_learning_rate": defaults.partition_learning_rate,
        "weight_decay": defaults.weight_decay,
        "global_clipnorm": defaults.global_clipnorm,
        "reduce_lr_factor": defaults.reduce_lr_factor,
        "reduce_lr_patience": defaults.reduce_lr_patience,
        "min_learning_rate": defaults.min_learning_rate,
        "early_stopping_patience": defaults.early_stopping_patience,
        "early_stopping_min_cycles": defaults.early_stopping_min_cycles,
        "early_stopping_min_delta": defaults.early_stopping_min_delta,
    }
    for key, expected_value in expected.items():
        actual_value = config.get(key)
        if isinstance(expected_value, float):
            try:
                if not np.isclose(float(actual_value), expected_value, rtol=0.0, atol=1e-12):
                    return False
            except (TypeError, ValueError):
                return False
        elif actual_value != expected_value:
            return False
    return True


def run_matches_era(run_dir: Path, run_era: str, old_run_before: str) -> bool:
    is_new_grid_run = NEW_GRID_RUN_PATTERN.search(run_dir.name) is not None
    is_current_retrain_era = run_dir.name >= old_run_before
    if run_era == "old":
        return not is_new_grid_run and not is_current_retrain_era
    if run_era == "current":
        return is_new_grid_run or is_current_retrain_era
    if run_era == "all":
        return True
    raise ValueError(f"Unknown run era: {run_era}")


def discover_runs(
    artifacts_dir: Path,
    scenarios: tuple[str, ...],
    models: tuple[str, ...],
    frame_sample_counts: tuple[int, ...],
    seeds: tuple[int, ...] | None,
    run_era: str,
    old_run_before: str,
    clean_protocol_only: bool,
) -> list[RunRecord]:
    runs: list[RunRecord] = []
    for config_path in sorted((artifacts_dir / "experiments").glob("*/*/hand/*/config.json")):
        run_dir = config_path.parent
        if not run_matches_era(run_dir, run_era, old_run_before):
            continue
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        config = read_json(config_path)
        summary = read_json(summary_path)
        if clean_protocol_only and not matches_clean_protocol(config):
            continue
        scenario = str(config.get("scenario", "")).strip()
        model = str(config.get("model", "")).strip()
        modality = str(config.get("modality", "")).strip()
        frame_sample_count = as_int(config.get("frame_sample_count"))
        seed = as_int(config.get("seed"))
        if modality != "hand":
            continue
        if scenario not in scenarios or model not in models:
            continue
        if frame_sample_count not in frame_sample_counts:
            continue
        if seeds is not None and seed not in seeds:
            continue
        if not is_completed_run(run_dir, summary):
            continue
        runs.append(
            RunRecord(
                scenario=scenario,
                model=model,
                frame_sample_count=int(frame_sample_count),
                seed=int(seed),
                run_dir=run_dir,
                config=config,
                summary=summary,
            )
        )
    return runs


def select_latest_runs(runs: list[RunRecord]) -> list[RunRecord]:
    selected: dict[tuple[str, str, int, int], RunRecord] = {}
    for run in runs:
        current = selected.get(run.key)
        if current is None or run.run_dir.name > current.run_dir.name:
            selected[run.key] = run
    return [selected[key] for key in sorted(selected)]


def select_latest_single_runs(runs: list[RunRecord]) -> list[RunRecord]:
    selected: dict[tuple[str, str, int], RunRecord] = {}
    for run in runs:
        key = (run.scenario, run.model, run.frame_sample_count)
        current = selected.get(key)
        if current is None or run.run_dir.name > current.run_dir.name:
            selected[key] = run
    return [selected[key] for key in sorted(selected)]


def build_missing_grid(
    runs: list[RunRecord],
    scenarios: tuple[str, ...],
    models: tuple[str, ...],
    frame_sample_counts: tuple[int, ...],
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    keys = {run.key for run in runs}
    rows = []
    for scenario in scenarios:
        for model in models:
            for frame_sample_count in frame_sample_counts:
                for seed in seeds:
                    key = (scenario, model, frame_sample_count, seed)
                    if key not in keys:
                        rows.append(
                            {
                                "scenario": scenario,
                                "model": model,
                                "modality": "hand",
                                "frame_sample_count": frame_sample_count,
                                "seed": seed,
                                "status": "missing_completed_run",
                            }
                        )
    return rows


def build_missing_single_grid(
    runs: list[RunRecord],
    scenarios: tuple[str, ...],
    models: tuple[str, ...],
    frame_sample_counts: tuple[int, ...],
) -> list[dict[str, Any]]:
    keys = {(run.scenario, run.model, run.frame_sample_count) for run in runs}
    rows = []
    for scenario in scenarios:
        for model in models:
            for frame_sample_count in frame_sample_counts:
                key = (scenario, model, frame_sample_count)
                if key not in keys:
                    rows.append(
                        {
                            "scenario": scenario,
                            "model": model,
                            "modality": "hand",
                            "frame_sample_count": frame_sample_count,
                            "seed": "",
                            "status": "missing_completed_run",
                        }
                    )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else ["message"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values: list[float]) -> tuple[int, float | None, float | None, str]:
    if not values:
        return 0, None, None, ""
    mean_value = statistics.fmean(values)
    sd_value = statistics.stdev(values) if len(values) > 1 else 0.0
    return len(values), mean_value, sd_value, f"{mean_value:.6f} ± {sd_value:.6f}"


def build_summary_by_frame(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["scenario"]), str(row["model"]), int(row["frame_sample_count"]))
        groups.setdefault(key, []).append(row)

    summary_rows = []
    for (scenario, model, frame_sample_count), group_rows in sorted(groups.items()):
        payload: dict[str, Any] = {
            "scenario": scenario,
            "model": model,
            "modality": "hand",
            "frame_sample_count": frame_sample_count,
            "n_runs": len(group_rows),
        }
        for metric in ("accuracy", "f1_score", "confidence", "loss"):
            values = [float(row[metric]) for row in group_rows if row.get(metric) not in (None, "")]
            count, mean_value, sd_value, formatted = mean_sd(values)
            payload[f"{metric}_n"] = count
            payload[f"{metric}_mean"] = mean_value
            payload[f"{metric}_sd"] = sd_value
            payload[f"{metric}_mean_sd"] = formatted
        summary_rows.append(payload)
    return summary_rows


def build_paper_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scenario_label = {
        "sd": "Signer-Dependent",
        "si": "Signer-Independent",
    }
    model_order = {
        "cnn1d": 0,
        "gcn": 1,
        "lstm": 2,
        "transformer": 3,
    }
    model_label = {
        "cnn1d": "CNN1D",
        "gcn": "GCN",
        "lstm": "LSTM",
        "transformer": "Transformer",
    }
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            str(row["scenario"]),
            int(row["frame_sample_count"]),
            model_order.get(str(row["model"]), 999),
        ),
    )
    return [
        {
            "Sc.": scenario_label.get(str(row["scenario"]), str(row["scenario"])),
            "Model": model_label.get(str(row["model"]), str(row["model"])),
            "Fr.": int(row["frame_sample_count"]),
            "Acc.": round(float(row["accuracy"]), 4),
            "F1": round(float(row["f1_score"]), 4),
            "Conf.": round(float(row["confidence"]), 4),
            "seed": int(row["seed"]),
            "run_dir": row["run_dir"],
        }
        for row in ordered_rows
    ]


def skeleton_path_for_sample(
    sample: RobustnessSample,
    *,
    video_root: Path,
    skeleton_root: Path,
    frame_sample_count: int,
) -> Path:
    video_path = sample.output_video_path
    if not video_path.is_absolute():
        video_path = Path("/") / video_path
    try:
        relative_path = video_path.relative_to(video_root)
    except ValueError:
        relative_path = Path(video_path.name)
    return skeleton_root / f"{frame_sample_count}frame" / "hand" / relative_path.with_suffix(".npy")


def load_batch_features(
    samples: list[RobustnessSample],
    *,
    video_root: Path,
    skeleton_root: Path,
    frame_sample_count: int,
) -> np.ndarray:
    features = [
        load_preprocessed_component(
            skeleton_path_for_sample(
                sample,
                video_root=video_root,
                skeleton_root=skeleton_root,
                frame_sample_count=frame_sample_count,
            ),
            "hand",
        )
        .reshape(frame_sample_count, -1)
        .astype(FEATURE_DTYPE, copy=False)
        for sample in samples
    ]
    return np.stack(features, axis=0).astype(FEATURE_DTYPE, copy=False)


def evaluate_run(
    run: RunRecord,
    samples: list[RobustnessSample],
    *,
    video_root: Path,
    skeleton_root: Path,
    output_dir: Path,
    batch_size: int,
) -> dict[str, Any]:
    run_output_dir = (
        output_dir
        / run.scenario
        / run.model
        / f"{run.frame_sample_count}frame"
        / f"seed{run.seed}"
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_output_dir / "predictions.csv"
    summary_path = run_output_dir / "summary.json"

    label_map = load_label_map(run.config, run.run_dir)
    labels_by_index = {index: label for label, index in label_map.items()}
    num_classes = len(label_map)
    weights_path = resolve_weights_path(run.run_dir, run.run_dir)
    model = load_model_from_config(run.config, weights_path)

    true_counts = np.zeros(num_classes, dtype=np.int64)
    predicted_counts = np.zeros(num_classes, dtype=np.int64)
    true_positives = np.zeros(num_classes, dtype=np.int64)
    total_loss = 0.0
    confidence_sum = 0.0
    total_samples = 0

    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "participant_id",
                "video_file",
                "output_video_path",
                "effect_variant",
                "pixel_variant",
                "true_label",
                "true_index",
                "predicted_label",
                "predicted_index",
                "confidence",
                "correct",
            ],
        )
        writer.writeheader()
        for start in range(0, len(samples), batch_size):
            batch_samples = samples[start : start + batch_size]
            batch_features = load_batch_features(
                batch_samples,
                video_root=video_root,
                skeleton_root=skeleton_root,
                frame_sample_count=run.frame_sample_count,
            )
            probabilities = np.asarray(model.predict_on_batch(batch_features))
            true_indices = np.asarray([label_map[sample.gloss] for sample in batch_samples], dtype=np.int64)
            predicted_indices = np.argmax(probabilities, axis=1).astype(np.int64)
            confidences = probabilities[np.arange(len(predicted_indices)), predicted_indices]
            true_probabilities = np.clip(
                probabilities[np.arange(len(true_indices)), true_indices],
                1e-12,
                1.0,
            )
            matches = true_indices == predicted_indices

            total_loss += float(-np.log(true_probabilities).sum())
            confidence_sum += float(confidences.sum())
            total_samples += len(true_indices)
            true_counts += np.bincount(true_indices, minlength=num_classes)
            predicted_counts += np.bincount(predicted_indices, minlength=num_classes)
            true_positives += np.bincount(true_indices[matches], minlength=num_classes)

            for sample, true_index, predicted_index, confidence, correct in zip(
                batch_samples,
                true_indices,
                predicted_indices,
                confidences,
                matches,
                strict=True,
            ):
                writer.writerow(
                    {
                        "participant_id": sample.participant_id,
                        "video_file": sample.video_file,
                        "output_video_path": str(sample.output_video_path),
                        "effect_variant": sample.effect_variant,
                        "pixel_variant": sample.pixel_variant,
                        "true_label": labels_by_index[int(true_index)],
                        "true_index": int(true_index),
                        "predicted_label": labels_by_index[int(predicted_index)],
                        "predicted_index": int(predicted_index),
                        "confidence": float(confidence),
                        "correct": bool(correct),
                    }
                )

    false_positives = predicted_counts - true_positives
    false_negatives = true_counts - true_positives
    precision = np.divide(
        true_positives,
        true_positives + false_positives,
        out=np.zeros(num_classes, dtype=np.float64),
        where=(true_positives + false_positives) > 0,
    )
    recall = np.divide(
        true_positives,
        true_positives + false_negatives,
        out=np.zeros(num_classes, dtype=np.float64),
        where=(true_positives + false_negatives) > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(num_classes, dtype=np.float64),
        where=(precision + recall) > 0,
    )
    correct_count = int(true_positives.sum())
    metrics = {
        "loss": total_loss / total_samples,
        "accuracy": correct_count / total_samples,
        "f1_score": float(f1.mean()),
        "confidence": confidence_sum / total_samples,
        "correct_count": correct_count,
        "incorrect_count": int(total_samples - correct_count),
        "test_size": total_samples,
    }
    payload = {
        "scenario": run.scenario,
        "model": run.model,
        "modality": "hand",
        "frame_sample_count": run.frame_sample_count,
        "seed": run.seed,
        "run_dir": str(run.run_dir),
        "weights_path": str(weights_path),
        "metrics": metrics,
        "predictions_csv": str(predictions_path),
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json(summary_path, payload)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate trained hand models on the robustness video set.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--skeleton-root", type=Path, default=DEFAULT_SKELETON_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument(
        "--frame-sample-counts",
        default=",".join(str(value) for value in DEFAULT_FRAME_SAMPLE_COUNTS),
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(value) for value in DEFAULT_SEEDS),
        help=(
            "Comma-separated training seeds to evaluate when --run-selection=seed."
        ),
    )
    parser.add_argument(
        "--run-selection",
        choices=("latest", "seed"),
        default="seed",
        help=(
            "seed evaluates the explicit --seeds list. latest selects one latest "
            "completed run per scenario/model/frame, regardless of seed."
        ),
    )
    parser.add_argument(
        "--run-era",
        choices=("old", "current", "all"),
        default="all",
        help=(
            "all allows clean runs from either era and is required for the four "
            "reused SI 30-frame seed-128 cells. old/current restrict by timestamp."
        ),
    )
    parser.add_argument(
        "--old-run-before",
        default=DEFAULT_OLD_RUN_BEFORE,
        help="Lexicographic run timestamp cutoff used by --run-era=old/current.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--clean-protocol-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require the final clean retraining protocol. Default: enabled; this "
            "excludes incompatible historical runs."
        ),
    )
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--limit-runs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scenarios = parse_csv_list(args.scenarios)
    models = parse_csv_list(args.models)
    frame_sample_counts = parse_int_list(args.frame_sample_counts)
    seeds = parse_int_list(args.seeds)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.limit_samples is not None and args.limit_samples < 1:
        raise ValueError("--limit-samples must be positive when provided.")
    if args.limit_runs is not None and args.limit_runs < 1:
        raise ValueError("--limit-runs must be positive when provided.")

    samples = read_manifest(args.manifest)
    if args.limit_samples is not None:
        samples = samples[: args.limit_samples]

    discovered_runs = discover_runs(
        artifacts_dir=args.artifacts_dir,
        scenarios=scenarios,
        models=models,
        frame_sample_counts=frame_sample_counts,
        seeds=seeds if args.run_selection == "seed" else None,
        run_era=args.run_era,
        old_run_before=args.old_run_before,
        clean_protocol_only=args.clean_protocol_only,
    )
    if args.run_selection == "seed":
        runs = select_latest_runs(discovered_runs)
        missing_grid = build_missing_grid(
            runs=runs,
            scenarios=scenarios,
            models=models,
            frame_sample_counts=frame_sample_counts,
            seeds=seeds,
        )
    else:
        runs = select_latest_single_runs(discovered_runs)
        missing_grid = build_missing_single_grid(
            runs=runs,
            scenarios=scenarios,
            models=models,
            frame_sample_counts=frame_sample_counts,
        )
    if args.limit_runs is not None:
        runs = runs[: args.limit_runs]

    expected_grid = len(scenarios) * len(models) * len(frame_sample_counts) * len(seeds)
    if args.run_selection == "latest":
        expected_grid = len(scenarios) * len(models) * len(frame_sample_counts)
    print(
        "Robustness evaluation plan: "
        f"samples={len(samples)} runs={len(runs)} expected_grid={expected_grid} "
        f"missing_grid={len(missing_grid)} "
        f"scenarios={','.join(scenarios)} models={','.join(models)} "
        f"frames={','.join(map(str, frame_sample_counts))} "
        f"run_selection={args.run_selection} run_era={args.run_era} "
        f"old_run_before={args.old_run_before} "
        f"seeds={','.join(map(str, seeds))}"
    )
    write_csv(
        args.output_dir / "missing_grid.csv",
        missing_grid,
        fieldnames=["scenario", "model", "modality", "frame_sample_count", "seed", "status"],
    )
    if args.dry_run:
        for run in runs[:20]:
            print(
                "would evaluate "
                f"{run.scenario}/{run.model}/hand "
                f"frames={run.frame_sample_count} seed={run.seed} run={run.run_dir}"
            )
        if len(runs) > 20:
            print(f"... {len(runs) - 20} more run(s)")
        return

    output_rows = []
    for index, run in enumerate(runs, start=1):
        print(
            "evaluating "
            f"{index}/{len(runs)} {run.scenario}/{run.model}/hand "
            f"frames={run.frame_sample_count} seed={run.seed}"
        )
        summary = evaluate_run(
            run,
            samples,
            video_root=args.video_root.resolve(),
            skeleton_root=args.skeleton_root.resolve(),
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )
        metrics = summary["metrics"]
        output_rows.append(
            {
                "scenario": run.scenario,
                "model": run.model,
                "modality": "hand",
                "frame_sample_count": run.frame_sample_count,
                "seed": run.seed,
                "run_dir": str(run.run_dir),
                "accuracy": metrics["accuracy"],
                "f1_score": metrics["f1_score"],
                "loss": metrics["loss"],
                "confidence": metrics["confidence"],
                "correct_count": metrics["correct_count"],
                "incorrect_count": metrics["incorrect_count"],
                "test_size": metrics["test_size"],
                "summary_json": str(
                    args.output_dir
                    / run.scenario
                    / run.model
                    / f"{run.frame_sample_count}frame"
                    / f"seed{run.seed}"
                    / "summary.json"
                ),
            }
        )

    summary_csv = args.output_dir / "robustness_summary.csv"
    write_csv(summary_csv, output_rows)
    write_csv(
        args.output_dir / "robustness_summary_by_frame.csv",
        build_summary_by_frame(output_rows),
    )
    write_csv(
        args.output_dir / "paper_table.csv",
        build_paper_table(output_rows),
    )
    print(f"Robustness evaluation complete: {summary_csv}")


if __name__ == "__main__":
    main()
