#!/usr/bin/env python3
"""Benchmark model-only latency for the final hand-only three-seed grid."""

from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
import pandas as pd
import torch

from evaluate_robustness import discover_runs, select_latest_runs
from training.data import load_sample_features, load_split_records
from training.testing_pipeline import load_model_from_config, resolve_weights_path


SCENARIOS = ("sd", "si")
MODELS = ("cnn1d", "gcn", "lstm", "transformer")
FRAME_COUNTS = (10, 20, 30)
SEEDS = (7, 42, 128)


def parse_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return values


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_run(
    run,
    feature_batch: np.ndarray,
    *,
    warmup: int,
    repetitions: int,
) -> dict[str, object]:
    weights_path = resolve_weights_path(run.run_dir, run.run_dir)
    model = load_model_from_config(run.config, weights_path)
    for _ in range(warmup):
        model.predict_on_batch(feature_batch)
    synchronize_cuda()

    durations_ms: list[float] = []
    for _ in range(repetitions):
        synchronize_cuda()
        started = time.perf_counter_ns()
        model.predict_on_batch(feature_batch)
        synchronize_cuda()
        durations_ms.append((time.perf_counter_ns() - started) / 1_000_000)

    result = {
        "scenario": run.scenario,
        "model": run.model,
        "modality": "hand",
        "frame_sample_count": run.frame_sample_count,
        "seed": run.seed,
        "run_dir": str(run.run_dir),
        "weights_path": str(weights_path),
        "batch_size": 1,
        "warmup_runs": warmup,
        "timed_runs": repetitions,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "mean_time_ms": float(np.mean(durations_ms)),
        "std_time_ms": float(np.std(durations_ms, ddof=1)) if len(durations_ms) > 1 else 0.0,
        "min_time_ms": float(np.min(durations_ms)),
        "max_time_ms": float(np.max(durations_ms)),
    }
    del model
    keras.backend.clear_session()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark batch-size-one model inference for the final three-seed grid."
    )
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/inference_reports"))
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--frame-sample-counts", default="10,20,30", type=parse_ints)
    parser.add_argument("--seeds", default="7,42,128", type=parse_ints)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--limit-runs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scenarios = tuple(item.strip() for item in args.scenarios.split(",") if item.strip())
    models = tuple(item.strip() for item in args.models.split(",") if item.strip())
    if args.warmup < 0 or args.repetitions < 1:
        raise ValueError("--warmup must be non-negative and --repetitions must be positive.")

    discovered = discover_runs(
        artifacts_dir=args.artifacts_dir,
        scenarios=scenarios,
        models=models,
        frame_sample_counts=args.frame_sample_counts,
        seeds=args.seeds,
        run_era="all",
        old_run_before="20261001T000000Z",
        clean_protocol_only=True,
    )
    runs = select_latest_runs(discovered)
    expected = len(scenarios) * len(models) * len(args.frame_sample_counts) * len(args.seeds)
    if len(runs) != expected:
        raise RuntimeError(f"Expected {expected} clean completed runs, found {len(runs)}.")
    if args.limit_runs is not None:
        runs = runs[: args.limit_runs]

    print(
        f"Inference benchmark plan: runs={len(runs)} batch_size=1 warmup={args.warmup} "
        f"repetitions={args.repetitions} device={'cuda' if torch.cuda.is_available() else 'cpu'}"
    )
    if args.dry_run:
        return

    feature_cache: dict[tuple[str, int], tuple[np.ndarray, str]] = {}
    rows: list[dict[str, object]] = []
    for index, run in enumerate(runs, start=1):
        cache_key = (run.scenario, run.frame_sample_count)
        if cache_key not in feature_cache:
            dataset_dir = Path(str(run.config["dataset_dir"]))
            record = load_split_records(dataset_dir, run.scenario, "test")[0]
            features = load_sample_features(
                dataset_dir,
                record,
                "hand",
                run.frame_sample_count,
            )
            feature_cache[cache_key] = (np.expand_dims(features, axis=0), record.video_file)
        feature_batch, video_file = feature_cache[cache_key]
        print(
            f"benchmarking {index}/{len(runs)} {run.scenario}/{run.model} "
            f"frames={run.frame_sample_count} seed={run.seed}"
        )
        row = benchmark_run(
            run,
            feature_batch,
            warmup=args.warmup,
            repetitions=args.repetitions,
        )
        row["test_video_file"] = video_file
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_path = args.output_dir / "inference_latency_by_seed.csv"
    summary_path = args.output_dir / "inference_latency_summary.csv"
    pd.DataFrame(rows).to_csv(seed_path, index=False)
    summary = (
        pd.DataFrame(rows)
        .groupby(["scenario", "model", "modality", "frame_sample_count"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_time_ms=("mean_time_ms", "mean"),
            sd_time_ms=("mean_time_ms", "std"),
            min_time_ms=("min_time_ms", "min"),
            max_time_ms=("max_time_ms", "max"),
        )
    )
    summary["mean_time_ms_pm_sd"] = summary.apply(
        lambda row: f"{row['mean_time_ms']:.3f} +/- {row['sd_time_ms']:.3f}",
        axis=1,
    )
    summary.to_csv(summary_path, index=False)
    metadata = args.output_dir / "inference_latency_notes.txt"
    metadata.write_text(
        "Model-only batch-size-one latency. Each model uses one preprocessed held-out "
        "test input, with disk loading and feature extraction excluded. CUDA is synchronized "
        "before and after every timed prediction. Mean +/- SD aggregates the three seed means.\n"
        f"generated_at_utc={datetime.now(UTC).isoformat()}\n",
        encoding="utf-8",
    )
    print(f"Wrote {seed_path} and {summary_path}")


if __name__ == "__main__":
    main()
