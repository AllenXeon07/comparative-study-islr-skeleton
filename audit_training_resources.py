from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.config import ALLOWED_MODELS, ALLOWED_SCENARIOS, ExperimentConfig

try:
    import pandas as pd
except ImportError:  # pragma: no cover - exercised only outside the Docker image.
    pd = None


SCENARIOS = ALLOWED_SCENARIOS
MODELS = ALLOWED_MODELS
FRAME_SAMPLE_COUNTS = (10, 20, 30)
SEEDS = (7, 42, 86, 128, 251)
HAND_FEATURE_COUNT = 210
FLOAT32_BYTES = 4
INT64_BYTES = 8
MODEL_WORKERS = {
    "cnn1d": 4,
    "gcn": 3,
    "lstm": 1,
    "transformer": 1,
}


@dataclass(frozen=True, slots=True)
class SplitStats:
    scenario: str
    train_size: int
    test_size: int
    train_participants: int
    test_participants: int
    train_classes: int
    test_classes: int


def _parse_requested_values(
    raw_value: str,
    allowed_values: tuple[str, ...],
    option_name: str,
) -> tuple[str, ...]:
    if raw_value.strip().lower() == "all":
        return allowed_values
    requested_values = tuple(
        value.strip()
        for value in raw_value.split(",")
        if value.strip()
    )
    invalid_values = sorted(set(requested_values) - set(allowed_values))
    if invalid_values:
        raise argparse.ArgumentTypeError(
            f"Invalid {option_name}: {', '.join(invalid_values)}."
        )
    if not requested_values:
        raise argparse.ArgumentTypeError(f"{option_name} cannot be empty.")
    return requested_values


def _parse_requested_ints(
    raw_value: str,
    allowed_values: tuple[int, ...],
    option_name: str,
) -> tuple[int, ...]:
    if raw_value.strip().lower() == "all":
        return allowed_values
    try:
        requested_values = tuple(
            int(value.strip())
            for value in raw_value.split(",")
            if value.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{option_name} must be a comma-separated integer list, or 'all'."
        ) from exc
    invalid_values = sorted(set(requested_values) - set(allowed_values))
    if invalid_values:
        invalid = ", ".join(str(value) for value in invalid_values)
        raise argparse.ArgumentTypeError(f"Invalid {option_name}: {invalid}.")
    if not requested_values:
        raise argparse.ArgumentTypeError(f"{option_name} cannot be empty.")
    return requested_values


def _read_split(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _split_stats(dataset_dir: Path, scenario: str) -> SplitStats:
    train_rows = _read_split(dataset_dir / "split" / f"{scenario}_train.csv")
    test_rows = _read_split(dataset_dir / "split" / f"{scenario}_test.csv")
    return SplitStats(
        scenario=scenario,
        train_size=len(train_rows),
        test_size=len(test_rows),
        train_participants=len({row.get("Participant ID", "") for row in train_rows}),
        test_participants=len({row.get("Participant ID", "") for row in test_rows}),
        train_classes=len({row.get("Gloss", "") for row in train_rows}),
        test_classes=len({row.get("Gloss", "") for row in test_rows}),
    )


def _bytes_to_mib(value: float) -> float:
    return value / 1024**2


def _try_gpu_info() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return {
            "gpu_name": None,
            "gpu_total_mib": None,
            "gpu_free_mib": None,
            "gpu_used_mib": None,
        }
    first_line = result.stdout.strip().splitlines()[0]
    name, total, free, used = [part.strip() for part in first_line.split(",")]
    return {
        "gpu_name": name,
        "gpu_total_mib": int(total),
        "gpu_free_mib": int(free),
        "gpu_used_mib": int(used),
    }


def _parameter_count(model: str, feature_count: int, num_classes: int) -> int:
    if model == "cnn1d":
        conv1 = 3 * feature_count * 128 + 128
        conv2 = 3 * 128 * 128 + 128
        dense = 128 * 128 + 128
        output = 128 * num_classes + num_classes
        return conv1 + conv2 + dense + output
    if model == "lstm":
        # Keras LSTM params: 4 * units * (input_dim + units + 1), doubled for BiLSTM.
        bilstm1 = 2 * 4 * 128 * (feature_count + 128 + 1)
        bilstm2 = 2 * 4 * 64 * (256 + 64 + 1)
        dense = 128 * 128 + 128
        output = 128 * num_classes + num_classes
        return bilstm1 + bilstm2 + dense + output
    if model == "transformer":
        input_dense = feature_count * 128 + 128
        block = 4 * (128 * 4 * 128 + 4 * 128) + 2 * (128 * 256 + 256 + 256 * 128 + 128) + 4 * 128
        output = 128 * num_classes + num_classes
        return input_dense + (2 * block) + output
    if model == "gcn":
        node_count = feature_count // 5
        graph1 = node_count * node_count + 5 * 64 + 64
        norm1 = 2 * 64
        graph2 = node_count * node_count + 64 * 128 + 128
        norm2 = 2 * 128
        dense = 128 * 128 + 128
        output = 128 * num_classes + num_classes
        return graph1 + norm1 + graph2 + norm2 + dense + output
    raise ValueError(f"Unsupported model: {model}")


def _estimated_training_vram_mib(
    model: str,
    frame_sample_count: int,
    batch_size: int,
    parameter_count: int,
) -> float:
    parameter_state_mib = _bytes_to_mib(parameter_count * FLOAT32_BYTES * 4)
    batch_feature_mib = _bytes_to_mib(batch_size * frame_sample_count * HAND_FEATURE_COUNT * FLOAT32_BYTES)
    activation_multiplier = {
        "cnn1d": 20,
        "gcn": 28,
        "lstm": 90,
        "transformer": 120,
    }[model]
    framework_overhead_mib = {
        "cnn1d": 900,
        "gcn": 1100,
        "lstm": 1900,
        "transformer": 2400,
    }[model]
    return framework_overhead_mib + parameter_state_mib + batch_feature_mib * activation_multiplier


def build_audit_rows(
    dataset_dir: Path,
    scenarios: tuple[str, ...],
    models: tuple[str, ...],
    frame_sample_counts: tuple[int, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    defaults = ExperimentConfig()
    gpu_info = _try_gpu_info()
    rows = []
    for scenario in scenarios:
        split = _split_stats(dataset_dir, scenario)
        largest_partition_size = math.ceil(split.train_size / defaults.train_partitions)
        validation_size = round(split.train_size * defaults.validation_fraction)
        for frame_sample_count in frame_sample_counts:
            sample_mib = _bytes_to_mib(frame_sample_count * HAND_FEATURE_COUNT * FLOAT32_BYTES)
            batch_mib = _bytes_to_mib(defaults.batch_size * frame_sample_count * HAND_FEATURE_COUNT * FLOAT32_BYTES)
            partition_cache_mib = _bytes_to_mib(
                largest_partition_size
                * (frame_sample_count * HAND_FEATURE_COUNT * FLOAT32_BYTES + INT64_BYTES)
            )
            validation_cache_mib = _bytes_to_mib(
                validation_size
                * (frame_sample_count * HAND_FEATURE_COUNT * FLOAT32_BYTES + INT64_BYTES)
            )
            for model in models:
                parameter_count = _parameter_count(model, HAND_FEATURE_COUNT, split.train_classes)
                estimated_vram_mib = _estimated_training_vram_mib(
                    model,
                    frame_sample_count,
                    defaults.batch_size,
                    parameter_count,
                )
                rows.append({
                    **gpu_info,
                    "scenario": scenario,
                    "model": model,
                    "frame_sample_count": frame_sample_count,
                    "train_size": split.train_size,
                    "test_size": split.test_size,
                    "train_classes": split.train_classes,
                    "train_participants": split.train_participants,
                    "test_participants": split.test_participants,
                    "batch_size": defaults.batch_size,
                    "train_partitions": defaults.train_partitions,
                    "largest_partition_size": largest_partition_size,
                    "validation_size_estimate": validation_size,
                    "feature_count": HAND_FEATURE_COUNT,
                    "sample_mib": round(sample_mib, 4),
                    "batch_feature_mib": round(batch_mib, 4),
                    "partition_cache_cpu_mib": round(partition_cache_mib, 2),
                    "validation_cache_cpu_mib": round(validation_cache_mib, 2),
                    "parameter_count_estimate": parameter_count,
                    "parameter_state_mib_estimate": round(_bytes_to_mib(parameter_count * FLOAT32_BYTES * 4), 2),
                    "estimated_training_vram_mib": round(estimated_vram_mib, 2),
                    "recommended_model_workers": MODEL_WORKERS[model],
                    "recommended_note": _recommendation(model),
                })
    schedule_rows = [
        {
            "model": model,
            "recommended_workers": worker_count,
            "reason": _recommendation(model),
        }
        for model, worker_count in MODEL_WORKERS.items()
    ]
    return rows, schedule_rows, gpu_info


def _recommendation(model: str) -> str:
    if model == "cnn1d":
        return "Small activation footprint; run up to 4 concurrently on 16 GB VRAM."
    if model == "gcn":
        return "Moderate graph ops; run up to 3 concurrently on 16 GB VRAM."
    if model == "lstm":
        return "RNN activations/workspace can spike; run one at a time."
    if model == "transformer":
        return "Attention/workspace is most sensitive; run one at a time."
    return ""


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_excel(path: Path, sheets: dict[str, list[dict[str, Any]]]) -> bool:
    if pd is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for sheet_name, rows in sheets.items():
                dataframe = pd.DataFrame(rows)
                if dataframe.empty:
                    dataframe = pd.DataFrame([{"message": "No rows"}])
                dataframe.to_excel(writer, sheet_name=sheet_name[:31], index=False)
                worksheet = writer.sheets[sheet_name[:31]]
                for column_index, column_name in enumerate(dataframe.columns, start=1):
                    values = dataframe[column_name].astype(str).tolist()
                    max_length = max([len(str(column_name)), *(len(value) for value in values)])
                    worksheet.column_dimensions[
                        worksheet.cell(row=1, column=column_index).column_letter
                    ].width = min(max_length + 2, 60)
    except ImportError:
        return False
    except ModuleNotFoundError:
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit estimated memory and recommended parallel schedule for hand-only training."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("Dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/resource_audits"))
    parser.add_argument("--scenarios", default="all")
    parser.add_argument("--models", default="all")
    parser.add_argument(
        "--frame-sample-counts",
        default=FRAME_SAMPLE_COUNTS,
        type=lambda value: _parse_requested_ints(value, FRAME_SAMPLE_COUNTS, "--frame-sample-counts"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scenarios = _parse_requested_values(args.scenarios, SCENARIOS, "--scenarios")
    models = _parse_requested_values(args.models, MODELS, "--models")
    rows, schedule_rows, gpu_info = build_audit_rows(
        dataset_dir=args.dataset_dir,
        scenarios=scenarios,
        models=models,
        frame_sample_counts=args.frame_sample_counts,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / "hand_training_resource_audit.csv"
    schedule_path = args.output_dir / "hand_training_schedule_recommendation.csv"
    excel_path = args.output_dir / "hand_training_resource_audit.xlsx"
    _write_csv(audit_path, rows)
    _write_csv(schedule_path, schedule_rows)
    wrote_excel = _write_excel(
        excel_path,
        {
            "resource_audit": rows,
            "schedule": schedule_rows,
        },
    )
    print(json.dumps(gpu_info, indent=2, sort_keys=True))
    print(f"Wrote audit: {audit_path}")
    print(f"Wrote schedule: {schedule_path}")
    if wrote_excel:
        print(f"Wrote Excel workbook: {excel_path}")
    else:
        print("Skipped Excel workbook because pandas/openpyxl is not installed.")
    print("Recommended launcher: python run_hand_all_experiments.py --schedule-profile balanced --max-workers 4")


if __name__ == "__main__":
    main()
