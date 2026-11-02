from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from training.config import ALLOWED_MODELS, ALLOWED_SCENARIOS, ExperimentConfig

try:
    import pandas as pd
except ImportError:  # pragma: no cover - exercised only outside the Docker image.
    pd = None

try:
    from scipy import stats
except ImportError:  # pragma: no cover - exercised only outside the Docker image.
    stats = None


SCENARIOS = ALLOWED_SCENARIOS
MODELS = ALLOWED_MODELS
MODALITY = "hand"
# Keep the final seed report aligned with the retraining grid.
SEEDS = (7, 42, 128)
FRAME_SAMPLE_COUNTS = (10, 20, 30)
METRICS = ("accuracy", "f1_score", "loss")
SHEET_FIELDNAMES = {
    "missing_grid": [
        "scenario",
        "frame_sample_count",
        "model",
        "seed",
        "status",
    ],
}


@dataclass(slots=True)
class RunRecord:
    run_dir: str
    timestamp: str
    scenario: str
    model: str
    modality: str
    frame_sample_count: int | None
    seed: int | None
    is_clean_protocol: bool
    clean_protocol_notes: str
    accuracy: float | None
    f1_score: float | None
    loss: float | None
    config: dict[str, Any]

    def key(self) -> tuple[str, str, int, int] | None:
        if self.frame_sample_count is None or self.seed is None:
            return None
        return (self.scenario, self.model, self.frame_sample_count, self.seed)

    def to_row(self) -> dict[str, Any]:
        defaults = ExperimentConfig()
        return {
            "run_dir": self.run_dir,
            "timestamp": self.timestamp,
            "scenario": self.scenario,
            "model": self.model,
            "modality": self.modality,
            "frame_sample_count": self.frame_sample_count,
            "seed": self.seed,
            "is_clean_protocol": self.is_clean_protocol,
            "clean_protocol_notes": self.clean_protocol_notes,
            "accuracy": self.accuracy,
            "f1_score": self.f1_score,
            "loss": self.loss,
            "epochs": self.config.get("epochs"),
            "batch_size": self.config.get("batch_size"),
            "train_partitions": self.config.get("train_partitions"),
            "partition_cycles": self.config.get("partition_cycles"),
            "validation_fraction": self.config.get("validation_fraction"),
            "optimizer": self.config.get("optimizer"),
            "learning_rate": self.config.get("learning_rate"),
            "partition_learning_rate": self.config.get("partition_learning_rate"),
            "clean_epochs": defaults.epochs,
            "clean_batch_size": defaults.batch_size,
            "clean_train_partitions": defaults.train_partitions,
            "clean_partition_cycles": defaults.partition_cycles,
        }


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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def _matches_clean_protocol(config: dict[str, Any]) -> tuple[bool, str]:
    defaults = ExperimentConfig()
    expected = {
        "modality": MODALITY,
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
    mismatches = []
    for key, expected_value in expected.items():
        actual_value = config.get(key)
        if isinstance(expected_value, float):
            try:
                if not math.isclose(float(actual_value), expected_value, rel_tol=0.0, abs_tol=1e-12):
                    mismatches.append(f"{key}={actual_value!r} expected {expected_value!r}")
            except (TypeError, ValueError):
                mismatches.append(f"{key}={actual_value!r} expected {expected_value!r}")
        elif actual_value != expected_value:
            mismatches.append(f"{key}={actual_value!r} expected {expected_value!r}")
    if mismatches:
        return False, "; ".join(mismatches)
    return True, "clean protocol"


def discover_runs(
    artifacts_dir: Path,
    scenarios: tuple[str, ...],
    models: tuple[str, ...],
    frame_sample_counts: tuple[int, ...],
    seeds: tuple[int, ...],
) -> list[RunRecord]:
    runs: list[RunRecord] = []
    experiments_dir = artifacts_dir / "experiments"
    for config_path in sorted(experiments_dir.glob("*/*/hand/*/config.json")):
        run_dir = config_path.parent
        config = _load_json(config_path)
        scenario = str(config.get("scenario", ""))
        model = str(config.get("model", ""))
        modality = str(config.get("modality", ""))
        frame_sample_count = _as_int(config.get("frame_sample_count"))
        seed = _as_int(config.get("seed"))
        if scenario not in scenarios or model not in models or modality != MODALITY:
            continue
        if frame_sample_count not in frame_sample_counts or seed not in seeds:
            continue

        summary_path = run_dir / "summary.json"
        metrics_path = run_dir / "metrics.json"
        summary = _load_json(summary_path) if summary_path.exists() else {}
        metrics = _load_json(metrics_path) if metrics_path.exists() else summary.get("metrics", {})
        is_clean, notes = _matches_clean_protocol(config)
        runs.append(
            RunRecord(
                run_dir=str(run_dir),
                timestamp=run_dir.name,
                scenario=scenario,
                model=model,
                modality=modality,
                frame_sample_count=frame_sample_count,
                seed=seed,
                is_clean_protocol=is_clean,
                clean_protocol_notes=notes,
                accuracy=_as_float(metrics.get("accuracy")),
                f1_score=_as_float(metrics.get("f1_score")),
                loss=_as_float(metrics.get("loss")),
                config=config,
            )
        )
    return runs


def select_latest_clean_runs(runs: list[RunRecord]) -> dict[tuple[str, str, int, int], RunRecord]:
    selected: dict[tuple[str, str, int, int], RunRecord] = {}
    for run in sorted(runs, key=lambda item: item.timestamp):
        key = run.key()
        if key is None or not run.is_clean_protocol:
            continue
        selected[key] = run
    return selected


def build_missing_grid(
    selected: dict[tuple[str, str, int, int], RunRecord],
    scenarios: tuple[str, ...],
    models: tuple[str, ...],
    frame_sample_counts: tuple[int, ...],
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows = []
    for scenario in scenarios:
        for frame_sample_count in frame_sample_counts:
            for model in models:
                for seed in seeds:
                    key = (scenario, model, frame_sample_count, seed)
                    if key not in selected:
                        rows.append({
                            "scenario": scenario,
                            "frame_sample_count": frame_sample_count,
                            "model": model,
                            "seed": seed,
                            "status": "missing_clean_run",
                        })
    return rows


def _metric_values(rows: list[RunRecord], metric: str) -> list[float]:
    values = []
    for row in rows:
        value = getattr(row, metric)
        if value is not None:
            values.append(value)
    return values


def _mean_sd_payload(values: list[float]) -> tuple[int, float | None, float | None, str]:
    if not values:
        return 0, None, None, ""
    value_mean = mean(values)
    value_std = stdev(values) if len(values) > 1 else 0.0
    return len(values), value_mean, value_std, f"{value_mean:.4f} +/- {value_std:.4f}"


def build_summary_rows(
    selected: dict[tuple[str, str, int, int], RunRecord],
    scenarios: tuple[str, ...],
    models: tuple[str, ...],
    frame_sample_counts: tuple[int, ...],
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows = []
    for scenario in scenarios:
        for frame_sample_count in frame_sample_counts:
            for model in models:
                records = [
                    selected[key]
                    for seed in seeds
                    if (key := (scenario, model, frame_sample_count, seed)) in selected
                ]
                rows.append(_build_summary_row("by_frame", scenario, model, frame_sample_count, records))

    for scenario in scenarios:
        for model in models:
            records = [
                selected[key]
                for frame_sample_count in frame_sample_counts
                for seed in seeds
                if (key := (scenario, model, frame_sample_count, seed)) in selected
            ]
            rows.append(_build_summary_row("across_frames", scenario, model, "all", records))
    return rows


def _build_summary_row(
    summary_scope: str,
    scenario: str,
    model: str,
    frame_sample_count: int | str,
    records: list[RunRecord],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "summary_scope": summary_scope,
        "scenario": scenario,
        "frame_sample_count": frame_sample_count,
        "model": model,
        "run_count": len(records),
    }
    for metric in METRICS:
        count, value_mean, value_std, formatted = _mean_sd_payload(_metric_values(records, metric))
        row[f"{metric}_n"] = count
        row[f"{metric}_mean"] = value_mean
        row[f"{metric}_std"] = value_std
        row[f"{metric}_mean_pm_std"] = formatted
    return row


def build_frame_seed_pivots(
    selected: dict[tuple[str, str, int, int], RunRecord],
    scenarios: tuple[str, ...],
    models: tuple[str, ...],
    frame_sample_counts: tuple[int, ...],
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows = []
    for scenario in scenarios:
        for frame_sample_count in frame_sample_counts:
            for seed in seeds:
                for metric in METRICS:
                    row: dict[str, Any] = {
                        "scenario": scenario,
                        "frame_sample_count": frame_sample_count,
                        "seed": seed,
                        "metric": metric,
                    }
                    for model in models:
                        record = selected.get((scenario, model, frame_sample_count, seed))
                        row[model] = getattr(record, metric) if record is not None else None
                    rows.append(row)
    return rows


def build_best_by_frame(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_rows = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in summary_rows:
        if row["summary_scope"] != "by_frame":
            continue
        grouped.setdefault((row["scenario"], row["model"]), []).append(row)
    for (scenario, model), rows in sorted(grouped.items()):
        complete_rows = [
            row for row in rows
            if row.get("f1_score_mean") is not None
        ]
        if not complete_rows:
            continue
        best = max(complete_rows, key=lambda row: row["f1_score_mean"])
        best_rows.append({
            "scenario": scenario,
            "model": model,
            "best_frame_sample_count": best["frame_sample_count"],
            "best_f1_score_mean": best["f1_score_mean"],
            "best_f1_score_std": best["f1_score_std"],
            "best_accuracy_mean": best["accuracy_mean"],
            "best_accuracy_std": best["accuracy_std"],
            "run_count": best["run_count"],
        })
    return best_rows


def _cohens_dz(differences: list[float]) -> float | None:
    if len(differences) < 2:
        return None
    diff_std = stdev(differences)
    if diff_std == 0:
        return None
    return mean(differences) / diff_std


def _pvalue(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def build_pairwise_stats(
    selected: dict[tuple[str, str, int, int], RunRecord],
    scenarios: tuple[str, ...],
    models: tuple[str, ...],
    frame_sample_counts: tuple[int, ...],
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows = []
    for scenario in scenarios:
        for frame_sample_count in frame_sample_counts:
            for metric in METRICS:
                for model_a, model_b in combinations(models, 2):
                    paired = []
                    for seed in seeds:
                        run_a = selected.get((scenario, model_a, frame_sample_count, seed))
                        run_b = selected.get((scenario, model_b, frame_sample_count, seed))
                        if run_a is None or run_b is None:
                            continue
                        value_a = getattr(run_a, metric)
                        value_b = getattr(run_b, metric)
                        if value_a is None or value_b is None:
                            continue
                        paired.append((seed, value_a, value_b))
                    differences = [value_a - value_b for _, value_a, value_b in paired]
                    t_p = None
                    wilcoxon_p = None
                    if stats is not None and len(differences) >= 2:
                        t_p = _pvalue(stats.ttest_rel(
                            [value_a for _, value_a, _ in paired],
                            [value_b for _, _, value_b in paired],
                            nan_policy="omit",
                        ).pvalue)
                        if any(difference != 0 for difference in differences):
                            try:
                                wilcoxon_p = _pvalue(stats.wilcoxon(differences).pvalue)
                            except ValueError:
                                wilcoxon_p = None
                    rows.append({
                        "scenario": scenario,
                        "frame_sample_count": frame_sample_count,
                        "metric": metric,
                        "model_a": model_a,
                        "model_b": model_b,
                        "n": len(paired),
                        "seeds": ",".join(str(seed) for seed, _, _ in paired),
                        "model_a_mean": mean([value_a for _, value_a, _ in paired]) if paired else None,
                        "model_b_mean": mean([value_b for _, _, value_b in paired]) if paired else None,
                        "mean_difference_a_minus_b": mean(differences) if differences else None,
                        "cohens_dz": _cohens_dz(differences),
                        "paired_t_p": t_p,
                        "wilcoxon_p": wilcoxon_p,
                    })
    _add_holm_adjusted_pvalues(rows, "paired_t_p", "paired_t_holm_p")
    _add_holm_adjusted_pvalues(rows, "wilcoxon_p", "wilcoxon_holm_p")
    return rows


def _add_holm_adjusted_pvalues(
    rows: list[dict[str, Any]],
    source_column: str,
    target_column: str,
) -> None:
    for row in rows:
        row[target_column] = None
    grouped: dict[tuple[str, int, str], list[tuple[int, float]]] = {}
    for index, row in enumerate(rows):
        pvalue = row.get(source_column)
        if pvalue is None:
            continue
        key = (row["scenario"], row["frame_sample_count"], row["metric"])
        grouped.setdefault(key, []).append((index, pvalue))
    for values in grouped.values():
        sorted_values = sorted(values, key=lambda item: item[1])
        total = len(sorted_values)
        running_max = 0.0
        for rank, (index, pvalue) in enumerate(sorted_values):
            adjusted = min(1.0, (total - rank) * pvalue)
            running_max = max(running_max, adjusted)
            rows[index][target_column] = running_max


def build_notes(
    selected: dict[tuple[str, str, int, int], RunRecord],
    missing_grid: list[dict[str, Any]],
    scenarios: tuple[str, ...],
    models: tuple[str, ...],
    frame_sample_counts: tuple[int, ...],
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    defaults = ExperimentConfig()
    return [
        {"item": "generated_at_utc", "value": datetime.now(UTC).isoformat()},
        {"item": "scenarios", "value": ",".join(scenarios)},
        {"item": "models", "value": ",".join(models)},
        {"item": "modality", "value": MODALITY},
        {"item": "frame_sample_counts", "value": ",".join(str(value) for value in frame_sample_counts)},
        {"item": "seeds", "value": ",".join(str(value) for value in seeds)},
        {"item": "selected_run_count", "value": len(selected)},
        {"item": "missing_grid_count", "value": len(missing_grid)},
        {"item": "clean_epochs", "value": defaults.epochs},
        {"item": "clean_batch_size", "value": defaults.batch_size},
        {"item": "clean_train_partitions", "value": defaults.train_partitions},
        {"item": "clean_partition_cycles", "value": defaults.partition_cycles},
        {"item": "clean_optimizer", "value": defaults.optimizer},
        {
            "item": "statistics_note",
            "value": (
                "Paired tests are exploratory because "
                f"n={len(seeds)} matched seeds."
            ),
        },
    ]


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fieldnames or sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            writer = csv.DictWriter(handle, fieldnames=["message"])
            writer.writeheader()
            writer.writerow({"message": "No rows"})
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _rows_to_dataframe(
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
):
    fieldnames = fieldnames or sorted({field for row in rows for field in row})
    if not fieldnames:
        return pd.DataFrame([{"message": "No rows"}])
    return pd.DataFrame(rows, columns=fieldnames)


def _write_excel(path: Path, sheets: dict[str, list[dict[str, Any]]]) -> bool:
    if pd is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for sheet_name, rows in sheets.items():
                dataframe = _rows_to_dataframe(
                    rows,
                    SHEET_FIELDNAMES.get(sheet_name),
                )
                dataframe.to_excel(writer, sheet_name=sheet_name[:31], index=False)
                worksheet = writer.sheets[sheet_name[:31]]
                for column_index, column_name in enumerate(dataframe.columns, start=1):
                    values = dataframe[column_name].astype(str).tolist()
                    max_length = max(
                        [len(str(column_name)), *(len(str(value)) for value in values)]
                    )
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
        description="Compile hand-only multi-seed, multi-frame experiment reports."
    )
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/seed_reports"))
    parser.add_argument(
        "--scenarios",
        default="all",
        help="Comma-separated scenario list, or 'all'. Default: all.",
    )
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model list, or 'all'. Default: all.",
    )
    parser.add_argument(
        "--seeds",
        default=SEEDS,
        type=lambda value: _parse_requested_ints(value, SEEDS, "--seeds"),
    )
    parser.add_argument(
        "--frame-sample-counts",
        default=FRAME_SAMPLE_COUNTS,
        type=lambda value: _parse_requested_ints(
            value,
            FRAME_SAMPLE_COUNTS,
            "--frame-sample-counts",
        ),
    )
    parser.add_argument("--prefix", default="hand_seed_frame_report")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scenarios = _parse_requested_values(args.scenarios, SCENARIOS, "--scenarios")
    models = _parse_requested_values(args.models, MODELS, "--models")
    seeds = args.seeds
    frame_sample_counts = args.frame_sample_counts

    runs = discover_runs(
        artifacts_dir=args.artifacts_dir,
        scenarios=scenarios,
        models=models,
        frame_sample_counts=frame_sample_counts,
        seeds=seeds,
    )
    selected = select_latest_clean_runs(runs)
    missing_grid = build_missing_grid(
        selected=selected,
        scenarios=scenarios,
        models=models,
        frame_sample_counts=frame_sample_counts,
        seeds=seeds,
    )
    selected_rows = [
        selected[key].to_row()
        for key in sorted(selected)
    ]
    inventory_rows = [run.to_row() for run in sorted(runs, key=lambda item: item.run_dir)]
    summary_rows = build_summary_rows(selected, scenarios, models, frame_sample_counts, seeds)
    pivot_rows = build_frame_seed_pivots(selected, scenarios, models, frame_sample_counts, seeds)
    pairwise_rows = build_pairwise_stats(selected, scenarios, models, frame_sample_counts, seeds)
    best_rows = build_best_by_frame(summary_rows)
    note_rows = build_notes(selected, missing_grid, scenarios, models, frame_sample_counts, seeds)

    sheets = {
        "run_inventory": inventory_rows,
        "selected_runs": selected_rows,
        "missing_grid": missing_grid,
        "summary_mean_sd": summary_rows,
        "frame_seed_pivots": pivot_rows,
        "pairwise_stats": pairwise_rows,
        "best_by_frame": best_rows,
        "notes": note_rows,
    }
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_stem = f"{args.prefix}_{timestamp}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sheet_name, rows in sheets.items():
        _write_csv(
            args.output_dir / f"{output_stem}_{sheet_name}.csv",
            rows,
            SHEET_FIELDNAMES.get(sheet_name),
        )
    excel_path = args.output_dir / f"{output_stem}.xlsx"
    wrote_excel = _write_excel(excel_path, sheets)

    print(f"Discovered hand runs: {len(inventory_rows)}")
    print(f"Selected clean runs: {len(selected_rows)}")
    print(f"Missing clean grid cells: {len(missing_grid)}")
    print(f"Wrote CSV reports to: {args.output_dir}")
    if wrote_excel:
        print(f"Wrote Excel workbook: {excel_path}")
    else:
        print("Skipped Excel workbook because pandas/openpyxl is not installed.")


if __name__ == "__main__":
    main()
