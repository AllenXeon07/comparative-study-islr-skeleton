from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from training.config import (
    ALLOWED_CACHE_DEVICES,
    ALLOWED_MODELS,
    ALLOWED_OPTIMIZERS,
    ALLOWED_SCENARIOS,
    ExperimentConfig,
)


SCENARIOS = ALLOWED_SCENARIOS
MODEL_ORDER = ("cnn1d", "gcn", "lstm", "transformer")
MODELS = MODEL_ORDER
MODALITIES = ("hand",)
# These three seeds preserve the paper-era baseline (42) and reuse the most
# completed clean-protocol cells already present in the artifact history.
SEEDS = (7, 42, 128)
FRAME_SAMPLE_COUNTS = (10, 20, 30)
MAX_WORKERS = 4
MODEL_WORKERS = {
    "cnn1d": 4,
    "gcn": 3,
    "lstm": 1,
    "transformer": 1,
}

if set(MODEL_ORDER) != set(ALLOWED_MODELS):
    raise RuntimeError("MODEL_ORDER must contain every allowed model exactly once.")


@dataclass(frozen=True, slots=True)
class ExperimentCommand:
    index: int
    scenario: str
    model: str
    modality: str
    frame_sample_count: int
    seed: int
    command: list[str]


@dataclass(frozen=True, slots=True)
class RunResult:
    experiment: ExperimentCommand
    returncode: int
    log_path: Path


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
            f"Invalid {option_name}: {', '.join(invalid_values)}. "
            f"Expected comma-separated values from: {', '.join(allowed_values)}, or 'all'."
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
        allowed = ", ".join(str(value) for value in allowed_values)
        invalid = ", ".join(str(value) for value in invalid_values)
        raise argparse.ArgumentTypeError(
            f"Invalid {option_name}: {invalid}. Expected values from: {allowed}, or 'all'."
        )
    if not requested_values:
        raise argparse.ArgumentTypeError(f"{option_name} cannot be empty.")
    return requested_values


def _parse_model_workers(raw_value: str) -> dict[str, int]:
    result = dict(MODEL_WORKERS)
    if not raw_value.strip():
        return result
    for item in raw_value.split(","):
        if not item.strip():
            continue
        if ":" not in item:
            raise argparse.ArgumentTypeError(
                "--model-workers entries must look like model:workers."
            )
        model, worker_count = item.split(":", maxsplit=1)
        model = model.strip()
        if model not in MODELS:
            raise argparse.ArgumentTypeError(
                f"Unknown model in --model-workers: {model}."
            )
        try:
            worker_count_int = int(worker_count.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid worker count for model {model}: {worker_count}."
            ) from exc
        if worker_count_int < 1:
            raise argparse.ArgumentTypeError(
                f"Worker count for model {model} must be at least 1."
            )
        result[model] = worker_count_int
    return result


def build_parser() -> argparse.ArgumentParser:
    defaults = ExperimentConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Run the clean hand-only multi-seed, multi-frame comparison grid."
        )
    )
    parser.add_argument("--python-bin", default=sys.executable)
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
        "--modalities",
        default="hand",
        help="Comma-separated modality list, or 'all'. Default: hand.",
    )
    parser.add_argument(
        "--seeds",
        default=SEEDS,
        type=lambda value: _parse_requested_ints(value, SEEDS, "--seeds"),
        help="Comma-separated seed list, or 'all'. Default: 7,42,128.",
    )
    parser.add_argument(
        "--frame-sample-counts",
        default=FRAME_SAMPLE_COUNTS,
        type=lambda value: _parse_requested_ints(
            value,
            FRAME_SAMPLE_COUNTS,
            "--frame-sample-counts",
        ),
        help="Comma-separated frame count list, or 'all'. Default: 10,20,30.",
    )
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument(
        "--schedule-profile",
        choices=("sequential", "balanced", "flat"),
        default="balanced",
        help=(
            "sequential runs one job at a time; balanced uses per-model worker "
            "limits; flat uses --max-workers for every selected job."
        ),
    )
    parser.add_argument(
        "--model-workers",
        default="cnn1d:4,gcn:3,lstm:1,transformer:1",
        type=_parse_model_workers,
        help=(
            "Balanced profile worker limits by model. Default: "
            "cnn1d:4,gcn:3,lstm:1,transformer:1."
        ),
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help=(
            "Skip clean completed runs already present under --artifacts-dir. "
            "Useful when resuming after an interrupted launcher run."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining batches after a failed child process.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("artifacts/run_logs/hand_grid"),
        help="Directory for per-experiment launcher logs.",
    )
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--cache-device", choices=ALLOWED_CACHE_DEVICES, default=defaults.cache_device)
    parser.add_argument(
        "--train-partitions",
        type=int,
        default=defaults.train_partitions,
    )
    parser.add_argument(
        "--partition-cycles",
        type=int,
        default=defaults.partition_cycles,
    )
    parser.add_argument(
        "--validate-during-training",
        action=argparse.BooleanOptionalAction,
        default=defaults.validate_during_training,
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=defaults.validation_fraction,
    )
    parser.add_argument(
        "--evaluation-batch-size",
        type=int,
        default=defaults.evaluation_batch_size,
    )
    parser.add_argument(
        "--test-partitions",
        type=int,
        default=defaults.test_partitions,
    )
    parser.add_argument("--optimizer", choices=ALLOWED_OPTIMIZERS, default=defaults.optimizer)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--partition-learning-rate", type=float, default=defaults.partition_learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--global-clipnorm", type=float, default=defaults.global_clipnorm)
    parser.add_argument("--reduce-lr-factor", type=float, default=defaults.reduce_lr_factor)
    parser.add_argument("--reduce-lr-patience", type=int, default=defaults.reduce_lr_patience)
    parser.add_argument("--min-learning-rate", type=float, default=defaults.min_learning_rate)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=defaults.early_stopping_patience,
    )
    parser.add_argument(
        "--early-stopping-min-cycles",
        type=int,
        default=defaults.early_stopping_min_cycles,
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=defaults.early_stopping_min_delta,
    )
    parser.add_argument("--dataset-dir", type=Path, default=defaults.dataset_dir)
    parser.add_argument("--artifacts-dir", type=Path, default=defaults.artifacts_dir)
    return parser


def _build_shared_flags(args: argparse.Namespace) -> list[str]:
    return [
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--cache-device",
        args.cache_device,
        "--train-partitions",
        str(args.train_partitions),
        "--partition-cycles",
        str(args.partition_cycles),
        "--validate-during-training" if args.validate_during_training else "--no-validate-during-training",
        "--validation-fraction",
        str(args.validation_fraction),
        "--evaluation-batch-size",
        str(args.evaluation_batch_size),
        "--test-partitions",
        str(args.test_partitions),
        "--optimizer",
        args.optimizer,
        "--learning-rate",
        str(args.learning_rate),
        "--partition-learning-rate",
        str(args.partition_learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--global-clipnorm",
        str(args.global_clipnorm),
        "--reduce-lr-factor",
        str(args.reduce_lr_factor),
        "--reduce-lr-patience",
        str(args.reduce_lr_patience),
        "--min-learning-rate",
        str(args.min_learning_rate),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--early-stopping-min-cycles",
        str(args.early_stopping_min_cycles),
        "--early-stopping-min-delta",
        str(args.early_stopping_min_delta),
        "--dataset-dir",
        str(args.dataset_dir),
        "--artifacts-dir",
        str(args.artifacts_dir),
    ]


def _experiment_key(experiment: ExperimentCommand) -> tuple[str, str, str, int, int]:
    return (
        experiment.scenario,
        experiment.model,
        experiment.modality,
        experiment.frame_sample_count,
        experiment.seed,
    )


def _values_match(left: object, right: object) -> bool:
    if isinstance(right, float):
        try:
            return abs(float(left) - right) <= 1e-12
        except (TypeError, ValueError):
            return False
    return left == right


def _clean_protocol_fields(args: argparse.Namespace) -> dict[str, object]:
    return {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_partitions": args.train_partitions,
        "partition_cycles": args.partition_cycles,
        "validate_during_training": args.validate_during_training,
        "validation_fraction": args.validation_fraction,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "partition_learning_rate": args.partition_learning_rate,
        "weight_decay": args.weight_decay,
        "global_clipnorm": args.global_clipnorm,
        "reduce_lr_factor": args.reduce_lr_factor,
        "reduce_lr_patience": args.reduce_lr_patience,
        "min_learning_rate": args.min_learning_rate,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_cycles": args.early_stopping_min_cycles,
        "early_stopping_min_delta": args.early_stopping_min_delta,
    }


def _discover_completed_keys(args: argparse.Namespace) -> set[tuple[str, str, str, int, int]]:
    protocol_fields = _clean_protocol_fields(args)
    completed_keys: set[tuple[str, str, str, int, int]] = set()
    experiments_dir = args.artifacts_dir / "experiments"
    for config_path in experiments_dir.glob("*/*/*/*/config.json"):
        run_dir = config_path.parent
        if not (run_dir / "summary.json").exists():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        try:
            key = (
                str(config["scenario"]),
                str(config["model"]),
                str(config["modality"]),
                int(config["frame_sample_count"]),
                int(config["seed"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        if all(
            _values_match(config.get(field), expected)
            for field, expected in protocol_fields.items()
        ):
            completed_keys.add(key)
    return completed_keys


def _log_path_for(logs_dir: Path, experiment: ExperimentCommand) -> Path:
    file_name = (
        f"{experiment.index:03d}_{experiment.scenario}_{experiment.model}_"
        f"{experiment.modality}_{experiment.frame_sample_count}frame_"
        f"seed{experiment.seed}.log"
    )
    return logs_dir / file_name


def _run_experiment_command(
    experiment: ExperimentCommand,
    logs_dir: Path,
) -> RunResult:
    log_path = _log_path_for(logs_dir, experiment)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(
            " ".join(experiment.command)
            + "\n\n"
            + (
                f"index={experiment.index} scenario={experiment.scenario} "
                f"model={experiment.model} modality={experiment.modality} "
                f"frames={experiment.frame_sample_count} seed={experiment.seed}\n\n"
            )
        )
        handle.flush()
        completed = subprocess.run(
            experiment.command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return RunResult(
        experiment=experiment,
        returncode=completed.returncode,
        log_path=log_path,
    )


def _raise_for_failures(failures: list[RunResult]) -> None:
    if not failures:
        return
    print("Failed experiments:")
    for failure in failures:
        experiment = failure.experiment
        print(
            f"  [{experiment.index}] scenario={experiment.scenario} "
            f"model={experiment.model} frames={experiment.frame_sample_count} "
            f"seed={experiment.seed} returncode={failure.returncode} "
            f"log={failure.log_path}"
        )
    raise RuntimeError(
        f"{len(failures)} experiment(s) failed. See the log paths above."
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.max_workers < 1:
        raise ValueError("--max-workers must be at least 1.")
    if args.start_index < 0:
        raise ValueError("--start-index must be at least 0.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1 when provided.")

    scenarios = _parse_requested_values(args.scenarios, SCENARIOS, "--scenarios")
    models = _parse_requested_values(args.models, MODELS, "--models")
    modalities = _parse_requested_values(args.modalities, MODALITIES, "--modalities")
    seeds = args.seeds
    frame_sample_counts = args.frame_sample_counts

    shared_flags = _build_shared_flags(args)
    all_commands: list[ExperimentCommand] = []
    command_index = 0
    for model in models:
        for scenario in scenarios:
            for frame_sample_count in frame_sample_counts:
                for seed in seeds:
                    for modality in modalities:
                        all_commands.append(
                            ExperimentCommand(
                                index=command_index,
                                scenario=scenario,
                                model=model,
                                modality=modality,
                                frame_sample_count=frame_sample_count,
                                seed=seed,
                                command=[
                                    args.python_bin,
                                    "-m",
                                    "training.run_experiments",
                                    "--scenario",
                                    scenario,
                                    "--model",
                                    model,
                                    "--modality",
                                    modality,
                                    "--frame-sample-count",
                                    str(frame_sample_count),
                                    "--seed",
                                    str(seed),
                                    *shared_flags,
                                ],
                            )
                        )
                        command_index += 1

    selected_commands = all_commands[args.start_index:]
    if args.limit is not None:
        selected_commands = selected_commands[:args.limit]
    skipped_completed = 0
    if args.skip_completed:
        completed_keys = _discover_completed_keys(args)
        before_skip = len(selected_commands)
        selected_commands = [
            experiment
            for experiment in selected_commands
            if _experiment_key(experiment) not in completed_keys
        ]
        skipped_completed = before_skip - len(selected_commands)

    if not selected_commands:
        if skipped_completed:
            print(f"No experiments selected. Skipped {skipped_completed} completed runs.")
        else:
            print("No experiments selected.")
        return

    print("Selected experiments:")
    for experiment in selected_commands:
        print(
            f"  [{experiment.index}] scenario={experiment.scenario} "
            f"model={experiment.model} modality={experiment.modality} "
            f"frames={experiment.frame_sample_count} seed={experiment.seed}"
        )
    print(
        "Training schedule: "
        f"selected={len(selected_commands)}/{len(all_commands)} "
        f"batch_size={args.batch_size} "
        f"train_partitions={args.train_partitions} "
        f"epochs_per_partition={args.epochs} "
        f"partition_cycles={args.partition_cycles} "
        f"effective_dataset_passes={args.epochs * args.partition_cycles} "
        f"evaluation_batch_size={args.evaluation_batch_size} "
        f"schedule_profile={args.schedule_profile} "
        f"max_workers={args.max_workers} "
        f"skipped_completed={skipped_completed}"
    )
    if args.schedule_profile == "balanced":
        model_workers = ", ".join(
            f"{model}:{args.model_workers[model]}" for model in models
        )
        print(f"Balanced model worker limits: {model_workers}")
        print(_balanced_plan_summary(selected_commands, args.max_workers, args.model_workers))
    if args.dry_run:
        print(f"Dry run only. {len(selected_commands)} experiments selected.")
        return

    launcher_log_dir = args.logs_dir / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    print(f"Per-experiment logs: {launcher_log_dir}")
    _run_selected_commands(
        selected_commands,
        args.max_workers,
        args.schedule_profile,
        args.model_workers,
        launcher_log_dir,
        args.continue_on_error,
    )

    print("All selected hand-only seed/frame experiments finished.")


def _run_selected_commands(
    selected_commands: list[ExperimentCommand],
    max_workers: int,
    schedule_profile: str,
    model_workers: dict[str, int],
    logs_dir: Path,
    continue_on_error: bool,
) -> None:
    failures: list[RunResult] = []
    if schedule_profile == "sequential":
        for experiment in selected_commands:
            print(f"Launching [{experiment.index}] {experiment.model} frames={experiment.frame_sample_count} seed={experiment.seed}")
            result = _run_experiment_command(experiment, logs_dir)
            if result.returncode != 0:
                failures.append(result)
                print(f"Failed [{experiment.index}] log={result.log_path}")
                if not continue_on_error:
                    _raise_for_failures(failures)
            else:
                print(f"Finished [{experiment.index}] log={result.log_path}")
        _raise_for_failures(failures)
        return

    if schedule_profile == "flat":
        print(f"Launching experiments with flat max_workers={max_workers}...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                executor.map(
                    lambda experiment: _run_experiment_command(experiment, logs_dir),
                    selected_commands,
                )
            )
        failures.extend(result for result in results if result.returncode != 0)
        _raise_for_failures(failures)
        return

    print("Launching experiments with balanced model-order batches...")
    for model in MODELS:
        model_commands = [
            experiment
            for experiment in selected_commands
            if experiment.model == model
        ]
        if not model_commands:
            continue
        worker_count = min(max_workers, model_workers.get(model, 1))
        print(
            f"Starting model group model={model} "
            f"jobs={len(model_commands)} workers={worker_count}"
        )
        for start in range(0, len(model_commands), worker_count):
            batch = model_commands[start:start + worker_count]
            batch_ids = ", ".join(str(experiment.index) for experiment in batch)
            print(f"Launching {model} batch: [{batch_ids}]")
            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                results = list(
                    executor.map(
                        lambda experiment: _run_experiment_command(experiment, logs_dir),
                        batch,
                    )
                )
            batch_failures = [
                result for result in results if result.returncode != 0
            ]
            for result in results:
                status = "finished" if result.returncode == 0 else "failed"
                print(f"{status.capitalize()} [{result.experiment.index}] log={result.log_path}")
            failures.extend(batch_failures)
            if batch_failures and not continue_on_error:
                _raise_for_failures(failures)
        print(f"Finished model group model={model}")
    _raise_for_failures(failures)


def _balanced_plan_summary(
    selected_commands: list[ExperimentCommand],
    max_workers: int,
    model_workers: dict[str, int],
) -> str:
    parts = []
    for model in MODELS:
        job_count = sum(1 for experiment in selected_commands if experiment.model == model)
        if job_count == 0:
            continue
        worker_count = min(max_workers, model_workers.get(model, 1))
        batch_count = (job_count + worker_count - 1) // worker_count
        parts.append(f"{model}: jobs={job_count}, workers={worker_count}, batches={batch_count}")
    return "Balanced execution order: " + " -> ".join(parts)


if __name__ == "__main__":
    main()
