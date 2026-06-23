from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor

import argparse
import subprocess
import sys
from pathlib import Path

from training.config import (
    ALLOWED_CACHE_DEVICES,
    ALLOWED_MODELS,
    ALLOWED_MODALITIES,
    ALLOWED_OPTIMIZERS,
    ALLOWED_SCENARIOS,
    ExperimentConfig,
)

SCENARIOS = ("si",)
MODELS = ["lstm"]#ALLOWED_MODELS
MODALITIES = ["body_hand_face"]#ALLOWED_MODALITIES

MAX_WORKERS = 1


def _parse_requested_values(raw_value: str, allowed_values: tuple[str, ...], option_name: str) -> tuple[str, ...]:
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


def build_parser() -> argparse.ArgumentParser:
    defaults = ExperimentConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Run all SI model and modality combinations using ExperimentConfig defaults."
        )
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument(
        "--scenarios",
        default="all",
        help="Comma-separated scenario list, or 'all'. Example: sd,si",
    )
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model list, or 'all'. Example: cnn1d,lstm",
    )
    parser.add_argument(
        "--modalities",
        default="all",
        help="Comma-separated modality list, or 'all'. Example: hand,hand_face",
    )
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--frame-sample-count", type=int, default=defaults.frame_sample_count)
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
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--dataset-dir", type=Path, default=defaults.dataset_dir)
    parser.add_argument("--artifacts-dir", type=Path, default=defaults.artifacts_dir)
    return parser


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

    shared_flags = [
        "--frame-sample-count",
        str(args.frame_sample_count),
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
        "--seed",
        str(args.seed),
        "--dataset-dir",
        str(args.dataset_dir),
        "--artifacts-dir",
        str(args.artifacts_dir),
    ]

    all_commands = []
    for scenario in scenarios:
        for model in models:
            for modality in modalities:
                all_commands.append([
                    args.python_bin, "-m", "training.run_experiments",
                    "--scenario", scenario, "--model", model,
                    "--modality", modality, *shared_flags,
                ])
    all_commands = all_commands[args.start_index:]
    if args.limit is not None:
        all_commands = all_commands[:args.limit]

    if not all_commands:
        print("No experiments selected.")
        return

    print("Selected experiments:")
    for index, command in enumerate(all_commands, start=args.start_index):
        scenario = command[4]
        model = command[6]
        modality = command[8]
        print(f"  [{index}] scenario={scenario} model={model} modality={modality}")
    print(
        "Training schedule: "
        f"batch_size={args.batch_size} "
        f"train_partitions={args.train_partitions} "
        f"epochs_per_partition={args.epochs} "
        f"partition_cycles={args.partition_cycles} "
        f"effective_dataset_passes={args.epochs * args.partition_cycles} "
        f"evaluation_batch_size={args.evaluation_batch_size}"
    )
    if args.dry_run:
        print(f"Dry run only. {len(all_commands)} experiments selected.")
        return

    print(f"Launching experiments with {args.max_workers} parallel workers...")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        list(
            executor.map(
                lambda command: subprocess.run(command, check=True),
                all_commands,
            )
        )

    print("All SI model and modality experiments finished.")

if __name__ == "__main__":
    main()
