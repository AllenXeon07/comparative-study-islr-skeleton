from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

from .config import (
    ALLOWED_CACHE_DEVICES,
    ALLOWED_MODELS,
    ALLOWED_MODALITIES,
    ALLOWED_OPTIMIZERS,
    ALLOWED_SCENARIOS,
    ExperimentConfig,
)


def build_parser() -> argparse.ArgumentParser:
    defaults = ExperimentConfig()
    parser = argparse.ArgumentParser(description="Run a sign-language recognition experiment.")
    parser.add_argument("--scenario", choices=ALLOWED_SCENARIOS, default=defaults.scenario)
    parser.add_argument("--model", choices=ALLOWED_MODELS, default=defaults.model)
    parser.add_argument("--modality", choices=ALLOWED_MODALITIES, default=defaults.modality)
    parser.add_argument("--frame-sample-count", type=int, default=defaults.frame_sample_count)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--cache-device", choices=ALLOWED_CACHE_DEVICES, default=defaults.cache_device)
    parser.add_argument("--train-partitions", type=int, default=defaults.train_partitions)
    parser.add_argument("--partition-cycles", type=int, default=defaults.partition_cycles)
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
    parser.add_argument("--test-partitions", type=int, default=defaults.test_partitions)
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
    from .experiment import run_experiment

    args = build_parser().parse_args()
    config = ExperimentConfig(
        scenario=args.scenario,
        model=args.model,
        modality=args.modality,
        frame_sample_count=args.frame_sample_count,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_device=args.cache_device,
        train_partitions=args.train_partitions,
        partition_cycles=args.partition_cycles,
        validate_during_training=args.validate_during_training,
        validation_fraction=args.validation_fraction,
        evaluation_batch_size=args.evaluation_batch_size,
        test_partitions=args.test_partitions,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        partition_learning_rate=args.partition_learning_rate,
        weight_decay=args.weight_decay,
        global_clipnorm=args.global_clipnorm,
        reduce_lr_factor=args.reduce_lr_factor,
        reduce_lr_patience=args.reduce_lr_patience,
        min_learning_rate=args.min_learning_rate,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_cycles=args.early_stopping_min_cycles,
        early_stopping_min_delta=args.early_stopping_min_delta,
        seed=args.seed,
        dataset_dir=args.dataset_dir,
        artifacts_dir=args.artifacts_dir,
    )
    result = run_experiment(config)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
