from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

from .config import ALLOWED_MODELS, ALLOWED_MODALITIES, ALLOWED_SCENARIOS, ExperimentConfig


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
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
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
        learning_rate=args.learning_rate,
        seed=args.seed,
        dataset_dir=args.dataset_dir,
        artifacts_dir=args.artifacts_dir,
    )
    result = run_experiment(config)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
