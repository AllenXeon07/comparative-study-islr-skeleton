from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ALLOWED_MODELS, ALLOWED_MODALITIES, ALLOWED_SCENARIOS, ExperimentConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a sign-language recognition experiment.")
    parser.add_argument("--scenario", required=True, choices=ALLOWED_SCENARIOS)
    parser.add_argument("--model", required=True, choices=ALLOWED_MODELS)
    parser.add_argument("--modality", required=True, choices=ALLOWED_MODALITIES)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-dir", type=Path, default=Path("Dataset"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    return parser


def main() -> None:
    from .experiment import run_experiment

    args = build_parser().parse_args()
    config = ExperimentConfig(
        scenario=args.scenario,
        model=args.model,
        modality=args.modality,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        dataset_dir=args.dataset_dir,
        artifacts_dir=args.artifacts_dir,
    )
    result = run_experiment(config)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
