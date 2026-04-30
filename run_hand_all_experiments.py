from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCENARIOS = ("sd", "si")
MODELS = ("cnn1d", "lstm", "transformer", "gcn")
MODALITY = "hand"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run all hand-only experiments across all models and both scenarios."
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-dir", type=Path, default=Path("Dataset"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    return parser


def main() -> None:
    args = build_parser().parse_args()

    shared_flags = [
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--seed",
        str(args.seed),
        "--dataset-dir",
        str(args.dataset_dir),
        "--artifacts-dir",
        str(args.artifacts_dir),
    ]

    for scenario in SCENARIOS:
        for model in MODELS:
            command = [
                args.python_bin,
                "-m",
                "training.run_experiments",
                "--scenario",
                scenario,
                "--model",
                model,
                "--modality",
                MODALITY,
                *shared_flags,
            ]
            print(f"Running: scenario={scenario} model={model} modality={MODALITY}")
            subprocess.run(command, check=True)

    print("All hand-only experiments finished.")


if __name__ == "__main__":
    main()
