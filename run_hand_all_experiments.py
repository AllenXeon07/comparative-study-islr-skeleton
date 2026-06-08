from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from threading import Semaphore

import argparse
import subprocess
import sys
from pathlib import Path

from training.config import ALLOWED_MODELS, ALLOWED_SCENARIOS, ALLOWED_MODALITIES, ExperimentConfig

SCENARIOS = ALLOWED_SCENARIOS
MODELS = ALLOWED_MODELS
MODALITIES = ALLOWED_MODALITIES


def build_parser() -> argparse.ArgumentParser:
    defaults = ExperimentConfig()
    parser = argparse.ArgumentParser(
        description="Run all hand-only experiments across all models and both scenarios."
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--frame-sample-count", type=int, default=defaults.frame_sample_count)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--dataset-dir", type=Path, default=defaults.dataset_dir)
    parser.add_argument("--artifacts-dir", type=Path, default=defaults.artifacts_dir)
    return parser


def run_command(command, semaphore):
    # Use the semaphore to limit concurrency
    with semaphore:
        scenario = command[4] # Extracting info for the print
        model = command[6]
        print(f"Starting: scenario={scenario} model={model}")
        
        # We still use check=True to catch errors locally
        subprocess.run(command, check=True)

def main() -> None:
    args = build_parser().parse_args()

    shared_flags = [
        "--frame-sample-count",
        str(args.frame_sample_count),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--learning-rate",
        str(args.learning_rate),
        "--seed",
        str(args.seed),
        "--dataset-dir",
        str(args.dataset_dir),
        "--artifacts-dir",
        str(args.artifacts_dir),
    ]

     # 1. Gather all commands first
    all_commands = []
    for scenario in SCENARIOS:
        for model in MODELS:
            for modality in MODALITIES:
                all_commands.append([
                    args.python_bin, "-m", "training.run_experiments",
                    "--scenario", scenario, "--model", model,
                    "--modality", modality, *shared_flags,
                ])

    # 2. Execute with a pool size of 2
    # Max_workers=4 ensures only 2 threads (and thus processes) run at once
    print(f"Launching experiments with 2 parallel workers...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit all commands to the pool
        executor.map(lambda cmd: subprocess.run(cmd, check=True), all_commands)

    print("All hand-only experiments finished.")

if __name__ == "__main__":
    main()
