#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.testing_pipeline import evaluate_prediction, predict_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full sign-language testing pipeline: load video, preprocess, "
            "load model from its run config, evaluate, and show the prediction."
        )
    )
    parser.add_argument(
        "video",
        type=Path,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "model_path",
        type=Path,
        help="Experiment run directory or a best/final .weights.h5 file.",
    )
    parser.add_argument(
        "--expected-label",
        help="Optional ground-truth gloss. When set, the top prediction is marked correct/incorrect.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of ranked predictions to show.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full result as JSON instead of a compact text report.",
    )
    return parser


def print_report(result: dict[str, object]) -> None:
    predictions = result["top_predictions"]
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("No predictions were produced.")

    top_prediction = predictions[0]
    print("\n=== Testing Pipeline Result ===")
    print(f"Video: {result['video_path']}")
    print(f"Run: {result['run_dir']}")
    print(f"Weights: {result['weights_path']}")
    print(
        "Config: "
        f"model={result['model']} "
        f"scenario={result['scenario']} "
        f"modality={result['modality']} "
        f"frames={result['frame_sample_count']} "
        f"input_shape={result['input_shape']}"
    )

    expected_label = result.get("expected_label")
    if expected_label is not None:
        print(f"Expected: {expected_label}")
        print(f"Correct: {result['correct']}")

    print(
        "Prediction: "
        f"{top_prediction['label']} "
        f"({top_prediction['confidence']:.4f})"
    )
    print("\nTop predictions:")
    for item in predictions:
        print(f"{item['rank']:>2}. {item['label']} [{item['index']}] {item['confidence']:.4f}")


def main() -> None:
    args = build_parser().parse_args()
    if args.top_k <= 0:
        raise ValueError(f"--top-k must be positive, got {args.top_k}.")

    result = predict_video(
        video_path=args.video,
        model_path=args.model_path,
        top_k=args.top_k,
    )
    result = evaluate_prediction(result, args.expected_label)

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_report(result)


if __name__ == "__main__":
    main()
