from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from keras.callbacks import ModelCheckpoint

from .config import ALLOWED_MODELS, ALLOWED_MODALITIES, ALLOWED_SCENARIOS, ExperimentConfig
from .data import (
    SkeletonSequence,
    build_label_map,
    infer_input_shape,
    load_split_records,
    validate_modality_support,
)
from .models import build_model, model_parameter_count


def _validate_config(config: ExperimentConfig) -> None:
    if config.scenario not in ALLOWED_SCENARIOS:
        raise ValueError(
            f"Unsupported scenario '{config.scenario}'. Expected one of: {', '.join(ALLOWED_SCENARIOS)}"
        )
    if config.modality not in ALLOWED_MODALITIES:
        raise ValueError(
            f"Unsupported modality '{config.modality}'. Expected one of: {', '.join(ALLOWED_MODALITIES)}"
        )
    if config.model not in ALLOWED_MODELS:
        raise ValueError(
            f"Unsupported model '{config.model}'. Expected one of: {', '.join(ALLOWED_MODELS)}"
        )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    _validate_config(config)

    train_records = load_split_records(config.dataset_dir, config.scenario, "train")
    test_records = load_split_records(config.dataset_dir, config.scenario, "test")
    validate_modality_support(config.dataset_dir, train_records + test_records, config.modality)

    label_map = build_label_map(train_records + test_records)
    input_shape = infer_input_shape(config.dataset_dir, train_records[0], config.modality)

    train_sequence = SkeletonSequence(
        records=train_records,
        dataset_dir=config.dataset_dir,
        label_map=label_map,
        modality=config.modality,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
    )
    test_sequence = SkeletonSequence(
        records=test_records,
        dataset_dir=config.dataset_dir,
        label_map=label_map,
        modality=config.modality,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed,
    )

    model = build_model(
        config.model,
        input_shape=input_shape,
        num_classes=len(label_map),
        learning_rate=config.learning_rate,
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (
        config.artifacts_dir
        / "experiments"
        / config.scenario
        / config.model
        / config.modality
        / timestamp
    )
    architecture_dir = config.artifacts_dir / "architectures" / config.model
    label_map_dir = config.artifacts_dir / "label_maps"
    checkpoint_path = run_dir / "best.weights.h5"
    run_architecture_path = run_dir / "architecture.json"
    architecture_path = architecture_dir / f"{config.model}_{config.modality}.json"

    callbacks = [
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_f1_score",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
        )
    ]

    history = model.fit(
        train_sequence,
        validation_data=test_sequence,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    if checkpoint_path.exists():
        model.load_weights(checkpoint_path)
    evaluation = model.evaluate(test_sequence, verbose=0, return_dict=True)
    final_weights_path = run_dir / "final.weights.h5"
    final_weights_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(final_weights_path)

    dataset_manifest = {
        "scenario": config.scenario,
        "train_size": len(train_records),
        "test_size": len(test_records),
        "train_samples": [record.to_dict() for record in train_records[:25]],
        "test_samples": [record.to_dict() for record in test_records[:25]],
    }
    config_payload = {
        **config.to_dict(),
        "input_shape": list(input_shape),
        "num_classes": len(label_map),
    }
    history_payload = {key: [float(value) for value in values] for key, values in history.history.items()}
    metrics_payload = {key: float(value) for key, value in evaluation.items()}
    summary_payload = {
        "scenario": config.scenario,
        "model": config.model,
        "modality": config.modality,
        "run_dir": str(run_dir),
        "input_shape": list(input_shape),
        "num_classes": len(label_map),
        "train_size": len(train_records),
        "test_size": len(test_records),
        "best_checkpoint": str(checkpoint_path),
        "final_weights": str(final_weights_path),
        "architecture_json": str(run_architecture_path),
        "metrics": metrics_payload,
        "parameter_count": model_parameter_count(model),
    }

    _write_json(run_dir / "config.json", config_payload)
    _write_json(run_dir / "history.json", history_payload)
    _write_json(run_dir / "metrics.json", metrics_payload)
    _write_json(run_dir / "dataset_manifest.json", dataset_manifest)
    _write_json(run_dir / "summary.json", summary_payload)
    _write_json(label_map_dir / f"{config.scenario}_labels.json", label_map)

    architecture_dir.mkdir(parents=True, exist_ok=True)
    architecture_json = model.to_json(indent=2)
    architecture_path.write_text(architecture_json, encoding="utf-8")
    run_architecture_path.write_text(architecture_json, encoding="utf-8")

    return {
        "run_dir": str(run_dir),
        "config": config_payload,
        "history": history_payload,
        "metrics": metrics_payload,
        "summary": summary_payload,
    }
