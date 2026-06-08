from __future__ import annotations

import json
import os
import random
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
import torch
from keras.callbacks import Callback, ModelCheckpoint
from torch.utils.data import DataLoader

from .config import ALLOWED_MODELS, ALLOWED_MODALITIES, ALLOWED_SCENARIOS, ExperimentConfig
from .data import (
    SkeletonDataset,
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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device() -> str:
    return "gpu" if torch.cuda.is_available() else "cpu"


def _log_runtime(device: str, num_workers: int, batch_size: int) -> None:
    backend = keras.backend.backend()
    print(f"[runtime] keras_backend={backend}")
    print(f"[runtime] requested_device={device}")
    print(f"[runtime] batch_size={batch_size} num_workers={num_workers}")
    print(f"[runtime] torch_cuda_available={torch.cuda.is_available()}")
    print(f"[runtime] torch_cuda_device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        current_index = torch.cuda.current_device()
        print(f"[runtime] torch_cuda_current_device={current_index}")
        print(f"[runtime] torch_cuda_device_name={torch.cuda.get_device_name(current_index)}")
        print(f"[runtime] torch_cuda_memory_allocated_mb={torch.cuda.memory_allocated() / 1024**2:.2f}")
        print(f"[runtime] torch_cuda_memory_reserved_mb={torch.cuda.memory_reserved() / 1024**2:.2f}")


def _keras_device_name(device: str) -> str:
    return "gpu:0" if device == "gpu" else "cpu:0"


def _print_cuda_memory(prefix: str) -> None:
    if not torch.cuda.is_available():
        print(f"{prefix} cuda_available=False")
        return
    print(f"{prefix} cuda_memory_allocated_mb={torch.cuda.memory_allocated() / 1024**2:.2f}")
    print(f"{prefix} cuda_memory_reserved_mb={torch.cuda.memory_reserved() / 1024**2:.2f}")


class TorchMemoryCallback(Callback):
    def on_train_begin(self, logs=None) -> None:
        del logs
        _print_cuda_memory("[runtime][train_begin]")

    def on_epoch_begin(self, epoch: int, logs=None) -> None:
        del logs
        _print_cuda_memory(f"[runtime][epoch_{epoch + 1}_begin]")

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        del logs
        _print_cuda_memory(f"[runtime][epoch_{epoch + 1}_end]")


def _build_dataloader(dataset, batch_size: int, shuffle: bool, num_workers: int, device: str) -> DataLoader:
    worker_count = max(0, num_workers)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": worker_count,
        "pin_memory": device == "gpu",
    }
    if worker_count > 0:
        # Keep workers alive between epochs to reduce file-loading overhead.
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **loader_kwargs)


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    _validate_config(config)
    _set_seed(config.seed)

    train_records = load_split_records(config.dataset_dir, config.scenario, "train")
    test_records = load_split_records(config.dataset_dir, config.scenario, "test")
    validate_modality_support(
        config.dataset_dir,
        train_records + test_records,
        config.modality,
        config.frame_sample_count,
    )

    label_map = build_label_map(train_records + test_records)
    input_shape = infer_input_shape(
        config.dataset_dir,
        train_records[0],
        config.modality,
        config.frame_sample_count,
    )
    device = _resolve_device()
    _log_runtime(device=device, num_workers=config.num_workers, batch_size=config.batch_size)
    keras_device_name = _keras_device_name(device)
    print(f"[runtime] keras_device_scope={keras_device_name}")

    train_dataset = SkeletonDataset(
        records=train_records,
        dataset_dir=config.dataset_dir,
        label_map=label_map,
        modality=config.modality,
        frame_sample_count=config.frame_sample_count,
    )
    test_dataset = SkeletonDataset(
        records=test_records,
        dataset_dir=config.dataset_dir,
        label_map=label_map,
        modality=config.modality,
        frame_sample_count=config.frame_sample_count,
    )

    train_loader = _build_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        device=device,
    )
    test_loader = _build_dataloader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        device=device,
    )

    with keras.device(keras_device_name):
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
    final_weights_path = run_dir / "final.weights.h5"
    run_architecture_path = run_dir / "architecture.json"
    architecture_path = architecture_dir / f"{config.model}_{config.modality}.json"

    callbacks = [
        TorchMemoryCallback(),
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_f1_score",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
        )
    ]

    with keras.device(keras_device_name):
        history = model.fit(
            train_loader,
            validation_data=test_loader,
            epochs=config.epochs,
            callbacks=callbacks,
            verbose=1,
        )
    if checkpoint_path.exists():
        model.load_weights(checkpoint_path)
    with keras.device(keras_device_name):
        evaluation_pairs = model.evaluate(test_loader, verbose=0, return_dict=True)
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
        "keras_backend": keras.backend.backend(),
        "device": device,
        "num_workers": config.num_workers,
    }
    history_payload = {key: [float(value) for value in values] for key, values in history.history.items()}
    metrics_payload = {key: float(value) for key, value in evaluation_pairs.items()}
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
        "keras_backend": keras.backend.backend(),
        "device": device,
        "num_workers": config.num_workers,
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
