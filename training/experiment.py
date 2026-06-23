from __future__ import annotations

import csv
import json
import os
import random
import gc
import math
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import keras
import numpy as np
import torch
from keras.callbacks import Callback, ReduceLROnPlateau
from torch.utils.data import DataLoader

from .config import (
    ALLOWED_CACHE_DEVICES,
    ALLOWED_MODELS,
    ALLOWED_MODALITIES,
    ALLOWED_OPTIMIZERS,
    ALLOWED_SCENARIOS,
    ExperimentConfig,
)
from .cycle_callbacks import ValidationCycleController
from .data import (
    PreloadedSkeletonDataset,
    SkeletonDataset,
    build_label_map,
    infer_input_shape,
    load_split_records,
    validate_modality_support,
)
from .models import build_model, model_parameter_count
from .scenario_splits import build_training_validation_split


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
    if config.cache_device not in ALLOWED_CACHE_DEVICES:
        raise ValueError(
            f"Unsupported cache_device '{config.cache_device}'. "
            f"Expected one of: {', '.join(ALLOWED_CACHE_DEVICES)}"
        )
    if config.optimizer not in ALLOWED_OPTIMIZERS:
        raise ValueError(
            f"Unsupported optimizer '{config.optimizer}'. "
            f"Expected one of: {', '.join(ALLOWED_OPTIMIZERS)}"
        )
    if config.train_partitions < 1:
        raise ValueError("train_partitions must be at least 1.")
    if config.partition_cycles < 1:
        raise ValueError("partition_cycles must be at least 1.")
    if config.evaluation_batch_size < 1:
        raise ValueError("evaluation_batch_size must be at least 1.")
    if not 0.0 < config.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if config.test_partitions < 1:
        raise ValueError("test_partitions must be at least 1.")
    if config.weight_decay < 0:
        raise ValueError("weight_decay cannot be negative.")
    if config.global_clipnorm <= 0:
        raise ValueError("global_clipnorm must be greater than 0.")
    if not 0.0 < config.reduce_lr_factor < 1.0:
        raise ValueError("reduce_lr_factor must be between 0 and 1.")
    if config.reduce_lr_patience < 1:
        raise ValueError("reduce_lr_patience must be at least 1.")
    if config.min_learning_rate <= 0:
        raise ValueError("min_learning_rate must be greater than 0.")
    if config.partition_learning_rate <= 0:
        raise ValueError("partition_learning_rate must be greater than 0.")
    if config.partition_learning_rate > config.learning_rate:
        raise ValueError("partition_learning_rate cannot exceed learning_rate.")
    if config.early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be at least 1.")
    if config.early_stopping_min_cycles < 1:
        raise ValueError("early_stopping_min_cycles must be at least 1.")
    validation_intervals = (
        config.epochs
        if config.train_partitions == 1
        else config.partition_cycles
    )
    if config.early_stopping_min_cycles > validation_intervals:
        raise ValueError(
            "early_stopping_min_cycles cannot exceed the number of validation intervals."
        )
    if config.early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta cannot be negative.")


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


def _log_runtime(device: str, num_workers: int, batch_size: int, cache_device: str) -> None:
    backend = keras.backend.backend()
    print(f"[runtime] keras_backend={backend}")
    print(f"[runtime] requested_device={device}")
    print(f"[runtime] batch_size={batch_size} num_workers={num_workers}")
    print(f"[runtime] cache_device={cache_device}")
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


class StreamingValidationCallback(Callback):
    """Run CPU-streamed validation and training control after every epoch."""

    def __init__(
        self,
        *,
        validation_loader: DataLoader,
        num_classes: int,
        controller: ValidationCycleController,
        checkpoint_path: Path,
        reduce_lr_factor: float,
        min_learning_rate: float,
    ) -> None:
        super().__init__()
        self.validation_loader = validation_loader
        self.num_classes = num_classes
        self.controller = controller
        self.checkpoint_path = checkpoint_path
        self.reduce_lr_factor = reduce_lr_factor
        self.min_learning_rate = min_learning_rate
        self.completed_epochs = 0
        self.stopped_early = False

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        logs = logs if logs is not None else {}
        validation_metrics = _evaluate_loader_metrics(
            self.model,
            self.validation_loader,
            num_classes=self.num_classes,
        )
        for key, value in validation_metrics.items():
            logs[f"val_{key}"] = float(value)

        epoch_number = epoch + 1
        train_accuracy = float(logs.get("accuracy", 0.0))
        validation_accuracy = float(validation_metrics["accuracy"])
        validation_f1 = float(validation_metrics["f1_score"])
        validation_loss = float(validation_metrics["loss"])
        print(
            "[runtime] epoch_validation "
            f"epoch={epoch_number} "
            f"train_accuracy={train_accuracy:.6f} "
            f"loss={validation_loss:.6f} "
            f"accuracy={validation_accuracy:.6f} "
            f"f1_score={validation_f1:.6f}"
        )
        decision = self.controller.update(
            cycle=epoch_number,
            train_accuracy=train_accuracy,
            validation_accuracy=validation_accuracy,
            validation_f1=validation_f1,
            validation_loss=validation_loss,
        )
        print(f"[runtime] epoch_callback {decision.message}")
        if decision.save_checkpoint:
            self.model.save_weights(self.checkpoint_path)
            print(
                "[runtime] saved best checkpoint "
                f"val_f1_score={self.controller.best_validation_f1:.6f}"
            )
        if decision.reduce_learning_rate:
            current_learning_rate = _optimizer_learning_rate(self.model)
            reduced_learning_rate = max(
                self.min_learning_rate,
                current_learning_rate * self.reduce_lr_factor,
            )
            if reduced_learning_rate < current_learning_rate:
                _set_optimizer_learning_rate(self.model, reduced_learning_rate)
        if decision.stop_training:
            self.model.stop_training = True
            self.stopped_early = True
            print(f"[runtime] early stopping after epoch {epoch_number}")

        self.completed_epochs = epoch_number
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _print_cuda_memory("[runtime][epoch_validation_released]")


def _dataset_device(dataset) -> torch.device:
    features = getattr(dataset, "features", None)
    if isinstance(features, torch.Tensor):
        return features.device
    return torch.device("cpu")


def _build_dataloader(dataset, batch_size: int, shuffle: bool, num_workers: int, device: str) -> DataLoader:
    worker_count = max(0, num_workers)
    dataset_device = _dataset_device(dataset)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": worker_count,
        "pin_memory": device == "gpu" and dataset_device.type == "cpu",
    }
    if worker_count > 0:
        # Keep workers alive between epochs to reduce file-loading overhead.
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **loader_kwargs)


def _build_dataset(
    records,
    config: ExperimentConfig,
    label_map: dict[str, int],
    cache_device: str,
):
    if cache_device == "gpu":
        print(f"[runtime] preloading {len(records)} samples to cuda")
        return PreloadedSkeletonDataset.from_records(
            records=records,
            dataset_dir=config.dataset_dir,
            label_map=label_map,
            modality=config.modality,
            frame_sample_count=config.frame_sample_count,
            device="cuda",
        )
    return SkeletonDataset(
        records=records,
        dataset_dir=config.dataset_dir,
        label_map=label_map,
        modality=config.modality,
        frame_sample_count=config.frame_sample_count,
    )


def _effective_training_cache_device(
    config: ExperimentConfig,
    input_shape: tuple[int, int],
    train_size: int,
) -> str:
    cache_device = config.cache_device
    if cache_device != "gpu":
        return cache_device

    partition_count = max(1, config.train_partitions)
    largest_partition_size = math.ceil(train_size / partition_count)
    time_steps, feature_count = input_shape
    estimated_cache_bytes = largest_partition_size * time_steps * feature_count * 4
    estimated_cache_bytes += largest_partition_size * 8
    free_bytes, _ = torch.cuda.mem_get_info()
    working_headroom_bytes = (
        8 * 1024**3
        if config.model == "gcn" and config.modality == "body_hand_face"
        else 4 * 1024**3
    )
    max_cache_bytes = min(
        int(free_bytes * 0.5),
        max(0, free_bytes - working_headroom_bytes),
    )
    print(
        "[runtime] gpu partition cache estimate "
        f"largest_partition_samples={largest_partition_size} "
        f"estimated_cache_gib={estimated_cache_bytes / 1024**3:.2f} "
        f"free_vram_gib={free_bytes / 1024**3:.2f} "
        f"reserved_working_headroom_gib={working_headroom_bytes / 1024**3:.2f}"
    )
    if estimated_cache_bytes > max_cache_bytes:
        print(
            "[runtime] overriding training cache_device=gpu to cpu because "
            "the largest partition would not leave enough VRAM for model "
            "activations, gradients, and optimizer state"
        )
        return "cpu"
    return cache_device


def _partition_records(records, partition_count: int, seed: int, shuffle: bool = True):
    shuffled_records = list(records)
    if shuffle:
        random.Random(seed).shuffle(shuffled_records)
    if partition_count <= 1:
        return [shuffled_records]
    partition_size = int(np.ceil(len(shuffled_records) / partition_count))
    return [
        shuffled_records[start:start + partition_size]
        for start in range(0, len(shuffled_records), partition_size)
    ]


def _extend_history(target: dict[str, list[float]], source) -> None:
    for key, values in source.history.items():
        target.setdefault(key, []).extend(float(value) for value in values)


def _release_dataset(dataset) -> None:
    if isinstance(dataset, PreloadedSkeletonDataset):
        dataset.features = torch.empty(0, dtype=torch.float32)
        dataset.labels = torch.empty(0, dtype=torch.long)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _set_optimizer_learning_rate(model, learning_rate: float) -> None:
    optimizer = model.optimizer
    current_learning_rate = optimizer.learning_rate
    try:
        current_learning_rate.assign(learning_rate)
    except AttributeError:
        optimizer.learning_rate = learning_rate
    print(f"[runtime] optimizer_learning_rate={learning_rate:.6g}")


def _optimizer_learning_rate(model) -> float:
    learning_rate = model.optimizer.learning_rate
    try:
        return float(learning_rate.numpy())
    except AttributeError:
        return float(learning_rate)


def _append_validation_history(
    history: dict[str, list[float]],
    metrics: dict[str, object],
) -> None:
    for key, value in metrics.items():
        history.setdefault(f"val_{key}", []).append(float(value))


def _evaluate_loader_metrics(
    model,
    loader: DataLoader,
    num_classes: int,
) -> dict[str, float]:
    true_counts = np.zeros(num_classes, dtype=np.int64)
    predicted_counts = np.zeros(num_classes, dtype=np.int64)
    true_positives = np.zeros(num_classes, dtype=np.int64)
    total_loss = 0.0
    total_samples = 0

    for batch_features, batch_labels in loader:
        probabilities = _to_numpy(model.predict_on_batch(batch_features))
        true_indices = _to_numpy(batch_labels).reshape(-1).astype(np.int64)
        predicted_indices = np.argmax(probabilities, axis=1).astype(np.int64)
        true_probabilities = np.clip(
            probabilities[np.arange(len(true_indices)), true_indices],
            1e-12,
            1.0,
        )
        matches = true_indices == predicted_indices
        total_loss += float(-np.log(true_probabilities).sum())
        total_samples += len(true_indices)
        true_counts += np.bincount(true_indices, minlength=num_classes)
        predicted_counts += np.bincount(predicted_indices, minlength=num_classes)
        true_positives += np.bincount(
            true_indices[matches],
            minlength=num_classes,
        )

    false_positives = predicted_counts - true_positives
    false_negatives = true_counts - true_positives
    precision = np.divide(
        true_positives,
        true_positives + false_positives,
        out=np.zeros(num_classes, dtype=np.float64),
        where=(true_positives + false_positives) > 0,
    )
    recall = np.divide(
        true_positives,
        true_positives + false_negatives,
        out=np.zeros(num_classes, dtype=np.float64),
        where=(true_positives + false_negatives) > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(num_classes, dtype=np.float64),
        where=(precision + recall) > 0,
    )
    return {
        "loss": total_loss / total_samples,
        "accuracy": float(true_positives.sum() / total_samples),
        "f1_score": float(f1.mean()),
    }


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _evaluate_test_partitions(
    model,
    test_records,
    config: ExperimentConfig,
    label_map: dict[str, int],
    device: str,
    run_dir: Path,
) -> tuple[dict[str, float | int], dict[str, object]]:
    labels_by_index = {index: label for label, index in label_map.items()}
    num_classes = len(label_map)
    true_counts = np.zeros(num_classes, dtype=np.int64)
    predicted_counts = np.zeros(num_classes, dtype=np.int64)
    true_positives = np.zeros(num_classes, dtype=np.int64)
    total_loss = 0.0
    total_samples = 0
    predictions_path = run_dir / "test_predictions.csv"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    test_partitions = _partition_records(
        test_records,
        config.test_partitions,
        config.seed + 10_000,
        shuffle=False,
    )
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "participant_id",
                "video_file",
                "true_label",
                "true_index",
                "predicted_label",
                "predicted_index",
                "confidence",
                "correct",
                "test_partition",
            ],
        )
        writer.writeheader()

        for partition_index, partition_records in enumerate(test_partitions, start=1):
            print(
                "[runtime] testing "
                f"partition={partition_index}/{len(test_partitions)} "
                f"samples={len(partition_records)}"
            )
            test_dataset = _build_dataset(
                records=partition_records,
                label_map=label_map,
                config=config,
                cache_device="cpu",
            )
            test_loader = _build_dataloader(
                test_dataset,
                batch_size=config.evaluation_batch_size,
                shuffle=False,
                num_workers=0,
                device=device,
            )

            record_offset = 0
            for batch_features, batch_labels in test_loader:
                probabilities = _to_numpy(model.predict_on_batch(batch_features))
                true_indices = _to_numpy(batch_labels).reshape(-1).astype(np.int64)
                predicted_indices = np.argmax(probabilities, axis=1).astype(np.int64)
                confidences = probabilities[
                    np.arange(len(predicted_indices)),
                    predicted_indices,
                ]
                true_probabilities = np.clip(
                    probabilities[np.arange(len(true_indices)), true_indices],
                    1e-12,
                    1.0,
                )
                total_loss += float(-np.log(true_probabilities).sum())
                total_samples += len(true_indices)
                true_counts += np.bincount(true_indices, minlength=num_classes)
                predicted_counts += np.bincount(predicted_indices, minlength=num_classes)
                matches = true_indices == predicted_indices
                true_positives += np.bincount(
                    true_indices[matches],
                    minlength=num_classes,
                )

                batch_records = partition_records[
                    record_offset:record_offset + len(true_indices)
                ]
                record_offset += len(true_indices)
                for record, true_index, predicted_index, confidence in zip(
                    batch_records,
                    true_indices,
                    predicted_indices,
                    confidences,
                    strict=True,
                ):
                    writer.writerow(
                        {
                            "participant_id": record.participant_id,
                            "video_file": record.video_file,
                            "true_label": labels_by_index[int(true_index)],
                            "true_index": int(true_index),
                            "predicted_label": labels_by_index[int(predicted_index)],
                            "predicted_index": int(predicted_index),
                            "confidence": float(confidence),
                            "correct": bool(true_index == predicted_index),
                            "test_partition": partition_index,
                        }
                    )

            del test_loader
            _release_dataset(test_dataset)

    false_positives = predicted_counts - true_positives
    false_negatives = true_counts - true_positives
    true_negatives = total_samples - true_positives - false_positives - false_negatives
    precision = np.divide(
        true_positives,
        true_positives + false_positives,
        out=np.zeros(num_classes, dtype=np.float64),
        where=(true_positives + false_positives) > 0,
    )
    recall = np.divide(
        true_positives,
        true_positives + false_negatives,
        out=np.zeros(num_classes, dtype=np.float64),
        where=(true_positives + false_negatives) > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(num_classes, dtype=np.float64),
        where=(precision + recall) > 0,
    )
    correct_count = int(true_positives.sum())
    metrics = {
        "loss": total_loss / total_samples,
        "accuracy": correct_count / total_samples,
        "f1_score": float(f1.mean()),
        "correct_count": correct_count,
        "incorrect_count": int(total_samples - correct_count),
        "test_size": total_samples,
    }
    class_report = {
        labels_by_index[index]: {
            "true_positive": int(true_positives[index]),
            "false_positive": int(false_positives[index]),
            "false_negative": int(false_negatives[index]),
            "true_negative": int(true_negatives[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1_score": float(f1[index]),
            "support": int(true_counts[index]),
        }
        for index in range(num_classes)
    }
    report = {
        "metrics": metrics,
        "test_partitions": len(test_partitions),
        "predictions_csv": str(predictions_path),
        "classes": class_report,
    }
    return metrics, report


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    _validate_config(config)
    _set_seed(config.seed)

    official_train_records = load_split_records(
        config.dataset_dir,
        config.scenario,
        "train",
    )
    test_records = load_split_records(config.dataset_dir, config.scenario, "test")
    validate_modality_support(
        config.dataset_dir,
        official_train_records + test_records,
        config.modality,
        config.frame_sample_count,
    )

    if config.validate_during_training:
        split_plan = build_training_validation_split(
            records=official_train_records,
            scenario=config.scenario,
            validation_fraction=config.validation_fraction,
            seed=config.seed,
        )
        train_records = split_plan.training_records
        validation_records = split_plan.validation_records
        split_payload = split_plan.to_dict()
    else:
        train_records = official_train_records
        validation_records = []
        split_payload = {
            "strategy": "disabled",
            "training_size": len(train_records),
            "validation_size": 0,
        }

    label_map = build_label_map(official_train_records + test_records)
    input_shape = infer_input_shape(
        config.dataset_dir,
        train_records[0],
        config.modality,
        config.frame_sample_count,
    )
    device = _resolve_device()
    cache_device = config.cache_device
    if cache_device == "gpu" and device != "gpu":
        raise RuntimeError("cache_device='gpu' requires torch.cuda.is_available().")
    training_cache_device = _effective_training_cache_device(
        config,
        input_shape,
        len(train_records),
    )
    effective_num_workers = 0 if training_cache_device == "gpu" else config.num_workers
    if training_cache_device == "gpu" and config.num_workers != 0:
        print("[runtime] cache_device=gpu uses CUDA tensors; forcing DataLoader num_workers=0")
    print(
        f"[runtime] validation_strategy={split_payload['strategy']} "
        f"training_samples={len(train_records)} "
        f"validation_samples={len(validation_records)}; "
        f"test data streams from CPU in {config.test_partitions} partitions "
        f"with batch_size={config.evaluation_batch_size} num_workers=0"
    )
    _log_runtime(
        device=device,
        num_workers=effective_num_workers,
        batch_size=config.batch_size,
        cache_device=training_cache_device,
    )
    if training_cache_device != cache_device:
        print(f"[runtime] requested_cache_device={cache_device}")
        print(f"[runtime] effective_training_cache_device={training_cache_device}")
    keras_device_name = _keras_device_name(device)
    print(f"[runtime] keras_device_scope={keras_device_name}")

    with keras.device(keras_device_name):
        model = build_model(
            config.model,
            input_shape=input_shape,
            num_classes=len(label_map),
            learning_rate=config.learning_rate,
            optimizer_name=config.optimizer,
            weight_decay=config.weight_decay,
            global_clipnorm=config.global_clipnorm,
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

    training_callbacks = [TorchMemoryCallback()]
    if not config.validate_during_training:
        training_callbacks.append(
            ReduceLROnPlateau(
                monitor="loss",
                factor=config.reduce_lr_factor,
                patience=config.reduce_lr_patience,
                min_lr=config.min_learning_rate,
                verbose=1,
            )
        )

    history_payload: dict[str, list[float]] = {}
    validation_dataset = None
    validation_loader = None
    if validation_records:
        validation_dataset = _build_dataset(
            records=validation_records,
            label_map=label_map,
            config=config,
            cache_device="cpu",
        )
        validation_loader = _build_dataloader(
            validation_dataset,
            batch_size=config.evaluation_batch_size,
            shuffle=False,
            num_workers=0,
            device=device,
        )

    initial_learning_rate = (
        config.learning_rate
        if config.train_partitions == 1
        else config.partition_learning_rate
    )
    _set_optimizer_learning_rate(model, initial_learning_rate)
    cycle_controller = ValidationCycleController(
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_min_cycles=config.early_stopping_min_cycles,
        early_stopping_min_delta=config.early_stopping_min_delta,
        reduce_lr_patience=config.reduce_lr_patience,
    )
    completed_cycles = 0
    stopped_early = False
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if config.train_partitions == 1:
        print(
            "[runtime] full_dataset_training "
            f"train_samples={len(train_records)} epochs={config.epochs} "
            "validation_frequency=each_epoch"
        )
        train_dataset = _build_dataset(
            records=train_records,
            label_map=label_map,
            config=config,
            cache_device=training_cache_device,
        )
        train_loader = _build_dataloader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=effective_num_workers,
            device=device,
        )
        fit_callbacks = list(training_callbacks)
        epoch_validation_callback = None
        if validation_loader is not None:
            epoch_validation_callback = StreamingValidationCallback(
                validation_loader=validation_loader,
                num_classes=len(label_map),
                controller=cycle_controller,
                checkpoint_path=checkpoint_path,
                reduce_lr_factor=config.reduce_lr_factor,
                min_learning_rate=config.min_learning_rate,
            )
            fit_callbacks.append(epoch_validation_callback)
        with keras.device(keras_device_name):
            history = model.fit(
                train_loader,
                epochs=config.epochs,
                callbacks=fit_callbacks,
                verbose=1,
            )
        _extend_history(history_payload, history)
        if epoch_validation_callback is not None:
            completed_cycles = epoch_validation_callback.completed_epochs
            stopped_early = epoch_validation_callback.stopped_early
        else:
            completed_cycles = len(history_payload.get("loss", []))
        del train_loader
        _release_dataset(train_dataset)

    for cycle_index in (
        range(config.partition_cycles)
        if config.train_partitions > 1
        else ()
    ):
        cycle_history_start = len(history_payload.get("accuracy", []))
        train_partitions = _partition_records(
            train_records,
            config.train_partitions,
            config.seed + cycle_index,
        )
        print(
            f"[runtime] partition_cycle={cycle_index + 1}/{config.partition_cycles}"
        )
        for partition_index, partition_records in enumerate(train_partitions, start=1):
            print(
                "[runtime] training "
                f"cycle={cycle_index + 1}/{config.partition_cycles} "
                f"partition={partition_index}/{len(train_partitions)} "
                f"train_samples={len(partition_records)} "
                f"epochs={config.epochs}"
            )
            train_dataset = _build_dataset(
                records=partition_records,
                label_map=label_map,
                config=config,
                cache_device=training_cache_device,
            )
            train_loader = _build_dataloader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=effective_num_workers,
                device=device,
            )
            with keras.device(keras_device_name):
                history = model.fit(
                    train_loader,
                    epochs=config.epochs,
                    callbacks=training_callbacks,
                    verbose=1,
                )
            _extend_history(history_payload, history)
            del train_loader
            _release_dataset(train_dataset)

        if validation_loader is not None:
            with keras.device(keras_device_name):
                validation_metrics = _evaluate_loader_metrics(
                    model,
                    validation_loader,
                    num_classes=len(label_map),
                )
            _append_validation_history(history_payload, validation_metrics)
            validation_loss = float(validation_metrics["loss"])
            validation_f1 = float(validation_metrics["f1_score"])
            cycle_train_accuracies = history_payload.get("accuracy", [])[
                cycle_history_start:
            ]
            cycle_train_accuracy = (
                float(np.mean(cycle_train_accuracies))
                if cycle_train_accuracies
                else 0.0
            )
            print(
                "[runtime] cycle_validation "
                f"cycle={cycle_index + 1}/{config.partition_cycles} "
                f"train_accuracy={cycle_train_accuracy:.6f} "
                f"loss={validation_loss:.6f} "
                f"accuracy={float(validation_metrics['accuracy']):.6f} "
                f"f1_score={validation_f1:.6f}"
            )
            cycle_decision = cycle_controller.update(
                cycle=cycle_index + 1,
                train_accuracy=cycle_train_accuracy,
                validation_accuracy=float(validation_metrics["accuracy"]),
                validation_f1=validation_f1,
                validation_loss=validation_loss,
            )
            print(f"[runtime] cycle_callback {cycle_decision.message}")
            if cycle_decision.save_checkpoint:
                model.save_weights(checkpoint_path)
                print(
                    "[runtime] saved best checkpoint "
                    f"val_f1_score={cycle_controller.best_validation_f1:.6f}"
                )
            if cycle_decision.reduce_learning_rate:
                current_learning_rate = _optimizer_learning_rate(model)
                reduced_learning_rate = max(
                    config.min_learning_rate,
                    current_learning_rate * config.reduce_lr_factor,
                )
                if reduced_learning_rate < current_learning_rate:
                    _set_optimizer_learning_rate(model, reduced_learning_rate)
            model.reset_metrics()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _print_cuda_memory("[runtime][cycle_validation_released]")
            if cycle_decision.stop_training:
                stopped_early = True
                print(
                    "[runtime] early stopping after complete partition cycle "
                    f"{cycle_index + 1}"
                )
                completed_cycles = cycle_index + 1
                break
        completed_cycles = cycle_index + 1

    if validation_loader is not None:
        del validation_loader
    if validation_dataset is not None:
        _release_dataset(validation_dataset)
    if checkpoint_path.exists():
        model.load_weights(checkpoint_path)
    final_weights_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(final_weights_path)
    with keras.device(keras_device_name):
        metrics_payload, test_report = _evaluate_test_partitions(
            model=model,
            test_records=test_records,
            config=config,
            label_map=label_map,
            device=device,
            run_dir=run_dir,
        )

    dataset_manifest = {
        "scenario": config.scenario,
        "official_train_size": len(official_train_records),
        "train_size": len(train_records),
        "validation_size": len(validation_records),
        "test_size": len(test_records),
        "validation_split": split_payload,
        "completed_cycles": completed_cycles,
        "stopped_early": stopped_early,
        "best_validation_f1": (
            cycle_controller.best_validation_f1
            if validation_records
            else None
        ),
        "train_samples": [record.to_dict() for record in train_records[:25]],
        "validation_samples": [
            record.to_dict() for record in validation_records[:25]
        ],
        "test_samples": [record.to_dict() for record in test_records[:25]],
    }
    config_payload = {
        **config.to_dict(),
        "input_shape": list(input_shape),
        "num_classes": len(label_map),
        "validation_split": split_payload,
        "keras_backend": keras.backend.backend(),
        "device": device,
        "num_workers": effective_num_workers,
        "cache_device": cache_device,
        "training_cache_device": training_cache_device,
    }
    summary_payload = {
        "scenario": config.scenario,
        "model": config.model,
        "modality": config.modality,
        "run_dir": str(run_dir),
        "input_shape": list(input_shape),
        "num_classes": len(label_map),
        "train_size": len(train_records),
        "validation_size": len(validation_records),
        "test_size": len(test_records),
        "validation_split": split_payload,
        "best_checkpoint": str(checkpoint_path),
        "final_weights": str(final_weights_path),
        "architecture_json": str(run_architecture_path),
        "metrics": metrics_payload,
        "parameter_count": model_parameter_count(model),
        "keras_backend": keras.backend.backend(),
        "device": device,
        "num_workers": effective_num_workers,
        "cache_device": cache_device,
        "training_cache_device": training_cache_device,
        "test_predictions": test_report["predictions_csv"],
    }

    _write_json(run_dir / "config.json", config_payload)
    _write_json(run_dir / "history.json", history_payload)
    _write_json(run_dir / "metrics.json", metrics_payload)
    _write_json(run_dir / "test_report.json", test_report)
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
