from __future__ import annotations

import csv
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import ALLOWED_MODALITIES
from .preprocessing import FEATURE_DTYPE, load_preprocessed_component


SPLIT_FILES = {
    "sd": {"train": "sd_train.csv", "test": "sd_test.csv"},
    "si": {"train": "si_train.csv", "test": "si_test.csv"},
}

MODALITY_COMPONENTS = {
    "hand": ("hand",),
    "hand_body": ("hand", "body"),
    "hand_face": ("hand", "face"),
    "body_hand_face": ("body", "hand", "face"),
}

_WARNED_NONFINITE_SAMPLES: set[tuple[str, str]] = set()


@dataclass(slots=True)
class SampleRecord:
    participant_id: str
    video_file: str
    gloss: str
    asl_lex_code: str
    split: str
    scenario: str

    def to_dict(self) -> dict[str, str]:
        return {
            "participant_id": self.participant_id,
            "video_file": self.video_file,
            "gloss": self.gloss,
            "asl_lex_code": self.asl_lex_code,
            "split": self.split,
            "scenario": self.scenario,
        }


def modality_components(modality: str) -> tuple[str, ...]:
    if modality not in MODALITY_COMPONENTS:
        raise ValueError(
            f"Unsupported modality '{modality}'. Expected one of: {', '.join(ALLOWED_MODALITIES)}"
        )
    return MODALITY_COMPONENTS[modality]


def load_split_records(dataset_dir: Path, scenario: str, split: str) -> list[SampleRecord]:
    split_file = dataset_dir / "split" / SPLIT_FILES[scenario][split]
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    records: list[SampleRecord] = []
    with split_file.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                SampleRecord(
                    participant_id=(row.get("Participant ID") or "").strip(),
                    video_file=(row.get("Video file") or "").strip(),
                    gloss=(row.get("Gloss") or "").strip(),
                    asl_lex_code=(row.get("ASL-LEX Code") or "").strip(),
                    split=split,
                    scenario=scenario,
                )
            )

    if not records:
        raise ValueError(f"No records found in split file: {split_file}")
    return records


def build_label_map(records: Iterable[SampleRecord]) -> dict[str, int]:
    labels = sorted({record.gloss for record in records if record.gloss})
    if not labels:
        raise ValueError("No non-empty Gloss labels were found in the provided records.")
    return {label: index for index, label in enumerate(labels)}


def skeleton_feature_root(dataset_dir: Path, frame_sample_count: int) -> Path:
    return dataset_dir / "skeleton" / f"{frame_sample_count}frame"


def sample_component_path(dataset_dir: Path, frame_sample_count: int, component: str, video_file: str) -> Path:
    return skeleton_feature_root(dataset_dir, frame_sample_count) / component / Path(video_file).with_suffix(".npy")


def validate_modality_support(
    dataset_dir: Path,
    records: Iterable[SampleRecord],
    modality: str,
    frame_sample_count: int,
) -> None:
    components = modality_components(modality)
    missing: list[str] = []
    sample_records = list(records)
    feature_root = skeleton_feature_root(dataset_dir, frame_sample_count)
    for component in components:
        component_dir = feature_root / component
        if not component_dir.exists():
            missing.append(f"Missing skeleton directory: {component_dir}")
            continue
        for record in sample_records[:10]:
            sample_path = sample_component_path(dataset_dir, frame_sample_count, component, record.video_file)
            if not sample_path.exists():
                missing.append(f"Missing {component} skeleton for sample '{record.video_file}'")
                break

    if missing:
        raise FileNotFoundError(
            f"Modality '{modality}' is unavailable.\n" + "\n".join(missing)
        )


def load_sample_features(
    dataset_dir: Path,
    record: SampleRecord,
    modality: str,
    frame_sample_count: int,
) -> np.ndarray:
    feature_parts: list[np.ndarray] = []
    for component in modality_components(modality):
        sample_path = sample_component_path(dataset_dir, frame_sample_count, component, record.video_file)
        try:
            array = load_preprocessed_component(sample_path, component)
        except Exception as exc:
            file_size = sample_path.stat().st_size if sample_path.exists() else "missing"
            raise RuntimeError(
                "Failed to load skeleton component "
                f"video_file='{record.video_file}', gloss='{record.gloss}', "
                f"participant_id='{record.participant_id}', scenario='{record.scenario}', "
                f"split='{record.split}', modality='{modality}', component='{component}', "
                f"path='{sample_path}', file_size_bytes={file_size}."
            ) from exc
        feature_parts.append(array)

    sequence_lengths = {part.shape[0] for part in feature_parts}
    if len(sequence_lengths) != 1:
        component_shapes = {
            component: part.shape
            for component, part in zip(modality_components(modality), feature_parts, strict=True)
        }
        raise ValueError(
            f"Sample '{record.video_file}' has inconsistent sequence lengths across modality '{modality}'. "
            f"Component shapes: {component_shapes}"
        )
    channel_counts = {part.shape[-1] for part in feature_parts}
    if channel_counts != {5}:
        raise ValueError(
            f"Sample '{record.video_file}' has unsupported channel counts across modality '{modality}': "
            f"{sorted(channel_counts)}"
        )

    features = (
        np.concatenate(feature_parts, axis=1)
        .reshape(feature_parts[0].shape[0], -1)
        .astype(FEATURE_DTYPE, copy=False)
    )
    finite_mask = np.isfinite(features)
    if finite_mask.all():
        return features

    key = (record.video_file, modality)
    if key not in _WARNED_NONFINITE_SAMPLES:
        non_finite_count = int(features.size - np.count_nonzero(finite_mask))
        warnings.warn(
            (
                f"Sample '{record.video_file}' for modality '{modality}' contains "
                f"{non_finite_count} non-finite values. Replacing them with 0.0 before training."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        _WARNED_NONFINITE_SAMPLES.add(key)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(FEATURE_DTYPE, copy=False)


class SkeletonDataset(Dataset):
    """Lazy-loading dataset that reads .npy files per sample."""

    def __init__(
        self,
        records: list[SampleRecord],
        dataset_dir: Path,
        label_map: dict[str, int],
        modality: str,
        frame_sample_count: int,
    ) -> None:
        self.records = list(records)
        self.dataset_dir = dataset_dir
        self.label_map = label_map
        self.modality = modality
        self.frame_sample_count = frame_sample_count

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        features = load_sample_features(
            self.dataset_dir,
            record,
            self.modality,
            self.frame_sample_count,
        )
        label_index = self.label_map[record.gloss]
        return (
            torch.from_numpy(features.astype(np.float32, copy=False)),
            torch.tensor(label_index, dtype=torch.long),
        )


class PreloadedSkeletonDataset(Dataset):
    """Dataset backed by already-loaded tensors on CPU or CUDA."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        if len(features) != len(labels):
            raise ValueError(
                f"Feature/label length mismatch: features={len(features)} labels={len(labels)}"
            )
        self.features = features
        self.labels = labels

    @classmethod
    def from_records(
        cls,
        records: list[SampleRecord],
        dataset_dir: Path,
        label_map: dict[str, int],
        modality: str,
        frame_sample_count: int,
        device: torch.device | str,
    ) -> "PreloadedSkeletonDataset":
        if not records:
            raise ValueError("Cannot preload an empty dataset.")

        first_features = load_sample_features(
            dataset_dir,
            records[0],
            modality,
            frame_sample_count,
        )
        feature_shape = first_features.shape
        target_device = torch.device(device)
        features = torch.empty(
            (len(records), *feature_shape),
            dtype=torch.float32,
            device=target_device,
        )
        labels = torch.empty(len(records), dtype=torch.long, device=target_device)

        features[0].copy_(torch.from_numpy(first_features.astype(np.float32, copy=False)))
        labels[0] = label_map[records[0].gloss]
        for index, record in enumerate(records[1:], start=1):
            sample_features = load_sample_features(
                dataset_dir,
                record,
                modality,
                frame_sample_count,
            )
            if sample_features.shape != feature_shape:
                raise ValueError(
                    f"Sample '{record.video_file}' has shape {sample_features.shape}, "
                    f"expected {feature_shape}."
                )
            features[index].copy_(torch.from_numpy(sample_features.astype(np.float32, copy=False)))
            labels[index] = label_map[record.gloss]

        return cls(features=features, labels=labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def dataloader_steps(dataset_size: int, batch_size: int) -> int:
    return math.ceil(dataset_size / batch_size)


def infer_input_shape(
    dataset_dir: Path,
    record: SampleRecord,
    modality: str,
    frame_sample_count: int,
) -> tuple[int, int]:
    features = load_sample_features(dataset_dir, record, modality, frame_sample_count)
    return int(features.shape[0]), int(features.shape[1])
