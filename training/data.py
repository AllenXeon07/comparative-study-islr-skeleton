from __future__ import annotations

import csv
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from keras.utils import Sequence

from .config import ALLOWED_MODALITIES
from .preprocessing import load_preprocessed_component


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


def sample_component_path(dataset_dir: Path, component: str, video_file: str) -> Path:
    return dataset_dir / "skeleton" / "raw" / component / Path(video_file).with_suffix(".npy")


def validate_modality_support(dataset_dir: Path, records: Iterable[SampleRecord], modality: str) -> None:
    components = modality_components(modality)
    missing: list[str] = []
    sample_records = list(records)
    for component in components:
        component_dir = dataset_dir / "skeleton" / "raw" / component
        if not component_dir.exists():
            missing.append(f"Missing skeleton directory: {component_dir}")
            continue
        for record in sample_records[:10]:
            sample_path = sample_component_path(dataset_dir, component, record.video_file)
            if not sample_path.exists():
                missing.append(f"Missing {component} skeleton for sample '{record.video_file}'")
                break

    if missing:
        raise FileNotFoundError(
            f"Modality '{modality}' is unavailable.\n" + "\n".join(missing)
        )


def load_sample_features(dataset_dir: Path, record: SampleRecord, modality: str) -> np.ndarray:
    feature_parts: list[np.ndarray] = []
    for component in modality_components(modality):
        array = load_preprocessed_component(sample_component_path(dataset_dir, component, record.video_file), component)
        feature_parts.append(array)

    sequence_lengths = {part.shape[0] for part in feature_parts}
    if len(sequence_lengths) != 1:
        raise ValueError(
            f"Sample '{record.video_file}' has inconsistent sequence lengths across modality '{modality}'."
        )
    channel_counts = {part.shape[-1] for part in feature_parts}
    if channel_counts != {5}:
        raise ValueError(
            f"Sample '{record.video_file}' has unsupported channel counts across modality '{modality}': "
            f"{sorted(channel_counts)}"
        )

    features = np.concatenate(feature_parts, axis=1).reshape(feature_parts[0].shape[0], -1).astype(np.float32)
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
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


class SkeletonSequence(Sequence):
    """Lazy-loading Keras sequence that loads .npy files per batch."""

    def __init__(
        self,
        records: list[SampleRecord],
        dataset_dir: Path,
        label_map: dict[str, int],
        modality: str,
        batch_size: int,
        shuffle: bool,
        seed: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.records = list(records)
        self.dataset_dir = dataset_dir
        self.label_map = label_map
        self.modality = modality
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.indexes = np.arange(len(self.records))
        self.on_epoch_end()

    def __len__(self) -> int:
        return math.ceil(len(self.records) / self.batch_size)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        start = index * self.batch_size
        stop = min(start + self.batch_size, len(self.records))
        batch_indexes = self.indexes[start:stop]
        batch_records = [self.records[i] for i in batch_indexes]

        batch_x = [load_sample_features(self.dataset_dir, record, self.modality) for record in batch_records]
        batch_y = [self.label_map[record.gloss] for record in batch_records]
        one_hot_y = np.eye(len(self.label_map), dtype=np.float32)[batch_y]
        return np.stack(batch_x, axis=0), one_hot_y

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.indexes)


def infer_input_shape(dataset_dir: Path, record: SampleRecord, modality: str) -> tuple[int, int]:
    features = load_sample_features(dataset_dir, record, modality)
    return int(features.shape[0]), int(features.shape[1])
