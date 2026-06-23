from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from .data import SampleRecord


@dataclass(slots=True)
class TrainingValidationSplit:
    training_records: list[SampleRecord]
    validation_records: list[SampleRecord]
    strategy: str
    training_participants: tuple[str, ...]
    validation_participants: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        training_labels = {record.gloss for record in self.training_records}
        validation_labels = {record.gloss for record in self.validation_records}
        return {
            "strategy": self.strategy,
            "training_size": len(self.training_records),
            "validation_size": len(self.validation_records),
            "validation_fraction": len(self.validation_records)
            / (len(self.training_records) + len(self.validation_records)),
            "training_class_count": len(training_labels),
            "validation_class_count": len(validation_labels),
            "training_participants": list(self.training_participants),
            "validation_participants": list(self.validation_participants),
            "participant_overlap": sorted(
                set(self.training_participants) & set(self.validation_participants)
            ),
            "classes_missing_from_training": sorted(validation_labels - training_labels),
            "classes_missing_from_validation": sorted(training_labels - validation_labels),
        }


def _participant_ids(records: list[SampleRecord]) -> tuple[str, ...]:
    return tuple(sorted({record.participant_id for record in records}))


def _split_signer_dependent(
    records: list[SampleRecord],
    validation_fraction: float,
    seed: int,
) -> TrainingValidationSplit:
    records_by_label: dict[str, list[SampleRecord]] = {}
    for record in records:
        records_by_label.setdefault(record.gloss, []).append(record)

    rng = random.Random(seed)
    training_records: list[SampleRecord] = []
    validation_records: list[SampleRecord] = []
    for label_records in records_by_label.values():
        shuffled = list(label_records)
        rng.shuffle(shuffled)
        if len(shuffled) < 2:
            training_records.extend(shuffled)
            continue
        validation_count = max(1, round(len(shuffled) * validation_fraction))
        validation_count = min(validation_count, len(shuffled) - 1)
        validation_records.extend(shuffled[:validation_count])
        training_records.extend(shuffled[validation_count:])

    rng.shuffle(training_records)
    rng.shuffle(validation_records)
    return TrainingValidationSplit(
        training_records=training_records,
        validation_records=validation_records,
        strategy="sd_class_stratified",
        training_participants=_participant_ids(training_records),
        validation_participants=_participant_ids(validation_records),
    )


def _split_signer_independent(
    records: list[SampleRecord],
    validation_fraction: float,
    seed: int,
    search_iterations: int = 20_000,
) -> TrainingValidationSplit:
    records_by_participant: dict[str, list[SampleRecord]] = {}
    for record in records:
        if not record.participant_id:
            raise ValueError("SI splitting requires a participant ID for every record.")
        records_by_participant.setdefault(record.participant_id, []).append(record)

    participants = sorted(records_by_participant)
    if len(participants) < 3:
        raise ValueError("SI splitting requires at least three participants.")

    labels = sorted({record.gloss for record in records})
    label_indices = {label: index for index, label in enumerate(labels)}
    participant_sizes: list[int] = []
    participant_label_bits: list[int] = []
    participant_index_by_label: dict[str, int] = {label: 0 for label in labels}
    for participant in participants:
        participant_records = records_by_participant[participant]
        participant_sizes.append(len(participant_records))
        label_bits = 0
        for label in {record.gloss for record in participant_records}:
            label_bits |= 1 << label_indices[label]
            participant_index_by_label[label] |= 1 << len(participant_label_bits)
        participant_label_bits.append(label_bits)

    target_validation_size = round(len(records) * validation_fraction)
    validation_participant_count = max(
        1,
        min(len(participants) - 1, round(len(participants) * validation_fraction)),
    )
    rng = random.Random(seed)
    best: tuple[float, tuple[int, ...]] | None = None
    forbidden_validation_masks = tuple(
        participant_mask
        for participant_mask in participant_index_by_label.values()
        if participant_mask.bit_count() <= validation_participant_count
    )
    for _ in range(search_iterations):
        selected_indices = tuple(
            sorted(rng.sample(range(len(participants)), validation_participant_count))
        )
        selected_participant_mask = sum(1 << index for index in selected_indices)
        if any(
            selected_participant_mask & label_mask == label_mask
            for label_mask in forbidden_validation_masks
        ):
            continue
        validation_size = sum(participant_sizes[index] for index in selected_indices)
        validation_label_bits = 0
        for index in selected_indices:
            validation_label_bits |= participant_label_bits[index]
        validation_class_count = validation_label_bits.bit_count()
        size_error = abs(validation_size - target_validation_size) / len(records)
        coverage_error = (len(labels) - validation_class_count) / len(labels)
        score = size_error + 0.05 * coverage_error
        candidate = (score, selected_indices)
        if best is None or candidate < best:
            best = candidate

    if best is None:
        raise RuntimeError("Could not select SI validation participants.")

    validation_participants = {
        participants[index] for index in best[1]
    }
    training_records = [
        record for record in records
        if record.participant_id not in validation_participants
    ]
    validation_records = [
        record for record in records
        if record.participant_id in validation_participants
    ]

    training_counts = Counter(record.gloss for record in training_records)
    validation_counts = Counter(record.gloss for record in validation_records)
    missing_training_labels = sorted(set(validation_counts) - set(training_counts))
    if missing_training_labels:
        raise ValueError(
            "SI participant holdout removed classes from training: "
            + ", ".join(missing_training_labels)
        )
    if not training_records or not validation_records:
        raise ValueError("SI split produced an empty training or validation set.")

    return TrainingValidationSplit(
        training_records=training_records,
        validation_records=validation_records,
        strategy="si_participant_disjoint",
        training_participants=_participant_ids(training_records),
        validation_participants=_participant_ids(validation_records),
    )


def build_training_validation_split(
    records: list[SampleRecord],
    scenario: str,
    validation_fraction: float,
    seed: int,
) -> TrainingValidationSplit:
    if not records:
        raise ValueError("Cannot split an empty record list.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if scenario == "si":
        return _split_signer_independent(records, validation_fraction, seed)
    if scenario == "sd":
        return _split_signer_dependent(records, validation_fraction, seed)
    raise ValueError(f"Unsupported scenario '{scenario}'.")
