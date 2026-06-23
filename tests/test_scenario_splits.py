from __future__ import annotations

import unittest

from training.data import SampleRecord
from training.scenario_splits import build_training_validation_split


def record(participant: str, gloss: str, index: int, scenario: str) -> SampleRecord:
    return SampleRecord(
        participant_id=participant,
        video_file=f"{participant}-{gloss}-{index}.mp4",
        gloss=gloss,
        asl_lex_code="",
        split="train",
        scenario=scenario,
    )


class ScenarioSplitTests(unittest.TestCase):
    def test_si_split_is_deterministic_and_participant_disjoint(self) -> None:
        records = [
            record(f"P{participant}", gloss, repeat, "si")
            for participant in range(10)
            for gloss in ("A", "B", "C")
            for repeat in range(participant + 1)
        ]

        first = build_training_validation_split(records, "si", 0.2, seed=128)
        second = build_training_validation_split(records, "si", 0.2, seed=128)

        self.assertEqual(first.validation_participants, second.validation_participants)
        self.assertFalse(
            set(first.training_participants) & set(first.validation_participants)
        )
        self.assertEqual(
            {item.gloss for item in first.training_records},
            {"A", "B", "C"},
        )
        self.assertEqual(
            {item.gloss for item in first.validation_records},
            {"A", "B", "C"},
        )

    def test_sd_split_is_class_stratified(self) -> None:
        records = [
            record(f"P{index % 3}", gloss, index, "sd")
            for gloss in ("A", "B", "C")
            for index in range(10)
        ]

        result = build_training_validation_split(records, "sd", 0.2, seed=128)

        self.assertEqual(len(result.training_records), 24)
        self.assertEqual(len(result.validation_records), 6)
        self.assertEqual(
            {item.gloss for item in result.validation_records},
            {"A", "B", "C"},
        )


if __name__ == "__main__":
    unittest.main()
