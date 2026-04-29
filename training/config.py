from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


ALLOWED_SCENARIOS = ("sd", "si")
ALLOWED_MODALITIES = ("hand", "hand_body", "hand_face", "body_hand_face")
ALLOWED_MODELS = ("cnn1d", "lstm", "transformer")


@dataclass(slots=True)
class ExperimentConfig:
    scenario: str
    model: str
    modality: str
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 0.01
    seed: int = 42
    dataset_dir: Path = Path("Dataset")
    artifacts_dir: Path = Path("artifacts")

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["dataset_dir"] = str(self.dataset_dir)
        data["artifacts_dir"] = str(self.artifacts_dir)
        return data
