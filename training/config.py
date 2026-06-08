from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


ALLOWED_SCENARIOS = ("si",...) #sd, si
ALLOWED_MODALITIES = ("hand_face", "body_hand_face") #"hand", "hand_body", "hand_face", "body_hand_face"
ALLOWED_MODELS = ("cnn1d", "lstm", "transformer", "gcn")


@dataclass(slots=True)
class ExperimentConfig:
    scenario: str = ALLOWED_SCENARIOS[0]
    model: str = ALLOWED_MODELS[0]
    modality: str = ALLOWED_MODALITIES[0]
    frame_sample_count: int = 30
    epochs: int = 100
    batch_size: int = 32
    # Keep notebook/container runs away from PyTorch DataLoader shared-memory
    # exhaustion. Increase explicitly on hosts with a larger /dev/shm.
    num_workers: int = 4
    learning_rate: float = 0.01
    seed: int = 42
    dataset_dir: Path = Path("Dataset")
    artifacts_dir: Path = Path("artifacts")

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["dataset_dir"] = str(self.dataset_dir)
        data["artifacts_dir"] = str(self.artifacts_dir)
        return data
