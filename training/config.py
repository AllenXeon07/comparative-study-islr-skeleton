from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


ALLOWED_SCENARIOS = ("sd", "si")
ALLOWED_MODALITIES = ("hand", "hand_body", "hand_face", "body_hand_face")
ALLOWED_MODELS = ("cnn1d", "lstm", "transformer", "gcn")
ALLOWED_CACHE_DEVICES = ("cpu", "gpu")
ALLOWED_OPTIMIZERS = ("adamw", "adam", "sgd")


@dataclass(slots=True)
class ExperimentConfig:
    scenario: str = "si"
    model: str = "gcn"
    modality: str = "body_hand_face"
    frame_sample_count: int = 30
    # The 553-landmark GCN is trained in six sequential partitions to keep
    # activation and dataset memory stable. Validation runs after each cycle,
    # once all six partitions have been seen.
    epochs: int = 2
    batch_size: int = 8
    # Docker defaults /dev/shm to 64 MiB. Keep collation in the main process
    # unless the container is started with a deliberately larger shm allocation.
    num_workers: int = 0
    cache_device: str = ALLOWED_CACHE_DEVICES[0]
    train_partitions: int = 6
    partition_cycles: int = 50
    validate_during_training: bool = True
    validation_fraction: float = 0.2
    evaluation_batch_size: int = 8
    test_partitions: int = 3
    optimizer: str = "adamw"
    learning_rate: float = 0.001
    partition_learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    global_clipnorm: float = 1.0
    reduce_lr_factor: float = 0.5
    reduce_lr_patience: int = 4
    min_learning_rate: float = 0.00001
    early_stopping_patience: int = 6
    early_stopping_min_cycles: int = 20
    early_stopping_min_delta: float = 0.001
    seed: int = 128
    dataset_dir: Path = Path("Dataset")
    artifacts_dir: Path = Path("artifacts")

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["dataset_dir"] = str(self.dataset_dir)
        data["artifacts_dir"] = str(self.artifacts_dir)
        return data
