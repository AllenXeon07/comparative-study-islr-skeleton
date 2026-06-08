from __future__ import annotations

import os
import subprocess
import sys
import time
import traceback

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
import torch
from keras import Input
from keras.layers import Conv1D, Dense, GlobalAveragePooling1D

from training.models import build_cnn1d


def print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def run_stage(name: str, fn) -> bool:
    print_header(name)
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - diagnostics script
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        return False
    print(f"[PASS] {name}")
    return True


def shell_text(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        return "<command not found>"
    except subprocess.CalledProcessError as exc:
        return exc.stderr.strip() or exc.stdout.strip() or f"<exit code {exc.returncode}>"
    return completed.stdout.strip() or "<no output>"


def stage_environment() -> None:
    print("Python:", sys.version.split()[0])
    print("Keras:", keras.__version__)
    print("Keras backend:", keras.backend.backend())
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))


def stage_nvidia_smi() -> None:
    print(shell_text(["nvidia-smi"]))


def stage_torch_matmul() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = torch.randn((4096, 4096), device=device)
    b = torch.randn((4096, 4096), device=device)
    start = time.perf_counter()
    c = a @ b
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print("Torch matmul device:", c.device)
    print("Torch matmul seconds:", round(elapsed, 4))


def stage_keras_dense_train() -> None:
    x = np.random.random((64, 64)).astype("float32")
    y = np.random.randint(0, 4, size=(64,))
    model = keras.Sequential(
        [
            Input(shape=(64,)),
            Dense(64, activation="relu"),
            Dense(4),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    history = model.fit(x, y, epochs=1, batch_size=8, verbose=0)
    print("Dense loss:", float(history.history["loss"][-1]))


def stage_keras_conv1d_build() -> None:
    model = keras.Sequential(
        [
            Input(shape=(32, 20)),
            Conv1D(32, kernel_size=3, padding="same", activation="relu"),
            GlobalAveragePooling1D(),
            Dense(3),
        ]
    )
    model.summary(print_fn=print)


def stage_repo_model_build() -> None:
    model = build_cnn1d(input_shape=(32, 20), num_classes=3)
    model.summary(print_fn=print)


def main() -> None:
    stages = [
        ("Environment", stage_environment),
        ("nvidia-smi", stage_nvidia_smi),
        ("Torch Matmul", stage_torch_matmul),
        ("Keras Dense Train", stage_keras_dense_train),
        ("Keras Conv1D Build", stage_keras_conv1d_build),
        ("Repo CNN1D Build", stage_repo_model_build),
    ]
    results = [(name, run_stage(name, fn)) for name, fn in stages]

    print_header("Summary")
    for name, ok in results:
        print(f"{'PASS' if ok else 'FAIL'}: {name}")


if __name__ == "__main__":
    main()
