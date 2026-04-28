#!/usr/bin/env python3
"""Runtime smoke test for the GPU dev container."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def run(command: list[str]) -> None:
    print(f"$ {' '.join(command)}")
    try:
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
    except FileNotFoundError:
        print(f"{command[0]}: not found")
        return

    output = (completed.stdout + completed.stderr).strip()
    if output:
        print(output)
    print(f"exit_code={completed.returncode}")


def main() -> None:
    if shutil.which("nvidia-smi"):
        run(["nvidia-smi"])
    else:
        print("nvidia-smi: not found")

    try:
        import tensorflow as tf
    except Exception as exc:
        print(f"tensorflow import failed: {exc}")
    else:
        print(f"tensorflow={tf.__version__}")
        print(f"tensorflow_gpus={tf.config.list_physical_devices('GPU')}")

    try:
        import mediapipe as mp
    except Exception as exc:
        print(f"mediapipe import failed: {exc}")
    else:
        print(f"mediapipe={mp.__version__}")
        delegate_names = [delegate.name for delegate in mp.tasks.BaseOptions.Delegate]
        print(f"mediapipe_delegates={delegate_names}")
        model_path = Path("hand_landmarker.task")
        if model_path.exists():
            try:
                options = mp.tasks.vision.HandLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(
                        model_asset_path=str(model_path),
                        delegate=mp.tasks.BaseOptions.Delegate.GPU,
                    ),
                    running_mode=mp.tasks.vision.RunningMode.IMAGE,
                    num_hands=2,
                )
                landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
                landmarker.close()
            except Exception as exc:
                print(f"mediapipe_gpu_delegate=failed: {exc}")
            else:
                print("mediapipe_gpu_delegate=ok")
        else:
            print("mediapipe_gpu_delegate=skipped: hand_landmarker.task not found")


if __name__ == "__main__":
    main()
