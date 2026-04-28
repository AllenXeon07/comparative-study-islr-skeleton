# GPU Devcontainer

This devcontainer is set up for TensorFlow GPU, MediaPipe Tasks, OpenCV, and Jupyter.

## Host requirements

- NVIDIA driver installed on the Windows/WSL host.
- Docker Desktop or Docker Engine with NVIDIA Container Toolkit support.
- `nvidia-smi` should work on the host or inside WSL before opening the container.

## Build and run

From this repository:

```bash
docker compose build
docker compose up
```

Or open the folder in VS Code and choose **Dev Containers: Rebuild and Reopen in Container**.

Jupyter is exposed on port `8890`.

## GPU checks

After the container starts:

```bash
python .devcontainer/check_gpu.py
```

Expected successful signals:

- `nvidia-smi` prints the NVIDIA GPU.
- `tensorflow_gpus=[PhysicalDevice(...)]` is not empty.
- `mediapipe_gpu_delegate=ok`.

If TensorFlow sees the GPU but MediaPipe does not, the usual missing pieces are NVIDIA container graphics capabilities or EGL/OpenGL ES support. This setup enables `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video,display` and installs the EGL/GLES runtime libraries MediaPipe needs.
