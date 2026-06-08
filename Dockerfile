FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV KERAS_BACKEND=torch

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libgles2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel
RUN python -m pip install --no-cache-dir \
    keras \
    jupyterlab \
    matplotlib \
    mediapipe \
    notebook \
    opencv-python \
    pandas \
    scikit-learn
RUN python -m pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

EXPOSE 8888

CMD ["python", "-m", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
