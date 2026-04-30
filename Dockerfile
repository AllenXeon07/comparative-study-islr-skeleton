# Use the CUDA 12 base image on Ubuntu 24.04
FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system essentials (Python and basic libs)
RUN apt-get update && apt-get install -y --no-install-recommends     python3-pip     python3-dev     git     curl     libgl1     libegl1     libgles2     libglib2.0-0     && rm -rf /var/lib/apt/lists/*


WORKDIR /app

# Install Python dependencies from a reproducible lockfile.
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install pip
RUN python3 -m pip install jupyter  
RUN python3 -m pip install --no-cache-dir -r requirements.txt

EXPOSE 8888

# Start Jupyter Notebook by default
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
