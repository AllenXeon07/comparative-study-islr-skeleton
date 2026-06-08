from __future__ import annotations

import os

# Keras backend must be selected before importing keras anywhere in this package.
os.environ.setdefault("KERAS_BACKEND", "torch")

"""Training helpers for modality-selectable sign recognition experiments."""
