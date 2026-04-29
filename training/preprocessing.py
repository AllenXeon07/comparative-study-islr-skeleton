from __future__ import annotations

from pathlib import Path

import numpy as np


LANDMARK_CHANNEL_COUNT = 5

_EXPECTED_SHAPES = {
    "hand": (2, 21),
    "body": (33,),
    "face": (478,),
}

_MISSING_POINT = np.zeros(LANDMARK_CHANNEL_COUNT, dtype=np.float32)


def _load_raw_component_array(path: Path) -> np.ndarray:
    try:
        array = np.load(path, allow_pickle=False)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported .npy payload at {path}. Expected numeric skeleton arrays; "
            "only keypoint index files should require allow_pickle=True."
        ) from exc
    return np.asarray(array, dtype=np.float32)


def _normalize_hand_shape(array: np.ndarray) -> np.ndarray:
    if array.ndim != 4:
        raise ValueError(
            f"Hand skeleton arrays must have shape (frames, 2, 21, channels); received {array.shape}."
        )

    frame_count, hand_count, point_count, channel_count = array.shape
    normalized = np.broadcast_to(_MISSING_POINT, (frame_count, 2, 21, LANDMARK_CHANNEL_COUNT)).copy()
    normalized[:, : min(hand_count, 2), : min(point_count, 21), : min(channel_count, LANDMARK_CHANNEL_COUNT)] = (
        array[:, : min(hand_count, 2), : min(point_count, 21), : min(channel_count, LANDMARK_CHANNEL_COUNT)]
    )
    return normalized


def _normalize_single_group_shape(array: np.ndarray, component: str) -> np.ndarray:
    expected_points = _EXPECTED_SHAPES[component][0]
    if array.ndim != 3:
        raise ValueError(
            f"{component.capitalize()} skeleton arrays must have shape (frames, {expected_points}, channels); "
            f"received {array.shape}."
        )

    frame_count, point_count, channel_count = array.shape
    normalized = np.broadcast_to(_MISSING_POINT, (frame_count, expected_points, LANDMARK_CHANNEL_COUNT)).copy()
    normalized[:, : min(point_count, expected_points), : min(channel_count, LANDMARK_CHANNEL_COUNT)] = (
        array[:, : min(point_count, expected_points), : min(channel_count, LANDMARK_CHANNEL_COUNT)]
    )
    return normalized


def _normalize_component_shape(component: str, array: np.ndarray) -> np.ndarray:
    if component == "hand":
        return _normalize_hand_shape(array)
    if component in {"body", "face"}:
        return _normalize_single_group_shape(array, component)
    raise ValueError(f"Unsupported component '{component}'.")


def _replace_nan_with_zero(array: np.ndarray) -> np.ndarray:
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def _flatten_hand_landmarks(array: np.ndarray) -> np.ndarray:
    return array.reshape(array.shape[0], -1, LANDMARK_CHANNEL_COUNT)


def preprocess_component_array(
    component: str,
    array: np.ndarray,
) -> np.ndarray:
    normalized = _normalize_component_shape(component, np.asarray(array, dtype=np.float32))
    processed = _replace_nan_with_zero(normalized)
    if component == "hand":
        return _flatten_hand_landmarks(processed)
    return processed


def load_preprocessed_component(
    path: Path,
    component: str,
) -> np.ndarray:
    return preprocess_component_array(
        component=component,
        array=_load_raw_component_array(path),
    )
