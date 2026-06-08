from __future__ import annotations

import json
import os
from contextlib import ExitStack
from pathlib import Path
from typing import Any

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np

from .preprocessing import FEATURE_DTYPE, preprocess_component_array


DEFAULT_ARTIFACTS_DIR = Path("artifacts")
MODALITY_COMPONENTS = {
    "hand": ("hand",),
    "hand_body": ("hand", "body"),
    "hand_face": ("hand", "face"),
    "body_hand_face": ("body", "hand", "face"),
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_run_dir(model_path: Path) -> Path:
    path = model_path.expanduser()
    if path.is_dir():
        run_dir = path
    elif path.is_file() and path.name.endswith(".weights.h5"):
        run_dir = path.parent
    else:
        raise FileNotFoundError(
            f"Expected an experiment run directory or .weights.h5 file, got: {model_path}"
        )

    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Could not find config.json beside model artifacts: {config_path}"
        )
    return run_dir


def load_run_config(model_path: Path) -> tuple[Path, dict[str, Any]]:
    run_dir = resolve_run_dir(model_path)
    return run_dir, _read_json(run_dir / "config.json")


def resolve_weights_path(model_path: Path, run_dir: Path) -> Path:
    path = model_path.expanduser()
    if path.is_file() and path.name.endswith(".weights.h5"):
        return path

    for filename in ("best.weights.h5", "final.weights.h5"):
        candidate = run_dir / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No weights file found in {run_dir}")


def load_label_map(config: dict[str, Any], run_dir: Path) -> dict[str, int]:
    scenario = str(config.get("scenario", "")).strip()
    artifacts_dir = Path(str(config.get("artifacts_dir", DEFAULT_ARTIFACTS_DIR)))

    candidates = []
    if scenario:
        candidates.append(artifacts_dir / "label_maps" / f"{scenario}_labels.json")
    candidates.append(run_dir / "label_map.json")

    for path in candidates:
        if path.exists():
            payload = _read_json(path)
            return {str(label): int(index) for label, index in payload.items()}

    num_classes = int(config["num_classes"])
    return {str(index): index for index in range(num_classes)}


def label_lookup(label_map: dict[str, int]) -> dict[int, str]:
    return {index: label for label, index in label_map.items()}


def modality_components(modality: str) -> tuple[str, ...]:
    if modality not in MODALITY_COMPONENTS:
        allowed = ", ".join(sorted(MODALITY_COMPONENTS))
        raise ValueError(f"Unsupported modality '{modality}'. Expected one of: {allowed}")
    return MODALITY_COMPONENTS[modality]


def _build_landmarker(component: str):
    from process_skeleton_dataset import (
        FaceLandmarker,
        HandLandmarker,
        PoseLandmarker,
        build_face_options,
        build_hand_options,
        build_pose_options,
        create_landmarker,
    )

    if component == "hand":
        return create_landmarker(HandLandmarker, build_hand_options, "hand landmarker")
    if component == "body":
        return create_landmarker(PoseLandmarker, build_pose_options, "pose landmarker")
    if component == "face":
        return create_landmarker(FaceLandmarker, build_face_options, "face landmarker")
    raise ValueError(f"Unsupported component: {component}")


def extract_video_components(
    video_path: Path,
    modality: str,
    frame_sample_count: int,
) -> dict[str, np.ndarray]:
    from process_skeleton_dataset import (
        EXPECTED_BODY_POINTS,
        EXPECTED_FACE_POINTS,
        EXPECTED_HAND_POINTS,
        classify_hands,
        ensure_requested_models,
        frame_to_mp_image,
        has_valid_x_coordinate,
        landmarks_to_array,
        load_video_frames_numpy,
        make_timestamps_ms,
        normalize_landmarks,
        sample_frame_indices,
        serialize_face_result,
        serialize_hand_result,
        serialize_pose_result,
    )

    components = modality_components(modality)
    include_hand = "hand" in components
    include_body = "body" in components
    include_face = "face" in components
    ensure_requested_models(include_hand, include_body, include_face)

    source_frames, fps = load_video_frames_numpy(video_path)
    sampled_indices = sample_frame_indices(len(source_frames), sample_count=frame_sample_count)
    sampled_frames = source_frames[sampled_indices]
    timestamps_ms = make_timestamps_ms(sampled_indices, fps)

    raw_parts: dict[str, list[np.ndarray]] = {component: [] for component in components}
    last_valid: dict[str, list[dict[str, float]] | None] = {
        "left_hand": None,
        "right_hand": None,
        "body": None,
        "face": None,
    }

    with ExitStack() as stack:
        landmarkers: dict[str, Any] = {}
        for component in components:
            landmarker, _delegate = _build_landmarker(component)
            landmarkers[component] = stack.enter_context(landmarker)

        for frame, timestamp_ms in zip(sampled_frames, timestamps_ms):
            mp_image = frame_to_mp_image(frame)

            if include_hand:
                hand_result = landmarkers["hand"].detect_for_video(mp_image, timestamp_ms)
                hands = classify_hands(serialize_hand_result(hand_result))
                left_hand = normalize_landmarks(
                    hands.get("left", []),
                    EXPECTED_HAND_POINTS,
                    fallback=last_valid["left_hand"],
                )
                right_hand = normalize_landmarks(
                    hands.get("right", []),
                    EXPECTED_HAND_POINTS,
                    fallback=last_valid["right_hand"],
                )
                if has_valid_x_coordinate(left_hand):
                    last_valid["left_hand"] = [dict(point) for point in left_hand]
                if has_valid_x_coordinate(right_hand):
                    last_valid["right_hand"] = [dict(point) for point in right_hand]
                raw_parts["hand"].append(
                    np.stack(
                        [landmarks_to_array(left_hand), landmarks_to_array(right_hand)],
                        axis=0,
                    )
                )

            if include_body:
                pose_result = landmarkers["body"].detect_for_video(mp_image, timestamp_ms)
                serialized_body = serialize_pose_result(pose_result)
                body = normalize_landmarks(
                    serialized_body[0] if serialized_body else [],
                    EXPECTED_BODY_POINTS,
                    fallback=last_valid["body"],
                )
                if has_valid_x_coordinate(body):
                    last_valid["body"] = [dict(point) for point in body]
                raw_parts["body"].append(landmarks_to_array(body))

            if include_face:
                face_result = landmarkers["face"].detect_for_video(mp_image, timestamp_ms)
                serialized_face = serialize_face_result(face_result)
                face = normalize_landmarks(
                    serialized_face[0] if serialized_face else [],
                    EXPECTED_FACE_POINTS,
                    fallback=last_valid["face"],
                )
                if has_valid_x_coordinate(face):
                    last_valid["face"] = [dict(point) for point in face]
                raw_parts["face"].append(landmarks_to_array(face))

    return {
        component: np.stack(component_frames, axis=0)
        for component, component_frames in raw_parts.items()
    }


def preprocess_video(
    video_path: Path,
    modality: str,
    frame_sample_count: int,
) -> np.ndarray:
    raw_components = extract_video_components(video_path, modality, frame_sample_count)
    feature_parts = [
        preprocess_component_array(component, raw_components[component])
        for component in modality_components(modality)
    ]
    features = (
        np.concatenate(feature_parts, axis=1)
        .reshape(feature_parts[0].shape[0], -1)
        .astype(FEATURE_DTYPE, copy=False)
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(FEATURE_DTYPE, copy=False)


def load_model_from_config(config: dict[str, Any], weights_path: Path):
    from .models import build_model

    input_shape = tuple(int(value) for value in config["input_shape"])
    num_classes = int(config["num_classes"])
    learning_rate = float(config.get("learning_rate", 0.01))

    model = build_model(
        model_name=str(config["model"]),
        input_shape=input_shape,
        num_classes=num_classes,
        learning_rate=learning_rate,
    )
    model.load_weights(weights_path)
    return model


def load_model_from_run(model_path: Path):
    run_dir, config = load_run_config(model_path)
    weights_path = resolve_weights_path(model_path, run_dir)
    model = load_model_from_config(config, weights_path)
    return model, config, weights_path, run_dir


def predict_video(
    video_path: Path,
    model_path: Path,
    top_k: int = 5,
) -> dict[str, Any]:
    run_dir, config = load_run_config(model_path)
    weights_path = resolve_weights_path(model_path, run_dir)
    label_map = load_label_map(config, run_dir)
    labels_by_index = label_lookup(label_map)

    modality = str(config["modality"])
    frame_sample_count = int(config["frame_sample_count"])
    expected_shape = tuple(int(value) for value in config["input_shape"])
    features = preprocess_video(video_path, modality, frame_sample_count)
    actual_shape = tuple(int(value) for value in features.shape)
    if actual_shape != expected_shape:
        raise ValueError(
            f"Preprocessed feature shape {actual_shape} does not match model input {expected_shape}."
        )

    model = load_model_from_config(config, weights_path)
    probabilities = np.asarray(model.predict(features[None, ...], verbose=0))[0]
    ordered_indices = np.argsort(probabilities)[::-1][:top_k]
    top_predictions = [
        {
            "rank": rank + 1,
            "label": labels_by_index.get(int(index), str(index)),
            "index": int(index),
            "confidence": float(probabilities[index]),
        }
        for rank, index in enumerate(ordered_indices)
    ]

    return {
        "video_path": str(video_path),
        "run_dir": str(run_dir),
        "weights_path": str(weights_path),
        "model": str(config["model"]),
        "scenario": str(config.get("scenario", "")),
        "modality": modality,
        "frame_sample_count": frame_sample_count,
        "input_shape": list(expected_shape),
        "top_predictions": top_predictions,
    }


def evaluate_prediction(result: dict[str, Any], expected_label: str | None) -> dict[str, Any]:
    if expected_label is None:
        return {**result, "expected_label": None, "correct": None}

    prediction = result["top_predictions"][0]["label"]
    return {
        **result,
        "expected_label": expected_label,
        "correct": prediction == expected_label,
    }
