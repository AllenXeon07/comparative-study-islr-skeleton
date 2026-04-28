#!/usr/bin/env python3
"""Process raw videos into body, hand, and face skeleton arrays."""

from __future__ import annotations

import urllib.request
from contextlib import ExitStack
from pathlib import Path
from typing import Iterable

import cv2
import mediapipe as mp
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Dataset"
RAW_DATASET_DIR = DATASET_DIR / "raw"
SKELETON_DATASET_DIR = DATASET_DIR / "skeleton" / "raw"

HAND_MODEL_PATH = BASE_DIR / "hand_landmarker.task"
POSE_MODEL_PATH = BASE_DIR / "pose_landmarker.task"
FACE_MODEL_PATH = BASE_DIR / "face_landmarker.task"

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
)
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

EXPECTED_HAND_POINTS = 21
EXPECTED_BODY_POINTS = 33
EXPECTED_FACE_POINTS = 478
TARGET_FRAME_COUNT = 10
LANDMARK_CHANNELS = {0: "x", 1: "y", 2: "z", 3: "visibility", 4: "presence"}

HAND_KEYPOINT_INDEX = {
    0: "wrist",
    1: "thumb_cmc",
    2: "thumb_mcp",
    3: "thumb_ip",
    4: "thumb_tip",
    5: "index_finger_mcp",
    6: "index_finger_pip",
    7: "index_finger_dip",
    8: "index_finger_tip",
    9: "middle_finger_mcp",
    10: "middle_finger_pip",
    11: "middle_finger_dip",
    12: "middle_finger_tip",
    13: "ring_finger_mcp",
    14: "ring_finger_pip",
    15: "ring_finger_dip",
    16: "ring_finger_tip",
    17: "pinky_mcp",
    18: "pinky_pip",
    19: "pinky_dip",
    20: "pinky_tip",
}

BODY_KEYPOINT_INDEX = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}

FACE_KEYPOINT_INDEX = {index: f"face_{index}" for index in range(EXPECTED_FACE_POINTS)}
HAND_SIDE_INDEX = {0: "left", 1: "right"}

MISSING_POINT = {
    "x": np.nan,
    "y": np.nan,
    "z": np.nan,
    "visibility": 0.0,
    "presence": 0.0,
}

VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def ensure_model(model_path: Path, download_url: str) -> None:
    if model_path.exists():
        return

    print(f"Downloading {model_path.name} ...")
    urllib.request.urlretrieve(download_url, model_path)


def ensure_models() -> None:
    ensure_model(HAND_MODEL_PATH, HAND_MODEL_URL)
    ensure_model(POSE_MODEL_PATH, POSE_MODEL_URL)
    ensure_model(FACE_MODEL_PATH, FACE_MODEL_URL)


def create_base_options(model_path: Path, delegate: BaseOptions.Delegate) -> BaseOptions:
    return BaseOptions(model_asset_path=str(model_path), delegate=delegate)


def build_hand_options(delegate: BaseOptions.Delegate) -> HandLandmarkerOptions:
    return HandLandmarkerOptions(
        base_options=create_base_options(HAND_MODEL_PATH, delegate),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def build_pose_options(delegate: BaseOptions.Delegate) -> PoseLandmarkerOptions:
    return PoseLandmarkerOptions(
        base_options=create_base_options(POSE_MODEL_PATH, delegate),
        running_mode=VisionRunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def build_face_options(delegate: BaseOptions.Delegate) -> FaceLandmarkerOptions:
    return FaceLandmarkerOptions(
        base_options=create_base_options(FACE_MODEL_PATH, delegate),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def create_landmarker(factory, options_builder, name: str):
    delegates = [BaseOptions.Delegate.GPU, BaseOptions.Delegate.CPU]
    errors = []
    for delegate in delegates:
        try:
            landmarker = factory.create_from_options(options_builder(delegate))
            return landmarker, delegate
        except (OSError, RuntimeError) as exc:
            errors.append(f"{delegate.name}: {exc}")

    raise RuntimeError(f"Could not create {name}. Attempts: {' | '.join(errors)}")


def frame_to_mp_image(frame_bgr: np.ndarray) -> mp.Image:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)


def load_video_frames_numpy(video_path: Path) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"No frames were read from {video_path}")

    return np.stack(frames, axis=0), float(fps)


def make_timestamps_ms(
    sampled_indices: list[int],
    fps: float,
    offset_ms: int = 0,
) -> list[int]:
    frame_ms = max(1, int(round(1000.0 / fps)))
    timestamps: list[int] = []
    previous = offset_ms - 1
    for source_frame_index in sampled_indices:
        timestamp = offset_ms + int(round((source_frame_index / fps) * 1000.0))
        if timestamp <= previous:
            timestamp = previous + frame_ms
        timestamps.append(timestamp)
        previous = timestamp
    return timestamps


def serialize_landmark(landmark) -> dict[str, float]:
    visibility = getattr(landmark, "visibility", 0.0) or 0.0
    presence = getattr(landmark, "presence", 0.0) or 0.0
    return {
        "x": float(landmark.x),
        "y": float(landmark.y),
        "z": float(landmark.z),
        "visibility": float(visibility),
        "presence": float(presence),
    }


def serialize_landmark_list(landmark_list) -> list[dict[str, float]]:
    return [serialize_landmark(landmark) for landmark in landmark_list]


def serialize_categories(category_list) -> list[dict[str, float | int | str | None]]:
    return [
        {
            "index": category.index,
            "score": float(category.score),
            "category_name": category.category_name,
            "display_name": category.display_name,
        }
        for category in category_list
    ]


def serialize_hand_result(hand_result) -> list[dict[str, object]]:
    hand_landmarks = getattr(hand_result, "hand_landmarks", []) or []
    handedness = getattr(hand_result, "handedness", []) or []
    return [
        {
            "landmarks": serialize_landmark_list(landmarks),
            "handedness": serialize_categories(handedness[index]) if index < len(handedness) else [],
        }
        for index, landmarks in enumerate(hand_landmarks)
    ]


def serialize_pose_result(pose_result) -> list[list[dict[str, float]]]:
    pose_landmarks = getattr(pose_result, "pose_landmarks", []) or []
    return [serialize_landmark_list(landmarks) for landmarks in pose_landmarks]


def serialize_face_result(face_result) -> list[list[dict[str, float]]]:
    face_landmarks = getattr(face_result, "face_landmarks", []) or []
    return [serialize_landmark_list(landmarks) for landmarks in face_landmarks]


def normalize_landmarks(
    landmarks: list[dict[str, float]] | None,
    expected_len: int,
    fallback: list[dict[str, float]] | None = None,
) -> list[dict[str, float]]:
    points = list(landmarks or [])
    if not points and fallback is not None:
        points = [dict(point) for point in fallback]

    if len(points) < expected_len:
        points.extend(dict(MISSING_POINT) for _ in range(expected_len - len(points)))
    elif len(points) > expected_len:
        points = points[:expected_len]

    normalized = []
    for point in points:
        normalized.append(
            {
                "x": float(point.get("x", np.nan)),
                "y": float(point.get("y", np.nan)),
                "z": float(point.get("z", np.nan)),
                "visibility": float(point.get("visibility", 0.0)),
                "presence": float(point.get("presence", 0.0)),
            }
        )
    return normalized


def classify_hands(hand_entries: list[dict[str, object]]) -> dict[str, list[dict[str, float]]]:
    hands = {"left": [], "right": []}
    for entry in hand_entries:
        handedness = entry.get("handedness", [])
        label = "unknown"
        if handedness:
            category_name = (handedness[0].get("category_name") or "").lower()
            if category_name in hands:
                label = category_name

        landmarks = entry.get("landmarks", [])
        if label in hands and not hands[label]:
            hands[label] = landmarks
        elif not hands["left"]:
            hands["left"] = landmarks
        elif not hands["right"]:
            hands["right"] = landmarks

    return hands


def has_valid_x_coordinate(landmarks: list[dict[str, float]]) -> bool:
    return not np.isnan([point["x"] for point in landmarks]).all()


def landmarks_to_array(landmarks: list[dict[str, float]]) -> np.ndarray:
    return np.array(
        [
            [
                point["x"],
                point["y"],
                point["z"],
                point["visibility"],
                point["presence"],
            ]
            for point in landmarks
        ],
        dtype=np.float32,
    )


def iter_video_paths(folder_path: Path) -> Iterable[Path]:
    for path in sorted(folder_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def sample_frame_indices(total_frames: int, sample_count: int = TARGET_FRAME_COUNT) -> list[int]:
    if total_frames <= 0 or sample_count <= 0:
        return []

    indices = np.rint(np.linspace(0, total_frames - 1, sample_count)).astype(np.int64)
    indices[0] = 0
    indices[-1] = total_frames - 1
    return indices.tolist()


def skeleton_output_paths(
    video_path: Path,
    input_root: Path,
    output_folder: Path,
) -> tuple[Path, Path, Path]:
    relative_path = video_path.relative_to(input_root).with_suffix(".npy")
    return (
        output_folder / "body" / relative_path,
        output_folder / "hand" / relative_path,
        output_folder / "face" / relative_path,
    )


def skeleton_outputs_exist(video_path: Path, input_root: Path, output_folder: Path) -> bool:
    return all(path.exists() for path in skeleton_output_paths(video_path, input_root, output_folder))


def write_index_maps(output_folder: Path) -> None:
    body_dir = output_folder / "body"
    hand_dir = output_folder / "hand"
    face_dir = output_folder / "face"

    body_dir.mkdir(parents=True, exist_ok=True)
    hand_dir.mkdir(parents=True, exist_ok=True)
    face_dir.mkdir(parents=True, exist_ok=True)

    np.save(
        body_dir / "keypoint_index.npy",
        {"keypoints": BODY_KEYPOINT_INDEX, "channels": LANDMARK_CHANNELS},
        allow_pickle=True,
    )
    np.save(
        hand_dir / "keypoint_index.npy",
        {
            "keypoints": HAND_KEYPOINT_INDEX,
            "channels": LANDMARK_CHANNELS,
            "hands": HAND_SIDE_INDEX,
        },
        allow_pickle=True,
    )
    np.save(
        face_dir / "keypoint_index.npy",
        {"keypoints": FACE_KEYPOINT_INDEX, "channels": LANDMARK_CHANNELS},
        allow_pickle=True,
    )


def process_video(
    video_path: Path,
    input_root: Path,
    output_folder: Path,
    hand_landmarker,
    pose_landmarker,
    face_landmarker,
    timestamp_offset_ms: int,
) -> int:
    body_output, hand_output, face_output = skeleton_output_paths(
        video_path,
        input_root,
        output_folder,
    )

    body_output.parent.mkdir(parents=True, exist_ok=True)
    hand_output.parent.mkdir(parents=True, exist_ok=True)
    face_output.parent.mkdir(parents=True, exist_ok=True)

    source_frames, fps = load_video_frames_numpy(video_path)
    sampled_frame_indices = sample_frame_indices(len(source_frames))
    sampled_frames = source_frames[sampled_frame_indices].copy()
    sampled_timestamps_ms = make_timestamps_ms(
        sampled_frame_indices,
        fps,
        offset_ms=timestamp_offset_ms,
    )

    body_frames: list[np.ndarray] = []
    hand_frames: list[np.ndarray] = []
    face_frames: list[np.ndarray] = []

    last_valid = {
        "left_hand": None,
        "right_hand": None,
        "body": None,
        "face": None,
    }

    print(
        "Processing "
        f"{video_path.relative_to(input_root)} "
        f"({len(source_frames)} source frames -> {TARGET_FRAME_COUNT} sampled frames)"
    )

    for frame, timestamp_ms in zip(sampled_frames, sampled_timestamps_ms):
        mp_image = frame_to_mp_image(frame)

        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        #face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

        serialized_hands = serialize_hand_result(hand_result)
        serialized_body = serialize_pose_result(pose_result)
        #serialized_face = serialize_face_result(face_result)

        hands = classify_hands(serialized_hands)
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
        body = normalize_landmarks(
            serialized_body[0] if serialized_body else [],
            EXPECTED_BODY_POINTS,
            fallback=last_valid["body"],
        )
        # face = normalize_landmarks(
        #     serialized_face[0] if serialized_face else [],
        #     EXPECTED_FACE_POINTS,
        #     fallback=last_valid["face"],
        # )

        if has_valid_x_coordinate(left_hand):
            last_valid["left_hand"] = [dict(point) for point in left_hand]
        if has_valid_x_coordinate(right_hand):
            last_valid["right_hand"] = [dict(point) for point in right_hand]
        if has_valid_x_coordinate(body):
            last_valid["body"] = [dict(point) for point in body]
        # if has_valid_x_coordinate(face):
        #     last_valid["face"] = [dict(point) for point in face]

        hand_frames.append(
            np.stack(
                [landmarks_to_array(left_hand), landmarks_to_array(right_hand)],
                axis=0,
            )
        )
        body_frames.append(landmarks_to_array(body))
        # face_frames.append(landmarks_to_array(face))

    if not body_frames:
        raise ValueError(f"No skeleton frames generated for {video_path}")

    np.save(body_output, np.stack(body_frames, axis=0))
    np.save(hand_output, np.stack(hand_frames, axis=0))
    # np.save(face_output, np.stack(face_frames, axis=0))

    frame_ms = max(1, int(round(1000.0 / fps)))
    return sampled_timestamps_ms[-1] + frame_ms


def main() -> None:
    input_root = RAW_DATASET_DIR.resolve()
    output_root = SKELETON_DATASET_DIR.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    video_paths = list(iter_video_paths(input_root))
    if not video_paths:
        raise FileNotFoundError(f"No supported video files found under {input_root}")

    write_index_maps(output_root)
    pending_video_paths = [
        video_path
        for video_path in video_paths
        if not skeleton_outputs_exist(video_path, input_root, output_root)
    ]
    skipped_count = len(video_paths) - len(pending_video_paths)

    for video_path in video_paths:
        if skeleton_outputs_exist(video_path, input_root, output_root):
            print(f"Skipping existing {video_path.relative_to(input_root)}")

    if not pending_video_paths:
        print(
            f"Finished. Skipped {skipped_count} existing video(s) from {input_root}; "
            f"outputs are in {output_root}/<body,hand,face>."
        )
        return

    ensure_models()
    with ExitStack() as stack:
        hand_landmarker, hand_delegate = create_landmarker(
            HandLandmarker, build_hand_options, "hand landmarker"
        )
        pose_landmarker, pose_delegate = create_landmarker(
            PoseLandmarker, build_pose_options, "pose landmarker"
        )
        face_landmarker, face_delegate = create_landmarker(
            FaceLandmarker, build_face_options, "face landmarker"
        )

        hand_landmarker = stack.enter_context(hand_landmarker)
        pose_landmarker = stack.enter_context(pose_landmarker)
        face_landmarker = stack.enter_context(face_landmarker)

        print(
            "Loaded models once. "
            f"Delegates -> hand: {hand_delegate.name}, "
            f"body: {pose_delegate.name}", 
            f"face: {face_delegate.name}"
        )

        timestamp_offset_ms = 0
        for video_path in pending_video_paths:
            timestamp_offset_ms = process_video(
                video_path,
                input_root,
                output_root,
                hand_landmarker,
                pose_landmarker,
                face_landmarker,
                timestamp_offset_ms,
            )

    print(
        f"Finished. Processed {len(pending_video_paths)} video(s), "
        f"skipped {skipped_count} existing video(s) from {input_root} "
        f"into {output_root}/<body,hand,face>."
    )


if __name__ == "__main__":
    main()
