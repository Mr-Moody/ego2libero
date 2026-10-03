# Ported from Hand-Tracking-Fusion src/hand_detector.py @ 07947d1
# https://github.com/Mr-Moody/Hand-Tracking-Fusion
# Changes: x-negation of world landmarks is now the `mirrored` flag (default False, phone
# video is not mirrored); handedness label and score are returned; timestamps come from the
# video clock (seconds) and are converted with round(t * 1000).
"""MediaPipe HandLandmarker wrapper (VIDEO running mode)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

Handedness = Literal["Left", "Right"]


@dataclass
class HandDetection:
    landmarks_2d: np.ndarray  # (21, 2) pixel coords
    landmarks_3d: np.ndarray  # (21, 3) MediaPipe world coords, metres, wrist-centred
    visible_mask: np.ndarray  # (21,) bool
    handedness: Handedness  # anatomical hand, corrected for mirroring
    handedness_score: float


def _visibility(lm) -> float:
    v = getattr(lm, "visibility", None)
    return float(v) if v is not None else 1.0


def to_detection(
    lm2d, lm3d, category, width: int, height: int, mirrored: bool = False
) -> HandDetection:
    """Convert one MediaPipe hand result to a HandDetection.

    `lm2d` / `lm3d` are sequences of landmarks with .x/.y/.z (normalised image coords and world
    metres); `category` has .category_name and .score. A mirrored (selfie) frame flips the
    chirality of the world landmarks, so x is negated to restore it. MediaPipe labels handedness
    assuming a mirrored image, so on an unmirrored frame the label is swapped."""
    landmarks_2d = np.array([[lm.x * width, lm.y * height] for lm in lm2d], dtype=np.float32)
    sign = -1.0 if mirrored else 1.0
    landmarks_3d = np.array([[sign * lm.x, lm.y, lm.z] for lm in lm3d], dtype=np.float32)

    vis_scores = np.array([_visibility(lm) for lm in lm2d])
    # Only threshold if visibility is actually populated; otherwise all visible
    visible_mask = vis_scores > 0.5 if vis_scores.max() < 1.0 else np.ones(21, dtype=bool)

    label = category.category_name
    if not mirrored:
        label = {"Left": "Right", "Right": "Left"}[label]
    return HandDetection(
        landmarks_2d=landmarks_2d,
        landmarks_3d=landmarks_3d,
        visible_mask=visible_mask,
        handedness=label,
        handedness_score=float(category.score),
    )


class HandDetector:
    def __init__(
        self,
        model_path: str | Path,
        num_hands: int = 2,
        mirrored: bool = False,
        min_detection_confidence: float = 0.7,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self.mirrored = mirrored
        self._last_ms = -1

    def detect_all(self, frame: np.ndarray, t: float) -> list[HandDetection]:
        """All hands in a BGR frame at video time t (seconds), in MediaPipe's order."""
        import mediapipe as mp

        timestamp_ms = round(t * 1000)
        if timestamp_ms <= self._last_ms:
            raise ValueError(
                f"timestamps must strictly increase: {timestamp_ms} <= {self._last_ms}"
            )
        self._last_ms = timestamp_ms

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        return [
            to_detection(lm2d, lm3d, hd[0], w, h, self.mirrored)
            for lm2d, lm3d, hd in zip(
                result.hand_landmarks, result.hand_world_landmarks, result.handedness, strict=True
            )
        ]

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
