# Ported from Hand-Tracking-Fusion src/hand_filter.py @ 07947d1
# https://github.com/Mr-Moody/Hand-Tracking-Fusion
# Changes: hands were matched on landmarks_3d[0], but world landmarks are wrist-centred so that
# is ~0 for every hand and the distance gate never separated them. Matching now uses the
# handedness label plus the 2D wrist pixel distance. The two near-duplicate filter methods are
# merged into `select`.
"""Keep tracking one hand when several are visible."""

from dataclasses import dataclass

import numpy as np

from e2l_perception.hand.detector import HandDetection, Handedness


@dataclass
class TrackedHand:
    detection: HandDetection
    wrist_px: np.ndarray  # (2,) wrist pixel position
    age: int  # frames since tracking started


class HandFilter:
    """Select the demonstrating hand from a frame's detections.

    - Only hands whose (mirror-corrected) handedness matches `hand` are candidates; pass None to
      accept either hand.
    - While tracking, the candidate whose wrist pixel is nearest the last tracked wrist is kept
      if it moved at most `max_jump_px` per frame since last seen; otherwise the frame is
      treated as a miss.
    - After `memory_frames` consecutive misses the track is dropped and the most confident
      candidate is acquired.
    """

    def __init__(
        self, hand: Handedness | None = "Right", max_jump_px: float = 150.0, memory_frames: int = 30
    ):
        self.hand = hand
        self.max_jump_px = max_jump_px
        self.memory_frames = memory_frames
        self._tracked: TrackedHand | None = None
        self._frames_since_lost = 0

    def select(self, detections: list[HandDetection]) -> HandDetection | None:
        """Pick the tracked hand from all detections in this frame (any camera), or None."""
        candidates = [d for d in detections if self.hand is None or d.handedness == self.hand]
        if not candidates:
            self._miss()
            return None

        if self._tracked is None or self._frames_since_lost >= self.memory_frames:
            best = max(candidates, key=lambda d: d.handedness_score)
            return self._accept(best, age=1)

        dists = [np.linalg.norm(d.landmarks_2d[0] - self._tracked.wrist_px) for d in candidates]
        nearest = int(np.argmin(dists))
        if dists[nearest] <= self.max_jump_px * (1 + self._frames_since_lost):
            return self._accept(candidates[nearest], age=self._tracked.age + 1)
        self._miss()
        return None

    def reset(self) -> None:
        self._tracked = None
        self._frames_since_lost = 0

    def _accept(self, detection: HandDetection, age: int) -> HandDetection:
        self._tracked = TrackedHand(detection, detection.landmarks_2d[0].astype(float), age)
        self._frames_since_lost = 0
        return detection

    def _miss(self) -> None:
        if self._tracked is not None:
            self._frames_since_lost += 1
            if self._frames_since_lost >= self.memory_frames:
                self._tracked = None

    @property
    def tracked_hand_age(self) -> int | None:
        """Number of frames the current hand has been tracked, or None."""
        return self._tracked.age if self._tracked is not None else None

    @property
    def frames_since_lost(self) -> int:
        """Number of frames since the tracked hand was last selected."""
        return self._frames_since_lost
