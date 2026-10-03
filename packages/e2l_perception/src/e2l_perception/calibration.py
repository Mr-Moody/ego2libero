# Ported from Hand-Tracking-Fusion src/calibration/calibration.py @ 07947d1
# https://github.com/Mr-Moody/Hand-Tracking-Fusion
# Changes: single-camera checkerboard calibration from a list of frames (e.g. a phone video)
# replaces the live stereo capture loop; intrinsics are saved as data/calib/<device>.json.
# The stereo projection / triangulation helpers are kept for HandFusion's UNTESTED stereo
# branch. stereo_cal.json is not ported.
"""Camera intrinsics: checkerboard calibration, save and load."""

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

BOARD_SIZE = (13, 8)  # interior corner count (columns, rows)
SQUARE_M = 0.020  # checkerboard square side in metres

_SUBPIX_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


@dataclass
class Intrinsics:
    K: np.ndarray  # (3, 3) camera matrix, pixels
    dist: np.ndarray  # (n,) OpenCV distortion coefficients
    size: tuple[int, int]  # (width, height) the calibration was made at
    rms: float = float("nan")  # reprojection error, pixels

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "K": self.K.tolist(),
                    "dist": self.dist.ravel().tolist(),
                    "width": self.size[0],
                    "height": self.size[1],
                    "rms": self.rms,
                },
                indent=2,
            )
        )
        return path


def load_intrinsics(path: str | Path) -> Intrinsics:
    """Read data/calib/<device>.json."""
    raw = json.loads(Path(path).read_text())
    return Intrinsics(
        K=np.array(raw["K"], dtype=np.float64),
        dist=np.array(raw["dist"], dtype=np.float64),
        size=(int(raw["width"]), int(raw["height"])),
        rms=float(raw.get("rms", float("nan"))),
    )


def board_object_points(
    board_size: tuple[int, int] = BOARD_SIZE, square: float = SQUARE_M
) -> np.ndarray:
    """(cols*rows, 3) checkerboard corner positions in the board frame, metres."""
    pts = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    pts[:, :2] = np.mgrid[: board_size[0], : board_size[1]].T.reshape(-1, 2)
    return pts * square


def find_corners(gray: np.ndarray, board_size: tuple[int, int] = BOARD_SIZE) -> np.ndarray | None:
    """Sub-pixel checkerboard corners (N, 1, 2) in a grayscale image, or None."""
    found, corners = cv2.findChessboardCorners(
        gray, board_size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not found:
        return None
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), _SUBPIX_CRIT)


def calibrate_from_frames(
    frames: Iterable[np.ndarray],
    board_size: tuple[int, int] = BOARD_SIZE,
    square: float = SQUARE_M,
    min_views: int = 10,
) -> Intrinsics:
    """Calibrate one camera from BGR frames that show the checkerboard."""
    obj = board_object_points(board_size, square)
    obj_pts, img_pts, size = [], [], None
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        corners = find_corners(gray, board_size)
        if corners is None:
            continue
        obj_pts.append(obj)
        img_pts.append(corners)
        size = gray.shape[::-1]
    if len(obj_pts) < min_views:
        raise RuntimeError(f"only {len(obj_pts)} checkerboard views found (need {min_views})")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    return Intrinsics(K=K, dist=dist.ravel(), size=(int(size[0]), int(size[1])), rms=float(rms))


# ---------------------------------------------------------------------------
# Stereo helpers (UNTESTED; only used by HandFusion's stereo branch)
# ---------------------------------------------------------------------------


def build_projection_matrices(
    cal: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (P0, P1, K0, d0, K1, d1) ready for triangulation.

    P0 = K0 @ [I | 0]  (camera-0 is the world origin)
    P1 = K1 @ [R | T]  (camera-1 relative to camera-0)
    """
    K0, d0 = cal["K0"], cal["d0"]
    K1, d1 = cal["K1"], cal["d1"]
    R, T = cal["R"], cal["T"].reshape(3, 1)
    P0 = K0 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = K1 @ np.hstack([R, T])
    return P0, P1, K0, d0, K1, d1


def triangulate_landmarks(
    pts0: np.ndarray,
    pts1: np.ndarray,
    P0: np.ndarray,
    P1: np.ndarray,
    K0: np.ndarray,
    d0: np.ndarray,
    K1: np.ndarray,
    d1: np.ndarray,
) -> np.ndarray:
    """Triangulate (21,3) 3D points from two sets of (21,2) image points.

    Returns wrist-centred metric coordinates in camera-0 frame. NOTE: the source negated x
    here to match its mirrored webcam feed; that is kept, so it is only correct for mirrored
    input.
    """
    u0 = cv2.undistortPoints(pts0.reshape(-1, 1, 2), K0, d0, P=K0).reshape(-1, 2)
    u1 = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K1, d1, P=K1).reshape(-1, 2)
    pts4d = cv2.triangulatePoints(P0, P1, u0.T, u1.T)
    pts3d = (pts4d[:3] / pts4d[3]).T.astype(np.float32)
    pts3d -= pts3d[0]  # wrist-centre (landmark 0)
    pts3d[:, 0] *= -1  # negate x (mirrored-feed convention from the source)
    return pts3d
