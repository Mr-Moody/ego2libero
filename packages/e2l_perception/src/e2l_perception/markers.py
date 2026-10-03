"""ArUco detection and marker poses in the camera frame (OpenCV `cam` frame: x right, y down,
z forward). The table frame sits at the centre of the GridBoard with z up out of the table."""

from dataclasses import dataclass, field

import cv2
import numpy as np

from e2l_common.config import CaptureConfig, TableBoard
from e2l_common.geometry import compose, make_T


def get_dictionary(name: str) -> cv2.aruco.Dictionary:
    """cv2.aruco predefined dictionary by name, e.g. "DICT_4X4_50"."""
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def make_board(board: TableBoard, dictionary: cv2.aruco.Dictionary) -> cv2.aruco.GridBoard:
    n = board.markers_x * board.markers_y
    ids = np.arange(board.first_id, board.first_id + n, dtype=np.int32)
    return cv2.aruco.GridBoard(
        (board.markers_x, board.markers_y),
        board.marker_length,
        board.marker_separation,
        dictionary,
        ids,
    )


def board_centre(board: cv2.aruco.GridBoard) -> np.ndarray:
    """T_board_table. OpenCV's board frame has its origin at the first marker's corner with
    x along the rows and y *down* the printed image, so z points into the table. The table
    frame sits at the board centre, rotated 180 deg about x so that z points up."""
    pts = np.concatenate([np.asarray(p).reshape(-1, 3) for p in board.getObjPoints()])
    return make_T(np.diag([1.0, -1.0, -1.0]), (pts.min(0) + pts.max(0)) / 2)


def marker_object_points(length: float) -> np.ndarray:
    """Marker corners (4,3) in the marker frame (centre origin, z out of the marker), in
    ArUco corner order: top-left, top-right, bottom-right, bottom-left."""
    h = length / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


def pnp_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    return make_T(R, np.asarray(tvec).ravel())


@dataclass
class MarkerObservation:
    T_cam_table: np.ndarray | None  # (4,4) or None if the board is not visible
    T_cam_obj: dict[str, np.ndarray] = field(default_factory=dict)  # only visible objects
    ids: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))


class MarkerTracker:
    """Per-frame table and object poses from ArUco markers."""

    def __init__(
        self, capture: CaptureConfig, K: np.ndarray, dist: np.ndarray, min_board_markers: int = 2
    ):
        self.capture = capture
        self.K = np.asarray(K, dtype=np.float64)
        self.dist = np.asarray(dist, dtype=np.float64)
        self.min_board_markers = min_board_markers
        self.dictionary = get_dictionary(capture.aruco_dictionary)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementMaxIterations = 100
        params.cornerRefinementMinAccuracy = 0.001
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, params)
        self.board = make_board(capture.table_board, self.dictionary)
        self.T_board_table = board_centre(self.board)
        self._obj_by_id = {o.marker_id: (name, o) for name, o in capture.objects.items()}

    def detect(self, frame: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        """ArUco corners (list of (1,4,2)) and ids (n,) in a BGR or grayscale frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        corners, ids, _ = self.detector.detectMarkers(gray)
        return list(corners), (ids.ravel() if ids is not None else np.zeros(0, np.int32))

    def estimate_T_cam_table(self, corners: list[np.ndarray], ids: np.ndarray) -> np.ndarray | None:
        """Table pose from all visible board markers, or None if fewer than
        `min_board_markers` are seen."""
        board_ids = set(self.board.getIds().ravel().tolist())
        if sum(int(i) in board_ids for i in ids) < self.min_board_markers:
            return None
        obj_pts, img_pts = self.board.matchImagePoints(corners, ids.reshape(-1, 1))
        if obj_pts is None or len(obj_pts) < 4:
            return None
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, self.K, self.dist)
        if not ok:
            return None
        return compose(pnp_to_T(rvec, tvec), self.T_board_table)

    def estimate_T_cam_marker(self, corners: np.ndarray, length: float) -> np.ndarray | None:
        """Single-marker pose T_cam_marker from its 4 corners (1,4,2) via IPPE_SQUARE."""
        ok, rvec, tvec = cv2.solvePnP(
            marker_object_points(length),
            np.asarray(corners, dtype=np.float64).reshape(4, 2),
            self.K,
            self.dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        return pnp_to_T(rvec, tvec) if ok else None

    def observe(self, frame: np.ndarray) -> MarkerObservation:
        """Table pose plus T_cam_obj = T_cam_marker @ T_marker_obj for each visible object."""
        corners, ids = self.detect(frame)
        obs = MarkerObservation(self.estimate_T_cam_table(corners, ids), ids=ids)
        for c, marker_id in zip(corners, ids, strict=True):
            if int(marker_id) not in self._obj_by_id:
                continue
            name, obj = self._obj_by_id[int(marker_id)]
            T_cam_marker = self.estimate_T_cam_marker(c, obj.marker_length)
            if T_cam_marker is not None:
                obs.T_cam_obj[name] = compose(T_cam_marker, obj.T_marker_obj.T)
        return obs


def estimate_T_cam_table(tracker: MarkerTracker, frame: np.ndarray) -> np.ndarray | None:
    """Convenience: table-board pose T_cam_table (4,4) in one frame, or None if not visible."""
    return tracker.estimate_T_cam_table(*tracker.detect(frame))
