"""One pydantic model per stage config in `configs/`. CLI flags override YAML through
`load_config(path, overrides)`, where overrides are `key.sub=value` strings or a dict."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from e2l_common.geometry import make_T, rotvec_to_matrix

__all__ = [
    "CONFIG_MODELS",
    "CaptureConfig",
    "EvalConfig",
    "GenerateConfig",
    "PerceptionConfig",
    "Pose",
    "RetargetConfig",
    "SegmentConfig",
    "SimConfig",
    "TrainConfig",
    "load_config",
]

Vec3 = tuple[float, float, float]


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Pose(_Config):
    """Rigid transform in config files: translation (m) + rotation vector (rad). No quaternions."""

    xyz: Vec3 = (0.0, 0.0, 0.0)
    rotvec: Vec3 = (0.0, 0.0, 0.0)

    @property
    def T(self) -> np.ndarray:
        return make_T(rotvec_to_matrix(np.array(self.rotvec)), np.array(self.xyz))


# --- capture ---------------------------------------------------------------------------------


class TableBoard(_Config):
    """ArUco GridBoard taped to the table; its centre defines the `table` frame."""

    markers_x: int = 4
    markers_y: int = 3
    marker_length: float = 0.04  # metres, printed size
    marker_separation: float = 0.01
    first_id: int = 0


class TaggedObject(_Config):
    marker_id: int
    marker_length: float  # metres
    T_marker_obj: Pose = Pose()  # canonical object frame in the marker frame


class CaptureConfig(_Config):
    intrinsics: Path = Path("data/calib/phone.json")
    aruco_dictionary: str = "DICT_4X4_50"
    table_board: TableBoard = TableBoard()
    objects: dict[str, TaggedObject] = Field(default_factory=dict)


# --- stages ----------------------------------------------------------------------------------


class PerceptionConfig(_Config):
    hand_model: Path = Path("models/hand_landmarker.task")
    fps: float = 30.0  # normalised constant frame rate
    mirrored: bool = False  # true only for selfie/webcam feeds
    hand: Literal["Left", "Right"] = "Right"
    num_hands: int = 2
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    max_wrist_jump_px: float = 150.0
    ekf_process_noise: float = 1e-3
    ekf_measurement_noise: float = 1e-2
    smooth: bool = True


class SegmentConfig(_Config):
    close_threshold: float = 0.035  # aperture (m) below which the grip closes
    open_threshold: float = 0.055  # aperture (m) above which it opens again
    min_dwell_s: float = 0.25


class RetargetConfig(_Config):
    T_hand_ee: Pose = Pose()  # wrist to Panda gripper centre
    canonicalise_top_down: bool = True
    workspace_min: Vec3 = (-0.4, -0.4, 0.0)  # table frame, metres
    workspace_max: Vec3 = (0.4, 0.4, 0.5)
    max_linear_speed: float = 0.5  # m/s
    max_angular_speed: float = 2.0  # rad/s


class SimConfig(_Config):
    suite: str = "libero_goal"
    task_id: int = 0
    task_name: str = ""
    camera_names: list[str] = Field(default_factory=lambda: ["agentview", "robot0_eye_in_hand"])
    image_size: int = 256
    control_freq: int = 20
    max_steps: int = 300
    seed: int = 0
    T_sim_table: Pose = Pose()
    object_map: dict[str, str] = Field(default_factory=dict)  # real object name -> LIBERO name
    transit_speed: float = 0.25  # m/s, free-space moves into each segment's first pose
    gripper_settle_steps: int = 10  # control steps held at each grasp / release
    max_steps_per_target: int = 3  # extra tracking steps while the grip site lags its target
    tracking_gain: float = 3.0  # pose error -> action gain (OSC lags a single delta)


class GenerateConfig(_Config):
    episodes_per_demo: int = 50
    xy_range: float = 0.08  # metres, uniform +/- around the source object pose
    yaw_range: float = 0.5  # radians
    transit_steps: int = 30
    workers: int = 4
    keep_failures: bool = False
    seed: int = 0


class TrainConfig(_Config):
    policy_path: str = "HuggingFaceVLA/smolvla_libero"
    dataset_repo_id: str = "local/ego2libero"
    dataset_root: Path = Path("data/lerobot/ego2libero")
    output_dir: Path = Path("outputs/train/smolvla")
    steps: int = 20000
    batch_size: int = 32
    num_workers: int = 4
    save_freq: int = 5000
    wandb: bool = False
    extra_args: list[str] = Field(default_factory=list)


class EvalConfig(_Config):
    policy_path: str = "HuggingFaceVLA/smolvla_libero"
    n_episodes: int = 50
    seed: int = 1000
    output_dir: Path = Path("outputs/eval")
    device: str = "cuda"
    demo_counts: list[int] = Field(default_factory=lambda: [5, 10, 20, 40])
    augmentation: list[bool] = Field(default_factory=lambda: [False, True])


CONFIG_MODELS: dict[str, type[BaseModel]] = {
    "capture": CaptureConfig,
    "perception": PerceptionConfig,
    "segment": SegmentConfig,
    "retarget": RetargetConfig,
    "sim": SimConfig,
    "generate": GenerateConfig,
    "smolvla": TrainConfig,
    "train": TrainConfig,
    "eval": EvalConfig,
}


def _parse_overrides(overrides: Mapping[str, Any] | Sequence[str]) -> dict[str, Any]:
    if isinstance(overrides, Mapping):
        return dict(overrides)
    parsed = {}
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"override {item!r} must look like key.sub=value")
        parsed[key.strip()] = yaml.safe_load(value)
    return parsed


def _set_dotted(data: dict[str, Any], key: str, value: Any) -> None:
    *parents, leaf = key.split(".")
    for part in parents:
        data = data.setdefault(part, {})
        if not isinstance(data, dict):
            raise ValueError(f"cannot override {key!r}: {part!r} is not a mapping")
    data[leaf] = value


def load_config[M: BaseModel](
    path: str | Path,
    overrides: Mapping[str, Any] | Sequence[str] | None = None,
    model: type[M] | None = None,
) -> M:
    """Load a YAML config into its stage model (picked from the file stem unless `model` is
    given), applying dotted overrides such as `["sim.task_id=3"]` or `{"seed": 1}` on top."""
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    for key, value in _parse_overrides(overrides or {}).items():
        _set_dotted(data, key, value)
    if model is None:
        if path.stem not in CONFIG_MODELS:
            raise ValueError(f"no config model for {path.name}; pass model=")
        model = CONFIG_MODELS[path.stem]  # type: ignore[assignment]
    return model.model_validate(data)  # type: ignore[union-attr]
