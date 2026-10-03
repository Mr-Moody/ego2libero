"""File schemas shared between stages. Arrays go in `<stem>.npz`, everything else (ids, object
order, fps, provenance) in a `<stem>.json` sidecar. `save()` and `load()` validate shapes."""

import json
from pathlib import Path
from typing import Any, Self, get_args, get_origin

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "DemoTrajectory",
    "HandSegment",
    "HandSegments",
    "RobotSegment",
    "RobotSegments",
    "SimEpisode",
]


def _check_shapes(specs: dict[str, tuple[np.ndarray, tuple]]) -> None:
    """Check each array against a shape spec. Ints must match exactly; strings are symbols
    (N, K, ...) that must agree across all arrays in `specs`."""
    sizes: dict[str, int] = {}
    for name, (arr, spec) in specs.items():
        if arr.ndim != len(spec):
            raise ValueError(f"{name}: expected {len(spec)} dims {spec}, got shape {arr.shape}")
        for dim, want in zip(arr.shape, spec, strict=True):
            if isinstance(want, int):
                if dim != want:
                    raise ValueError(f"{name}: expected shape {spec}, got {arr.shape}")
            elif sizes.setdefault(want, dim) != dim:
                raise ValueError(f"{name}: {want}={dim} disagrees with {want}={sizes[want]}")


def _is_model_list(annotation: Any) -> bool:
    args = get_args(annotation)
    return (
        get_origin(annotation) is list
        and bool(args)
        and isinstance(args[0], type)
        and (issubclass(args[0], BaseModel))
    )


class ArrayModel(BaseModel):
    """Base for schemas with numpy fields. Supports `np.ndarray`, `dict[str, np.ndarray]` and
    `list[ArrayModel]` fields; any other field is JSON metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _split(self, prefix: str = "") -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        arrays: dict[str, np.ndarray] = {}
        meta: dict[str, Any] = {}
        for name, field in type(self).model_fields.items():
            value = getattr(self, name)
            if isinstance(value, np.ndarray):
                arrays[prefix + name] = value
            elif (
                isinstance(value, dict)
                and value
                and all(isinstance(v, np.ndarray) for v in value.values())
            ):
                meta[name] = {"__array_keys__": list(value)}
                arrays.update({f"{prefix}{name}.{k}": v for k, v in value.items()})
            elif _is_model_list(field.annotation):
                items = [item._split(f"{prefix}{name}.{i}.") for i, item in enumerate(value)]
                for item_arrays, _ in items:
                    arrays.update(item_arrays)
                meta[name] = [item_meta for _, item_meta in items]
            else:
                meta[name] = value
        return arrays, meta

    @classmethod
    def _join(cls, arrays: dict[str, np.ndarray], meta: dict[str, Any], prefix: str = "") -> Self:
        data: dict[str, Any] = {}
        for name, field in cls.model_fields.items():
            if prefix + name in arrays:
                data[name] = arrays[prefix + name]
            elif _is_model_list(field.annotation):
                sub = get_args(field.annotation)[0]
                data[name] = [
                    sub._join(arrays, m, f"{prefix}{name}.{i}.") for i, m in enumerate(meta[name])
                ]
            elif isinstance(meta.get(name), dict) and "__array_keys__" in meta[name]:
                keys = meta[name]["__array_keys__"]
                data[name] = {k: arrays[f"{prefix}{name}.{k}"] for k in keys}
            elif name in meta:
                data[name] = meta[name]
        return cls(**data)

    def save(self, path: str | Path) -> tuple[Path, Path]:
        """Write `<path>.npz` and `<path>.json` (any suffix on `path` is replaced)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays, meta = self._split()
        npz, sidecar = path.with_suffix(".npz"), path.with_suffix(".json")
        np.savez_compressed(npz, **arrays)
        sidecar.write_text(json.dumps({"schema": type(self).__name__, **meta}, indent=2))
        return npz, sidecar

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Read and validate `<path>.npz` + `<path>.json`."""
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        if meta.pop("schema", cls.__name__) != cls.__name__:
            raise ValueError(f"{path} does not hold a {cls.__name__}")
        with np.load(path.with_suffix(".npz")) as npz:
            arrays = {k: npz[k] for k in npz.files}
        return cls._join(arrays, meta)


class DemoTrajectory(ArrayModel):
    """Table-frame hand and object trajectories for one demo (output of `e2l perception run`)."""

    demo_id: str
    fps: float
    object_names: list[str]
    config_hash: str = ""
    t: np.ndarray  # (N,) seconds from frame index / fps
    T_table_hand: np.ndarray  # (N,4,4)
    hand_valid: np.ndarray  # (N,) bool
    keypoints_table: np.ndarray  # (N,21,3) metres
    aperture: np.ndarray  # (N,) thumb-tip to index-tip distance, metres
    T_table_obj: np.ndarray  # (N,K,4,4)
    obj_valid: np.ndarray  # (N,K) bool

    @model_validator(mode="after")
    def _shapes(self) -> Self:
        _check_shapes(
            {
                "t": (self.t, ("N",)),
                "T_table_hand": (self.T_table_hand, ("N", 4, 4)),
                "hand_valid": (self.hand_valid, ("N",)),
                "keypoints_table": (self.keypoints_table, ("N", 21, 3)),
                "aperture": (self.aperture, ("N",)),
                "T_table_obj": (self.T_table_obj, ("N", "K", 4, 4)),
                "obj_valid": (self.obj_valid, ("N", "K")),
            }
        )
        if self.T_table_obj.shape[1] != len(self.object_names):
            raise ValueError("T_table_obj K must equal len(object_names)")
        return self


class HandSegment(ArrayModel):
    """One object-centric hand segment: hand pose relative to `ref_object` and binary grip."""

    ref_object: str
    t_start: float
    t_end: float
    T_obj_hand: np.ndarray  # (M,4,4)
    grip: np.ndarray  # (M,) 0 open, 1 closed

    @model_validator(mode="after")
    def _shapes(self) -> Self:
        _check_shapes({"T_obj_hand": (self.T_obj_hand, ("M", 4, 4)), "grip": (self.grip, ("M",))})
        return self


class HandSegments(ArrayModel):
    demo_id: str
    fps: float
    segments: list[HandSegment] = Field(default_factory=list)


class RobotSegment(ArrayModel):
    """One object-centric Panda end-effector segment."""

    ref_object: str
    T_obj_ee: np.ndarray  # (M,4,4)
    gripper: np.ndarray  # (M,) robosuite convention: -1 open, +1 closed
    feasible: np.ndarray  # (M,) bool

    @model_validator(mode="after")
    def _shapes(self) -> Self:
        _check_shapes(
            {
                "T_obj_ee": (self.T_obj_ee, ("M", 4, 4)),
                "gripper": (self.gripper, ("M",)),
                "feasible": (self.feasible, ("M",)),
            }
        )
        return self


class RobotSegments(ArrayModel):
    demo_id: str
    fps: float
    segments: list[RobotSegment] = Field(default_factory=list)


class SimEpisode(ArrayModel):
    """A replayed LIBERO episode with provenance back to the human demos."""

    task: str
    success: bool
    source_demo_ids: list[str]
    images: dict[str, np.ndarray]  # camera name -> (T,H,W,3) uint8
    state: np.ndarray  # (T,8)
    actions: np.ndarray  # (T,7) LIBERO delta actions
    object_poses: dict[str, np.ndarray]  # object name -> initial T_sim_obj (4,4)

    @model_validator(mode="after")
    def _shapes(self) -> Self:
        specs: dict[str, tuple[np.ndarray, tuple]] = {
            "state": (self.state, ("T", 8)),
            "actions": (self.actions, ("T", 7)),
        }
        specs |= {f"images.{k}": (v, ("T", "H", "W", 3)) for k, v in self.images.items()}
        specs |= {f"object_poses.{k}": (v, (4, 4)) for k, v in self.object_poses.items()}
        _check_shapes(specs)
        if any(v.dtype != np.uint8 for v in self.images.values()):
            raise ValueError("images must be uint8")
        return self
