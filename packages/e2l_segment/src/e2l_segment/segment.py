"""Cut a demo at grip events and re-express each piece relative to its reference object."""

import numpy as np

from e2l_common.config import SegmentConfig
from e2l_common.geometry import interpolate_poses, inv_T
from e2l_common.schemas import DemoTrajectory, HandSegment, HandSegments
from e2l_segment.anchor import anchor_depth
from e2l_segment.assign import Side, nearest_valid_index, reference_object
from e2l_segment.events import aperture_to_grip, grip_events


def segment_demo(demo: DemoTrajectory, cfg: SegmentConfig) -> HandSegments:
    """DemoTrajectory -> HandSegments.

    Hand depth is first anchored to the carried objects (`anchor.anchor_depth`, using the grip
    from the raw aperture).

    Segments run from the previous event (or start) to each grasp/release event (or end). Each
    segment stores T_obj_hand = inv(T_table_obj[ref]) @ T_table_hand over its frames, where the
    object pose is held at its value at the segment's closing event (objects are static until
    grasped), plus the binary grip from `events.aperture_to_grip`.

    Segments share their boundary event frame. Each runs from its first to its last hand-valid
    frame (segments with fewer than 2 valid frames are dropped); invalid frames in between are
    filled by pose interpolation so the segment stays uniform in time (see `largest_gap`). The
    object pose is taken from the last valid frame at or before a grasp (the hand may cover the
    tag), the nearest at a release of a static target, and the first at or after a release when
    the reference is the carried object itself. The trailing segment after the last event reuses
    that event's object and pose."""
    if not demo.hand_valid.any():
        raise ValueError(f"{demo.demo_id}: hand is never valid")
    grip = aperture_to_grip(demo.t, demo.aperture, cfg)
    demo = anchor_depth(demo, grip, cfg)
    found = grip_events(demo.t, grip)
    events = [e.index for e in found]
    last = len(demo.t) - 1
    bounds = [0, *events, last]
    # (reference object, pose lookup side) per closing event; a release skips the object grasped
    # before it, unless that is the only object ever seen (then its post-release pose is used).
    refs: list[tuple[str, Side]] = []
    held: frozenset[str] = frozenset()
    for e in found:
        if e.kind == "grasp":
            refs.append((reference_object(demo, e.index, side="before"), "before"))
            held = frozenset({refs[-1][0]})
        else:
            ref = reference_object(demo, e.index, held)
            refs.append((ref, "after" if ref in held else "nearest"))
            held = frozenset()
    segments = []
    for s, (i0, i1) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        valid = np.flatnonzero(demo.hand_valid[i0 : i1 + 1]) + i0
        if len(valid) < 2:
            continue
        frames = np.arange(valid[0], valid[-1] + 1)
        ref, side = (
            refs[min(s, len(refs) - 1)] if refs else (reference_object(demo, last), "nearest")
        )
        closing = events[min(s, len(events) - 1)] if events else last
        k = demo.object_names.index(ref)
        T_obj_table = inv_T(
            demo.T_table_obj[nearest_valid_index(demo.obj_valid[:, k], closing, side), k]
        )
        T_hand = demo.T_table_hand[frames].copy()
        gap = ~demo.hand_valid[frames]
        if gap.any():
            T_hand[gap] = interpolate_poses(
                demo.t[valid], demo.T_table_hand[valid], demo.t[frames[gap]]
            )
        segments.append(
            HandSegment(
                ref_object=ref,
                t_start=float(demo.t[frames[0]]),
                t_end=float(demo.t[frames[-1]]),
                T_obj_hand=T_obj_table @ T_hand,
                grip=grip[frames],
            )
        )
    return HandSegments(demo_id=demo.demo_id, fps=demo.fps, segments=segments)


def largest_gap(demo: DemoTrajectory, segment: HandSegment) -> float:
    """Longest stretch (s) between consecutive hand-valid frames inside `segment`; 1/fps if none
    was filled."""
    i0, i1 = (int(round(x * demo.fps)) for x in (segment.t_start, segment.t_end))
    valid = np.flatnonzero(demo.hand_valid[i0 : i1 + 1])
    return float(np.diff(valid).max() / demo.fps) if len(valid) > 1 else 0.0


def object_tilt_deg(demo: DemoTrajectory) -> dict[str, float]:
    """Median angle (deg) between each object's z axis and the table normal over its valid
    frames; objects never seen are omitted."""
    tilt = {}
    for k, name in enumerate(demo.object_names):
        valid = demo.obj_valid[:, k]
        if valid.any():
            cos = np.clip(demo.T_table_obj[valid, k, 2, 2], -1.0, 1.0)
            tilt[name] = float(np.median(np.degrees(np.arccos(cos))))
    return tilt
