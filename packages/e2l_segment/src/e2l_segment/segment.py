"""Cut a demo at grip events and re-express each piece relative to its reference object."""

import numpy as np

from e2l_common.config import SegmentConfig
from e2l_common.geometry import inv_T
from e2l_common.schemas import DemoTrajectory, HandSegment, HandSegments
from e2l_segment.assign import nearest_valid_index, reference_object
from e2l_segment.events import aperture_to_grip, grip_events


def segment_demo(demo: DemoTrajectory, cfg: SegmentConfig) -> HandSegments:
    """DemoTrajectory -> HandSegments.

    Segments run from the previous event (or start) to each grasp/release event (or end). Each
    segment stores T_obj_hand = inv(T_table_obj[ref]) @ T_table_hand over its frames, where the
    object pose is held at its value at the segment's closing event (objects are static until
    grasped), plus the binary grip from `events.aperture_to_grip`.

    Segments share their boundary event frame; only hand-valid frames are kept and segments with
    fewer than 2 are dropped. The trailing segment after the last event reuses that event's
    object and pose."""
    grip = aperture_to_grip(demo.t, demo.aperture, cfg)
    found = grip_events(demo.t, grip)
    events = [e.index for e in found]
    last = len(demo.t) - 1
    bounds = [0, *events, last]
    # Reference object per closing event; a release skips the object grasped before it.
    refs: list[str] = []
    held: frozenset[str] = frozenset()
    for e in found:
        refs.append(reference_object(demo, e.index, held if e.kind == "release" else frozenset()))
        held = frozenset({refs[-1]}) if e.kind == "grasp" else frozenset()
    segments = []
    for s, (i0, i1) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        frames = np.arange(i0, i1 + 1)
        frames = frames[demo.hand_valid[frames]]
        if len(frames) < 2:
            continue
        closing = events[s] if s < len(events) else (events[-1] if events else last)
        ref = refs[s] if s < len(refs) else (refs[-1] if refs else reference_object(demo, last))
        k = demo.object_names.index(ref)
        T_obj_table = inv_T(demo.T_table_obj[nearest_valid_index(demo.obj_valid[:, k], closing), k])
        segments.append(
            HandSegment(
                ref_object=ref,
                t_start=float(demo.t[frames[0]]),
                t_end=float(demo.t[frames[-1]]),
                T_obj_hand=T_obj_table @ demo.T_table_hand[frames],
                grip=grip[frames],
            )
        )
    return HandSegments(demo_id=demo.demo_id, fps=demo.fps, segments=segments)
