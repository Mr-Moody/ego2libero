"""Cut a demo at grip events and re-express each piece relative to its reference object."""

from e2l_common.config import SegmentConfig
from e2l_common.schemas import DemoTrajectory, HandSegments
from e2l_common.stub import not_implemented


def segment_demo(demo: DemoTrajectory, cfg: SegmentConfig) -> HandSegments:
    """DemoTrajectory -> HandSegments.

    Segments run from the previous event (or start) to each grasp/release event (or end). Each
    segment stores T_obj_hand = inv(T_table_obj[ref]) @ T_table_hand over its frames, where the
    object pose is held at its value at the segment's closing event (objects are static until
    grasped), plus the binary grip from `events.aperture_to_grip`."""
    not_implemented("e2l_segment.segment.segment_demo")
