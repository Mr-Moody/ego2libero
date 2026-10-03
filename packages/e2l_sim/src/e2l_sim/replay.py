"""Execute RobotSegments in LIBERO and record a SimEpisode."""

from e2l_common.config import SimConfig
from e2l_common.schemas import RobotSegments, SimEpisode
from e2l_common.stub import not_implemented


def replay(segments: RobotSegments, cfg: SimConfig) -> SimEpisode:
    """Place objects at their nominal sim poses, express each segment in the sim frame via its
    reference object, track it with `control.track`, and record images, state, actions and the
    task success flag."""
    not_implemented("e2l_sim.replay.replay")
