"""Marker for functions whose algorithm is filled in after initialisation."""

from typing import NoReturn


def not_implemented(qualname: str) -> NoReturn:
    """Raise a NotImplementedError that names the stub, e.g. `e2l_segment.events.grip_events`."""
    raise NotImplementedError(f"{qualname} is a stub; see the project spec for its contract")
