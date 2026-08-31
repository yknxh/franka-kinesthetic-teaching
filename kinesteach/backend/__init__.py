"""Robot backends. Nothing outside this package sees polymetis types."""
from __future__ import annotations

from typing import Optional

from ..config import BackendConfig
from .base import (
    GRIPPER_ARRAY_FIELDS,
    ROBOT_ARRAY_FIELDS,
    EpisodeBuffer,
    GripperBuffer,
    RobotBackend,
    RobotSpec,
    TelemetrySample,
)
from .mock import MockBackend

__all__ = [
    "EpisodeBuffer",
    "GripperBuffer",
    "RobotBackend",
    "RobotSpec",
    "TelemetrySample",
    "ROBOT_ARRAY_FIELDS",
    "GRIPPER_ARRAY_FIELDS",
    "MockBackend",
    "make_backend",
]


def make_backend(cfg: Optional[BackendConfig] = None) -> RobotBackend:
    """Construct the backend named by `cfg.kind`."""
    cfg = cfg or BackendConfig()
    if cfg.kind == "mock":
        return MockBackend(cfg)
    # Imported lazily: polymetis is not needed to work with recorded episodes.
    from .polymetis import PolymetisBackend

    return PolymetisBackend(cfg)
