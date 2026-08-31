"""Turn a finished capture into an episode on disk.

The core of the project never sees a protobuf: `EpisodeBuffer` arrives here as
plain arrays and leaves as npz (invariant 7).

Metadata is written with reproducibility in mind. EE pose is not in the robot
state and has to be reconstructed offline by forward kinematics (plan 2.1), so
the URDF and `ee_link_name` the log was produced under are recorded alongside
it -- without them the episode cannot be turned back into Cartesian data.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, Optional

from . import KINESTEACH_VERSION
from .backend.base import EpisodeBuffer, GripperBuffer, RobotSpec
from .config import Config
from .dataset import URDF, VALIDATION, Episode, new_episode
from .validate import validate_buffer

__all__ = ["KINESTEACH_VERSION", "save_teaching_episode", "save_replay_pass", "build_metadata"]


def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def build_metadata(
    spec: RobotSpec,
    cfg: Config,
    buf: EpisodeBuffer,
    kind: str,
    controller: Dict[str, Any],
    gripper: Optional[GripperBuffer] = None,
    notes: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "kinesteach_version": KINESTEACH_VERSION,
        "kind": kind,  # teaching | replay
        "created_at": _now(),
        "notes": notes,
        # ---- robot: never hardcoded, always from the server (invariant 3)
        "urdf_path": URDF,
        "controller": controller,
        # ---- capture
        "n_states": buf.n,
        "duration_s": buf.duration,
        "effective_hz": buf.effective_hz,
        "gripper": {
            "enabled": cfg.backend.gripper_enabled,
            "model": "robotiq_2f" if cfg.backend.gripper_enabled else None,
            "poll_hz": cfg.backend.gripper_poll_hz,
            "n_samples": 0 if gripper is None else gripper.n,
        },
        # ---- Phase 2, deliberately unset for now (plan 2.2)
        "payload_mass": None,
        "payload_com": None,
        "tcp": None,
        "ee_frame": "flange",
        "config": cfg.to_dict(),
    }
    meta.update(spec.to_metadata())
    if extra:
        meta.update(extra)
    return meta


def _save(
    ep: Episode,
    buf: EpisodeBuffer,
    spec: RobotSpec,
    cfg: Config,
    kind: str,
    controller: Dict[str, Any],
    gripper: Optional[GripperBuffer],
    notes: str,
    extra: Optional[Dict[str, Any]],
) -> Episode:
    checksums = ep.write_raw(buf, gripper=gripper, urdf_text=spec.urdf_text)
    meta = build_metadata(spec, cfg, buf, kind, controller, gripper, notes, extra)
    meta["raw_sha256"] = checksums
    ep.write_metadata(meta)

    report = validate_buffer(buf, spec, max_duration_s=cfg.teach.max_duration_s)
    ep.write_json(VALIDATION, report)
    return ep


def save_teaching_episode(
    root,
    buf: EpisodeBuffer,
    spec: RobotSpec,
    cfg: Config,
    controller: Dict[str, Any],
    gripper: Optional[GripperBuffer] = None,
    notes: str = "",
    episode: Optional[Episode] = None,
) -> Episode:
    ep = episode or new_episode(root)
    return _save(ep, buf, spec, cfg, "teaching", controller, gripper, notes, None)


def save_replay_pass(
    source: Episode,
    buf: EpisodeBuffer,
    spec: RobotSpec,
    cfg: Config,
    controller: Dict[str, Any],
    gripper: Optional[GripperBuffer] = None,
    notes: str = "",
) -> Episode:
    """Store a replay pass inside the episode it replays.

    Kept as a full episode directory of its own so validate.py and the WebUI
    treat a replay exactly like a demonstration.
    """
    ep = source.new_replay_pass()
    extra = {"source_episode": str(source.path)}
    return _save(ep, buf, spec, cfg, "replay", controller, gripper, notes, extra)
