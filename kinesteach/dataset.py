"""Episode layout on disk, and the guard that keeps raw data raw.

    episodes/episode_0001/
      metadata.json          provenance: robot, gains, rates, checksums
      robot.urdf             the exact model the log was produced under
      robot_raw.npz          untouched server log
      gripper_raw.npz        polled Robotiq states (separate clock, plan 2.8)
      validation.json        report from validate.py
      processed.npz          filtered / resampled / FK -- always derived
      cutoff_sweep.npz       one filtered q per candidate cutoff
      replay/pass_0001/      a replay pass, itself a full episode directory

INVARIANT 2 (plan 6, baseline 16/18): raw arrays are never rewritten. This
module enforces that rather than documenting it -- `write_raw` refuses to
overwrite, and `verify_raw` re-checks the checksum recorded at capture time.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .backend.base import EpisodeBuffer, GripperBuffer

__all__ = [
    "Episode",
    "find_episode",
    "RawImmutableError",
    "RawCorruptedError",
    "new_episode",
    "list_episodes",
    "METADATA",
    "ROBOT_RAW",
    "GRIPPER_RAW",
    "PROCESSED",
]

METADATA = "metadata.json"
ROBOT_RAW = "robot_raw.npz"
GRIPPER_RAW = "gripper_raw.npz"
PROCESSED = "processed.npz"
CUTOFF_SWEEP = "cutoff_sweep.npz"
VALIDATION = "validation.json"
URDF = "robot.urdf"
REPLAY_DIR = "replay"

#: Files that are written exactly once, at capture time.
RAW_FILES = frozenset({ROBOT_RAW, GRIPPER_RAW, URDF})

_EPISODE_RE = re.compile(r"^episode_(\d+)$")


class RawImmutableError(RuntimeError):
    """Raised on any attempt to rewrite a raw capture file."""


class RawCorruptedError(RuntimeError):
    """Raised when a raw file no longer matches its recorded checksum."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Episode:
    """A single episode directory. Cheap to construct; touches disk lazily."""

    def __init__(self, path):
        self.path = Path(path)

    def __repr__(self) -> str:
        return "Episode(%r)" % str(self.path)

    def __eq__(self, other) -> bool:
        return isinstance(other, Episode) and self.path == other.path

    def __hash__(self) -> int:
        # Defining __eq__ alone would make Episode unhashable, so a set or a
        # dict key of episodes would raise rather than de-duplicate.
        return hash(self.path)

    @property
    def name(self) -> str:
        return self.path.name

    def file(self, name: str) -> Path:
        return self.path / name

    def exists(self) -> bool:
        return self.file(METADATA).exists()

    # ---- metadata ------------------------------------------------------

    def read_metadata(self) -> Dict[str, Any]:
        return json.loads(self.file(METADATA).read_text())

    def write_metadata(self, meta: Dict[str, Any]) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        # Same encoder as write_json: metadata is assembled from spec, config
        # and controller dicts, and one caller passing a numpy scalar through
        # would otherwise fail the save *after* the raw files are on disk.
        self.file(METADATA).write_text(
            json.dumps(meta, indent=2, sort_keys=False, default=_json_default))

    def update_metadata(self, **changes: Any) -> Dict[str, Any]:
        meta = self.read_metadata()
        meta.update(changes)
        self.write_metadata(meta)
        return meta

    def read_json(self, name: str) -> Dict[str, Any]:
        return json.loads(self.file(name).read_text())

    def write_json(self, name: str, obj: Dict[str, Any]) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        self.file(name).write_text(json.dumps(obj, indent=2, default=_json_default))

    # ---- raw (write once) ----------------------------------------------

    def write_raw(
        self,
        robot: EpisodeBuffer,
        gripper: Optional[GripperBuffer] = None,
        urdf_text: Optional[str] = None,
    ) -> Dict[str, str]:
        """Write the capture files. Refuses to touch one that already exists.

        Returns the checksums, for the caller to store in metadata.
        """
        self.path.mkdir(parents=True, exist_ok=True)
        for name in RAW_FILES:
            if self.file(name).exists():
                raise RawImmutableError(
                    "%s already exists in %s; raw capture files are written "
                    "once (invariant 2). Write derived data to %s instead."
                    % (name, self.path, PROCESSED)
                )
        np.savez_compressed(self.file(ROBOT_RAW), **robot.as_dict())
        if gripper is not None:
            np.savez_compressed(self.file(GRIPPER_RAW), **gripper.as_dict())
        if urdf_text:
            self.file(URDF).write_text(urdf_text)
        return self.raw_checksums()

    def raw_checksums(self) -> Dict[str, str]:
        return {
            name: _sha256(self.file(name))
            for name in sorted(RAW_FILES)
            if self.file(name).exists()
        }

    def verify_raw(self) -> Dict[str, str]:
        """Re-check raw files against the checksums stored at capture time.

        Raises RawCorruptedError on any mismatch. This is the runtime half of
        invariant 2: the write guard stops us, the checksum catches everything
        else (a stray script, an editor, a partial copy).
        """
        recorded = self.read_metadata().get("raw_sha256") or {}
        if not recorded:
            raise RawCorruptedError(
                "%s has no raw_sha256 in metadata; cannot verify" % self.path
            )
        actual = self.raw_checksums()
        bad = {
            k: (v, actual.get(k))
            for k, v in recorded.items()
            if actual.get(k) != v
        }
        if bad:
            raise RawCorruptedError(
                "raw file(s) changed since capture in %s: %s" % (self.path, bad)
            )
        return actual

    def read_raw(self, verify: bool = True) -> EpisodeBuffer:
        if verify:
            self.verify_raw()
        with np.load(self.file(ROBOT_RAW)) as z:
            return EpisodeBuffer.from_dict({k: z[k] for k in z.files})

    def read_gripper_raw(self) -> Optional[GripperBuffer]:
        p = self.file(GRIPPER_RAW)
        if not p.exists():
            return None
        with np.load(p) as z:
            return GripperBuffer.from_dict({k: z[k] for k in z.files})

    def urdf_path(self) -> Optional[Path]:
        p = self.file(URDF)
        return p if p.exists() else None

    # ---- derived (rewritable) ------------------------------------------

    def write_arrays(self, name: str, arrays: Dict[str, np.ndarray]) -> None:
        if name in RAW_FILES:
            raise RawImmutableError(
                "refusing to write derived data to raw file %r (invariant 2)" % name
            )
        self.path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.file(name), **arrays)

    def read_arrays(self, name: str) -> Dict[str, np.ndarray]:
        with np.load(self.file(name)) as z:
            return {k: z[k] for k in z.files}

    def write_processed(self, arrays: Dict[str, np.ndarray]) -> None:
        self.write_arrays(PROCESSED, arrays)

    def read_processed(self) -> Dict[str, np.ndarray]:
        return self.read_arrays(PROCESSED)

    def has_processed(self) -> bool:
        return self.file(PROCESSED).exists()

    # ---- replay passes -------------------------------------------------

    def replay_passes(self) -> List["Episode"]:
        root = self.path / REPLAY_DIR
        if not root.is_dir():
            return []
        return [Episode(p) for p in sorted(root.iterdir()) if (p / METADATA).exists()]

    def new_replay_pass(self) -> "Episode":
        root = self.path / REPLAY_DIR
        root.mkdir(parents=True, exist_ok=True)
        n = 1 + max(
            [int(p.name.split("_")[-1]) for p in root.iterdir() if p.is_dir() and p.name.startswith("pass_")]
            or [0]
        )
        ep = Episode(root / ("pass_%04d" % n))
        ep.path.mkdir(parents=True, exist_ok=True)
        return ep


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError("not JSON serialisable: %r" % (type(o),))


def find_episode(root, name: str) -> Episode:
    """The episode called `name` under `root`.

    Raises KeyError; callers turn that into whatever their front end needs --
    `SystemExit` on the CLI, `CommandRejected` in the WebUI worker.
    """
    for ep in list_episodes(root):
        if ep.name == name:
            return ep
    raise KeyError("no episode named %r under %s" % (name, root))


def list_episodes(root) -> List[Episode]:
    root = Path(root)
    if not root.is_dir():
        return []
    out = [Episode(p) for p in sorted(root.iterdir()) if _EPISODE_RE.match(p.name)]
    return [e for e in out if e.exists()]


def new_episode(root, name: Optional[str] = None) -> Episode:
    """Allocate the next `episode_NNNN` directory under `root`."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if name is None:
        used = [
            int(m.group(1))
            for m in (_EPISODE_RE.match(p.name) for p in root.iterdir())
            if m
        ]
        name = "episode_%04d" % (1 + max(used or [0]))
    ep = Episode(root / name)
    if ep.path.exists() and any(ep.path.iterdir()):
        raise RawImmutableError("episode directory %s already has content" % ep.path)
    ep.path.mkdir(parents=True, exist_ok=True)
    return ep
