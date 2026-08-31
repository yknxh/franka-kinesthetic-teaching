"""Shared plumbing for the command line entry points.

Its own module because both halves of the CLI need it and neither owns it:
`cli.py` runs the teach/process/replay pipeline, `cli_payload.py` runs the
hand-walked envelope and the load identification that stands beside it.

Callers reach these through the module (`cli_common._connected(cfg)`) rather
than binding the name at import time, so a test that replaces one replaces it
for every command.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from ..config import Config
from ..dataset import find_episode

log = logging.getLogger("kinesteach")

#: How close counts as home. Wider than the replay start-pose tolerance on
#: purpose: homing is a convenience, not a safety gate.
HOME_ARRIVAL_TOL_RAD = 0.05


def _cfg(args) -> Config:
    cfg = Config.load(args.config)
    if getattr(args, "backend", None):
        cfg.backend.kind = args.backend
    if getattr(args, "data_root", None):
        cfg.data_root = args.data_root
    if getattr(args, "ip", None):
        cfg.backend.ip_address = args.ip
    if getattr(args, "port", None):
        cfg.backend.port = args.port
    if getattr(args, "gripper", False):
        cfg.backend.gripper_enabled = True
    # Sweeping this from the shell beats editing the YAML between runs; the
    # gains that were actually used are recorded in the episode either way.
    if getattr(args, "kqd_scale", None) is not None:
        cfg.teach.Kqd_scale = args.kqd_scale
    return cfg


def _find(cfg: Config, name: str):
    try:
        return find_episode(cfg.data_root, name)
    except KeyError as e:
        raise SystemExit(str(e.args[0]))


@contextmanager
def _connected(cfg: Config):
    """Backend with the process-level termination backstop installed.

    A context manager rather than a pair, so the teardown -- stop the policy,
    put the signal handlers back, close the connection -- is written once. It
    used to be copied verbatim into the `finally` of every command that moves
    the arm, which is four chances to fix three of them.
    """
    from ..backend import make_backend
    from ..safety import EmergencyTermination

    backend = make_backend(cfg.backend)
    backend.connect()
    guard = EmergencyTermination(backend).install()
    try:
        yield backend, guard
    finally:
        guard.fire()
        guard.uninstall()
        backend.close()


# ---------------------------------------------------------------- commands



def _checked_home_pose(backend, spec, limit_margin_rad: float = 0.25,
                       min_flange_z: float = 0.20):
    """The home pose, but only once the way there has been checked.

    `go_home` is autonomous motion that verifies nothing, and the moment an
    operator reaches for it -- the arm was left somewhere they would rather not
    start from -- is the moment the path deserves a look. Raises SystemExit
    rather than returning a flag: there is no sensible way to carry on homing
    when the route is not clear.
    """
    from ..envelope import path_is_safe
    from ..kinematics import fk_from_urdf_text

    if spec.home_pose is None:
        raise SystemExit("the server reported no home pose")
    home = np.asarray(spec.home_pose, dtype=np.float64)
    if not spec.urdf_text:
        raise SystemExit("the server sent no URDF; cannot check the way home")
    fk = fk_from_urdf_text(spec.urdf_text, spec.ee_link_name)
    here = backend.get_joint_positions()
    ok, why = path_is_safe(here, home, fk, spec,
                           limit_margin_rad=limit_margin_rad,
                           min_flange_z=min_flange_z)
    if not ok:
        raise SystemExit(
            "the path from where the arm is now to its home pose is not safe: "
            "%s.\nMove it clear by hand first, or start from where it stands."
            % why
        )
    return home



def default_envelope(cfg) -> Path:
    return Path(cfg.data_root).parent / "workspace" / "envelope.json"


def print_issues(rep) -> None:
    for e in rep.get("errors", []):
        print("  ERROR   %s" % e)
    for w in rep.get("warnings", []):
        print("  warning %s" % w)
