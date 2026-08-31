"""The region the arm is allowed to drive itself through, and the check that
a move stays inside it.

THE SAFE ENVELOPE, AS A CONVEX SET IN JOINT SPACE.

`move_to_joint_positions` plans in joint space, so the path between two poses
is the straight segment joining them *there*. Describe the safe region in the
same space and containment becomes free: a convex set contains every segment
between its members, so a sweep that only ever visits convex combinations of
demonstrated configurations never leaves the region the operator walked.

A Cartesian box cannot make that promise -- measured on the lab robot, legs
between two in-box poses bulged up to 0.43 m outside it (progress 8.9).

Two things this does NOT prove, both left to the caller:

  - the hull of collision-free poses can contain colliding ones, if the
    operator guided *around* something. The plan is printed for confirmation.
  - forward kinematics is nonlinear, so the Cartesian image of the hull can
    exceed the demonstrated Cartesian extent. `path_is_safe` keeps a floor
    check for that, and for the approach from wherever the arm was left.

Kept out of `payload.py` on purpose. Everything here is a *safety* artefact
used by any tool that moves the arm on its own -- the payload sweep is only its
first caller, and a homing routine reaching into a load-estimation module to
ask whether its path is clear reads like the wrong dependency, because it is.
"""
from __future__ import annotations

import json
import logging
import pathlib
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "save_envelope",
    "load_envelope",
    "sample_hull",
    "path_is_safe",
]


def save_envelope(path: str, qs: np.ndarray, spec: Any, fk: Any) -> Dict[str, Any]:
    """Store demonstrated configurations as the vertices of a safe region."""
    qs = np.atleast_2d(np.asarray(qs, dtype=np.float64))
    pos, _ = fk.fk(qs)
    data = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "robot_model": spec.robot_model,
        "num_dofs": int(spec.num_dofs),
        "ee_link_name": spec.ee_link_name,
        "n_vertices": int(qs.shape[0]),
        "vertices_q": qs.tolist(),
        "joint_range_rad": {
            "min": qs.min(axis=0).tolist(),
            "max": qs.max(axis=0).tolist(),
        },
        "flange_extent_m": {
            "min": pos.min(axis=0).tolist(),
            "max": pos.max(axis=0).tolist(),
        },
    }
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(path).write_text(json.dumps(data, indent=2))
    return data


def load_envelope(path: str) -> Dict[str, Any]:
    data = json.loads(pathlib.Path(path).read_text())
    if not data.get("vertices_q"):
        raise ValueError("%s carries no vertices_q" % path)
    data["vertices_q"] = np.asarray(data["vertices_q"], dtype=np.float64)
    return data


def sample_hull(
    vertices: np.ndarray,
    n: int,
    rng: Optional[np.random.Generator] = None,
    k: int = 3,
    alpha: float = 0.3,
) -> List[np.ndarray]:
    """`n` points guaranteed to lie in the convex hull of `vertices`.

    Each is a convex combination of `k` randomly chosen vertices with sparse
    Dirichlet weights: `alpha < 1` pushes the mass onto one or two of them, so
    the samples spread towards the boundary instead of piling up in the middle
    the way uniform weights would.
    """
    rng = rng or np.random.default_rng(0)
    V = np.atleast_2d(np.asarray(vertices, dtype=np.float64))
    if len(V) < 2:
        raise ValueError("need at least 2 vertices, got %d" % len(V))
    k = int(min(k, len(V)))
    out = []
    for _ in range(n):
        idx = rng.choice(len(V), size=k, replace=False)
        w = rng.dirichlet([alpha] * k)
        out.append(w @ V[idx])
    return out


def path_is_safe(
    q_from: np.ndarray,
    q_to: np.ndarray,
    fk: Any,
    spec: Any,
    limit_margin_rad: float = 0.25,
    min_flange_z: float = 0.20,
    max_reach_m: float = 0.80,
    steps: int = 60,
) -> Tuple[bool, str]:
    """Check the whole straight line between two poses, not just its ends.

    `move_to_joint_positions` plans a min-jerk profile *in joint space*, which
    only reparameterises time -- the path really is this straight line. Its
    Cartesian shape is not controlled, though, so a move between two perfectly
    good poses can still swing the flange down through the table. Same argument
    as the replay pre-flight check (progress 5): find that here, where nothing
    has moved yet.
    """
    q_from = np.asarray(q_from, dtype=np.float64).ravel()
    q_to = np.asarray(q_to, dtype=np.float64).ravel()
    lo, hi = spec.joint_pos_min, spec.joint_pos_max
    if lo is None or hi is None:
        return False, "spec carries no joint limits to check against"
    lo = np.asarray(lo, dtype=np.float64) + limit_margin_rad
    hi = np.asarray(hi, dtype=np.float64) - limit_margin_rad

    ts = np.linspace(0.0, 1.0, steps)[:, None]
    path = q_from[None, :] * (1 - ts) + q_to[None, :] * ts
    if np.any(path < lo) or np.any(path > hi):
        j = int(np.argmax(np.max((lo - path).clip(0) + (path - hi).clip(0), axis=0)))
        return False, "joint %d passes within %.2f rad of its limit" % (j + 1, limit_margin_rad)

    pos, _ = fk.fk(path)
    if pos[:, 2].min() < min_flange_z:
        return False, "flange dips to z=%.3f m, below the %.2f m floor" % (
            pos[:, 2].min(), min_flange_z)
    r = np.linalg.norm(pos[:, :2], axis=1)
    if r.max() > max_reach_m:
        return False, "flange reaches %.3f m out, past the %.2f m limit" % (r.max(), max_reach_m)
    return True, "ok"
