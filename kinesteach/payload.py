"""Estimating the load the control box does not know about.

The Desk End Effector settings declare 0.9 kg at [0, 0, 0.057] m, which covers
the Robotiq but not the ZED 2i, its bracket or the cabling. Gravity
compensation is therefore short by roughly 0.1-0.3 kg, and the arm sinks
whenever the operator lets go of it.

No teaching gain fixes that. Stiffness would hold a pose but teaching requires
`Kq = 0` (invariant 4), and damping only sets the speed of the descent, never
stops it -- both settings tried on the robot sank (progress 8.7/8.8). The only
mechanism that holds against gravity is gravity compensation itself, so the
missing mass has to be measured and registered.

MODEL. An unmodelled point mass `m` sitting at `c` in the flange frame pulls on
the arm with a wrench at the flange: force `m*g` and moment `(R c) x m*g`. With
the world-frame Jacobian `J = [J_v; J_w]` that is

    tau = m * (J_v^T g)  -  J_w^T skew(g) R (m c)

which is *linear* in `theta = [m, m*c]`, four unknowns. Add one constant offset
per joint for the torque sensors' own bias and the whole thing is an ordinary
least squares problem in `4 + dof` parameters.

Joint 1 is the check that the geometry is right: its axis is vertical, so no
mass anywhere on the arm can produce a torque about it, and its row of the
regressor comes out exactly zero. Whatever joint 1 reads is pure bias.

SIGN. Whether `tau_external` reports the load's torque or its negation is a
convention we do not get to assume, and the fit cannot tell us -- flipping the
model just flips `theta`. Physics decides instead: mass is positive, so a
negative `m` means the convention is inverted. `estimate_payload` says so
rather than silently returning a negative mass.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "GRAVITY",
    "envelope_report",
    "gravity_regressor",
    "design_matrix",
    "estimate_payload",
    "select_poses",
    "sample_poses",
]

#: World-frame gravity vector (m/s^2). Z up, as in the URDF.
GRAVITY = np.array([0.0, 0.0, -9.81])


# Torque noise to quote uncertainties at. Test A measured the static bias
# reproducible to 0.003 Nm, so this is deliberately several times pessimistic:
# `tau_external` also carries model error, which noise alone does not cover.
_TAU_NOISE_NM = 0.02

# The unregistered load is expected between 0.1 and 0.3 kg. Pinning it to 10 g
# is already far finer than Desk's own entry resolution.
_MASS_STDERR_MAX_KG = 0.01

# A flange axis whose projection on gravity covers less than half of its full
# [-1, 1] range never turned over, and the first moment along it stays tangled
# with the mass.
_ORIENTATION_SPAN_MIN = 1.0

# Reach below this spans too little lever to weigh anything.
_REACH_SPAN_MIN_M = 0.15

# Wrist turning confined to less than this share of the reach that was covered
# means the two families were demonstrated one after the other instead of
# together, and they have to be seen together.
_TURN_REACH_FRACTION = 0.6


def envelope_report(
    qs: Sequence[np.ndarray], fk: Any, gravity: np.ndarray = GRAVITY,
    tau_noise_nm: float = _TAU_NOISE_NM,
) -> Dict[str, Any]:
    """How well a chosen set of poses pins each parameter down.

    Returned per-parameter standard errors are for unit measurement noise, so
    multiply by the torque noise to get real ones. The point is which entry is
    *worst*: `m` is the mass, and `m*c_x/y/z` are the first moments along the
    flange's own axes, so a bad one names the flange direction the poses failed
    to rotate through.
    """
    D, dof = design_matrix(qs, fk, gravity)
    sv = np.linalg.svd(D, compute_uv=False)
    cond = float(sv.max() / sv.min()) if sv.min() > 0 else float("inf")
    try:
        unit = np.sqrt(np.diag(np.linalg.inv(D.T @ D)))
    except np.linalg.LinAlgError:  # pragma: no cover
        unit = np.full(D.shape[1], np.inf)
    names = ["mass", "m*c_x", "m*c_y", "m*c_z"]
    worst = int(np.argmax(unit[:4]))

    # Naming the worst parameter is not the same as naming the fix. The first
    # real walk covered 0.32-0.74 m of reach -- plenty -- yet reported `mass`
    # as worst and advised reaching further, because mass had nothing to do
    # with it: the gripper pointed downwards at every pose, so `m` and `m*c`
    # could imitate each other. Measure both families and let the narrower one
    # speak.
    # A bare condition number is not a verdict. The second usable walk scored
    # 82, which the old `cond < 40` called a failure -- but it pinned the mass
    # to 3 g of an expected 0.1-0.3 kg load, and acting on it would have cost
    # a third walk for nothing. Judge the answer, not the matrix.
    mass_sd = float(unit[0] * tau_noise_nm)
    ok = bool(np.isfinite(cond) and mass_sd < _MASS_STDERR_MAX_KG)

    reach, axis_span, turn_reach = _pose_spread(qs, fk, gravity)
    tight = int(np.argmin(axis_span))
    if reach < _REACH_SPAN_MIN_M and axis_span.max() < _ORIENTATION_SPAN_MIN:
        fix = ("the poses barely differ (%.2f m of reach, %.2f of turn) -- they "
               "have to be spread through both" % (reach, axis_span.max()))
    elif reach > 0 and turn_reach < _TURN_REACH_FRACTION * reach:
        # The failure the first real walk actually had. Its reach covered
        # 0.32-0.74 m and its wrist turned through 64 deg, so every marginal
        # statistic looked healthy -- but the turning all happened at
        # 0.53-0.58 m. Adding two poses that turn the wrist with the arm
        # extended took it from 96 to 32; going past horizontal was never
        # needed. The parameters are fixed by the combination, not the margins.
        fix = ("reach spans %.2f m but the wrist only turns within %.2f m of "
               "it -- turn the wrist again with the arm extended, and again "
               "folded in, rather than only in the middle" % (reach, turn_reach))
    elif axis_span[tight] < _ORIENTATION_SPAN_MIN:
        fix = ("the flange never turned much about its %s axis (span %.2f of "
               "2.0) -- lay the gripper over further in that direction"
               % ("xyz"[tight], axis_span[tight]))
    elif reach < _REACH_SPAN_MIN_M:
        fix = ("reach varies by only %.2f m -- reach further out and fold back "
               "in, mass only shows up through the moment arm" % reach)
    elif worst == 0:
        fix = "reach further out and fold back in"
    else:
        fix = ("rotate the flange so its %s axis points up and down in "
               "different poses" % names[worst][-1])
    return {
        "n_poses": len(qs),
        "condition_number": cond,
        "unit_stderr": {n: float(unit[i]) for i, n in enumerate(names)},
        "worst_parameter": names[worst],
        "reach_span_m": float(reach),
        "turning_reach_span_m": float(turn_reach),
        "axis_vs_gravity_span": {a: float(v) for a, v in zip("xyz", axis_span)},
        "mass_stderr_kg": mass_sd,
        "tau_noise_nm": float(tau_noise_nm),
        "well_conditioned": ok,
        "advice": ("good enough: mass to +/-%.3f kg at %.3f Nm of torque noise"
                   % (mass_sd, tau_noise_nm) if ok else
                   "mass only to +/-%.3f kg (condition %.0f); %s is the least "
                   "determined -- %s" % (mass_sd, cond, names[worst], fix)),
    }


def _pose_spread(qs: Sequence[np.ndarray], fk: Any, gravity: np.ndarray):
    """How far the poses travelled, and how far they turned over.

    Returns the horizontal reach span in metres, the span of each flange axis'
    projection on gravity -- the quantity the regressor actually sees -- and the
    reach span of just those poses whose wrist is turned away from vertical,
    which is how far out the turning itself was demonstrated.
    """
    Q = np.atleast_2d(np.asarray(qs, dtype=np.float64))
    pos, quat = fk.fk(Q)
    reach = np.linalg.norm(pos[:, :2], axis=1)
    ghat = gravity / np.linalg.norm(gravity)
    R = np.array([_rotation(q) for q in quat])
    proj = np.einsum("nji,j->ni", R, ghat)      # (n, 3): each flange axis . g
    tilt = np.arccos(np.clip(proj[:, 2], -1.0, 1.0))
    turned = tilt >= 0.5 * float(tilt.max()) if len(tilt) else np.zeros(0, bool)
    turn_reach = float(np.ptp(reach[turned])) if turned.sum() > 1 else 0.0
    return float(np.ptp(reach)), np.ptp(proj, axis=0), turn_reach


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _rotation(quat: np.ndarray) -> np.ndarray:
    """Rotation matrix from an xyzw quaternion (the order `fk()` returns)."""
    x, y, z, w = np.asarray(quat, dtype=np.float64)
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        raise ValueError("degenerate quaternion")
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def gravity_regressor(q: np.ndarray, fk: Any, gravity: np.ndarray = GRAVITY) -> np.ndarray:
    """(dof, 4) mapping [m, m*cx, m*cy, m*cz] to joint torques at `q`."""
    q = np.asarray(q, dtype=np.float64).reshape(1, -1)
    J = fk.jacobian(q)[0]
    dof = J.shape[1]
    Jv, Jw = J[:3], J[3:]
    _, quat = fk.fk(q)
    R = _rotation(quat[0])

    A = np.empty((dof, 4), dtype=np.float64)
    A[:, 0] = Jv.T @ gravity
    A[:, 1:] = -Jw.T @ _skew(gravity) @ R
    return A


def design_matrix(
    qs: Sequence[np.ndarray], fk: Any, gravity: np.ndarray = GRAVITY
) -> Tuple[np.ndarray, int]:
    """((dof*N, 4+dof), dof). Load columns first, then one bias column per joint."""
    blocks = [gravity_regressor(q, fk, gravity) for q in qs]
    dof = blocks[0].shape[0]
    n = len(blocks)
    D = np.zeros((dof * n, 4 + dof), dtype=np.float64)
    for i, A in enumerate(blocks):
        if A.shape[0] != dof:
            raise ValueError("poses disagree on DOF")
        D[i * dof:(i + 1) * dof, :4] = A
        D[i * dof:(i + 1) * dof, 4:] = np.eye(dof)
    return D, dof


def estimate_payload(
    qs: Sequence[np.ndarray],
    taus: Sequence[np.ndarray],
    fk: Any,
    gravity: np.ndarray = GRAVITY,
) -> Dict[str, Any]:
    """Least-squares fit of the unmodelled load and the per-joint torque bias.

    `qs` and `taus` are one settled, untouched pose each: the arm at rest with
    nobody holding it, so `tau_external` is the missing gravity plus bias and
    nothing else.
    """
    qs = [np.asarray(q, dtype=np.float64).ravel() for q in qs]
    taus = [np.asarray(t, dtype=np.float64).ravel() for t in taus]
    if len(qs) != len(taus):
        raise ValueError("got %d pose(s) but %d torque vector(s)" % (len(qs), len(taus)))
    if len(qs) < 2:
        raise ValueError("need at least 2 poses; %d given" % len(qs))

    D, dof = design_matrix(qs, fk, gravity)
    y = np.concatenate(taus)
    if y.size != D.shape[0]:
        raise ValueError("torque vectors do not match the robot's %d DOF" % dof)

    sol, _, rank, sv = np.linalg.lstsq(D, y, rcond=None)
    resid = y - D @ sol
    dofs_left = max(D.shape[0] - D.shape[1], 1)
    sigma2 = float(resid @ resid) / dofs_left
    try:
        cov = sigma2 * np.linalg.inv(D.T @ D)
        stderr = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:  # pragma: no cover - singular designs
        stderr = np.full(D.shape[1], np.nan)

    m, u = float(sol[0]), sol[1:4]
    bias = sol[4:]
    flipped = m < 0.0
    mass = abs(m)
    # c = u / m is ill-conditioned when m is small; report it but say how sure.
    com = (u / m) if abs(m) > 1e-6 else np.full(3, np.nan)

    return {
        "mass_kg": mass,
        "com_flange_m": com.tolist(),
        "first_moment_kg_m": (u if not flipped else -u).tolist(),
        # Not negated when `flipped`, unlike the load terms. If the sensor
        # reports the load's torque negated then `y = -(A theta_true) + b`, so
        # the fit returns `-theta_true` for the load columns but `b` itself for
        # the bias columns -- the bias is already in the reported convention.
        "bias_nm": bias.tolist(),
        "sign_flipped": bool(flipped),
        "n_poses": len(qs),
        "residual_rms_nm": float(np.sqrt(np.mean(resid ** 2))),
        "residual_max_nm": float(np.abs(resid).max()),
        "signal_rms_nm": float(np.sqrt(np.mean(y ** 2))),
        "mass_stderr_kg": float(stderr[0]),
        "bias_stderr_nm": stderr[4:].tolist(),
        "condition_number": float(sv.max() / sv.min()) if sv.min() > 0 else float("inf"),
        "rank": int(rank),
        "full_rank": bool(rank == D.shape[1]),
    }


# ---------------------------------------------------------------------------
# Choosing where to measure
# ---------------------------------------------------------------------------


def _logdet(M: np.ndarray) -> float:
    sign, val = np.linalg.slogdet(M)
    return val if sign > 0 else -np.inf


def select_poses(
    candidates: Sequence[np.ndarray],
    fk: Any,
    n: int,
    gravity: np.ndarray = GRAVITY,
    seed_indices: Optional[Sequence[int]] = None,
) -> List[int]:
    """Greedy D-optimal subset: the `n` poses that pin the parameters down best.

    Maximising `det(D^T D)` shrinks the confidence ellipsoid of the whole
    parameter vector, which in practice means picking poses whose gravity
    signatures point in different directions -- extended against folded, and
    wrists rolled to opposite sides. Guessing at that by hand is exactly what
    this replaces.
    """
    if n < 2:
        raise ValueError("need at least 2 poses")
    blocks = [gravity_regressor(q, fk, gravity) for q in candidates]
    dof = blocks[0].shape[0]
    eye = np.eye(dof)
    rows = [np.hstack([A, eye]) for A in blocks]

    chosen = list(seed_indices or [])
    # A tiny ridge keeps the determinant finite before the design has rank.
    acc = 1e-9 * np.eye(4 + dof)
    for i in chosen:
        acc = acc + rows[i].T @ rows[i]

    while len(chosen) < min(n, len(rows)):
        best, best_score = None, -np.inf
        for i in range(len(rows)):
            if i in chosen:
                continue
            score = _logdet(acc + rows[i].T @ rows[i])
            if score > best_score:
                best, best_score = i, score
        if best is None:  # pragma: no cover - exhausted
            break
        chosen.append(best)
        acc = acc + rows[best].T @ rows[best]
    return chosen


def sample_poses(
    spec: Any,
    fk: Any,
    n: int,
    rng: Optional[np.random.Generator] = None,
    limit_margin_rad: float = 0.35,
    min_flange_z: float = 0.25,
    max_reach_m: float = 0.75,
    min_reach_m: float = 0.30,
) -> List[np.ndarray]:
    """Random joint configurations that are safe to visit.

    Rejects anything close to a joint limit or with the flange low, far out or
    folded into the column. The margin is deliberately wider than the server's
    own 0.2 rad reflex margin so the sweep never provokes it -- a pose where
    the safety controller is pushing back is a pose whose `tau_external` is not
    the payload.
    """
    rng = rng or np.random.default_rng(0)
    lo, hi = spec.joint_pos_min, spec.joint_pos_max
    if lo is None or hi is None:
        raise ValueError("spec has no joint limits; refusing to sample poses")
    lo = np.asarray(lo, dtype=np.float64) + limit_margin_rad
    hi = np.asarray(hi, dtype=np.float64) - limit_margin_rad
    if np.any(lo >= hi):
        raise ValueError("limit_margin_rad %.2f leaves no room" % limit_margin_rad)

    out: List[np.ndarray] = []
    for _ in range(200 * n):
        if len(out) >= n:
            break
        q = rng.uniform(lo, hi)
        pos, _ = fk.fk(q[None])
        p = pos[0]
        r = float(np.linalg.norm(p[:2]))
        if p[2] < min_flange_z or not (min_reach_m <= r <= max_reach_m):
            continue
        out.append(q)
    if len(out) < n:
        log.warning("only found %d/%d valid poses; loosen the box", len(out), n)
    return out
