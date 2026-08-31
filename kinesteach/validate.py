"""Did the capture actually work?

Every episode gets a report at save time. The failures this is looking for are
the quiet ones -- a log that silently lost its first minute to the server's
ring buffer, a session that ran at 640 Hz instead of 1000, a joint that spent
the demo pinned against a limit with the safety controller pushing back.

Also holds the acceptance tests A-D from baseline 25. They are named `check_*`
rather than `test_*` so pytest does not try to collect them as its own tests.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .backend.base import EpisodeBuffer, RobotSpec

__all__ = [
    "validate_buffer",
    "MAX_CIRCULAR_BUFFER_SIZE",
    "check_a_static_bias",
    "check_b_handguiding",
    "check_c_repeatability",
    "check_d_replay_comparison",
]

#: polymetis_server.hpp: `#define MAX_CIRCULAR_BUFFER_SIZE 300000`
#: -- 5 minutes at 1 kHz, after which the head of the episode is overwritten
#: without any error being raised (plan 2.4).
MAX_CIRCULAR_BUFFER_SIZE = 300000

#: Nm below which `tau_safened_prev - tau_computed_prev` is taken as "the
#: server changed nothing". The two are equal bit-for-bit when the safety layer
#: contributes zero, so this only has to clear float32 round-off.
SAFETY_REFLEX_EPS_NM = 1e-6


def _longest_run(mask: np.ndarray) -> int:
    """Length of the longest run of True in a 1-D boolean array."""
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        if run > best:
            best = run
    return int(best)


def validate_buffer(
    buf: EpisodeBuffer,
    spec: Optional[RobotSpec] = None,
    max_duration_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Summarise a capture and flag anything suspicious.

    `errors` means the data is probably not usable; `warnings` means look at it
    before trusting it.
    """
    errors: List[str] = []
    warnings: List[str] = []
    rep: Dict[str, Any] = {
        "n_states": buf.n,
        "num_dofs": buf.num_dofs,
        "duration_s": buf.duration,
        "effective_hz": buf.effective_hz,
    }

    if buf.n < 2:
        errors.append("log has %d state(s); nothing to validate" % buf.n)
        rep.update(ok=False, errors=errors, warnings=warnings)
        return rep

    nominal_hz = float(spec.control_hz) if spec is not None else buf.effective_hz
    rep["nominal_hz"] = nominal_hz

    # ---- timing --------------------------------------------------------
    dt = buf.dt
    nominal_dt = 1.0 / nominal_hz if nominal_hz > 0 else float(np.median(dt))
    rep["dt"] = {
        "mean_ms": float(np.mean(dt) * 1e3),
        "std_ms": float(np.std(dt) * 1e3),
        "min_ms": float(np.min(dt) * 1e3),
        "median_ms": float(np.median(dt) * 1e3),
        "p99_ms": float(np.percentile(dt, 99) * 1e3),
        "max_ms": float(np.max(dt) * 1e3),
        "nominal_ms": float(nominal_dt * 1e3),
    }
    # Histogram in units of the nominal period, so "1.0" is a healthy sample.
    ratios = dt / nominal_dt
    edges = np.array([0.0, 0.5, 0.9, 1.1, 1.5, 2.5, 5.5, np.inf])
    counts, _ = np.histogram(ratios, bins=edges)
    rep["dt_ratio_histogram"] = {
        "edges": edges[:-1].tolist() + ["inf"],
        "counts": counts.astype(int).tolist(),
    }

    dropped = int(np.sum(np.maximum(np.round(ratios) - 1.0, 0.0)))
    rep["estimated_dropped_samples"] = dropped
    rep["max_gap_ms"] = float(np.max(dt) * 1e3)
    rep["n_gaps_over_2x"] = int(np.sum(ratios > 2.0))

    if np.any(dt <= 0):
        errors.append("timestamps are not strictly increasing")
    if nominal_hz > 0:
        ratio = buf.effective_hz / nominal_hz
        rep["hz_ratio"] = float(ratio)
        if ratio < 0.98:
            warnings.append(
                "effective rate %.1f Hz is %.1f%% of the nominal %.0f Hz"
                % (buf.effective_hz, 100 * ratio, nominal_hz)
            )
    if dropped:
        warnings.append("about %d sample(s) missing from the log" % dropped)

    # ---- ring buffer ---------------------------------------------------
    rep["buffer_capacity"] = MAX_CIRCULAR_BUFFER_SIZE
    rep["buffer_fill"] = buf.n / float(MAX_CIRCULAR_BUFFER_SIZE)
    if buf.n >= MAX_CIRCULAR_BUFFER_SIZE:
        errors.append(
            "log holds %d states, at or above the server ring buffer capacity "
            "(%d): the beginning of this episode was overwritten (plan 2.4)"
            % (buf.n, MAX_CIRCULAR_BUFFER_SIZE)
        )
    elif buf.n > 0.9 * MAX_CIRCULAR_BUFFER_SIZE:
        warnings.append(
            "log is at %.0f%% of the server ring buffer capacity"
            % (100 * rep["buffer_fill"])
        )
    if max_duration_s is not None and buf.duration > max_duration_s:
        warnings.append(
            "episode ran %.1f s, past the configured limit of %.1f s"
            % (buf.duration, max_duration_s)
        )

    # ---- controller health ---------------------------------------------
    codes, code_counts = np.unique(buf.error_code, return_counts=True)
    rep["error_codes"] = {int(c): int(n) for c, n in zip(codes, code_counts)}
    n_err = int(np.sum(buf.error_code != 0))
    n_fail = int(np.sum(~buf.command_successful))
    rep["n_error_states"] = n_err
    rep["n_failed_commands"] = n_fail
    if n_err:
        errors.append("%d state(s) carry a non-zero error_code" % n_err)

    # Dropped command packets.
    #
    # `prev_command_successful` is not an error report from the robot: the
    # client infers it (franka_panda_client.cpp, updateServerCommand) from the
    # desired torques being bit-identical to the previous tick, because that is
    # what libfranka leaves behind when a command packet misses its slot and the
    # last torque is held. A stationary arm at zero stiffness is the case where
    # that inference could plausibly false-positive, so the *count* alone says
    # little.
    #
    # What matters is how many land in a row: one held tick is 1 ms of stale
    # torque, while a long run is what trips the robot's own communication
    # watchdog. Report the run length, and let error_code above be the robot's
    # verdict on whether it minded.
    longest = _longest_run(~buf.command_successful)
    rep["max_consecutive_failed_commands"] = longest
    rep["frac_failed_commands"] = float(n_fail) / buf.n
    if n_fail:
        warnings.append(
            "%d of %d command packet(s) appear dropped (%.2f%%), longest run %d "
            "tick(s); the previous torque was held for those"
            % (n_fail, buf.n, 100.0 * rep["frac_failed_commands"], longest)
        )
    rep["latency_ms"] = {
        "mean": float(np.mean(buf.latency_ms)),
        "max": float(np.max(buf.latency_ms)),
    }

    # ---- server safety-layer intervention -------------------------------
    # The robot client does not send our policy's torque straight through: it
    # adds a reflex torque of its own when a limit is approached, then clamps
    # the sum to the configured torque limits (franka_panda_client.cpp,
    # `checkStateLimits` + `postprocessTorques`). `tau_computed_prev` is what
    # the controller server asked for and `tau_safened_prev` is what was
    # actually applied, so their difference *is* that intervention.
    #
    # Worth its own check because the limits driving it live in a config file
    # on the server machine that this workstation cannot read. Rather than
    # trusting a copy of those numbers, every episode measures whether they
    # fired. Anything non-zero here also means tau_external for those samples
    # contains the server pushing back, not just the operator (plan 2.7).
    reflex = buf.tau_safened_prev - buf.tau_computed_prev
    per_joint = np.abs(reflex).max(axis=0)
    active = np.abs(reflex).max(axis=1) > SAFETY_REFLEX_EPS_NM
    rep["safety_reflex_absmax_nm"] = per_joint.tolist()
    rep["frac_states_with_safety_reflex"] = float(np.mean(active))
    if np.any(active):
        joints = np.flatnonzero(per_joint > SAFETY_REFLEX_EPS_NM).tolist()
        warnings.append(
            "the server's safety layer altered the applied torque on %.1f%% of "
            "states (joint(s) %s, up to %.2f Nm); tau_external there is not the "
            "operator alone (plan 2.7)"
            % (100 * rep["frac_states_with_safety_reflex"], joints, per_joint.max())
        )

    # ---- data sanity ---------------------------------------------------
    nonfinite = {
        name: int(np.sum(~np.isfinite(getattr(buf, name))))
        for name in ("q", "dq", "tau_measured", "tau_external")
    }
    rep["nonfinite"] = nonfinite
    if any(nonfinite.values()):
        errors.append("non-finite values present: %s" % nonfinite)

    rep["q_min"] = buf.q.min(axis=0).tolist()
    rep["q_max"] = buf.q.max(axis=0).tolist()
    rep["dq_absmax"] = np.abs(buf.dq).max(axis=0).tolist()
    rep["tau_external_absmax"] = np.abs(buf.tau_external).max(axis=0).tolist()
    rep["tau_measured_absmax"] = np.abs(buf.tau_measured).max(axis=0).tolist()

    # ---- joint limit proximity -----------------------------------------
    # The safety controller is active on the real robot and starts pushing at
    # a 0.2 rad margin. Its torque shows up in tau_external, so an episode that
    # grazed a limit needs that noted, not silently averaged in (plan 2.7).
    if spec is not None and spec.joint_pos_min is not None and spec.joint_pos_max is not None:
        d = min(buf.num_dofs, spec.joint_pos_min.shape[0])
        margin = np.minimum(
            buf.q[:, :d] - spec.joint_pos_min[:d], spec.joint_pos_max[:d] - buf.q[:, :d]
        )
        min_margin = margin.min(axis=0)
        rep["joint_limit_margin_min_rad"] = min_margin.tolist()
        rep["frac_within_safety_margin"] = float(np.mean(margin.min(axis=1) < 0.2))
        if np.any(min_margin < 0.0):
            errors.append(
                "joint(s) %s went past their configured position limits"
                % np.flatnonzero(min_margin < 0.0).tolist()
            )
        elif np.any(min_margin < 0.2):
            warnings.append(
                "joint(s) %s came within the 0.2 rad safety-controller margin; "
                "its torque is mixed into tau_external there (plan 2.7)"
                % np.flatnonzero(min_margin < 0.2).tolist()
            )

    rep["ok"] = not errors
    rep["errors"] = errors
    rep["warnings"] = warnings
    return rep


# ---------------------------------------------------------------------------
# Acceptance tests (baseline 25)
# ---------------------------------------------------------------------------


def check_a_static_bias(
    buf: EpisodeBuffer, speed_thresh: float = 1e-3, min_samples: int = 100
) -> Dict[str, Any]:
    """A: with the robot at rest and untouched, what does tau_external read?

    Any non-zero mean here is a bias that every later torque number inherits.
    """
    still = np.abs(buf.dq).max(axis=1) < speed_thresh
    n = int(np.sum(still))
    out: Dict[str, Any] = {"test": "A_static_bias", "n_static_samples": n}
    if n < min_samples:
        out["ok"] = False
        out["note"] = (
            "only %d sample(s) below %.1e rad/s; hold the arm still and rerun"
            % (n, speed_thresh)
        )
        return out
    tau = buf.tau_external[still]
    out.update(
        mean=tau.mean(axis=0).tolist(),
        std=tau.std(axis=0).tolist(),
        absmax=np.abs(tau).max(axis=0).tolist(),
        ok=True,
    )
    return out


def check_b_handguiding(
    buf: EpisodeBuffer, spec: RobotSpec, hz_tol: float = 0.02
) -> Dict[str, Any]:
    """B: a free-space hand-guiding pass logged at the full control rate."""
    rep = validate_buffer(buf, spec)
    ratio = rep.get("hz_ratio", 0.0)
    ok = (
        rep["ok"]
        and abs(1.0 - ratio) <= hz_tol
        and rep["estimated_dropped_samples"] == 0
    )
    return {
        "test": "B_handguiding",
        "ok": bool(ok),
        "effective_hz": rep["effective_hz"],
        "nominal_hz": rep.get("nominal_hz"),
        "hz_ratio": ratio,
        "estimated_dropped_samples": rep["estimated_dropped_samples"],
        "max_gap_ms": rep["max_gap_ms"],
        "errors": rep["errors"],
        "warnings": rep["warnings"],
    }


def _normalised(buf: EpisodeBuffer, n: int = 500) -> np.ndarray:
    """Resample q onto n points of normalised time in [0, 1]."""
    t = buf.t
    if t[-1] <= 0:
        raise ValueError("buffer has zero duration")
    u = t / t[-1]
    grid = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(grid, u, buf.q[:, j]) for j in range(buf.num_dofs)], axis=1)


def check_c_repeatability(
    buffers: Sequence[EpisodeBuffer], n_points: int = 500
) -> Dict[str, Any]:
    """C: repeat the same demonstration; how close are the trajectories?

    Compared on normalised time, so this measures spatial repeatability and
    deliberately ignores that the operator moved at a different pace each time.
    """
    if len(buffers) < 2:
        return {"test": "C_repeatability", "ok": False, "note": "need >= 2 demonstrations"}
    stack = np.stack([_normalised(b, n_points) for b in buffers])  # (K, n, D)
    std = stack.std(axis=0)  # (n, D)
    return {
        "test": "C_repeatability",
        "ok": True,
        "n_demos": len(buffers),
        "per_joint_mean_std_rad": std.mean(axis=0).tolist(),
        "per_joint_max_std_rad": std.max(axis=0).tolist(),
        "overall_rms_std_rad": float(np.sqrt(np.mean(std ** 2))),
        "durations_s": [b.duration for b in buffers],
    }


def check_d_replay_comparison(
    teach: EpisodeBuffer, replay: EpisodeBuffer, n_points: int = 500
) -> Dict[str, Any]:
    """D: compare the taught pass against an unattended replay of it.

    During teaching, tau_external contains the operator's hand; during replay it
    does not. A large difference in free space is therefore expected and is not
    an environment force (baseline 12/13). What is informative is where the two
    differ during the *contact* phase.
    """
    grid = np.linspace(0.0, 1.0, n_points)

    def interp(buf, arr):
        u = buf.t / buf.t[-1]
        return np.stack([np.interp(grid, u, arr[:, j]) for j in range(arr.shape[1])], axis=1)

    if teach.duration <= 0 or replay.duration <= 0:
        return {"test": "D_replay_comparison", "ok": False, "note": "zero-duration buffer"}

    dof = min(teach.num_dofs, replay.num_dofs)
    q_t, q_r = interp(teach, teach.q[:, :dof]), interp(replay, replay.q[:, :dof])
    e_t, e_r = interp(teach, teach.tau_external[:, :dof]), interp(replay, replay.tau_external[:, :dof])
    dq = q_r - q_t
    return {
        "test": "D_replay_comparison",
        "ok": True,
        "teach_duration_s": teach.duration,
        "replay_duration_s": replay.duration,
        "tracking_rms_rad": float(np.sqrt(np.mean(dq ** 2))),
        "tracking_max_rad": float(np.abs(dq).max()),
        "per_joint_tracking_rms_rad": np.sqrt(np.mean(dq ** 2, axis=0)).tolist(),
        "teach_tau_external_rms": np.sqrt(np.mean(e_t ** 2, axis=0)).tolist(),
        "replay_tau_external_rms": np.sqrt(np.mean(e_r ** 2, axis=0)).tolist(),
        "note": (
            "tau_external during teaching includes the operator's hand; a large "
            "free-space difference is expected (baseline 12)."
        ),
    }
