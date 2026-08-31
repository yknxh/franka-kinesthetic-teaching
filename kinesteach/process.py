"""Offline processing: resample, filter, differentiate, FK.

Everything here is derived data and is written to `processed.npz`. The raw
capture is opened read-only and never rewritten (invariant 2, baseline 16/18) --
which is what makes it safe to rerun this with a different cutoff as often as
you like.

Order matters. Server timestamps are near-uniform but not exactly uniform, and
`filtfilt` assumes a fixed sample period, so the log is interpolated onto a
uniform grid at the nominal control rate *before* any filtering.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter

from .backend.base import EpisodeBuffer
from .config import Config, ProcessConfig
from .dataset import CUTOFF_SWEEP, Episode
from .kinematics import EE_FRAME, ForwardKinematics

log = logging.getLogger(__name__)

__all__ = [
    "resample_uniform",
    "butterworth",
    "savgol",
    "filter_signal",
    "derivative",
    "build_replay_trajectory",
    "cutoff_sweep",
    "process_episode",
]


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------


def resample_uniform(
    t: np.ndarray, X: np.ndarray, hz: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate `X(t)` onto a uniform grid at `hz`.

    `t` is relative seconds. Returns (t_uniform, X_uniform).
    """
    t = np.asarray(t, dtype=np.float64)
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    if t.size < 2:
        raise ValueError("need at least 2 samples to resample")
    if X.shape[0] != t.size:
        raise ValueError("X has %d rows but t has %d" % (X.shape[0], t.size))
    # floor, not round: the grid must stay inside [t[0], t[-1]] so nothing is
    # extrapolated, and it must stay exactly uniform, because
    # JointTrajectoryExecutor consumes one waypoint per control tick and a
    # short final interval would be executed as if it were a full one.
    n = max(int(np.floor((t[-1] - t[0]) * hz)) + 1, 2)
    grid = t[0] + np.arange(n) / hz
    out = np.stack([np.interp(grid, t, X[:, j]) for j in range(X.shape[1])], axis=1)
    return grid, out


def _check_cutoff(cutoff_hz: float, fs: float) -> float:
    nyq = fs / 2.0
    if not (0 < cutoff_hz < nyq):
        raise ValueError(
            "cutoff %.3f Hz must be in (0, %.3f) for a %.1f Hz sample rate"
            % (cutoff_hz, nyq, fs)
        )
    return cutoff_hz / nyq


def butterworth(X: np.ndarray, fs: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """Zero-phase low-pass. `filtfilt` so the trajectory is not time-shifted."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    b, a = butter(order, _check_cutoff(cutoff_hz, fs), btype="low")
    padlen = 3 * max(len(a), len(b))
    if X.shape[0] <= padlen:
        raise ValueError(
            "signal has %d samples; filtfilt at order %d needs more than %d"
            % (X.shape[0], order, padlen)
        )
    return filtfilt(b, a, X, axis=0)


def savgol(X: np.ndarray, fs: float, window_s: float = 0.05, polyorder: int = 3) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    w = int(round(window_s * fs))
    w = max(w + (w + 1) % 2, polyorder + 2 + (polyorder % 2))  # odd, > polyorder
    if w > X.shape[0]:
        raise ValueError(
            "savgol window of %d samples exceeds the %d-sample signal" % (w, X.shape[0])
        )
    return savgol_filter(X, w, polyorder, axis=0)


def filter_signal(
    X: np.ndarray, fs: float, pcfg: ProcessConfig, cutoff_hz: Optional[float] = None
) -> np.ndarray:
    """Apply the configured filter."""
    if pcfg.filter == "butterworth":
        return butterworth(X, fs, pcfg.cutoff_hz if cutoff_hz is None else cutoff_hz, pcfg.order)
    return savgol(X, fs, pcfg.savgol_window_s, pcfg.savgol_polyorder)


def derivative(X: np.ndarray, dt: float) -> np.ndarray:
    """d/dt by central differences (one-sided at the ends).

    Taken from the *filtered* signal. Differentiating raw 1 kHz joint angles
    would amplify exactly the noise the filter was there to remove.
    """
    return np.gradient(np.asarray(X, dtype=np.float64), dt, axis=0, edge_order=2)


def build_replay_trajectory(
    t: np.ndarray, q: np.ndarray, control_hz: float, time_scale: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample a trajectory onto the server's control period.

    `JointTrajectoryExecutor` consumes one waypoint per control tick, so the
    trajectory has to be at the server's own rate, whatever that is -- read it
    from the robot, do not assume 1000 Hz (plan 2.5, 2.6).

    `time_scale` stretches the timeline: 2.0 replays at half speed, which is
    the slowdown baseline 20 asks for. Velocities are recomputed on the
    stretched timeline, not merely rescaled, so they stay consistent with the
    positions the executor is asked to track.
    """
    if time_scale <= 0:
        raise ValueError("time_scale must be positive, got %r" % (time_scale,))
    t = np.asarray(t, dtype=np.float64)
    t = t - t[0]
    t_scaled = t * time_scale
    t_out, q_out = resample_uniform(t_scaled, q, control_hz)
    dq_out = derivative(q_out, 1.0 / control_hz)
    return t_out, q_out, dq_out


def cutoff_sweep(
    X: np.ndarray, fs: float, cutoffs: Sequence[float], order: int = 4
) -> Dict[str, np.ndarray]:
    """One filtered copy per candidate cutoff, for the comparison in baseline 17."""
    out: Dict[str, np.ndarray] = {}
    for c in cutoffs:
        try:
            out["cutoff_%g" % c] = butterworth(X, fs, float(c), order)
        except ValueError as e:
            log.warning("skipping cutoff %g Hz: %s", c, e)
    return out


def sweep_summary(raw: np.ndarray, sweep: Dict[str, np.ndarray], dt: float) -> Dict[str, Any]:
    """How much each cutoff smooths, and how far it moves the trajectory.

    The trade-off to look at: `rms_deviation` is how much of the demonstration
    the filter threw away, `accel_rms` is how much high-frequency content is
    left for the replay controller to chase.
    """
    out: Dict[str, Any] = {}
    for name, q in sweep.items():
        d = q - raw
        acc = derivative(derivative(q, dt), dt)
        out[name] = {
            "rms_deviation_rad": float(np.sqrt(np.mean(d ** 2))),
            "max_deviation_rad": float(np.abs(d).max()),
            "accel_rms_rad_s2": float(np.sqrt(np.mean(acc ** 2))),
        }
    return out


# ---------------------------------------------------------------------------
# episode-level
# ---------------------------------------------------------------------------


def process_episode(
    ep: Episode,
    cfg: Config,
    write: bool = True,
    do_fk: bool = True,
    do_sweep: bool = True,
) -> Dict[str, np.ndarray]:
    """Read an episode's raw log and produce its derived arrays."""
    meta = ep.read_metadata()
    buf: EpisodeBuffer = ep.read_raw()  # verifies the checksum on the way in
    if buf.n < 3:
        raise ValueError("episode %s has only %d states" % (ep.path, buf.n))

    fs = float(meta.get("control_hz") or buf.effective_hz)
    pcfg = cfg.process
    dt = 1.0 / fs

    t_rel = buf.t
    t_u, q_u = resample_uniform(t_rel, buf.q, fs)
    _, dq_u = resample_uniform(t_rel, buf.dq, fs)
    _, tau_ext_u = resample_uniform(t_rel, buf.tau_external, fs)
    _, tau_meas_u = resample_uniform(t_rel, buf.tau_measured, fs)

    q_f = filter_signal(q_u, fs, pcfg)
    out: Dict[str, np.ndarray] = {
        "t": t_u,
        "q_uniform": q_u,
        "q_filtered": q_f,
        "dq_from_q_filtered": derivative(q_f, dt),
        "ddq_from_q_filtered": derivative(derivative(q_f, dt), dt),
        "dq_filtered": filter_signal(dq_u, fs, pcfg),
        "tau_external_filtered": filter_signal(tau_ext_u, fs, pcfg, pcfg.tau_cutoff_hz),
        "tau_measured_filtered": filter_signal(tau_meas_u, fs, pcfg, pcfg.tau_cutoff_hz),
    }

    # The trajectory the replay would actually execute, at the configured
    # slowdown. replay.py can rebuild this with a different time_scale.
    t_r, q_r, dq_r = build_replay_trajectory(t_u, q_f, fs, cfg.replay.time_scale)
    out.update(t_replay=t_r, q_replay=q_r, dq_replay=dq_r)

    processing: Dict[str, Any] = {
        "filter": pcfg.filter,
        "cutoff_hz": pcfg.cutoff_hz,
        "order": pcfg.order,
        "tau_cutoff_hz": pcfg.tau_cutoff_hz,
        "resampled_hz": fs,
        "n_uniform": int(t_u.size),
        "replay_time_scale": cfg.replay.time_scale,
        "ee_frame": None,
    }

    if do_fk:
        fk = ForwardKinematics.from_episode(ep, arm_dofs=meta.get("num_dofs"))
        if fk is None:
            log.warning("no URDF stored in %s; skipping FK", ep.path)
        else:
            pos, quat = fk.fk(q_f)
            out["ee_pos_flange"] = pos
            out["ee_quat_flange"] = quat  # xyzw
            processing.update(
                ee_frame=EE_FRAME,
                ee_link_name=fk.ee_link_name,
                fk_source="q_filtered",
            )

    if write:
        ep.write_processed(out)
        if do_sweep:
            sweep = cutoff_sweep(q_u, fs, pcfg.sweep_cutoffs, pcfg.order)
            ep.write_arrays(CUTOFF_SWEEP, dict(sweep, t=t_u))
            processing["cutoff_sweep"] = sweep_summary(q_u, sweep, dt)
        ep.update_metadata(processing=processing)
    return out
