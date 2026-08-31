"""Replaying a demonstration.

The trajectory goes to the server in one piece, as a `JointTrajectoryExecutor`.
Streaming waypoints from a python loop is one RPC per control tick and cannot
hold 1 kHz (baseline 9, plan 2.5).

This is the one part of the system that moves the robot on its own, so the
gates in front of it matter more than the code behind it:

  1. the trajectory is checked against the robot's position and velocity limits
     before anything is sent;
  2. the arm is brought to the trajectory's first waypoint under a slow,
     bounded move, and the replay refuses to start if it is not there;
  3. the policy is started inside `policy_guard` + `EmergencyTermination`, so a
     crash, a Ctrl-C or a dead WebUI backend still stops the arm (invariant 5).

Replay is also the only place a stop button earns its keep. Teaching is already
the most harmless state the arm has -- zero stiffness, a hand on it -- and
stopping there makes it *stiffer*. Here the arm moves under its own power with
nobody holding it, so stopping mid-motion is worth the machinery. What stopping
does is described in safety.py: it freezes the arm where it is, at the default
stiffness. It is not the physical E-stop and never replaces it.
"""
from __future__ import annotations

import logging
import time
from contextlib import ExitStack
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .backend.base import EpisodeBuffer, RobotBackend, RobotSpec
from .config import Config
from .dataset import Episode
from .process import build_replay_trajectory
from .record import save_replay_pass
from .safety import EmergencyTermination, policy_guard

log = logging.getLogger(__name__)

__all__ = [
    "POLICY_WATCHDOG_MARGIN_S",
    "resolve_replay_gains",
    "replay_controller_metadata",
    "load_replay_trajectory",
    "check_trajectory",
    "check_start_pose",
    "start_pose_offsets",
    "start_pose_torque",
    "start_pose_ok",
    "verify_start_pose",
    "replay_episode",
    "TrajectorySafetyError",
    "StartPoseError",
]

#: Slack on top of a motion's expected duration before a watchdog gives up on
#: it. One number, because the CLI replay loop, the WebUI replay loop and the
#: WebUI approach/home watchdogs are all answering the same question -- "the
#: server should have finished by now, has it?" -- and drifting apart would
#: mean the same overrun is tolerated for different lengths of time depending
#: on which front end started it.
POLICY_WATCHDOG_MARGIN_S = 5.0


class TrajectorySafetyError(RuntimeError):
    """The trajectory would violate a robot limit; nothing was sent."""


class StartPoseError(RuntimeError):
    """The arm is not at the trajectory's first waypoint."""


def resolve_replay_gains(
    spec: RobotSpec, cfg: Config
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(Kq, Kqd, Kx, Kxd), scaled from the robot's own defaults.

    Kx/Kxd are zero unless `replay.use_cartesian_stiffness` is set:
    `JointTrajectoryExecutor` takes them too, and the same surprise that makes
    `adaptive=True` unusable for teaching applies here (plan 2.3, 2.5).
    """
    r = cfg.replay
    Kq = np.asarray(spec.default_Kq, dtype=np.float64) * r.Kq_scale
    Kqd = np.asarray(spec.default_Kqd, dtype=np.float64) * r.Kqd_scale
    if r.use_cartesian_stiffness:
        Kx = np.asarray(spec.default_Kx, dtype=np.float64) * r.Kx_scale
        Kxd = np.asarray(spec.default_Kxd, dtype=np.float64) * r.Kxd_scale
    else:
        Kx = np.zeros(6)
        Kxd = np.zeros(6)
    return Kq, Kqd, Kx, Kxd


def replay_controller_metadata(
    cfg: Config,
    Kq: np.ndarray,
    Kqd: np.ndarray,
    Kx: np.ndarray,
    Kxd: np.ndarray,
    q_traj: np.ndarray,
    expected_s: float,
    traj_report: Dict[str, Any],
) -> Dict[str, Any]:
    """What a replay pass records about how it was run.

    Shared with the WebUI worker rather than written out twice: this dict is
    the only description an episode keeps of the gains it was replayed under,
    and a field added on one path but not the other would silently produce two
    kinds of replay pass that cannot be compared.
    """
    return {
        "type": "JointTrajectoryExecutor",
        "Kq": np.asarray(Kq).tolist(),
        "Kqd": np.asarray(Kqd).tolist(),
        "Kx": np.asarray(Kx).tolist(),
        "Kxd": np.asarray(Kxd).tolist(),
        "use_cartesian_stiffness": cfg.replay.use_cartesian_stiffness,
        "time_scale": cfg.replay.time_scale,
        "source": cfg.replay.source,
        "n_waypoints": int(np.asarray(q_traj).shape[0]),
        "expected_duration_s": float(expected_s),
        "ignore_gravity": True,
        "trajectory_check": traj_report,
    }


def load_replay_trajectory(
    ep: Episode, cfg: Config, control_hz: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t, q_traj, dq_traj) at the target control rate.

    Built from the filtered trajectory rather than from the stored `q_replay`,
    so the replay rate and time scale can be chosen at replay time, and so an
    episode stays replayable on a differently-clocked server (plan 2.6).
    """
    meta = ep.read_metadata()
    hz = float(control_hz or meta.get("control_hz") or 1000.0)
    if not ep.has_processed():
        raise FileNotFoundError(
            "%s has no processed.npz; run process_episode() first" % ep.path
        )
    proc = ep.read_processed()
    source = cfg.replay.source
    if source not in proc:
        raise KeyError(
            "replay.source=%r not in processed.npz (have: %s)"
            % (source, sorted(proc))
        )
    return build_replay_trajectory(proc["t"], proc[source], hz, cfg.replay.time_scale)


def check_trajectory(
    q_traj: np.ndarray, dq_traj: np.ndarray, spec: RobotSpec, margin_rad: float = 0.05
) -> Dict[str, Any]:
    """Refuse a trajectory that leaves the robot's limits.

    The server's safety controller would push back on a limit violation, but by
    then the arm is already moving into it; catching it here means nothing is
    sent at all.
    """
    q_traj = np.asarray(q_traj, dtype=np.float64)
    dq_traj = np.asarray(dq_traj, dtype=np.float64)
    if q_traj.ndim != 2 or q_traj.shape != dq_traj.shape:
        raise TrajectorySafetyError(
            "trajectory shapes disagree: q %r, dq %r" % (q_traj.shape, dq_traj.shape)
        )
    if q_traj.shape[1] != spec.num_dofs:
        raise TrajectorySafetyError(
            "trajectory has %d DOF but the robot has %d"
            % (q_traj.shape[1], spec.num_dofs)
        )
    if not np.isfinite(q_traj).all() or not np.isfinite(dq_traj).all():
        raise TrajectorySafetyError("trajectory contains non-finite values")

    report: Dict[str, Any] = {
        "n_waypoints": int(q_traj.shape[0]),
        "q_min": q_traj.min(axis=0).tolist(),
        "q_max": q_traj.max(axis=0).tolist(),
        "dq_absmax": np.abs(dq_traj).max(axis=0).tolist(),
    }
    if spec.joint_pos_min is not None and spec.joint_pos_max is not None:
        lo = spec.joint_pos_min + margin_rad
        hi = spec.joint_pos_max - margin_rad
        bad = np.flatnonzero((q_traj < lo).any(axis=0) | (q_traj > hi).any(axis=0))
        if bad.size:
            raise TrajectorySafetyError(
                "joint(s) %s leave their position limits (with a %.2f rad margin)"
                % (bad.tolist(), margin_rad)
            )
    if spec.joint_vel_max is not None:
        over = np.flatnonzero((np.abs(dq_traj) > spec.joint_vel_max).any(axis=0))
        if over.size:
            peak = np.abs(dq_traj).max(axis=0)
            raise TrajectorySafetyError(
                "joint(s) %s exceed their velocity limits (peak %s vs limit %s); "
                "raise replay.time_scale to slow the replay down"
                % (over.tolist(), peak[over].round(3).tolist(),
                   np.asarray(spec.joint_vel_max)[over].round(3).tolist())
            )
    return report


def start_pose_offsets(backend: RobotBackend, q_start: np.ndarray) -> np.ndarray:
    """Signed per-joint distance from the arm's pose to `q_start`, in rad."""
    q_start = np.asarray(q_start, dtype=np.float64)
    return np.asarray(backend.get_joint_positions(), dtype=np.float64) - q_start


def start_pose_torque(offsets: np.ndarray, Kq: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """The torque the executor would apply at t=0, per joint.

    This is what the gate is actually protecting against, and it is not
    proportional to the angle: the joints do not share a stiffness.
    """
    if Kq is None:
        return None
    return np.abs(np.asarray(Kq, dtype=np.float64) * np.asarray(offsets, dtype=np.float64))


def verify_start_pose(
    err: float,
    cfg: Config,
    after_move: bool,
    offsets: Optional[np.ndarray] = None,
    Kq: Optional[np.ndarray] = None,
) -> None:
    """Raise unless the arm is close enough to start from where it stands.

    Two thresholds, and both have to hold:

    - `start_pose_tol_nm` on `Kq * error`, the transient the executor would
      apply the instant it takes over. This is the real check.
    - `start_pose_tol_rad` as a gross cap, for a joint so far off that the wrong
      episode is the likelier explanation.

    Passing `offsets`/`Kq` enables the torque check; without them only the cap
    applies, which is the behaviour every caller had before it existed.
    """
    where = ("approach move finished" if after_move else "arm is")
    tail = ("; not starting the replay" if after_move else "")

    tau = start_pose_torque(offsets, Kq) if offsets is not None else None
    if tau is not None and cfg.replay.start_pose_tol_nm is not None:
        lim = float(cfg.replay.start_pose_tol_nm)
        if tau.max() > lim:
            j = int(np.argmax(tau))
            raise StartPoseError(
                "%s %.4f rad from the first waypoint, which is %.2f Nm on joint "
                "%d (tolerance %.2f Nm)%s. If the angle is already small, the "
                "approach has hit the friction floor rather than missed: it "
                "settles at friction/Kq_default, and the transient is that "
                "times Kq_scale, so raising Kq_scale alone cannot satisfy a "
                "tight limit."
                % (where, abs(float(offsets[j])), tau[j], j + 1, lim, tail)
            )

    if err > cfg.replay.start_pose_tol_rad:
        raise StartPoseError(
            "%s %.4f rad from the first waypoint (cap %.4f rad)%s"
            % (where, err, cfg.replay.start_pose_tol_rad, tail)
        )


def start_pose_ok(
    offsets: np.ndarray, err: float, cfg: Config, Kq: Optional[np.ndarray]
) -> bool:
    """Whether `verify_start_pose` would accept this, without raising."""
    tau = start_pose_torque(offsets, Kq)
    if tau is not None and cfg.replay.start_pose_tol_nm is not None:
        if tau.max() > float(cfg.replay.start_pose_tol_nm):
            return False
    return err <= cfg.replay.start_pose_tol_rad


def check_start_pose(
    backend: RobotBackend,
    q_start: np.ndarray,
    cfg: Config,
    move: bool = True,
    Kq: Optional[np.ndarray] = None,
) -> float:
    """Bring the arm to the trajectory's first waypoint, or refuse to continue.

    A `JointTrajectoryExecutor` starting from the wrong pose snaps towards the
    first waypoint at whatever the stiffness allows. The approach move is slow
    and explicit precisely so that motion is not a surprise.

    `Kq` is the stiffness the replay itself will use, so the check can be made
    on the transient rather than on an angle that means different things on
    different joints.

    This blocks for the length of the move. That is fine for the CLI, where a
    signal handler is still live, but not for the WebUI worker: see
    `start_pose_offsets` / `verify_start_pose`, which let it run the same two
    steps across its own loop and keep the stop button answering throughout.
    """
    off = start_pose_offsets(backend, q_start)
    err = float(np.abs(off).max())
    if start_pose_ok(off, err, cfg, Kq):
        return err
    if not move:
        verify_start_pose(err, cfg, after_move=False, offsets=off, Kq=Kq)
    log.info(
        "approaching first waypoint: %.4f rad away, moving over %.1f s",
        err, cfg.replay.approach_time_s,
    )
    backend.move_to_joint_positions(q_start, time_to_go=cfg.replay.approach_time_s)
    off = start_pose_offsets(backend, q_start)
    err = float(np.abs(off).max())
    verify_start_pose(err, cfg, after_move=True, offsets=off, Kq=Kq)
    return err


def replay_episode(
    backend: RobotBackend,
    ep: Episode,
    cfg: Config,
    save: bool = True,
    notes: str = "",
    poll_s: float = 0.05,
    timeout_margin_s: float = POLICY_WATCHDOG_MARGIN_S,
    stop: Optional[EmergencyTermination] = None,
) -> Tuple[EpisodeBuffer, Optional[Episode]]:
    """Replay a stored episode. Returns (log, saved replay pass or None).

    Ctrl-C once stops the arm where it is and still saves the pass; twice
    terminates immediately. `stop` reuses a guard the caller already installed.
    """
    spec = backend.spec()
    t_traj, q_traj, dq_traj = load_replay_trajectory(ep, cfg, spec.control_hz)
    traj_report = check_trajectory(q_traj, dq_traj, spec)
    Kq, Kqd, Kx, Kxd = resolve_replay_gains(spec, cfg)
    expected_s = float(q_traj.shape[0]) / spec.control_hz

    controller = replay_controller_metadata(
        cfg, Kq, Kqd, Kx, Kxd, q_traj, expected_s, traj_report)

    log.info(
        "replay: %d waypoints, %.1f s at %.0f Hz (time_scale %.2f), Kq=%s",
        q_traj.shape[0], expected_s, spec.control_hz, cfg.replay.time_scale,
        np.round(Kq, 1),
    )
    check_start_pose(backend, q_traj[0], cfg, Kq=Kq)

    ended_by = "completed"
    with ExitStack() as stack:
        # Install a guard only if the caller has not already got one. A second
        # EmergencyTermination would replace the first one's signal handlers
        # while the first one holds the cooperative block, so its `_coop` is 0
        # and it fires on the *first* Ctrl-C: measured on the mock backend, the
        # policy was terminated before `stop_requested` was even read. That
        # defeats the two-level stop this module documents, and the CLI, which
        # always passes `stop`, was the one path taking it.
        stopper = stop
        if stopper is None:
            stopper = stack.enter_context(EmergencyTermination(backend))
        stack.enter_context(policy_guard(backend))
        # Inside cooperative(), the first Ctrl-C only sets stop_requested. That
        # is what lets the loop below reach terminate_policy() itself and keep
        # the partial log -- a run you had to stop is the one worth reading.
        with stopper.cooperative():
            backend.start_replay(q_traj, dq_traj, Kq, Kqd, Kx, Kxd)
            deadline = time.time() + expected_s + timeout_margin_s
            while backend.is_running_policy():
                if stopper.stop_requested:
                    ended_by = "stop_requested"
                    log.warning(
                        "stop requested; the arm will hold the pose it stops in"
                    )
                    break
                if time.time() > deadline:
                    ended_by = "overrun"
                    log.error("replay overran %.1f s; terminating", expected_s)
                    break
                stopper.wait_stop(poll_s)  # wakes at once on a stop
            buf = backend.terminate_policy()

    controller["ended_by"] = ended_by
    controller["aborted"] = ended_by != "completed"
    log.info("replay %s: %d states, %.2f s", ended_by, buf.n, buf.duration)
    saved = None
    if save:
        saved = save_replay_pass(ep, buf, spec, cfg, controller=controller, notes=notes)
    return buf, saved
