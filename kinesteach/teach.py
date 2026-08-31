"""Kinesthetic teaching sessions.

Near-zero joint stiffness with a small damping term, so the operator moves the
arm by hand while the server logs at its full control rate (baseline 6).

Two things here are load bearing:

* The stiffness gains are resolved from the robot's own metadata, never
  hardcoded: they have to match the robot's DOF (invariant 3).
* The session length is capped below the server's 300 s ring buffer, because
  overrunning it discards the *start* of the demonstration silently (plan 2.4).
"""
from __future__ import annotations

import logging
import threading
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .backend.base import EpisodeBuffer, GripperBuffer, RobotBackend, RobotSpec
from .config import Config
from .safety import policy_guard

log = logging.getLogger(__name__)


__all__ = [
    "resolve_teaching_gains",
    "TeachingSession",
    "run_teaching",
    "TeachingAborted",
]


class TeachingAborted(RuntimeError):
    """A session that failed *after* the policy was already running.

    Carries whatever the server had logged. The demonstration itself is usually
    fine -- the failure is in the shutdown path around it, which is where the
    first hardware run lost 2 s to a `TypeError` inside `stop()`. Losing four
    minutes of someone else's demonstration to a bug on the way out would be
    its own failure, so the data comes out with the exception.
    """

    def __init__(self, cause, buf, gripper, controller):
        super().__init__("teaching session aborted: %r" % (cause,))
        self.cause = cause
        self.buf = buf
        self.gripper = gripper
        self.controller = controller


def resolve_teaching_gains(spec: RobotSpec, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """(Kq, Kqd) for hand guiding.

    Kq defaults to zero: the arm should not pull towards any pose. Kqd defaults
    to the robot's own default damping, which is the sanest starting point we
    have before M5-6 tuning; `teach.Kqd_scale` moves it from there.
    """
    d = spec.num_dofs
    Kq = (
        np.zeros(d)
        if cfg.teach.Kq is None
        else np.asarray(cfg.teach.Kq, dtype=np.float64)
    )
    Kqd = (
        np.asarray(spec.default_Kqd, dtype=np.float64) * cfg.teach.Kqd_scale
        if cfg.teach.Kqd is None
        else np.asarray(cfg.teach.Kqd, dtype=np.float64)
    )
    for name, g in (("Kq", Kq), ("Kqd", Kqd)):
        if g.shape != (d,):
            raise ValueError(
                "teach.%s has %d entries but the robot has %d DOF" % (name, g.size, d)
            )
    if np.any(Kq < 0) or np.any(Kqd < 0):
        raise ValueError("teaching gains must be non-negative")
    return Kq, Kqd


class _GripperPoller(threading.Thread):
    """The Robotiq has no streaming log, so its states have to be polled.

    Kept on its own clock and stored in its own file; alignment with the arm is
    an offline problem (plan 2.8).
    """

    def __init__(self, backend: Any, hz: float):
        super().__init__(daemon=True, name="gripper-poller")
        self.backend = backend
        self.period = 1.0 / max(hz, 1e-6)
        # NOT `_stop`: `threading.Thread` already has a private `_stop()` that
        # `join()` calls once the thread has finished, and shadowing it with an
        # Event makes every join raise `TypeError: 'Event' object is not
        # callable`. That fires on the way *out* of a session, after the arm is
        # already under a zero-stiffness policy.
        self._stopping = threading.Event()
        self._rows: List[Tuple[float, float, bool, bool, int]] = []

    def run(self) -> None:
        while not self._stopping.is_set():
            t0 = time.time()
            try:
                s = self.backend.get_gripper_sample()
                if s is not None:
                    self._rows.append(s)
            except Exception:
                log.exception("gripper poll failed")
            self._stopping.wait(max(0.0, self.period - (time.time() - t0)))

    def request_stop(self) -> None:
        self._stopping.set()

    def buffer(self) -> GripperBuffer:
        """The rows collected so far. Safe to call without joining."""
        if not self._rows:
            return GripperBuffer.empty()
        return GripperBuffer(
            timestamp_ns=np.array([r[0] for r in self._rows], dtype=np.int64),
            width=np.array([r[1] for r in self._rows], dtype=np.float64),
            is_grasped=np.array([r[2] for r in self._rows], dtype=bool),
            is_moving=np.array([r[3] for r in self._rows], dtype=bool),
            error_code=np.array([r[4] for r in self._rows], dtype=np.int32),
        )

    def stop(self) -> GripperBuffer:
        self.request_stop()
        self.join(timeout=2.0)
        return self.buffer()


class TeachingSession:
    """One hand-guiding session, start to stop.

    Usable both from a script (`run_teaching`) and from the WebUI worker, which
    starts and stops it from separate HTTP requests.
    """

    def __init__(self, backend: RobotBackend, cfg: Config):
        self.backend = backend
        self.cfg = cfg
        self.spec = backend.spec()
        self.Kq, self.Kqd = resolve_teaching_gains(self.spec, cfg)
        self._t0: Optional[float] = None
        self._poller: Optional[_GripperPoller] = None
        self._stopped = False

    # ---- description ---------------------------------------------------

    @property
    def controller_metadata(self) -> Dict[str, Any]:
        return {
            "type": "JointImpedanceControl",
            "Kq": self.Kq.tolist(),
            "Kqd": self.Kqd.tolist(),
            # Recorded explicitly: an episode taught with a Cartesian stiffness
            # riding along is a different experiment (plan 2.3, invariant 4).
            "adaptive": False,
            "Kx": None,
            "Kxd": None,
            "ignore_gravity": True,
        }

    @property
    def elapsed(self) -> float:
        return 0.0 if self._t0 is None else time.time() - self._t0

    @property
    def remaining(self) -> float:
        return max(0.0, self.cfg.teach.max_duration_s - self.elapsed)

    def should_stop(self) -> bool:
        return self._t0 is not None and self.elapsed >= self.cfg.teach.max_duration_s

    # ---- lifecycle -----------------------------------------------------

    def start(self) -> None:
        if self._t0 is not None:
            raise RuntimeError("teaching session already started")
        if self.backend.is_running_policy():
            raise RuntimeError(
                "another policy is already running; stop it before teaching"
            )
        log.info(
            "teaching: Kq=%s Kqd=%s (max %.0f s)",
            np.round(self.Kq, 3), np.round(self.Kqd, 3), self.cfg.teach.max_duration_s,
        )
        self.backend.start_teaching(self.Kq, self.Kqd)
        self._t0 = time.time()
        if self.cfg.backend.gripper_enabled:
            self._poller = _GripperPoller(self.backend, self.cfg.backend.gripper_poll_hz)
            self._poller.start()

    def stop(self) -> Tuple[EpisodeBuffer, Optional[GripperBuffer]]:
        if self._t0 is None:
            raise RuntimeError("teaching session was never started")
        self._stopped = True
        gripper = self._poller.stop() if self._poller is not None else None
        self._poller = None
        buf = self.backend.terminate_policy()
        if self.cfg.teach.settle_s > 0 and buf.n:
            buf = buf.trim_seconds(head=self.cfg.teach.settle_s)
        log.info(
            "teaching stopped: %d states, %.2f s, %.1f Hz",
            buf.n, buf.duration, buf.effective_hz,
        )
        return buf, gripper

    def salvage(self) -> Tuple[EpisodeBuffer, Optional[GripperBuffer]]:
        """Best-effort recovery once `stop()` has failed.

        The server holds the log either way: `terminate_policy()` returns it
        directly if the policy is still running, and falls back to
        `get_previous_log()` if the guard has already stopped it. The gripper
        rows are read straight off the poller rather than joining it, since a
        failing join is one of the ways we get here.
        """
        self._stopped = True
        gripper = None
        if self._poller is not None:
            try:
                self._poller.request_stop()
                gripper = self._poller.buffer()
            except Exception:
                log.exception("could not salvage the gripper rows")
            self._poller = None
        return self.backend.terminate_policy(), gripper

    def __enter__(self) -> "TeachingSession":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        if not self._stopped and self._t0 is not None:
            try:
                self.stop()
            except Exception:
                log.exception("failed to stop teaching session cleanly")


def run_teaching(
    backend: RobotBackend,
    cfg: Config,
    duration_s: Optional[float] = None,
    stop: Optional[Any] = None,
) -> Tuple[EpisodeBuffer, Optional[GripperBuffer], Dict[str, Any]]:
    """Blocking teaching session. Returns (robot, gripper, controller).

    `duration_s` is an upper bound, not a fixed length. Pass `stop` (an
    `EmergencyTermination`) to make Ctrl-C end the demonstration and *keep* the
    log: "I am done" is the natural reason to interrupt a teaching session, and
    losing four minutes of demonstration to it would be its own failure.
    """
    limit = cfg.teach.max_duration_s if duration_s is None else min(duration_s, cfg.teach.max_duration_s)
    if duration_s is not None and duration_s > cfg.teach.max_duration_s:
        log.warning(
            "requested %.0f s but the ring buffer only holds %.0f s; capping (plan 2.4)",
            duration_s, cfg.teach.max_duration_s,
        )
    session = TeachingSession(backend, cfg)
    ended_by = "duration_limit"
    with policy_guard(backend):
        with (stop.cooperative() if stop is not None else nullcontext()):
            session.start()
            deadline = time.time() + limit
            while time.time() < deadline:
                if stop is not None and stop.stop_requested:
                    ended_by = "stop_requested"
                    log.info("stop requested; ending the demonstration")
                    break
                remaining = max(0.0, deadline - time.time())
                if stop is not None:
                    stop.wait_stop(min(0.05, remaining))
                else:
                    time.sleep(min(0.05, remaining))
            try:
                buf, gripper = session.stop()
            except Exception as exc:
                # The demonstration is already on the server; only the shutdown
                # path failed. Get the data out with the exception rather than
                # letting a bug on the way out cost someone their recording.
                #
                # Deliberately `Exception`, not `BaseException`: Ctrl-C has its
                # own designed path above (`stop_requested`) that keeps the log,
                # and wrapping KeyboardInterrupt here would break the two-level
                # stop in safety.py.
                log.exception("teaching session failed during shutdown; salvaging")
                try:
                    buf, gripper = session.salvage()
                except Exception:
                    log.exception("salvage failed too; the log is lost")
                    raise exc
                controller = dict(session.controller_metadata)
                controller["ended_by"] = "exception"
                controller["error"] = repr(exc)
                raise TeachingAborted(exc, buf, gripper, controller) from exc
    controller = dict(session.controller_metadata)
    controller["ended_by"] = ended_by
    return buf, gripper, controller
