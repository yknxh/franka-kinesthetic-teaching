"""The thread that owns the robot.

polymetis calls block, so none of them may run inside an HTTP handler. The
worker is the single thread allowed to touch the backend; the web layer only
posts commands to it and reads snapshots back.

The loop never blocks for long. Teaching and replay are *states* it advances
one tick at a time, not calls it waits on, which is what lets the stop button
and the telemetry stream keep working while the robot is moving.

    IDLE -> HOMING -> TEACHING -> SAVING -> REPLAY_READY -> REPLAYING
                                     |                          |
                                     +--------- IDLE <----------+

Invariant 1 (plan 6, baseline 9): the 1 kHz record does not travel this path.
Telemetry is a low-rate display poll; the real log is fetched in one batch from
the server buffer when the session ends.
"""
from __future__ import annotations

import collections
import logging
import threading
import time
import traceback
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from ..backend import make_backend
from ..backend.base import EpisodeBuffer, TelemetrySample
from ..config import Config
from ..dataset import Episode, find_episode
from ..process import process_episode
from ..record import save_teaching_episode
from ..replay import (
    POLICY_WATCHDOG_MARGIN_S,
    StartPoseError,
    check_trajectory,
    load_replay_trajectory,
    replay_controller_metadata,
    resolve_replay_gains,
    start_pose_offsets,
    start_pose_ok,
    verify_start_pose,
)
from ..safety import EmergencyTermination
from ..teach import TeachingSession

log = logging.getLogger(__name__)

__all__ = ["Worker", "State", "CommandRejected"]


class CommandRejected(RuntimeError):
    """The command was not valid right now.

    Distinct from a failure: nothing was attempted, so the robot is exactly
    where it was. A double-clicked button must not knock a running teaching
    session into ERROR and take the stop button away with it.
    """


class State:
    IDLE = "IDLE"
    HOMING = "HOMING"
    TEACHING = "TEACHING"
    SAVING = "SAVING"
    REPLAY_READY = "REPLAY_READY"
    APPROACHING = "APPROACHING"
    REPLAYING = "REPLAYING"
    ERROR = "ERROR"
    DISCONNECTED = "DISCONNECTED"

    #: States in which the arm can be moving under its own power. The stop
    #: button must answer in all of them, which is why none of them is entered
    #: by a blocking call.
    MOVING = (HOMING, APPROACHING, REPLAYING)


class Worker(threading.Thread):
    def __init__(self, cfg: Config, telemetry_hz: float = 30.0, history_s: float = 60.0):
        super().__init__(daemon=True, name="kinesteach-worker")
        self.cfg = cfg
        self.period = 1.0 / telemetry_hz
        self.backend = None
        self.spec = None
        self._emergency: Optional[EmergencyTermination] = None

        self._lock = threading.RLock()
        self._state = State.DISCONNECTED
        self._queue: Deque[Dict[str, Any]] = collections.deque()
        self._shutdown = threading.Event()
        self._estop = threading.Event()
        # Set alongside _estop and _shutdown so the loop's inter-tick sleep ends
        # at once instead of running out the full tick period. Without it a stop
        # waits up to one tick (33 ms at 30 Hz) before anyone looks at it.
        self._wake = threading.Event()

        self._telemetry: Optional[TelemetrySample] = None
        self._history: Deque[Dict[str, Any]] = collections.deque(
            maxlen=int(history_s * telemetry_hz)
        )
        self._session: Optional[TeachingSession] = None
        self._replay_ctx: Optional[Dict[str, Any]] = None
        self._move_ctx: Optional[Dict[str, Any]] = None
        self._pending_replay: Optional[Dict[str, Any]] = None
        self._selected: Optional[str] = None
        self._message = ""
        self._error = ""
        self._events: Deque[Dict[str, Any]] = collections.deque(maxlen=200)
        self._seq = 0

    # ---- public API (called from HTTP handlers) -------------------------

    def submit(self, name: str, **args: Any) -> Dict[str, Any]:
        """Queue a command. Returns immediately."""
        # "estop" is kept as an alias so an already-open browser tab from a
        # previous version still stops the robot. It is the wrong name -- this
        # is not an emergency stop -- so the UI says "stop".
        if name in ("stop", "estop"):  # out of band: never waits behind a queue
            self._estop.set()
            self._wake.set()
            self._note("stop requested", level="warning")
            return {"accepted": True, "queued": 0}
        with self._lock:
            self._queue.append({"name": name, "args": args})
            return {"accepted": True, "queued": len(self._queue)}

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            t = self._telemetry
            return {
                "seq": self._seq,
                "state": self._state,
                "backend": self.cfg.backend.kind,
                "connected": self.backend is not None,
                "message": self._message,
                "error": self._error,
                "selected_episode": self._selected,
                "queued": len(self._queue),
                "robot": None if self.spec is None else {
                    "model": self.spec.robot_model,
                    "num_dofs": self.spec.num_dofs,
                    "control_hz": self.spec.control_hz,
                    "ee_link_name": self.spec.ee_link_name,
                },
                "session": self._session_info(),
                "telemetry": None if t is None else t.to_json(),
                "events": list(self._events)[-20:],
            }

    def history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._history)

    def shutdown(self) -> None:
        self._shutdown.set()
        self._wake.set()

    # ---- internals ------------------------------------------------------

    def _session_info(self) -> Optional[Dict[str, Any]]:
        if self._state == State.TEACHING and self._session is not None:
            return {
                "elapsed_s": self._session.elapsed,
                "remaining_s": self._session.remaining,
                "max_duration_s": self.cfg.teach.max_duration_s,
            }
        if self._state == State.REPLAYING and self._replay_ctx is not None:
            c = self._replay_ctx
            elapsed = time.time() - c["t0"]
            return {
                "elapsed_s": elapsed,
                "remaining_s": max(0.0, c["expected_s"] - elapsed),
                "expected_s": c["expected_s"],
                "episode": c["episode"],
            }
        return None

    def _note(self, msg: str, level: str = "info") -> None:
        getattr(log, level, log.info)("%s", msg)
        with self._lock:
            self._message = msg
            self._events.append({"t": time.time(), "level": level, "text": msg})

    def _set_state(self, state: str, msg: str = "") -> None:
        with self._lock:
            self._state = state
        if msg:
            self._note(msg)

    def _fail(self, what: str, exc: BaseException) -> None:
        detail = "%s: %s" % (what, exc)
        log.error("%s\n%s", detail, traceback.format_exc())
        with self._lock:
            self._error = detail
            self._state = State.ERROR if self.backend is not None else State.DISCONNECTED
            self._events.append({"t": time.time(), "level": "error", "text": detail})

    # ---- main loop ------------------------------------------------------

    def run(self) -> None:
        log.info("worker started (backend=%s)", self.cfg.backend.kind)
        while not self._shutdown.is_set():
            t0 = time.time()
            try:
                if self._estop.is_set():
                    self._estop.clear()
                    self._do_stop()
                self._poll_telemetry()
                self._advance()
                self._pump_one_command()
            except Exception as e:  # a worker that dies leaves the robot running
                self._fail("worker loop", e)
            self._wake.wait(max(0.0, self.period - (time.time() - t0)))
            self._wake.clear()
        self._teardown()
        log.info("worker stopped")

    def _teardown(self) -> None:
        try:
            if self.backend is not None:
                self.backend.close()
        except Exception:
            log.exception("backend close failed")
        finally:
            if self._emergency is not None:
                self._emergency.uninstall()

    def _poll_telemetry(self) -> None:
        if self.backend is None:
            return
        try:
            t = self.backend.get_telemetry()
        except Exception as e:
            self._fail("telemetry", e)
            return
        with self._lock:
            self._telemetry = t
            self._seq += 1
            self._history.append(
                {
                    "t": t.timestamp,
                    "q": t.q.tolist(),
                    "tau_external": t.tau_external.tolist(),
                    "state": self._state,
                }
            )

    def _advance(self) -> None:
        """Progress whichever long-running state we are in.

        Every branch here is a poll, never a wait. That is the whole reason the
        stop button works: the worker is the only thread allowed to touch the
        robot, so any blocking call it makes is time during which nobody is
        reading the stop request -- and the calls worth stopping are exactly the
        ones that block longest (a 4 s approach move, a whole replay).
        """
        if self._state == State.TEACHING and self._session is not None:
            if self._session.should_stop():
                self._note(
                    "teaching hit the %.0f s limit; stopping before the server "
                    "ring buffer overwrites the start (plan 2.4)"
                    % self.cfg.teach.max_duration_s,
                    level="warning",
                )
                self._cmd_stop_teaching(save=True, notes="auto-stopped at duration limit")
        elif self._state == State.HOMING:
            self._advance_move()
        elif self._state == State.APPROACHING:
            self._advance_approach()
        elif self._state == State.REPLAYING:
            self._advance_replay()

    #: Slack on top of a move's expected duration before the watchdog fires.
    #: Shared with the CLI replay loop so an overrun means the same thing
    #: whichever front end started the motion.
    MOVE_MARGIN_S = POLICY_WATCHDOG_MARGIN_S

    def _move_overran(self, ctx: Optional[Dict[str, Any]], what: str) -> bool:
        """Watchdog for a non-blocking move.

        A blocking move reports its own failure by raising. A non-blocking one
        simply never reports finishing, so without this the worker would sit in
        HOMING or APPROACHING forever, refusing every other command. The stop
        button still answers -- it is out of band -- but nothing else does.
        """
        if ctx is None or time.time() <= ctx["deadline"]:
            return False
        self._move_ctx = None
        self._pending_replay = None
        try:
            self.backend.terminate_policy()
        except Exception:
            log.exception("failed to terminate an overrunning %s move", what)
        self._fail(
            what,
            RuntimeError(
                "%s move did not finish within %.1f s; the arm has been stopped "
                "and is holding its pose" % (what, ctx["limit_s"])
            ),
        )
        return True

    def _advance_move(self) -> None:
        """A non-blocking go_home has finished, or has overrun."""
        if self._move_overran(self._move_ctx, "home"):
            return
        if self.backend.is_running_policy():
            return
        self.backend.terminate_policy()  # collects the log we do not keep
        self._move_ctx = None
        self._set_state(State.IDLE, "home")

    def _advance_approach(self) -> None:
        """The approach move has finished; re-check the pose, then replay."""
        if self._move_overran(self._pending_replay, "approach"):
            return
        if self.backend.is_running_policy():
            return
        self.backend.terminate_policy()
        ctx, self._pending_replay = self._pending_replay, None
        if ctx is None:
            self._set_state(State.IDLE)
            return
        off = start_pose_offsets(self.backend, ctx["q"][0])
        err = float(np.abs(off).max())
        try:
            verify_start_pose(err, ctx["cfg"], after_move=True,
                              offsets=off, Kq=ctx["K"][0])
        except StartPoseError as e:
            # Deliberately not a retry loop: an approach that did not land is a
            # reason to stop and look, not to try again while the arm moves.
            self._fail("approach", e)
            return
        self._begin_replay(ctx, err)

    def _advance_replay(self) -> None:
        c = self._replay_ctx
        if c is None:
            self._set_state(State.IDLE)
            return
        overrun = time.time() > c["t0"] + c["expected_s"] + self.MOVE_MARGIN_S
        if self.backend.is_running_policy() and not overrun:
            return
        if overrun:
            self._note("replay overran its expected duration; terminating", "warning")
        c["controller"]["ended_by"] = "overrun" if overrun else "completed"
        c["controller"]["aborted"] = overrun
        buf = self.backend.terminate_policy()
        self._replay_ctx = None
        # Save before announcing the state: REPLAY_READY has to mean the pass is
        # on disk, or a client that reacts to it races the write.
        if self._save_replay_pass(c, buf, "overran" if overrun else "finished"):
            self._set_state(State.REPLAY_READY)

    def _pump_one_command(self) -> None:
        with self._lock:
            if not self._queue:
                return
            cmd = self._queue.popleft()
        name, args = cmd["name"], cmd["args"]
        handler = getattr(self, "_cmd_" + name, None)
        if handler is None:
            self._fail("command", ValueError("unknown command %r" % name))
            return
        try:
            with self._lock:
                self._error = ""
            handler(**args)
        except CommandRejected as e:
            self._note("ignored %s: %s" % (name, e), level="warning")
            with self._lock:
                self._error = str(e)
        except Exception as e:
            self._fail("command %s" % name, e)

    # ---- commands -------------------------------------------------------

    def _cmd_connect(self) -> None:
        if self.backend is not None:
            return
        backend = make_backend(self.cfg.backend)
        backend.connect()
        self.backend = backend
        self.spec = backend.spec()
        # Process-level backstop, in addition to the per-call policy_guard.
        self._emergency = EmergencyTermination(backend).install()
        self._set_state(
            State.IDLE,
            "connected to %s backend: %s, %d DOF at %.0f Hz"
            % (self.cfg.backend.kind, self.spec.robot_model, self.spec.num_dofs, self.spec.control_hz),
        )

    def _cmd_disconnect(self) -> None:
        self._teardown()
        self.backend = None
        self.spec = None
        self._emergency = None
        self._set_state(State.DISCONNECTED, "disconnected")

    def _require_idle(self, *ok_states: str) -> None:
        if self.backend is None:
            raise CommandRejected("not connected")
        allowed = ok_states or (State.IDLE, State.REPLAY_READY, State.ERROR)
        if self._state not in allowed:
            raise CommandRejected("cannot do that while %s" % self._state)

    def _cmd_home(self) -> None:
        self._require_idle()
        # blocking=False: homing is autonomous motion, so the loop has to stay
        # free to answer a stop for the whole of it.
        limit_s = self.backend.expected_move_time_s(self.spec.home_pose)
        self.backend.go_home(blocking=False)
        self._move_ctx = self._move_deadline(limit_s)
        self._set_state(
            State.HOMING,
            "homing -- the arm is moving on its own (about %.1f s)" % limit_s,
        )

    def _move_deadline(self, limit_s: float) -> Dict[str, Any]:
        return {
            "t0": time.time(),
            "limit_s": float(limit_s),
            "deadline": time.time() + float(limit_s) + self.MOVE_MARGIN_S,
        }

    def _cmd_start_teaching(self) -> None:
        self._require_idle()
        session = TeachingSession(self.backend, self.cfg)
        session.start()
        self._session = session
        self._set_state(
            State.TEACHING,
            "teaching: Kq=%s Kqd=%s"
            % (np.round(session.Kq, 2).tolist(), np.round(session.Kqd, 2).tolist()),
        )

    def _cmd_stop_teaching(self, save: bool = True, notes: str = "") -> None:
        if self._session is None:
            raise CommandRejected("no teaching session is running")
        session, self._session = self._session, None
        self._set_state(State.SAVING, "stopping teaching")
        # A bug in the shutdown path must not cost the demonstration. The CLI
        # learned this on the robot -- `stop()` raised, the policy guard stopped
        # the arm, and the log survived only because it was fetched by hand
        # afterwards. Salvage the same way here, then re-raise so the failure is
        # still reported rather than swallowed.
        aborted = None
        try:
            buf, gripper = session.stop()
        except Exception as exc:
            aborted = exc
            try:
                buf, gripper = session.salvage()
            except Exception:
                log.exception("could not salvage the demonstration")
                raise exc
        if not save:
            self._set_state(State.IDLE, "discarded %d states" % buf.n)
            return
        if buf.n < 2:
            self._set_state(State.IDLE, "nothing to save (%d states)" % buf.n)
            return
        ep = save_teaching_episode(
            self.cfg.data_root, buf, self.spec, self.cfg,
            controller=session.controller_metadata, gripper=gripper, notes=notes,
        )
        report = ep.read_json("validation.json")
        self._selected = ep.name
        level = "warning" if (report["warnings"] or not report["ok"]) else "info"
        self._note(
            "saved %s: %d states, %.1f s, %.0f Hz%s"
            % (ep.name, buf.n, buf.duration, buf.effective_hz,
               "" if report["ok"] and not report["warnings"]
               else " -- " + "; ".join(report["errors"] + report["warnings"])),
            level=level,
        )
        self._set_state(State.REPLAY_READY)
        if aborted is not None:
            self._note("teaching ended badly but %s was recovered: %r"
                       % (ep.name, aborted), level="warning")
            raise aborted

    def _cmd_process(self, episode: str, cutoff_hz: Optional[float] = None) -> None:
        cfg = self.cfg
        if cutoff_hz is not None:
            cfg = Config.from_dict(dict(cfg.to_dict()))
            cfg.process.cutoff_hz = float(cutoff_hz)
        ep = self._episode(episode)
        process_episode(ep, cfg)
        self._note("processed %s at %.1f Hz cutoff" % (ep.name, cfg.process.cutoff_hz))

    def _cmd_select(self, episode: str) -> None:
        with self._lock:
            self._selected = self._episode(episode).name

    def _cmd_replay(self, episode: str, time_scale: Optional[float] = None) -> None:
        self._require_idle()
        ep = self._episode(episode)
        cfg = self.cfg
        if time_scale is not None:
            cfg = Config.from_dict(cfg.to_dict())
            cfg.replay.time_scale = float(time_scale)
        if not ep.has_processed():
            process_episode(ep, cfg)

        t, q, dq = load_replay_trajectory(ep, cfg, self.spec.control_hz)
        report = check_trajectory(q, dq, self.spec)
        Kq, Kqd, Kx, Kxd = resolve_replay_gains(self.spec, cfg)
        expected_s = q.shape[0] / self.spec.control_hz

        ctx = {
            "cfg": cfg,
            "q": q, "dq": dq, "K": (Kq, Kqd, Kx, Kxd),
            "expected_s": expected_s,
            "episode": ep.name,
            "episode_path": str(ep.path),
            "controller": replay_controller_metadata(
                cfg, Kq, Kqd, Kx, Kxd, q, expected_s, report),
        }

        off = start_pose_offsets(self.backend, q[0])
        err = float(np.abs(off).max())
        if start_pose_ok(off, err, cfg, Kq):
            self._begin_replay(ctx, err)
            return
        # Too far to start. The approach is real autonomous motion, so it gets
        # its own state rather than a blocking call inside this handler.
        self.backend.move_to_joint_positions(
            q[0], time_to_go=cfg.replay.approach_time_s, blocking=False
        )
        ctx.update(self._move_deadline(cfg.replay.approach_time_s))
        self._pending_replay = ctx
        self._set_state(
            State.APPROACHING,
            "approaching the first waypoint: %.3f rad away, moving over %.1f s"
            % (err, cfg.replay.approach_time_s),
        )

    def _begin_replay(self, ctx: Dict[str, Any], start_err: float) -> None:
        cfg, q, dq = ctx["cfg"], ctx["q"], ctx["dq"]
        Kq, Kqd, Kx, Kxd = ctx["K"]
        expected_s = ctx["expected_s"]
        ctx["controller"]["start_pose_error_rad"] = start_err
        self.backend.start_replay(q, dq, Kq, Kqd, Kx, Kxd)
        self._replay_ctx = {
            "t0": time.time(),
            "expected_s": expected_s,
            "episode": ctx["episode"],
            "episode_path": ctx["episode_path"],
            "controller": ctx["controller"],
        }
        ep_name = ctx["episode"]
        self._set_state(
            State.REPLAYING,
            "replaying %s: %d waypoints, %.1f s at %.1fx slowdown"
            % (ep_name, q.shape[0], expected_s, cfg.replay.time_scale),
        )

    def _do_stop(self) -> None:
        """Stop whatever the robot is doing, right now.

        Deliberately not queued: it runs at the top of the loop, and the loop
        never blocks, so it takes effect within one tick even if commands are
        backed up. What it does to the arm is not a power cut -- the server
        falls back to a PD hold at the default stiffness, so the arm freezes
        where it is (see safety.py). The physical E-stop is a separate thing.
        """
        if self.backend is None:
            return
        replay, self._replay_ctx = self._replay_ctx, None
        teaching = self._session is not None
        self._session = None
        self._pending_replay = None
        self._move_ctx = None
        try:
            buf = self.backend.terminate_policy()
        except Exception as e:
            self._fail("stop", e)
            return
        if replay is not None:
            # A replay you had to stop is the one worth reading afterwards.
            replay["controller"]["ended_by"] = "stop_requested"
            replay["controller"]["aborted"] = True
            if self._save_replay_pass(replay, buf, "stopped"):
                self._set_state(State.IDLE)
            return
        self._set_state(State.IDLE)
        if teaching:
            self._note(
                "stopped: the arm is holding its pose. The %d recorded states "
                "were discarded -- use Stop & save to keep a demonstration."
                % buf.n,
                "warning",
            )
        else:
            self._note("stopped: the arm is holding its pose", "warning")

    def _save_replay_pass(
        self, c: Dict[str, Any], buf: EpisodeBuffer, how: str
    ) -> bool:
        """Write the pass. False means it failed and the state is now ERROR."""
        from ..record import save_replay_pass

        try:
            saved = save_replay_pass(
                Episode(c["episode_path"]), buf, self.spec, self.cfg,
                controller=c["controller"],
            )
        except Exception as e:
            self._fail("saving replay pass", e)
            return False
        self._note(
            "replay %s: %d states saved to %s" % (how, buf.n, saved.path.name),
            "info" if how == "finished" else "warning",
        )
        return True

    def _episode(self, name: str) -> Episode:
        try:
            return find_episode(self.cfg.data_root, name)
        except KeyError as e:
            raise CommandRejected(str(e.args[0]))
