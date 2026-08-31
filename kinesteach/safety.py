"""Making sure a policy always stops.

INVARIANT 5 (plan 6, 3.3). The RT loop lives in the polymetis server process,
not ours. If our process dies, the server keeps running whatever policy it was
given. During teaching that leaves a limp arm, which is survivable. During
replay it means a `JointTrajectoryExecutor` runs the trajectory to the end with
nobody watching -- so every code path that starts a policy goes through here.

WHAT A SOFTWARE STOP ACTUALLY DOES. `terminate_policy()` does not leave the arm
limp. The polymetis server falls back to the `DefaultController` from the robot
metadata, which is a PD loop around whatever pose the arm was in at the moment
of termination, at the full `default_Kq` [40 30 50 25 35 25 10]
(polymetis_server.cpp `ControlUpdate`, torchcontrol default_controller.py). So a
software stop means "freeze here, stiffly":

  - during replay that is what you want -- the motion stops where it is;
  - during teaching it is a step change. The arm was compliant and is suddenly
    holding a setpoint. If you are pushing on it, it now pushes back.

It stops commanded motion. It does not remove power, engage the brakes, or
release anything the arm is already pressing on. It is not a substitute for the
physical E-stop, and nothing in this project may present it as one.

TWO LEVELS OF STOP. Inside a `cooperative()` block the first SIGINT only *asks*
the running loop to stop, so teaching can end its session normally and keep the
log -- losing a four minute demonstration because the natural way to say "I am
done" is Ctrl-C would be its own kind of failure. A second SIGINT, or any signal
outside such a block, terminates the policy immediately.
"""
from __future__ import annotations

import atexit
import logging
import signal
import threading
from contextlib import contextmanager
from typing import Any, Iterator

log = logging.getLogger(__name__)

__all__ = ["policy_guard", "EmergencyTermination"]


@contextmanager
def policy_guard(backend: Any, terminate_on_success: bool = False):
    """Terminate the backend's policy if the block does not do so itself.

    Normal exit leaves an already-stopped policy alone (the caller usually
    wants the log that `terminate_policy()` returns). Any exception, including
    KeyboardInterrupt, stops the robot on the way out.
    """
    try:
        yield backend
    except BaseException:
        _safe_terminate(backend, "exception in policy block")
        raise
    else:
        if terminate_on_success:
            _safe_terminate(backend, "block exit")


def _safe_terminate(backend: Any, why: str) -> None:
    try:
        if backend.is_running_policy():
            log.warning("terminating running policy (%s)", why)
            backend.terminate_policy()
    except Exception:  # never mask the original failure
        log.exception("failed to terminate policy (%s)", why)


class EmergencyTermination:
    """Process-level backstop: stop the robot on exit or on a signal.

    Installed for the lifetime of a backend connection. SIGINT/SIGTERM stop the
    policy first and then run whatever handler was there before, so Ctrl-C
    still behaves like Ctrl-C.

    Inside `cooperative()` the first signal is downgraded to a request: it sets
    `stop_requested` and returns, letting the loop that opened the block shut
    the session down in an orderly way. See the module docstring.
    """

    def __init__(self, backend: Any):
        self.backend = backend
        self._lock = threading.Lock()
        self._fired = False
        self._prev = {}
        self._installed = False
        self._stop = threading.Event()
        self._coop = 0

    def install(self) -> "EmergencyTermination":
        if self._installed:
            return self
        atexit.register(self.fire)
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    self._prev[sig] = signal.signal(sig, self._on_signal)
                except (ValueError, OSError):  # pragma: no cover
                    pass
        self._installed = True
        return self

    def uninstall(self) -> None:
        if not self._installed:
            return
        atexit.unregister(self.fire)
        for sig, prev in self._prev.items():
            try:
                signal.signal(sig, prev)
            except (ValueError, OSError):  # pragma: no cover
                pass
        self._prev.clear()
        self._installed = False

    # ---- cooperative stop ----------------------------------------------

    @property
    def stop_requested(self) -> bool:
        """True once a stop has been asked for. Poll this from a run loop."""
        return self._stop.is_set()

    def request_stop(self) -> None:
        """Ask the running loop to stop. Commands nothing by itself."""
        self._stop.set()

    def wait_stop(self, timeout: float) -> bool:
        """Sleep up to `timeout`, returning early if a stop is requested."""
        return self._stop.wait(timeout)

    @contextmanager
    def cooperative(self) -> "Iterator[EmergencyTermination]":
        """Downgrade the first signal to a stop *request* inside this block.

        The loop inside is expected to poll `stop_requested` and end the policy
        itself, which is the only way it gets to keep the server-side log. If it
        does not -- or if the user presses Ctrl-C a second time -- the next
        signal terminates the policy outright.
        """
        self._coop += 1
        self._stop.clear()
        try:
            yield self
        finally:
            self._coop -= 1
            if self._coop == 0:
                self._stop.clear()

    # ---- hard stop -------------------------------------------------------

    def fire(self) -> None:
        with self._lock:
            if self._fired:
                return
            self._fired = True
        _safe_terminate(self.backend, "emergency termination")

    def _on_signal(self, signum, frame):
        if self._coop > 0 and not self._stop.is_set():
            # First Ctrl-C of a cooperative block: ask, do not command, and do
            # not chain -- chaining to SIG_DFL here would kill the process and
            # take the un-fetched log with it.
            self._stop.set()
            log.warning(
                "stop requested; finishing cleanly. Press Ctrl-C again to "
                "terminate the policy immediately."
            )
            return
        self.fire()
        prev = self._prev.get(signum)
        if callable(prev):
            prev(signum, frame)
        elif prev == signal.SIG_DFL:
            signal.signal(signum, signal.SIG_DFL)
            signal.raise_signal(signum)

    def __enter__(self) -> "EmergencyTermination":
        return self.install()

    def __exit__(self, *exc) -> None:
        self.fire()
        self.uninstall()
