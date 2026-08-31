"""The rules from implementation_plan.md 6, as executable checks."""
import ast
import pathlib
import signal
from pathlib import Path

import numpy as np
import pytest

from kinesteach.config import Config
from kinesteach.teach import TeachingSession

PKG = Path(__file__).resolve().parents[1] / "kinesteach"
BACKEND = PKG / "backend"


def _imports(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module


def test_invariant_7_protobuf_stays_in_backend():
    """polymetis types must not escape kinesteach/backend/."""
    offenders = {}
    for path in PKG.rglob("*.py"):
        if BACKEND in path.parents or path.parent == BACKEND:
            continue
        bad = [
            m for m in _imports(path)
            if m == "polymetis" or m.startswith("polymetis.") or m.startswith("polymetis_pb2")
        ]
        if bad:
            offenders[str(path.relative_to(PKG))] = bad
    assert not offenders, (
        "these modules import polymetis outside the backend seam: %s" % offenders
    )


def test_invariant_4_teaching_is_never_adaptive():
    with pytest.raises(ValueError, match="adaptive"):
        Config.from_dict({"teach": {"adaptive": True}})


def test_invariant_4_controller_metadata_records_no_cartesian_stiffness(backend, cfg):
    meta = TeachingSession(backend, cfg).controller_metadata
    assert meta["adaptive"] is False
    assert meta["Kx"] is None and meta["Kxd"] is None
    assert meta["type"] == "JointImpedanceControl"


def test_invariant_3_nothing_hardcodes_dof_or_rate(cfg):
    """A robot the code has never seen must flow through unchanged.

    13 DOF at 240 Hz: nothing in the pipeline may assume the 7-at-1000 Hz shape
    of the arm in this lab (plan 2.6).
    """
    import time

    from kinesteach.backend.mock import MockBackend
    from kinesteach.record import save_teaching_episode
    from kinesteach.teach import TeachingSession

    b = MockBackend(cfg.backend, num_dofs=13, control_hz=240.0)
    b.connect()
    s = TeachingSession(b, cfg)
    s.start()
    time.sleep(0.4)
    buf, _ = s.stop()
    assert buf.num_dofs == 13
    ep = save_teaching_episode(cfg.data_root, buf, b.spec(), cfg, controller=s.controller_metadata)
    m = ep.read_metadata()
    assert m["num_dofs"] == 13 and m["control_hz"] == 240.0
    assert 200 < buf.effective_hz < 280


def test_invariant_2_raw_is_write_once(episode):
    from kinesteach.dataset import RawCorruptedError, RawImmutableError

    buf = episode.read_raw()
    with pytest.raises(RawImmutableError):
        episode.write_raw(buf)
    with pytest.raises(RawImmutableError):
        episode.write_arrays("robot_raw.npz", {"q": buf.q})

    episode.verify_raw()
    with episode.file("robot_raw.npz").open("ab") as f:
        f.write(b"tamper")
    with pytest.raises(RawCorruptedError):
        episode.read_raw()


def test_invariant_2_processing_leaves_raw_untouched(episode, cfg):
    from kinesteach.process import process_episode

    before = episode.raw_checksums()
    process_episode(episode, cfg)
    process_episode(episode, cfg)  # rerunning must also be harmless
    assert episode.raw_checksums() == before


def test_invariant_5_policy_guard_terminates_on_exception(backend):
    backend.start_teaching(np.zeros(7), np.ones(7))
    from kinesteach.safety import policy_guard

    with pytest.raises(RuntimeError):
        with policy_guard(backend):
            raise RuntimeError("boom")
    assert not backend.is_running_policy()


def test_invariant_5_emergency_termination_fires_on_exit(backend):
    from kinesteach.safety import EmergencyTermination

    with EmergencyTermination(backend):
        backend.start_teaching(np.zeros(7), np.ones(7))
        assert backend.is_running_policy()
    assert not backend.is_running_policy()


def test_invariant_5_session_with_gripper_stops_cleanly(cfg):
    """A gripper-enabled session must survive its own shutdown.

    The gripper poller is a `threading.Thread`, and every previous test ran with
    `gripper_enabled=False`, so the poller was never constructed. The first
    hardware run found what that hid: naming its flag `_stop` shadowed
    `Thread._stop()`, and `join()` raised `TypeError: 'Event' object is not
    callable` on the way out -- while the arm was under a live policy. The
    policy guard caught it, but the log was lost.
    """
    import time

    from kinesteach.backend.mock import MockBackend

    cfg.backend.gripper_enabled = True
    cfg.backend.gripper_poll_hz = 50.0

    b = MockBackend(cfg.backend)
    b.connect()
    s = TeachingSession(b, cfg)
    s.start()
    assert b.is_running_policy()
    time.sleep(0.3)
    buf, gripper = s.stop()

    assert not b.is_running_policy()  # invariant 5
    assert buf.n > 0
    assert gripper is not None and gripper.n > 0
    b.close()


def test_no_thread_subclass_shadows_a_thread_internal():
    """Guard the whole class of bug, not just the one instance of it.

    Checked with the AST rather than by inspecting a class, because the bug is
    an *instance* attribute assigned in `__init__` -- `vars(cls)` never sees it.
    """
    import threading

    # Single-underscore only: overriding `__init__`/`run` is the point of
    # subclassing, but `_stop`, `_bootstrap`, `_wait_for_tstate_lock` and the
    # rest are machinery `join()` calls on itself.
    reserved = {
        n for n in dir(threading.Thread)
        if n.startswith("_") and not n.startswith("__")
    }

    def _is_thread(base):
        return (isinstance(base, ast.Name) and base.id == "Thread") or (
            isinstance(base, ast.Attribute) and base.attr == "Thread"
        )

    offenders = []
    checked = []
    for path in PKG.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or not any(
                _is_thread(b) for b in node.bases
            ):
                continue
            checked.append(node.name)
            for sub in ast.walk(node):
                if not isinstance(sub, ast.Assign):
                    continue
                for t in sub.targets:
                    if (
                        isinstance(t, ast.Attribute)
                        and isinstance(t.value, ast.Name)
                        and t.value.id == "self"
                        and t.attr in reserved
                    ):
                        offenders.append(
                            "%s.%s (%s)" % (node.name, t.attr, path.name)
                        )

    assert checked, "found no threading.Thread subclasses to check"
    assert not offenders, "shadows Thread internals: %s" % sorted(offenders)


def test_a_crash_after_the_policy_started_still_yields_the_log(cfg, monkeypatch):
    """A bug in the shutdown path must not cost the demonstration.

    This is the first hardware failure turned into a rule: `stop()` raised, the
    policy guard stopped the arm, and 2 s of log survived only because it was
    fetched by hand afterwards.
    """
    import time

    from kinesteach.backend.mock import MockBackend
    from kinesteach.teach import TeachingAborted, _GripperPoller, run_teaching

    cfg.backend.gripper_enabled = True

    def boom(self):
        raise TypeError("'Event' object is not callable")

    monkeypatch.setattr(_GripperPoller, "stop", boom)

    b = MockBackend(cfg.backend)
    b.connect()
    with pytest.raises(TeachingAborted) as caught:
        run_teaching(b, cfg, duration_s=0.3)

    exc = caught.value
    assert isinstance(exc.cause, TypeError)
    assert exc.buf.n > 0                       # the demonstration survived
    assert exc.gripper is not None and exc.gripper.n > 0
    assert exc.controller["ended_by"] == "exception"
    assert "Event" in exc.controller["error"]
    assert not b.is_running_policy()            # invariant 5 still holds
    b.close()


def test_no_command_homes_the_arm_without_checking_the_way(cfg, monkeypatch, stub_connected):
    """`go_home` verifies nothing, so nothing user-facing may call it directly.

    Homing is the one move an operator reaches for precisely when the arm is
    somewhere they would rather not start from -- which is when the path most
    deserves a look. Both `teach --home` and `payload-sweep --home-first` route
    it through `path_is_safe` instead.
    """
    import argparse

    from kinesteach.backend.mock import MockBackend
    from kinesteach.cli.main import cmd_teach

    # No CLI command may reach go_home() on its own. Taken from the package's
    # own __path__ rather than a glob, so moving the modules cannot quietly
    # empty the set being checked -- which is exactly what a hardcoded
    # "cli*.py" did when they became a package.
    import kinesteach.cli

    cli_modules = sorted(
        f for d in kinesteach.cli.__path__ for f in pathlib.Path(d).rglob("*.py")
    )
    assert len(cli_modules) >= 3, (
        "found only %d CLI module(s); the check would cover nothing"
        % len(cli_modules)
    )
    for mod in cli_modules:
        assert "go_home()" not in mod.read_text(), (
            "%s still homes without a path check" % mod.name
        )

    b = MockBackend(cfg.backend)
    b.connect()
    if not b.spec().urdf_text or b.spec().joint_pos_min is None:
        pytest.skip("mock backend has no URDF or limits")

    moved = []
    monkeypatch.setattr(b, "move_to_joint_positions", lambda *a, **k: moved.append(a))
    monkeypatch.setattr(b, "go_home", lambda *a, **k: moved.append(("go_home",)))
    monkeypatch.setattr("kinesteach.envelope.path_is_safe",
                        lambda *a, **k: (False, "would clip the table"))

    monkeypatch.setattr("kinesteach.cli.common._cfg", lambda a: cfg)
    stub_connected(b)

    args = argparse.Namespace(config=None, duration=0.2, notes="", home=True,
                              kqd_scale=None)
    with pytest.raises(SystemExit) as caught:
        cmd_teach(args)
    assert "home" in str(caught.value)
    assert not moved, "the arm moved despite the refusal"
    b.close()


def test_homing_says_so_when_it_did_not_arrive(cfg, monkeypatch, caplog, stub_connected):
    """A move that stalls must not be reported as a move that finished.

    The controller has no integral term, so a joint stops where `Kq * error`
    balances what resists it. On the lab arm j7 stalls ~0.26 rad short of home
    against its gripper cabling; `teach --home` used to fall straight through to
    teaching, and the operator reasonably read that as "homing did nothing".
    """
    import argparse
    import logging

    from kinesteach.backend.mock import MockBackend
    from kinesteach.cli.main import cmd_teach

    b = MockBackend(cfg.backend)
    b.connect()
    spec = b.spec()
    if spec.home_pose is None or not spec.urdf_text:
        pytest.skip("mock backend has no home pose or URDF")

    stalled = np.asarray(spec.home_pose, dtype=float).copy()
    stalled[-1] += 0.26                       # the joint that does not arrive
    monkeypatch.setattr(b, "move_to_joint_positions", lambda *a, **k: None)
    monkeypatch.setattr(b, "get_joint_positions", lambda: stalled)

    from kinesteach.safety import EmergencyTermination

    monkeypatch.setattr("kinesteach.cli.common._cfg", lambda a: cfg)
    stub_connected(b, EmergencyTermination(b))

    args = argparse.Namespace(config=None, duration=0.2, notes="", home=True,
                              kqd_scale=None)
    with caplog.at_level(logging.WARNING):
        cmd_teach(args)
    text = caplog.text
    assert "homing stopped" in text and "NOT at its home pose" in text
    assert "j7" in text
    b.close()


def test_a_command_that_crashes_still_puts_the_robot_down(cfg, monkeypatch):
    """`_connected` owns the teardown, so no command can forget part of it.

    Stopping the policy, restoring the signal handlers and closing the
    connection used to be copied into the `finally` of every command that moves
    the arm. Four copies is three chances to fix the wrong one, and the copy
    that gets missed is the one that leaves a policy running in the server after
    our process has gone (invariant 5).
    """
    from kinesteach.backend.mock import MockBackend
    from kinesteach.cli.common import _connected

    b = MockBackend(cfg.backend)
    seen = {"closed": 0}
    real_close = b.close

    def counted():
        seen["closed"] += 1
        real_close()

    monkeypatch.setattr(b, "close", counted)
    monkeypatch.setattr("kinesteach.backend.make_backend", lambda c: b)

    with pytest.raises(RuntimeError, match="the command failed"):
        with _connected(cfg) as (backend, guard):
            assert backend is b
            assert guard.stop_requested is False
            raise RuntimeError("the command failed")

    assert seen["closed"] == 1, "the backend was not closed on the way out"
    assert not b.is_running_policy()
    # The handlers must go back, or the next command inherits a guard pointing
    # at a closed backend.
    assert signal.getsignal(signal.SIGINT) is not None
