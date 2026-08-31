"""The worker state machine, driven the way the browser drives it.

Exercised without HTTP: the risk in the WebUI is the state machine and the
termination guarantee, not FastAPI's routing.
"""
import json
import time

import pytest

from kinesteach.webui.worker import State, Worker


def wait_for(worker, *states, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = worker.snapshot()
        if s["state"] in states:
            return s
        if s["state"] == State.ERROR and State.ERROR not in states:
            pytest.fail("worker errored: %s" % s["error"])
        time.sleep(0.02)
    pytest.fail("timed out waiting for %s; still %s" % (states, worker.snapshot()["state"]))


@pytest.fixture
def worker(cfg):
    w = Worker(cfg, telemetry_hz=50.0)
    w.start()
    w.submit("connect")
    wait_for(w, State.IDLE)
    yield w
    w.shutdown()
    w.join(timeout=5.0)


def test_connect_reports_the_robot(worker):
    s = worker.snapshot()
    assert s["connected"] and s["robot"]["num_dofs"] == 7
    assert s["robot"]["control_hz"] == 1000.0


def test_telemetry_streams_without_a_policy(worker):
    first = worker.snapshot()["seq"]
    time.sleep(0.3)
    s = worker.snapshot()
    assert s["seq"] > first
    assert len(s["telemetry"]["q"]) == 7
    assert worker.history(), "history feeds the rolling plot"


def test_full_cycle_teach_save_process_replay(worker, cfg):
    from kinesteach.dataset import list_episodes

    # Homing no longer blocks the worker, so it is a state to pass through
    # rather than a call that has already finished when submit() returns.
    worker.submit("home")
    wait_for(worker, State.HOMING)
    wait_for(worker, State.IDLE)

    worker.submit("start_teaching")
    wait_for(worker, State.TEACHING)
    time.sleep(0.5)
    assert worker.snapshot()["session"]["elapsed_s"] > 0.3

    worker.submit("stop_teaching", save=True, notes="webui test")
    wait_for(worker, State.REPLAY_READY)

    eps = list_episodes(cfg.data_root)
    assert len(eps) == 1
    ep = eps[0]
    assert ep.read_metadata()["notes"] == "webui test"
    assert worker.snapshot()["selected_episode"] == ep.name

    worker.submit("process", episode=ep.name, cutoff_hz=15.0)
    deadline = time.time() + 10
    while not ep.has_processed() and time.time() < deadline:
        time.sleep(0.05)
    assert ep.read_metadata()["processing"]["cutoff_hz"] == 15.0

    worker.submit("replay", episode=ep.name, time_scale=2.0)
    wait_for(worker, State.REPLAYING)
    wait_for(worker, State.REPLAY_READY, timeout=30.0)
    assert len(ep.replay_passes()) == 1
    assert ep.replay_passes()[0].read_metadata()["controller"]["time_scale"] == 2.0


def test_discarding_a_session_saves_nothing(worker, cfg):
    from kinesteach.dataset import list_episodes

    worker.submit("start_teaching")
    wait_for(worker, State.TEACHING)
    time.sleep(0.2)
    worker.submit("stop_teaching", save=False)
    wait_for(worker, State.IDLE)
    assert list_episodes(cfg.data_root) == []


def test_stop_stops_a_teaching_session(worker):
    worker.submit("start_teaching")
    wait_for(worker, State.TEACHING)
    worker.submit("stop")
    wait_for(worker, State.IDLE)
    assert not worker.backend.is_running_policy()


def test_stop_freezes_a_replay_and_keeps_the_partial_log(worker, cfg, episode):
    from kinesteach.process import process_episode

    process_episode(episode, cfg)
    worker.submit("replay", episode=episode.name, time_scale=4.0)
    wait_for(worker, State.REPLAYING)
    worker.submit("stop")
    wait_for(worker, State.IDLE)
    assert not worker.backend.is_running_policy()

    # A replay you had to stop is the one worth reading afterwards, so the
    # partial log is written out and marked as not having run to the end.
    passes = sorted((episode.path / "replay").glob("pass_*"))
    assert passes, "the stopped replay should still have been saved"
    meta = json.loads((passes[-1] / "metadata.json").read_text())
    assert meta["controller"]["ended_by"] == "stop_requested"
    assert meta["controller"]["aborted"] is True


def test_a_rejected_command_does_not_disturb_the_session(worker):
    """A double-clicked button must not take the stop button away."""
    worker.submit("start_teaching")
    wait_for(worker, State.TEACHING)
    worker.submit("start_teaching")  # invalid: already teaching
    time.sleep(0.3)
    s = worker.snapshot()
    assert s["state"] == State.TEACHING, "the running session must survive"
    assert "cannot do that" in s["error"]
    assert worker.backend.is_running_policy()

    worker.submit("stop_teaching", save=False)
    wait_for(worker, State.IDLE)


def test_replaying_an_unknown_episode_is_rejected_not_fatal(worker):
    worker.submit("replay", episode="episode_9999")
    time.sleep(0.3)
    s = worker.snapshot()
    assert s["state"] == State.IDLE and "episode_9999" in s["error"]


def test_worker_shutdown_leaves_no_policy_running(cfg):
    w = Worker(cfg, telemetry_hz=50.0)
    w.start()
    w.submit("connect")
    wait_for(w, State.IDLE)
    w.submit("start_teaching")
    wait_for(w, State.TEACHING)
    backend = w.backend
    w.shutdown()
    w.join(timeout=5.0)
    assert not backend.is_running_policy(), "invariant 5: shutdown must stop the robot"


def test_stop_answers_while_the_arm_approaches_the_first_waypoint(worker, cfg, episode):
    """The regression this whole change exists for.

    The approach move is autonomous motion lasting `approach_time_s` (4 s by
    default). It used to be a blocking call inside the command handler, so for
    its whole duration the worker -- the only thread allowed to touch the robot
    -- was parked inside gRPC and nobody read the stop request. The stop button
    was dead for exactly as long as the arm was moving on its own.
    """
    from kinesteach.process import process_episode

    process_episode(episode, cfg)
    worker.submit("home")  # park the arm away from the trajectory's start
    wait_for(worker, State.HOMING)
    wait_for(worker, State.IDLE)

    worker.submit("replay", episode=episode.name, time_scale=2.0)
    wait_for(worker, State.APPROACHING)

    t0 = time.time()
    worker.submit("stop")
    wait_for(worker, State.IDLE, timeout=5.0)
    latency = time.time() - t0

    assert not worker.backend.is_running_policy()
    assert latency < 0.5, (
        "stop took %.2f s; it must not wait out the %.1f s approach move"
        % (latency, cfg.replay.approach_time_s)
    )
    # Nothing was replayed, so there is nothing to write.
    assert not episode.replay_passes()


def test_a_stopped_approach_does_not_go_on_to_replay(worker, cfg, episode):
    from kinesteach.process import process_episode

    process_episode(episode, cfg)
    worker.submit("home")
    wait_for(worker, State.HOMING)
    wait_for(worker, State.IDLE)

    worker.submit("replay", episode=episode.name, time_scale=2.0)
    wait_for(worker, State.APPROACHING)
    worker.submit("stop")
    wait_for(worker, State.IDLE, timeout=5.0)

    time.sleep(1.0)  # well past when the approach would have finished
    assert worker.snapshot()["state"] == State.IDLE
    assert not worker.backend.is_running_policy()
    assert not episode.replay_passes()


def test_a_move_that_never_finishes_is_stopped_by_the_watchdog(worker):
    """Non-blocking moves gave up polymetis' own error reporting.

    A blocking move raises when it goes wrong. A non-blocking one just never
    reports finishing, and the worker would sit in HOMING forever refusing every
    command. The stop button would still answer -- it is out of band -- but
    nothing else would.
    """
    worker.MOVE_MARGIN_S = 0.3
    worker.backend.is_running_policy = lambda: True  # the move never reports done

    worker.submit("home")
    wait_for(worker, State.HOMING)
    s = wait_for(worker, State.ERROR, timeout=5.0)
    assert "did not finish" in s["error"]
    assert "holding its pose" in s["error"]


def test_the_approach_carries_a_watchdog_deadline(worker, cfg, episode):
    from kinesteach.process import process_episode

    process_episode(episode, cfg)
    worker.submit("home")
    wait_for(worker, State.HOMING)
    wait_for(worker, State.IDLE)

    worker.submit("replay", episode=episode.name, time_scale=2.0)
    wait_for(worker, State.APPROACHING)
    ctx = worker._pending_replay
    assert ctx is not None
    assert ctx["limit_s"] == cfg.replay.approach_time_s
    assert ctx["deadline"] > time.time()
    worker.submit("stop")
    wait_for(worker, State.IDLE, timeout=5.0)


def test_a_crash_stopping_teaching_still_saves_the_episode(cfg, monkeypatch):
    """The WebUI must salvage what the CLI salvages.

    `stop()` raising is not hypothetical: it happened on the robot, under a live
    policy, with the gripper enabled -- the configuration `configs/real.yaml`
    uses. The CLI kept the demonstration; this path used to lose it, and the
    browser was told only that the command had failed.
    """
    from kinesteach.dataset import list_episodes
    from kinesteach.teach import TeachingSession

    cfg.backend.gripper_enabled = True

    def boom(self):
        raise TypeError("'Event' object is not callable")

    monkeypatch.setattr(TeachingSession, "stop", boom)

    w = Worker(cfg, telemetry_hz=50.0)
    w.start()
    try:
        w.submit("connect")
        wait_for(w, State.IDLE)
        before = len(list_episodes(cfg.data_root))

        w.submit("start_teaching")
        wait_for(w, State.TEACHING)
        time.sleep(0.3)
        w.submit("stop_teaching", save=True, notes="salvaged")

        # The failure is still reported -- salvaging is not swallowing.
        s = wait_for(w, State.ERROR)
        assert "Event" in s["error"]

        eps = list_episodes(cfg.data_root)
        assert len(eps) == before + 1, "the demonstration was lost"
        assert eps[-1].read_raw().n > 0
        assert not w.backend.is_running_policy()      # invariant 5
    finally:
        w.shutdown()
        w.join(timeout=5.0)
