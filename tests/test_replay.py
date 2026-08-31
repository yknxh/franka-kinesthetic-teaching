import numpy as np
import pytest

from kinesteach.config import Config
from kinesteach.process import process_episode
from kinesteach.replay import (
    StartPoseError,
    TrajectorySafetyError,
    check_start_pose,
    check_trajectory,
    load_replay_trajectory,
    replay_episode,
    resolve_replay_gains,
)


def test_cartesian_stiffness_is_off_unless_asked_for(backend, cfg):
    Kq, Kqd, Kx, Kxd = resolve_replay_gains(backend.spec(), cfg)
    assert not np.any(Kx) and not np.any(Kxd), "plan 2.3: no Kx unless explicitly enabled"
    np.testing.assert_allclose(Kq, backend.spec().default_Kq * cfg.replay.Kq_scale)

    cfg2 = Config.from_dict({"replay": {"use_cartesian_stiffness": True, "Kx_scale": 0.5}})
    _, _, Kx2, _ = resolve_replay_gains(backend.spec(), cfg2)
    np.testing.assert_allclose(Kx2, backend.spec().default_Kx * 0.5)


def test_trajectory_outside_position_limits_is_refused(backend):
    spec = backend.spec()
    q = np.tile(spec.joint_pos_max + 0.5, (10, 1))
    with pytest.raises(TrajectorySafetyError, match="position limits"):
        check_trajectory(q, np.zeros_like(q), spec)


def test_trajectory_over_the_velocity_limit_is_refused(backend):
    spec = backend.spec()
    q = np.tile(spec.home_pose, (10, 1))
    dq = np.tile(spec.joint_vel_max * 2, (10, 1))
    with pytest.raises(TrajectorySafetyError, match="velocity limits"):
        check_trajectory(q, dq, spec)


def test_trajectory_with_wrong_dof_is_refused(backend):
    q = np.zeros((10, 3))
    with pytest.raises(TrajectorySafetyError, match="DOF"):
        check_trajectory(q, np.zeros_like(q), backend.spec())


def test_start_pose_gate_refuses_when_it_cannot_move(backend, cfg):
    far = backend.get_joint_positions() + 1.0
    with pytest.raises(StartPoseError):
        check_start_pose(backend, far, cfg, move=False)


def test_replay_round_trip_on_mock(backend, cfg, episode):
    process_episode(episode, cfg)
    buf, saved = replay_episode(backend, episode, cfg)
    assert buf.n > 0
    assert saved is not None and saved.read_metadata()["kind"] == "replay"

    ctrl = saved.read_metadata()["controller"]
    assert ctrl["type"] == "JointTrajectoryExecutor"
    assert ctrl["time_scale"] == cfg.replay.time_scale
    assert ctrl["Kx"] == [0.0] * 6

    # a replay pass is a full episode: it validates like any other
    v = saved.read_json("validation.json")
    assert v["n_states"] == buf.n


def test_replay_needs_a_processed_episode(backend, cfg, episode):
    with pytest.raises(FileNotFoundError, match="processed"):
        load_replay_trajectory(episode, cfg)


def test_the_start_gate_measures_torque_not_angle(cfg):
    """The transient is `Kq * error`, and the joints do not share a stiffness.

    On the lab arm joint 7 has a fifth of joint 3's stiffness, so a single angle
    threshold held it to a fifth of the standard -- and it was the joint that
    kept refusing replays while carrying the least resistance of all seven.
    """
    import numpy as np
    import pytest

    from kinesteach.replay import (
        StartPoseError, start_pose_ok, start_pose_torque, verify_start_pose,
    )

    cfg.replay.start_pose_tol_nm = 0.8
    cfg.replay.start_pose_tol_rad = 0.25
    Kq = 0.5 * np.array([40, 30, 50, 25, 35, 25, 10.0])

    # 0.10 rad on joint 7 is 0.50 Nm -- inside the transient the gate allows,
    # and twice what the old 0.05 rad angle threshold permitted there.
    soft = np.zeros(7); soft[6] = 0.10
    assert start_pose_torque(soft, Kq).max() == pytest.approx(0.5)
    assert start_pose_ok(soft, float(np.abs(soft).max()), cfg, Kq)

    # The same angle on joint 3 is 1.25 Nm and must be refused.
    stiff = np.zeros(7); stiff[2] = 0.10
    assert not start_pose_ok(stiff, float(np.abs(stiff).max()), cfg, Kq)
    with pytest.raises(StartPoseError, match="joint 3"):
        verify_start_pose(float(np.abs(stiff).max()), cfg, after_move=True,
                          offsets=stiff, Kq=Kq)

    # The gross cap still catches a pose that is simply wrong.
    gross = np.zeros(7); gross[6] = 0.9          # only 4.5 Nm... but 0.9 rad
    with pytest.raises(StartPoseError):
        verify_start_pose(float(np.abs(gross).max()), cfg, after_move=True,
                          offsets=gross, Kq=Kq)

    # Without gains, only the cap applies -- the behaviour callers had before.
    assert start_pose_ok(stiff, float(np.abs(stiff).max()), cfg, None)


def test_the_start_gate_threshold_clears_the_friction_floor(cfg):
    """A gate the hardware cannot satisfy is a broken gate, not a strict one.

    The approach move runs at the servo's `Kq_default` and settles where joint
    friction balances it, so it leaves `friction / Kq_default` of angle behind.
    The replay then applies that angle times *its* own stiffness: the smallest
    reachable transient is `friction * Kq_scale`, and it grows with the very
    knob a user turns to track better. A 0.8 Nm limit refused replays whose arm
    was 0.02 rad away at Kq_scale 0.8 and 1.0.
    """
    import numpy as np

    from kinesteach.config import Config
    from kinesteach.replay import start_pose_ok

    # Peak friction measured across ten settled poses on the lab FR3 (Nm).
    friction = np.array([0.79, 0.82, 0.86, 1.13, 0.90, 0.86, 0.23])
    Kq_default = np.array([40, 30, 50, 25, 35, 25, 10.0])

    real = Config.load("configs/real.yaml")
    for name, c in (("default", cfg), ("configs/real.yaml", real)):
        lim = c.replay.start_pose_tol_nm
        if lim is None:
            continue
        for scale in (0.3, 0.5, 0.8, 1.0):
            off = friction / Kq_default          # what the approach leaves
            reachable = (friction * scale).max()
            assert reachable < lim, (
                "%s: at Kq_scale %.1f the best reachable transient is %.2f Nm "
                "but the gate allows %.2f" % (name, scale, reachable, lim)
            )
            assert start_pose_ok(off, float(off.max()), c, scale * Kq_default)


def test_the_first_ctrl_c_asks_the_replay_to_stop_rather_than_cutting_it(cfg, episode):
    """The two-level stop safety.py documents has to hold on the CLI path too.

    `replay_episode` used to open an `EmergencyTermination` of its own even when
    the caller passed one in. The inner guard replaced the outer's signal
    handlers while the outer held the cooperative block, so the inner's `_coop`
    was zero and the *first* SIGINT went straight to `fire()`: the policy was
    terminated before anyone read `stop_requested`. Measured, not reasoned --
    the guard logged "terminating running policy (emergency termination)".
    """
    import os
    import signal

    from kinesteach.backend.mock import MockBackend
    from kinesteach.process import process_episode
    from kinesteach.replay import replay_episode
    from kinesteach.safety import EmergencyTermination

    process_episode(episode, cfg)
    b = MockBackend(cfg.backend)
    b.connect()
    guard = EmergencyTermination(b).install()

    # Whether a stop had been *requested* at each termination. A cooperative
    # first Ctrl-C means the loop stops the policy, so the flag is already set;
    # a guard firing on the signal itself terminates while it is still False.
    requested_at_terminate = []
    real_terminate = b.terminate_policy

    def watched():
        requested_at_terminate.append(guard.stop_requested)
        return real_terminate()

    b.terminate_policy = watched

    fired = []

    def one_shot(running=b.is_running_policy):
        if not fired:
            fired.append(True)
            os.kill(os.getpid(), signal.SIGINT)
        return running()

    b.is_running_policy = one_shot
    try:
        buf, _ = replay_episode(b, episode, cfg, save=False, stop=guard)
    finally:
        guard.uninstall()
        b.close()

    assert fired, "the test never delivered a SIGINT"
    assert requested_at_terminate, "the policy was never terminated"
    assert requested_at_terminate[0] is True, (
        "the first Ctrl-C terminated the policy before the loop saw the stop "
        "request; the cooperative block was bypassed"
    )
    # Deliberately no assertion on buf.n: the signal lands on the very first
    # poll, so the mock has under one control period of trajectory to report.
    # How much log survives is a property of this stub's clock, not of the
    # guard composition being tested.
    assert buf is not None


def test_both_replay_paths_describe_a_pass_the_same_way(cfg, episode):
    """The CLI and the WebUI must not write two shapes of replay pass.

    The controller dict is the only record an episode keeps of the gains it was
    replayed under. It was written out longhand in `replay_episode` and again in
    the worker, so a field added to one would silently produce passes that
    cannot be compared against the other's.
    """
    import inspect

    from kinesteach.replay import replay_controller_metadata
    from kinesteach.webui import worker as worker_mod

    src = inspect.getsource(worker_mod._cmd_replay if hasattr(worker_mod, "_cmd_replay")
                            else worker_mod.Worker._cmd_replay)
    assert "replay_controller_metadata(" in src
    assert '"type": "JointTrajectoryExecutor"' not in src, (
        "the worker builds the controller dict itself again"
    )

    import numpy as np

    meta = replay_controller_metadata(
        cfg, np.zeros(7), np.zeros(7), np.zeros(6), np.zeros(6),
        np.zeros((10, 7)), 1.0, {"n_waypoints": 10},
    )
    assert meta["type"] == "JointTrajectoryExecutor"
    assert meta["n_waypoints"] == 10
    assert meta["time_scale"] == cfg.replay.time_scale
