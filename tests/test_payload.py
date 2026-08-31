"""The unmodelled-load estimator, checked against ground truth it cannot see."""
import pathlib

import numpy as np
import pytest

from kinesteach.backend.mock import MockBackend
from kinesteach.envelope import path_is_safe
from kinesteach.payload import (
    GRAVITY,
    design_matrix,
    estimate_payload,
    gravity_regressor,
    sample_poses,
    select_poses,
)


@pytest.fixture(scope="module")
def fk():
    from kinesteach.kinematics import fk_from_urdf_text

    urdf = MockBackend().spec().urdf_text
    if not urdf:
        pytest.skip("polymetis URDF unavailable")
    return fk_from_urdf_text(urdf, "panda_link8")


@pytest.fixture(scope="module")
def spec():
    return MockBackend().spec()


def _synthesise(qs, fk, mass, com, bias, noise=0.0, seed=0):
    """Torques a load of (mass, com) would produce, plus bias and noise."""
    theta = np.concatenate([[mass], mass * np.asarray(com, dtype=np.float64)])
    rng = np.random.default_rng(seed)
    out = []
    for q in qs:
        tau = gravity_regressor(q, fk) @ theta + bias
        out.append(tau + rng.normal(0.0, noise, size=tau.shape))
    return out


def test_joint_one_cannot_feel_gravity(fk):
    """Its axis is vertical, so its regressor row must be exactly zero.

    This is the geometry check for the whole model: if the Jacobian convention
    were wrong, this row would not vanish.
    """
    rng = np.random.default_rng(3)
    for _ in range(5):
        q = rng.uniform(-1.5, 1.5, size=7)
        q[3] = -1.5  # keep the elbow in a legal region
        A = gravity_regressor(q, fk)
        np.testing.assert_allclose(A[0], 0.0, atol=1e-9)


def test_recovers_a_known_load(fk, spec):
    mass, com = 0.23, np.array([0.012, -0.028, 0.075])
    bias = np.array([-0.19, -0.43, 0.08, 0.63, 0.21, 0.14, -0.13])

    qs = sample_poses(spec, fk, 40, rng=np.random.default_rng(1))
    keep = select_poses(qs, fk, 12)
    qs = [qs[i] for i in keep]
    taus = _synthesise(qs, fk, mass, com, bias)

    r = estimate_payload(qs, taus, fk)

    assert r["full_rank"]
    assert not r["sign_flipped"]
    assert r["mass_kg"] == pytest.approx(mass, abs=1e-6)
    np.testing.assert_allclose(r["com_flange_m"], com, atol=1e-6)
    np.testing.assert_allclose(r["bias_nm"], bias, atol=1e-6)
    assert r["residual_rms_nm"] < 1e-9


def test_survives_realistic_sensor_noise(fk, spec):
    """Test A measured a per-joint std around 0.013 Nm; use twice that."""
    mass, com = 0.23, np.array([0.012, -0.028, 0.075])
    bias = np.zeros(7)

    qs = sample_poses(spec, fk, 60, rng=np.random.default_rng(2))
    qs = [qs[i] for i in select_poses(qs, fk, 12)]
    taus = _synthesise(qs, fk, mass, com, bias, noise=0.026, seed=5)

    r = estimate_payload(qs, taus, fk)

    assert r["mass_kg"] == pytest.approx(mass, abs=0.03)
    assert r["mass_stderr_kg"] < 0.03
    np.testing.assert_allclose(r["com_flange_m"], com, atol=0.05)


def test_d_optimal_selection_beats_taking_the_first_poses(fk, spec):
    """Otherwise there is no reason to compute the poses instead of guessing."""
    mass, com = 0.23, np.array([0.012, -0.028, 0.075])
    pool = sample_poses(spec, fk, 60, rng=np.random.default_rng(7))

    chosen = [pool[i] for i in select_poses(pool, fk, 10)]
    arbitrary = pool[:10]

    def err(qs):
        taus = _synthesise(qs, fk, mass, com, np.zeros(7), noise=0.026, seed=11)
        return abs(estimate_payload(qs, taus, fk)["mass_kg"] - mass)

    assert err(chosen) < err(arbitrary)

    d_sel, _ = design_matrix(chosen, fk)
    d_arb, _ = design_matrix(arbitrary, fk)
    assert np.linalg.cond(d_sel) < np.linalg.cond(d_arb)


def test_flipped_sign_convention_is_reported_not_hidden(fk, spec):
    qs = sample_poses(spec, fk, 30, rng=np.random.default_rng(4))
    qs = [qs[i] for i in select_poses(qs, fk, 8)]
    taus = [-t for t in _synthesise(qs, fk, 0.2, [0, 0, 0.06], np.zeros(7))]

    r = estimate_payload(qs, taus, fk)

    assert r["sign_flipped"] is True
    assert r["mass_kg"] == pytest.approx(0.2, abs=1e-6)  # never a negative mass


def test_one_pose_is_refused(fk, spec):
    q = sample_poses(spec, fk, 1, rng=np.random.default_rng(9))
    with pytest.raises(ValueError, match="at least 2 poses"):
        estimate_payload(q, [np.zeros(7)], fk)


def test_path_check_rejects_a_swing_through_the_floor(fk, spec):
    q = np.array(spec.home_pose, dtype=np.float64)
    ok, why = path_is_safe(q, q, fk, spec)
    assert ok, why

    # Straightening the elbow from the rest pose drags the flange down.
    far = q.copy()
    far[1] = 1.0
    far[3] = -0.3
    ok, why = path_is_safe(q, far, fk, spec, min_flange_z=0.45)
    assert not ok and ("floor" in why or "limit" in why)


def test_sampled_poses_stay_clear_of_the_limits(fk, spec):
    margin = 0.35
    for q in sample_poses(spec, fk, 15, rng=np.random.default_rng(6), limit_margin_rad=margin):
        assert np.all(q >= np.asarray(spec.joint_pos_min) + margin - 1e-9)
        assert np.all(q <= np.asarray(spec.joint_pos_max) - margin + 1e-9)


# ---- the safe envelope ----------------------------------------------------


def _in_hull(point, vertices, tol=1e-7):
    """Exact hull membership by LP: is `point` a convex combination?"""
    from scipy.optimize import linprog

    V = np.asarray(vertices, dtype=np.float64)
    n = len(V)
    A_eq = np.vstack([V.T, np.ones(n)])
    b_eq = np.concatenate([np.asarray(point, dtype=np.float64), [1.0]])
    r = linprog(np.zeros(n), A_eq=A_eq, b_eq=b_eq, bounds=[(0, 1)] * n,
                method="highs")
    return bool(r.success) and float(np.abs(A_eq @ r.x - b_eq).max()) < tol


def test_every_move_between_sampled_poses_stays_inside_the_walked_region():
    """The whole reason the envelope lives in joint space.

    `move_to_joint_positions` interpolates there, so a convex region contains
    the entire path, not just its endpoints. A Cartesian box cannot promise
    this -- on the lab robot such legs bulged 0.43 m outside one.
    """
    from kinesteach.envelope import sample_hull

    rng = np.random.default_rng(0)
    V = rng.uniform(-1.0, 1.0, size=(25, 7))
    poses = sample_hull(V, 6, rng=rng)

    for q in poses:
        assert _in_hull(q, V), "a sampled pose left the hull"
    for a, b in zip(poses, poses[1:]):
        for t in np.linspace(0.0, 1.0, 11):
            assert _in_hull(a * (1 - t) + b * t, V), "a leg left the hull"


def test_envelope_round_trips(fk, spec, tmp_path):
    from kinesteach.envelope import load_envelope, save_envelope

    qs = np.asarray(sample_poses(spec, fk, 12, rng=np.random.default_rng(0)))
    path = tmp_path / "envelope.json"
    save_envelope(str(path), qs, spec, fk)
    back = load_envelope(str(path))

    assert back["num_dofs"] == spec.num_dofs
    assert back["n_vertices"] == 12
    np.testing.assert_allclose(back["vertices_q"], qs)
    lo = back["flange_extent_m"]["min"]
    assert len(lo) == 3


def test_envelope_report_names_the_right_motion_to_add(fk, spec):
    """Mass and the first moments are fixed by different motions."""
    from kinesteach.envelope import sample_hull
    from kinesteach.payload import envelope_report

    good = sample_poses(spec, fk, 60, rng=np.random.default_rng(0))
    good = [good[i] for i in select_poses(good, fk, 12)]
    assert envelope_report(good, fk)["well_conditioned"]

    # A pinched envelope: poses that barely differ cannot separate anything.
    base = good[0]
    pinched = sample_hull(np.array([base, base + 0.02]), 8,
                          rng=np.random.default_rng(1))
    r = envelope_report(pinched, fk)
    assert not r["well_conditioned"]
    assert r["worst_parameter"] in ("mass", "m*c_x", "m*c_y", "m*c_z")
    # Poses 0.02 rad apart differ in neither reach nor orientation, so naming
    # one of the two would send the operator half way to a fix. Say both.
    assert "barely differ" in r["advice"]
    assert r["reach_span_m"] < 0.05


def test_hull_sampling_needs_more_than_one_vertex():
    from kinesteach.envelope import sample_hull

    with pytest.raises(ValueError, match="at least 2 vertices"):
        sample_hull(np.zeros((1, 7)), 3)


def test_a_walk_in_which_the_arm_never_moved_is_refused(cfg, tmp_path, monkeypatch, stub_connected):
    """A stationary walk is a failed session, not a small region.

    The first real walk on the robot saved 203 configurations spanning
    0.0008 deg -- encoder noise, because the joints were still locked -- and
    reported it as a success. `payload-sweep` would then have sampled inside a
    single point and fitted a load from one pose.
    """
    import argparse

    import numpy as np
    import pytest

    from kinesteach.backend.mock import MockBackend
    from kinesteach.cli.calibrate import cmd_workspace

    b = MockBackend(cfg.backend)
    b.connect()
    if not b.spec().urdf_text:
        pytest.skip("mock backend has no URDF to build kinematics from")

    monkeypatch.setattr("kinesteach.cli.common._cfg", lambda a: cfg)
    stub_connected(b)

    q = np.array([0.1, -0.2, 0.3, -1.5, 0.0, 1.6, 0.4])
    still = np.repeat(q[None, :], 500, axis=0)

    class _Buf:
        n = 500
        duration = 5.0
        q = still

    monkeypatch.setattr("kinesteach.teach.run_teaching",
                        lambda *a, **k: (_Buf(), None, None))

    out = tmp_path / "envelope.json"
    args = argparse.Namespace(config=None, out=str(out), duration=5.0,
                              vertices=200, guided=False, kqd_scale=None)
    with pytest.raises(SystemExit) as caught:
        cmd_workspace(args)
    assert "did not move" in str(caught.value)
    assert not out.exists()  # nothing saved
    b.close()



def test_the_advice_names_the_motion_that_is_missing_not_the_worst_parameter():
    """A report that says "reach further" when reach was fine sends you nowhere.

    The first real walk covered 0.42 m of reach and still reported `mass` as the
    least determined parameter, because the gripper pointed downwards at every
    pose: `m` and `m*c` could imitate each other. The old advice read that as a
    reach problem. It is an orientation problem.
    """
    import numpy as np

    from kinesteach.payload import envelope_report

    class _FK:
        """Poses at varying reach, all with the flange pointing straight down."""

        def __init__(self, rolls):
            self.rolls = rolls

        def fk(self, qs):
            n = len(np.atleast_2d(qs))
            pos = np.zeros((n, 3))
            pos[:, 0] = np.linspace(0.30, 0.75, n)   # 0.45 m of reach
            pos[:, 2] = 0.5
            quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
            return pos, quat

        def jacobian(self, q):
            seed = abs(int(np.asarray(q).ravel()[0] * 1e6)) % 2**31
            rng = np.random.default_rng(seed)
            return (rng.normal(size=(6, 7)),)

    qs = [np.full(7, 0.1 * i) for i in range(1, 13)]
    r = envelope_report(qs, _FK(None))

    assert r["reach_span_m"] > 0.4                      # reach was never the issue
    assert min(r["axis_vs_gravity_span"].values()) < 0.1  # nothing ever turned over
    assert "never turned much" in r["advice"]
    assert "reach further" not in r["advice"]


def test_advice_catches_a_wrist_turned_in_only_one_place():
    """The failure the first usable walk actually had.

    Its reach covered 0.32-0.74 m and its wrist turned through 64 deg, so both
    margins looked healthy and the report blamed the mass. But the turning all
    happened at 0.53-0.58 m: two poses that turned the wrist with the arm
    extended took the condition number from 96 to 32, and going past horizontal
    was never needed. Margins cannot see that; the combination has to be checked.
    """
    import numpy as np

    from kinesteach.payload import envelope_report

    rng = np.random.default_rng(7)
    base, delta = rng.normal(size=(6, 7)), rng.normal(size=(6, 7))

    class _FK:
        """Reach 0.30-0.75 m, but the wrist only tilts near 0.55 m."""

        def fk(self, qs):
            Q = np.atleast_2d(np.asarray(qs, dtype=np.float64))
            n, reach = len(Q), Q[:, 0]
            pos = np.stack([reach, np.zeros(n), np.full(n, 0.5)], axis=1)
            # Tilt away from straight down, and only within 3 cm of 0.55 m.
            theta = np.where(np.abs(reach - 0.55) < 0.03, np.deg2rad(64.0), 0.0)
            half = (np.pi - theta) / 2.0
            quat = np.stack([np.sin(half), np.zeros(n), np.zeros(n),
                             np.cos(half)], axis=1)
            return pos, quat

        def jacobian(self, q):
            return (base + 0.2 * float(np.asarray(q).ravel()[0]) * delta,)

    qs = [np.array([r] + [0.1] * 6) for r in np.linspace(0.30, 0.75, 14)]
    r = envelope_report(qs, _FK())

    assert r["reach_span_m"] > 0.4                        # plenty of reach
    assert max(r["axis_vs_gravity_span"].values()) > 0.5  # and real turning
    assert r["turning_reach_span_m"] < 0.1                # but all in one place
    assert not r["well_conditioned"]
    assert "with the arm extended" in r["advice"]
    assert "reach further" not in r["advice"]


def test_home_first_refuses_an_approach_it_cannot_clear(cfg, tmp_path, monkeypatch, stub_connected):
    """Homing from an awkward pose is exactly when the path must be checked.

    `go_home` moves autonomously without checking anything, so `--home-first`
    routes home through `path_is_safe` like any other leg. If that leg does not
    clear, the run stops before the arm moves at all -- it does not fall back on
    approaching directly, because the operator asked to go home for a reason.
    """
    import argparse
    import json

    import numpy as np
    import pytest

    from kinesteach.backend.mock import MockBackend
    from kinesteach.cli.calibrate import cmd_payload_sweep
    from kinesteach.kinematics import fk_from_urdf_text
    from kinesteach.envelope import save_envelope
    from kinesteach.payload import sample_poses

    b = MockBackend(cfg.backend)
    b.connect()
    spec = b.spec()
    if not spec.urdf_text or spec.joint_pos_min is None:
        pytest.skip("mock backend has no URDF or limits")

    fk = fk_from_urdf_text(spec.urdf_text, spec.ee_link_name)
    env = tmp_path / "envelope.json"
    save_envelope(str(env), np.array(sample_poses(spec, fk, 30,
                                                  rng=np.random.default_rng(0))),
                  spec, fk)

    moved = []
    monkeypatch.setattr(b, "move_to_joint_positions",
                        lambda *a, **k: moved.append(a))
    monkeypatch.setattr("kinesteach.envelope.path_is_safe",
                        lambda *a, **k: (False, "would clip the table"))

    monkeypatch.setattr("kinesteach.cli.common._cfg", lambda a: cfg)
    stub_connected(b)

    args = argparse.Namespace(
        config=None, envelope=str(env), no_envelope=False, poses=6,
        candidates=100, seed=0, hull_k=3, hull_alpha=0.3, limit_margin=0.35,
        min_z=0.25, settle=0.0, dwell=0.0, slow=1.0, still_thresh=1e-2,
        min_samples=1, yes=True, home_first=True,
    )
    with pytest.raises(SystemExit) as caught:
        cmd_payload_sweep(args)

    assert "home" in str(caught.value)
    assert not moved, "the arm moved despite the refusal"
    b.close()


def test_building_kinematics_from_text_leaves_no_file_behind(cfg):
    """FK from URDF text used to leak one /tmp file per call.

    Pinocchio only reads from a path, so the text is spilled to a temporary
    file -- which was created with `delete=False` and never removed. Every
    `workspace`, every `payload-sweep`, every `teach --home` left one behind,
    and the workstation had accumulated 355 before anyone looked.
    """
    import glob
    import tempfile

    from kinesteach.backend.mock import MockBackend
    from kinesteach.kinematics import fk_from_urdf_text

    b = MockBackend(cfg.backend)
    b.connect()
    spec = b.spec()
    if not spec.urdf_text:
        pytest.skip("mock backend has no URDF")

    pattern = str(pathlib.Path(tempfile.gettempdir()) / "kinesteach-*.urdf")
    before = set(glob.glob(pattern))
    fk = fk_from_urdf_text(spec.urdf_text, spec.ee_link_name)
    leaked = set(glob.glob(pattern)) - before
    b.close()

    assert not leaked, "left %d temporary URDF file(s) behind: %s" % (
        len(leaked), sorted(leaked))
    # The model still works after its source file is gone: pinocchio parses in
    # the constructor and never reopens the path.
    pos, _ = fk.fk(np.zeros((1, spec.num_dofs)))
    assert pos.shape == (1, 3)


def test_a_swept_pose_records_the_target_it_was_given_not_just_where_it_landed():
    """The row is the whole measurement, and the fit is built back out of it.

    The first sweep kept only the reached pose, so recovering the commanded one
    afterwards meant re-deriving the plan and matching by distance -- which
    mismatched two poses of ten. It also has to carry the health fields: a pose
    sampled while the control loop was struggling is not a pose to fit.
    """
    from kinesteach.backend.base import TelemetrySample
    from kinesteach.cli.calibrate import _measure_pose

    target = np.array([0.1, -0.2, 0.3, -1.5, 0.0, 1.6, 0.4])
    landed = target + 0.02                       # the friction offset
    tau = np.arange(7) * 0.1

    still = [
        TelemetrySample(
            timestamp=0.0, q=landed, dq=np.zeros(7), tau_external=tau,
            error_code=0, command_successful=(k != 0),   # one dropped packet
            controller_latency_ms=0.2,
        )
        for k in range(4)
    ]

    row = _measure_pose(2, target, still)
    assert row["pose"] == 3
    assert row["n_still"] == 4
    np.testing.assert_allclose(row["q_target"], target)
    np.testing.assert_allclose(row["q"], landed)
    np.testing.assert_allclose(row["q_error_rad"], np.full(7, 0.02), atol=1e-12)
    assert row["q_error_max_rad"] == pytest.approx(0.02)
    assert row["latency_ms_mean"] == pytest.approx(0.2)
    assert row["frac_failed_commands"] == pytest.approx(0.25)

    # The fit reads its inputs back out of the rows, so those keys are the
    # contract between measuring and fitting, not incidental bookkeeping.
    np.testing.assert_allclose(np.asarray(row["q"]), landed)
    np.testing.assert_allclose(np.asarray(row["tau_external"]), tau)
