import numpy as np
import pytest

from kinesteach.backend.base import ROBOT_ARRAY_FIELDS, EpisodeBuffer
from kinesteach.backend.mock import MockBackend
from kinesteach.validate import (
    MAX_CIRCULAR_BUFFER_SIZE,
    check_a_static_bias,
    check_b_handguiding,
    check_c_repeatability,
    check_d_replay_comparison,
    validate_buffer,
)


def _buf(n, hz=1000.0, dof=7, t=None):
    d = {
        nm: (np.zeros((n, dof)) if pj else np.zeros(n)).astype(dt)
        for nm, dt, pj in ROBOT_ARRAY_FIELDS
    }
    d["command_successful"] = np.ones(n, dtype=bool)
    d["timestamp_ns"] = (
        (np.arange(n) * (1e9 / hz)).astype(np.int64) if t is None
        else np.asarray(t, dtype=np.int64)
    )
    return EpisodeBuffer.from_dict(d)


def _ns(seconds):
    return np.round(np.asarray(seconds) * 1e9).astype(np.int64)


def test_clean_log_passes():
    rep = validate_buffer(_buf(2000))
    assert rep["ok"] and not rep["warnings"]
    assert rep["effective_hz"] == pytest.approx(1000.0)


def test_dropped_samples_are_counted():
    t = np.delete(np.arange(1000) / 1000.0, [100, 101, 500])  # three missing ticks
    rep = validate_buffer(_buf(t.size, t=_ns(t)))
    assert rep["estimated_dropped_samples"] == 3
    assert any("missing" in w for w in rep["warnings"])


def test_ring_buffer_overflow_is_an_error():
    """A full buffer means the start of the episode is gone (plan 2.4)."""
    n = MAX_CIRCULAR_BUFFER_SIZE
    rep = validate_buffer(_buf(n))
    assert not rep["ok"]
    assert any("ring buffer" in e for e in rep["errors"])


def test_non_monotonic_timestamps_are_an_error():
    t = np.arange(100) / 1000.0
    t[50] = t[40]
    rep = validate_buffer(_buf(100, t=_ns(t)))
    assert not rep["ok"]


def test_error_codes_are_reported():
    b = _buf(500)
    b.error_code[10:20] = 3
    rep = validate_buffer(b)
    assert rep["n_error_states"] == 10 and rep["error_codes"][3] == 10
    assert not rep["ok"]


def test_joint_limit_proximity_is_flagged(backend):
    spec = backend.spec()
    b = _buf(500)
    b.q[:] = spec.joint_pos_max - 0.1  # inside the 0.2 rad safety margin
    rep = validate_buffer(b, spec)
    assert any("safety-controller margin" in w for w in rep["warnings"])


def test_rate_below_nominal_is_flagged(backend):
    """640 Hz where 1000 Hz was expected is the failure this must catch."""
    rep = validate_buffer(_buf(640, hz=640.0), backend.spec())
    assert rep["effective_hz"] == pytest.approx(640.0, rel=1e-3)
    assert rep["hz_ratio"] == pytest.approx(0.64, rel=1e-2)
    assert any("nominal" in w for w in rep["warnings"])


def test_acceptance_tests_run_on_mock_data():
    b = MockBackend()
    b.connect()
    log = b._make_log(0.0, 3.0, "teach", None, None)

    a = check_a_static_bias(log)
    assert a["test"] == "A_static_bias"  # mock is always moving, so ok=False is fine

    bb = check_b_handguiding(log, b.spec())
    assert bb["ok"] and bb["hz_ratio"] == pytest.approx(1.0, abs=0.02)

    c = check_c_repeatability([log, b._make_log(0.0, 3.0, "teach", None, None)])
    assert c["ok"] and c["n_demos"] == 2
    assert c["overall_rms_std_rad"] < 1e-9  # same seed -> identical demos

    d = check_d_replay_comparison(log, log)
    assert d["ok"] and d["tracking_rms_rad"] == pytest.approx(0.0, abs=1e-12)


def test_safety_layer_intervention_is_flagged():
    """`tau_safened_prev` differing from `tau_computed_prev` is the server
    having overridden our policy -- the one signal we get about limits that
    live in a config file on the server machine."""
    buf = _buf(1000)
    buf.tau_computed_prev[:] = 1.0
    buf.tau_safened_prev[:] = 1.0
    buf.tau_safened_prev[400:450, 5] = 1.0 + 2.5  # reflex on joint 5

    rep = validate_buffer(buf)

    assert rep["frac_states_with_safety_reflex"] == pytest.approx(0.05)
    assert rep["safety_reflex_absmax_nm"][5] == pytest.approx(2.5)
    assert rep["safety_reflex_absmax_nm"][0] == 0.0
    assert any("safety layer" in w for w in rep["warnings"])
    assert rep["ok"]  # informative, not disqualifying


def test_untouched_torques_report_no_intervention():
    buf = _buf(1000)
    buf.tau_computed_prev[:] = 0.7
    buf.tau_safened_prev[:] = 0.7

    rep = validate_buffer(buf)

    assert rep["frac_states_with_safety_reflex"] == 0.0
    assert not any("safety layer" in w for w in rep["warnings"])


def test_dropped_commands_report_their_longest_run():
    """The count alone is not the risk; consecutive held ticks are."""
    buf = _buf(1000)
    buf.command_successful[:] = True
    buf.command_successful[[10, 200, 201, 202, 500]] = False

    rep = validate_buffer(buf)

    assert rep["n_failed_commands"] == 5
    assert rep["max_consecutive_failed_commands"] == 3
    assert rep["frac_failed_commands"] == pytest.approx(0.005)
    assert any("longest run 3" in w for w in rep["warnings"])


def test_no_dropped_commands_reports_zero_run():
    rep = validate_buffer(_buf(500))
    assert rep["max_consecutive_failed_commands"] == 0
    assert not any("dropped" in w for w in rep["warnings"])
