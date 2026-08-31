import numpy as np
import pytest

from kinesteach.config import Config, ProcessConfig
from kinesteach.process import (
    build_replay_trajectory,
    butterworth,
    cutoff_sweep,
    derivative,
    process_episode,
    resample_uniform,
)


def _signal(fs=1000.0, dur=2.0, f_lo=1.0, f_hi=80.0):
    t = np.arange(0, dur, 1 / fs)
    lo = np.sin(2 * np.pi * f_lo * t)
    hi = 0.2 * np.sin(2 * np.pi * f_hi * t)
    return t, (lo + hi)[:, None], lo[:, None]


def test_butterworth_removes_the_high_band_and_keeps_the_low():
    t, x, lo = _signal()
    y = butterworth(x, 1000.0, 10.0, 4)
    edge = 200  # ignore filtfilt edge transients
    assert np.abs(y - lo)[edge:-edge].max() < 0.05


def test_filtfilt_does_not_shift_the_signal_in_time():
    """Zero phase: the peak must not move (this is why filtfilt, not lfilter)."""
    t, x, lo = _signal()
    y = butterworth(x, 1000.0, 10.0, 4)
    edge = 200
    i_in = int(np.argmax(lo[edge:-edge]))
    i_out = int(np.argmax(y[edge:-edge]))
    assert abs(i_in - i_out) <= 2


def test_butterworth_rejects_a_cutoff_above_nyquist():
    _, x, _ = _signal()
    with pytest.raises(ValueError, match="must be in"):
        butterworth(x, 1000.0, 600.0, 4)


def test_lower_cutoff_smooths_more():
    _, x, _ = _signal()
    sweep = cutoff_sweep(x, 1000.0, [5, 10, 20, 40])
    accel = [
        np.sqrt(np.mean(derivative(derivative(v, 1e-3), 1e-3) ** 2))
        for v in (sweep["cutoff_%g" % c] for c in (5, 10, 20, 40))
    ]
    assert accel == sorted(accel), "a lower cutoff must leave less acceleration"


def test_resample_uniform_lands_on_a_fixed_period():
    rng = np.random.default_rng(0)
    t = np.cumsum(rng.normal(1e-3, 5e-5, 1000))
    t -= t[0]
    x = np.sin(2 * np.pi * t)[:, None]
    tu, xu = resample_uniform(t, x, 1000.0)
    assert np.allclose(np.diff(tu)[:-1], 1e-3)
    assert tu[-1] <= t[-1]  # never extrapolates


def test_time_scale_slows_the_replay_proportionally():
    t = np.arange(0, 1.0, 1e-3)
    q = np.sin(2 * np.pi * t)[:, None]
    _, q1, dq1 = build_replay_trajectory(t, q, 1000.0, 1.0)
    _, q2, dq2 = build_replay_trajectory(t, q, 1000.0, 2.0)
    assert q2.shape[0] == pytest.approx(2 * q1.shape[0], rel=0.01)
    assert np.abs(dq2).max() == pytest.approx(np.abs(dq1).max() / 2, rel=0.02)


def test_build_replay_trajectory_matches_the_server_rate():
    t = np.arange(0, 1.0, 1e-3)
    q = np.zeros((t.size, 7))
    for hz in (500.0, 1000.0):
        tr, qr, _ = build_replay_trajectory(t, q, hz, 1.0)
        assert np.diff(tr).mean() == pytest.approx(1 / hz, rel=1e-6)


def test_process_episode_writes_derived_arrays_and_fk(episode, cfg):
    out = process_episode(episode, cfg)
    for key in ("t", "q_filtered", "dq_from_q_filtered", "tau_external_filtered",
                "q_replay", "dq_replay"):
        assert key in out
    info = episode.read_metadata()["processing"]
    assert info["cutoff_hz"] == cfg.process.cutoff_hz
    if "ee_pos_flange" in out:  # needs polymetis' URDF
        assert out["ee_pos_flange"].shape == (out["q_filtered"].shape[0], 3)
        assert info["ee_frame"] == "flange", "FK gives the flange, not the TCP (plan 2.2)"
    assert set(episode.read_processed()) == set(out)
