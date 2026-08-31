import numpy as np
import pytest

from kinesteach.backend.base import ROBOT_ARRAY_FIELDS, EpisodeBuffer, RobotBackend


def _buf(n=100, d=7, hz=1000.0):
    return EpisodeBuffer(
        **{
            nm: (np.zeros((n, d)) if pj else np.zeros(n)).astype(dt)
            for nm, dt, pj in ROBOT_ARRAY_FIELDS
        }
    )


def test_mock_satisfies_the_protocol(backend):
    assert isinstance(backend, RobotBackend)


def test_buffer_rejects_ragged_input():
    d = {nm: (np.zeros((10, 7)) if pj else np.zeros(10)) for nm, _, pj in ROBOT_ARRAY_FIELDS}
    d["q"] = np.zeros((9, 7))
    with pytest.raises(ValueError, match="length mismatch"):
        EpisodeBuffer.from_dict(d)


def test_buffer_rejects_missing_field():
    d = {nm: (np.zeros((10, 7)) if pj else np.zeros(10)) for nm, _, pj in ROBOT_ARRAY_FIELDS}
    del d["tau_external"]
    with pytest.raises(ValueError, match="missing robot log field"):
        EpisodeBuffer.from_dict(d)


def test_effective_hz_counts_intervals_not_samples():
    b = _buf(n=1001)
    b.timestamp_ns[:] = np.arange(1001) * 1_000_000
    assert b.duration == pytest.approx(1.0)
    assert b.effective_hz == pytest.approx(1000.0)


def test_protobuf_boundary_carries_every_field():
    """Whatever the server sends must survive the conversion (plan 1.2)."""
    pytest.importorskip("polymetis_pb2")
    from polymetis_pb2 import RobotState

    from kinesteach.backend.polymetis import states_to_buffer

    states = []
    for i in range(4):
        s = RobotState()
        s.timestamp.seconds = 1_750_000_000
        s.timestamp.nanos = i * 1_000_000
        s.joint_positions.extend([0.1 * i] * 7)
        s.joint_velocities.extend([0.2] * 7)
        s.motor_torques_measured.extend([1.0] * 7)
        s.motor_torques_external.extend([0.5 * i] * 7)
        s.joint_torques_computed.extend([2.0] * 7)
        s.motor_torques_desired.extend([3.0] * 7)
        s.prev_joint_torques_computed.extend([4.0] * 7)
        s.prev_joint_torques_computed_safened.extend([5.0] * 7)
        s.prev_controller_latency_ms = 0.3
        s.prev_command_successful = True
        s.error_code = 0
        states.append(s)

    buf = states_to_buffer(states)
    assert buf.n == 4 and buf.num_dofs == 7
    # exact: nanoseconds survive the conversion without rounding
    assert buf.effective_hz == 1000.0
    assert buf.timestamp_ns[1] - buf.timestamp_ns[0] == 1_000_000
    # every RobotState field has a home in EpisodeBuffer
    proto_fields = {f.name for f in RobotState.DESCRIPTOR.fields}
    assert len(proto_fields) == len(ROBOT_ARRAY_FIELDS)
    np.testing.assert_allclose(buf.tau_external[:, 0], [0.0, 0.5, 1.0, 1.5])
    np.testing.assert_allclose(buf.tau_safened_prev[0], [5.0] * 7)


def test_mock_replay_log_follows_the_commanded_trajectory(backend):
    import time

    spec = backend.spec()
    q = np.tile(spec.home_pose, (300, 1))  # 0.3 s at 1 kHz
    backend.start_replay(q, np.zeros_like(q), spec.default_Kq, spec.default_Kqd,
                         np.zeros(6), np.zeros(6))
    time.sleep(0.35)
    buf = backend.terminate_policy()
    assert buf.n > 0
    assert np.abs(buf.q - spec.home_pose).max() < 1e-2


# ---- joint limit overrides ------------------------------------------------
#
# The lab server runs a Panda URDF against FR3 hardware and its metadata carries
# no limits, so the URDF's numbers are wrong and nothing on the wire says so.


def test_config_limits_replace_the_urdf_limits():
    from kinesteach.backend import make_backend
    from kinesteach.config import BackendConfig

    fr3_low = [-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159]
    fr3_high = [2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159]

    plain = make_backend(BackendConfig(kind="mock")).spec()
    assert plain.joint_limits_source == "urdf"
    assert plain.joint_pos_min[5] < 0.0  # the Panda value we are replacing

    b = make_backend(
        BackendConfig(kind="mock", joint_pos_min=fr3_low, joint_pos_max=fr3_high)
    )
    spec = b.spec()

    assert spec.joint_limits_source == "config"
    np.testing.assert_allclose(spec.joint_pos_min, fr3_low)
    np.testing.assert_allclose(spec.joint_pos_max, fr3_high)
    # Untouched overrides keep the server's own value.
    np.testing.assert_allclose(spec.joint_vel_max, plain.joint_vel_max)
    # And the episode records which limits it was checked against.
    md = spec.to_metadata()
    assert md["joint_limits_source"] == "config"
    assert md["joint_pos_min"][5] == pytest.approx(0.5445)


def test_limit_override_must_match_the_reported_dof(backend):
    """Invariant 3 still holds: the server decides the DOF, not the file."""
    spec = backend.spec()
    with pytest.raises(ValueError, match="7 DOF"):
        spec.with_limit_overrides(joint_pos_min=[0.0] * 6, joint_pos_max=[1.0] * 6)


def test_bad_limit_overrides_are_rejected_by_config():
    from kinesteach.config import BackendConfig

    with pytest.raises(ValueError, match="set together"):
        BackendConfig(kind="real", joint_pos_min=[0.0] * 7)
    with pytest.raises(ValueError, match="below joint_pos_max"):
        BackendConfig(
            kind="real", joint_pos_min=[1.0] * 7, joint_pos_max=[0.0] * 7
        )
    with pytest.raises(ValueError, match="disagree on length"):
        BackendConfig(
            kind="real", joint_pos_min=[0.0] * 7, joint_pos_max=[1.0] * 6
        )


def test_kqd_scale_flag_overrides_the_config(tmp_path):
    """M5-5 sweeps this per run, so it has to beat the YAML."""
    import argparse

    from kinesteach.backend.mock import MockBackend
    from kinesteach.cli.common import _cfg
    from kinesteach.config import Config
    from kinesteach.teach import resolve_teaching_gains

    path = tmp_path / "c.yaml"
    Config.from_dict({"teach": {"Kqd_scale": 1.0}}).save(str(path))
    args = argparse.Namespace(config=str(path), data_root=None, log_level=None)

    base = _cfg(args)
    assert base.teach.Kqd_scale == 1.0

    args.kqd_scale = 0.0
    cfg = _cfg(args)
    assert cfg.teach.Kqd_scale == 0.0

    spec = MockBackend(cfg.backend).spec()
    _, Kqd = resolve_teaching_gains(spec, cfg)
    np.testing.assert_allclose(Kqd, np.zeros(spec.num_dofs))


def test_telemetry_carries_the_health_fields_the_sweep_quotes(cfg):
    """A field the sweep reads must exist, or it silently records zeros.

    `payload-sweep` reports latency and dropped commands beside each measured
    pose. The first version read `prev_controller_latency_ms` and
    `prev_command_successful` off the sample -- the names the *protobuf* uses,
    not the ones `TelemetrySample` exposes -- so `getattr` defaults would have
    filled the file with zeros that looked like a healthy run.
    """
    import numpy as np

    from kinesteach.backend.mock import MockBackend

    b = MockBackend(cfg.backend)
    b.connect()
    s = b.get_telemetry()
    for field in ("controller_latency_ms", "command_successful"):
        assert hasattr(s, field), "TelemetrySample lost %s" % field
    assert isinstance(float(s.controller_latency_ms), float)
    assert "controller_latency_ms" in s.to_json()
    b.close()
