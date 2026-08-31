"""A robot that exists only in this process.

The mock exercises every code path the real backend does -- policy lifecycle,
batched log retrieval, telemetry polling, trajectory execution -- without a
gRPC server, so M1-M4 can be built and unit tested on a desk.

The numbers it produces are *synthetic*, not physical. Joint angles are a sum
of sinusoids and the torques are shaped noise. Use it to check that the
pipeline is correct, never to draw a conclusion about the robot.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

from ..config import BackendConfig
from .base import EpisodeBuffer, RobotSpec, TelemetrySample

# Real franka_panda values, so mock episodes have the same shape and scale as
# real ones (conf/robot_model/franka_panda.yaml, conf/robot_client/franka_hardware.yaml).
PANDA_REST_POSE = np.array(
    [-0.13935426, -0.02048170, -0.05201414, -2.06912565, 0.05058914, 2.00286508, -0.91678745]
)
PANDA_DEFAULT_KQ = np.array([40.0, 30.0, 50.0, 25.0, 35.0, 25.0, 10.0])
PANDA_DEFAULT_KQD = np.array([4.0, 6.0, 5.0, 5.0, 3.0, 2.0, 1.0])
PANDA_DEFAULT_KX = np.array([750.0, 750.0, 750.0, 15.0, 15.0, 15.0])
PANDA_DEFAULT_KXD = np.array([37.0, 37.0, 37.0, 2.0, 2.0, 2.0])
PANDA_JOINT_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
PANDA_JOINT_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
# conf/robot_client/franka_hardware.yaml `limits.joint_vel` (franka limits minus margin)
PANDA_JOINT_VEL_MAX = np.array([2.075, 2.075, 2.075, 2.075, 2.51, 2.51, 2.51])


def _panda_urdf_text() -> Optional[str]:
    """The real panda URDF, if polymetis is importable.

    Loading it lets the mock exercise the offline FK path (M2) for real. When
    polymetis is absent -- a bare unit-test environment -- FK is simply skipped.
    """
    try:
        from polymetis.utils.data_dir import get_full_path_to_urdf

        return Path(get_full_path_to_urdf("franka_panda/panda_arm.urdf")).read_text()
    except Exception:
        return None


def synthetic_motion(
    t: np.ndarray, num_dofs: int, seed: int = 0, home: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic (q, dq, tau_external) for a vector of times.

    Pure function of `t` and `seed`, so a test can reproduce an episode exactly
    and an episode's length can be varied without changing its content.
    """
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    rng = np.random.default_rng(seed)
    home_v = (
        np.resize(PANDA_REST_POSE, num_dofs) if home is None else np.asarray(home, float)
    )

    amp = rng.uniform(0.08, 0.25, num_dofs)
    freq = rng.uniform(0.10, 0.45, num_dofs)
    phase = rng.uniform(0, 2 * np.pi, num_dofs)

    w = 2 * np.pi * freq
    ang = w * t[:, None] + phase
    q = home_v + amp * np.sin(ang)
    dq = amp * w * np.cos(ang)

    # A slow "human push" plus broadband sensor noise, scaled per joint the way
    # the proximal joints of a 7-DOF arm see larger torques than the wrist.
    scale = np.linspace(6.0, 1.5, num_dofs)
    push = np.sin(2 * np.pi * 0.25 * t[:, None] + phase * 0.5)
    noise = np.random.default_rng(seed + 1).normal(0.0, 0.05, size=(t.size, num_dofs))
    tau_external = scale * (0.6 * push + noise)
    return q, dq, tau_external


class MockBackend:
    """In-process RobotBackend implementation."""

    def __init__(
        self,
        cfg: Optional[BackendConfig] = None,
        num_dofs: int = 7,
        control_hz: float = 1000.0,
        seed: int = 0,
        jitter_s: float = 2e-5,
        drop_rate: float = 0.0,
        move_time_s: float = 0.2,
        clock: Optional[Callable[[], float]] = None,
    ):
        self.cfg = cfg or BackendConfig(kind="mock")
        self._num_dofs = int(num_dofs)
        self._control_hz = float(control_hz)
        self._seed = int(seed)
        self._jitter_s = float(jitter_s)
        self._drop_rate = float(drop_rate)
        self._move_time_s = float(move_time_s)
        self._clock = clock or time.time

        self._connected = False
        self._policy: Optional[str] = None  # None | "teach" | "replay" | "move"
        self._policy_t0 = 0.0
        self._policy_deadline: Optional[float] = None
        self._replay_traj: Optional[np.ndarray] = None
        self._replay_dtraj: Optional[np.ndarray] = None
        self._q_current = np.resize(PANDA_REST_POSE, self._num_dofs).copy()
        self._urdf = _panda_urdf_text()

    # ---- lifecycle -----------------------------------------------------

    def connect(self) -> None:
        self._connected = True

    def close(self) -> None:
        if self._policy is not None:
            self.terminate_policy()
        self._connected = False

    def spec(self) -> RobotSpec:
        d = self._num_dofs
        spec = RobotSpec(
            backend="mock",
            robot_model="franka_panda(mock)",
            num_dofs=d,
            control_hz=self._control_hz,
            ee_link_name="panda_link8",
            urdf_text=self._urdf,
            home_pose=np.resize(PANDA_REST_POSE, d),
            default_Kq=np.resize(PANDA_DEFAULT_KQ, d),
            default_Kqd=np.resize(PANDA_DEFAULT_KQD, d),
            default_Kx=PANDA_DEFAULT_KX.copy(),
            default_Kxd=PANDA_DEFAULT_KXD.copy(),
            joint_pos_min=np.resize(PANDA_JOINT_MIN, d),
            joint_pos_max=np.resize(PANDA_JOINT_MAX, d),
            joint_vel_max=np.resize(PANDA_JOINT_VEL_MAX, d),
            polymetis_version="mock",
        )
        # Honoured here too, so the override path is exercised by the tests
        # rather than first running on the robot.
        return spec.with_limit_overrides(
            joint_pos_min=self.cfg.joint_pos_min,
            joint_pos_max=self.cfg.joint_pos_max,
            joint_vel_max=self.cfg.joint_vel_max,
        )

    # ---- read ----------------------------------------------------------

    def _state_at(self, wall: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._policy == "replay" and self._replay_traj is not None:
            i = int((wall - self._policy_t0) * self._control_hz)
            i = int(np.clip(i, 0, self._replay_traj.shape[0] - 1))
            q = self._replay_traj[i]
            dq = self._replay_dtraj[i]
            tau = np.zeros(self._num_dofs)
            return q, dq, tau
        if self._policy is None or self._policy == "move":
            # Idle means idle: a robot that drifts while no policy is running
            # would make the replay start-pose gate impossible to satisfy. A
            # non-blocking move is already at its target, so it reads the same.
            z = np.zeros(self._num_dofs)
            return self._q_current.copy(), z, z
        elapsed = wall - self._policy_t0
        q, dq, tau = synthetic_motion(
            np.array([elapsed]), self._num_dofs, self._seed, self._q_current
        )
        return q[0], dq[0], tau[0]

    def get_joint_positions(self) -> np.ndarray:
        return self._state_at(self._clock())[0]

    def get_telemetry(self) -> TelemetrySample:
        now = self._clock()
        q, dq, tau = self._state_at(now)
        g = self.get_gripper_sample()
        return TelemetrySample(
            timestamp=now,
            q=q,
            dq=dq,
            tau_external=tau,
            error_code=0,
            command_successful=True,
            gripper_width=None if g is None else g[1],
            gripper_is_grasped=None if g is None else g[2],
        )

    def get_gripper_sample(self):
        if not self.cfg.gripper_enabled:
            return None
        now = self._clock()
        width = 0.04 + 0.03 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * now))
        return (int(now * 1e9), float(width), bool(width < 0.05), False, 0)

    # ---- motion --------------------------------------------------------

    def go_home(
        self, time_to_go: Optional[float] = None, blocking: bool = True
    ) -> None:
        self.move_to_joint_positions(
            np.resize(PANDA_REST_POSE, self._num_dofs), time_to_go, blocking
        )

    def expected_move_time_s(
        self, positions: np.ndarray, time_to_go: Optional[float] = None
    ) -> float:
        return self._move_time_s if time_to_go is None else float(time_to_go)

    def move_to_joint_positions(
        self,
        positions: np.ndarray,
        time_to_go: Optional[float] = None,
        blocking: bool = True,
    ) -> None:
        if self._policy is not None:
            raise RuntimeError("a policy is already running: %s" % self._policy)
        self._q_current = np.asarray(positions, dtype=np.float64).copy()
        if blocking:
            return
        # Non-blocking: the arm is already where it was asked to go, but the
        # caller polls is_running_policy() to know when the move is done, so the
        # mock has to actually take time. Without this the worker's APPROACHING
        # state would be over before a test could ever observe it.
        self._policy = "move"
        self._policy_t0 = self._clock()
        self._policy_deadline = self._policy_t0 + (
            self._move_time_s if time_to_go is None else float(time_to_go)
        )

    def start_teaching(self, Kq: np.ndarray, Kqd: np.ndarray) -> None:
        if self._policy is not None:
            raise RuntimeError("a policy is already running: %s" % self._policy)
        self._policy = "teach"
        self._policy_t0 = self._clock()
        self._policy_deadline = None

    def start_replay(self, q_traj, dq_traj, Kq, Kqd, Kx, Kxd) -> None:
        if self._policy is not None:
            raise RuntimeError("a policy is already running: %s" % self._policy)
        q_traj = np.asarray(q_traj, dtype=np.float64)
        dq_traj = np.asarray(dq_traj, dtype=np.float64)
        if q_traj.shape != dq_traj.shape:
            raise ValueError(
                "q_traj %r and dq_traj %r must have the same shape"
                % (q_traj.shape, dq_traj.shape)
            )
        self._policy = "replay"
        self._policy_t0 = self._clock()
        self._replay_traj = q_traj
        self._replay_dtraj = dq_traj
        self._policy_deadline = self._policy_t0 + q_traj.shape[0] / self._control_hz

    def is_running_policy(self) -> bool:
        if self._policy is None:
            return False
        if self._policy_deadline is not None and self._clock() >= self._policy_deadline:
            return False
        return True

    # ---- log -----------------------------------------------------------

    def terminate_policy(self) -> EpisodeBuffer:
        if self._policy is None:
            return EpisodeBuffer.empty(self._num_dofs)
        if self._policy == "move":
            # A point-to-point move is not a recorded episode, and its log must
            # not move _q_current: the arm is already at the target and the
            # replay start-pose gate is about to check exactly that.
            self._policy = None
            self._policy_deadline = None
            return EpisodeBuffer.empty(self._num_dofs)
        t0, kind = self._policy_t0, self._policy
        t1 = self._clock()
        if self._policy_deadline is not None:
            t1 = min(t1, self._policy_deadline)
        traj, dtraj = self._replay_traj, self._replay_dtraj
        self._policy = None
        self._replay_traj = self._replay_dtraj = None
        self._policy_deadline = None

        buf = self._make_log(t0, max(t1 - t0, 0.0), kind, traj, dtraj)
        if buf.n:
            self._q_current = buf.q[-1].copy()
        return buf

    def _make_log(self, t0, duration, kind, traj, dtraj) -> EpisodeBuffer:
        """Build the log the server would have collected over `duration`."""
        n_nominal = int(duration * self._control_hz)
        if n_nominal < 2:
            return EpisodeBuffer.empty(self._num_dofs)

        rng = np.random.default_rng(self._seed + 7)
        idx = np.arange(n_nominal)
        if self._drop_rate > 0:
            keep = rng.random(n_nominal) >= self._drop_rate
            keep[0] = keep[-1] = True
            idx = idx[keep]

        t_rel = idx / self._control_hz
        if self._jitter_s > 0:
            t_rel = t_rel + rng.normal(0.0, self._jitter_s, idx.size)
            t_rel = np.maximum.accumulate(t_rel)  # timestamps never go backwards
        timestamp_ns = np.round((t0 + t_rel) * 1e9).astype(np.int64)
        n = idx.size

        if kind == "replay" and traj is not None:
            j = np.clip(idx, 0, traj.shape[0] - 1)
            q = traj[j] + rng.normal(0.0, 2e-4, (n, self._num_dofs))
            dq = dtraj[j]
            tau_external = rng.normal(0.0, 0.15, (n, self._num_dofs))
        else:
            q, dq, tau_external = synthetic_motion(
                t_rel, self._num_dofs, self._seed, self._q_current
            )

        # Rough inverse-dynamics stand-in: enough structure that plots and
        # filters have something to bite on. Not physical.
        tau_computed = 0.5 * dq + 0.05 * q
        tau_measured = tau_computed + tau_external
        return EpisodeBuffer(
            timestamp_ns=timestamp_ns,
            q=q,
            dq=dq,
            tau_measured=tau_measured,
            tau_external=tau_external,
            tau_computed=tau_computed,
            tau_desired=tau_computed,
            tau_computed_prev=tau_computed,
            tau_safened_prev=tau_computed,
            latency_ms=np.abs(rng.normal(0.25, 0.05, n)),
            command_successful=np.ones(n, dtype=bool),
            error_code=np.zeros(n, dtype=np.int32),
        )
