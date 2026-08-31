"""The backend seam: the only place that is allowed to know about polymetis.

INVARIANT 7 (implementation_plan.md 6). `polymetis_pb2.RobotState` must not
escape this package. Backends convert protobuf to plain numpy *here*, at the
boundary, and everything downstream (record, process, replay, validate, webui)
sees only the dataclasses defined below.

That conversion has to be written either way in order to save npz, so putting
it at the boundary costs nothing today and keeps the core independent of the
transport -- which is what makes a second backend, a unit test without a robot,
and an eventual public release cheap.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # py3.8
    from typing import Protocol, runtime_checkable
except ImportError:  # pragma: no cover
    from typing_extensions import Protocol, runtime_checkable  # type: ignore

__all__ = [
    "RobotSpec",
    "EpisodeBuffer",
    "GripperBuffer",
    "TelemetrySample",
    "RobotBackend",
    "ROBOT_ARRAY_FIELDS",
    "GRIPPER_ARRAY_FIELDS",
]


# ---------------------------------------------------------------------------
# Robot description
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RobotSpec:
    """Everything the core needs to know about the robot it is talking to.

    Read from the server's own metadata, never hardcoded (invariant 3, plan
    2.6). The server is the only authority on how many joints it has and how
    fast it runs: a URDF that includes the gripper's joints, a different arm,
    or a server started at another rate all change these numbers, and code that
    assumed 7 and 1000 would corrupt episodes rather than fail.
    """

    backend: str  # mock | real
    robot_model: str
    num_dofs: int
    control_hz: float
    ee_link_name: Optional[str]
    urdf_text: Optional[str]  # the URDF itself, so an episode is self-contained
    home_pose: np.ndarray
    default_Kq: np.ndarray
    default_Kqd: np.ndarray
    default_Kx: np.ndarray
    default_Kxd: np.ndarray
    joint_pos_min: Optional[np.ndarray] = None
    joint_pos_max: Optional[np.ndarray] = None
    joint_vel_max: Optional[np.ndarray] = None
    polymetis_version: str = ""

    #: Where the three limit arrays above came from. "urdf" is the server's own
    #: model; "config" means a `backend.joint_*` override replaced part of it,
    #: which the lab FR3 needs because its server runs a Panda URDF.
    joint_limits_source: str = "urdf"

    def with_limit_overrides(
        self,
        joint_pos_min: Optional[Sequence[float]] = None,
        joint_pos_max: Optional[Sequence[float]] = None,
        joint_vel_max: Optional[Sequence[float]] = None,
    ) -> "RobotSpec":
        """A copy with hand-supplied joint limits in place of the URDF's.

        Invariant 3 says not to hardcode DOF or rate, and this does not: the
        override is rejected unless it matches the DOF the *server* reported.
        The limits themselves are a different matter -- the server does not
        send them at all, so there is no authority to defer to.
        """
        overrides = {
            "joint_pos_min": joint_pos_min,
            "joint_pos_max": joint_pos_max,
            "joint_vel_max": joint_vel_max,
        }
        given = {k: v for k, v in overrides.items() if v is not None}
        if not given:
            return self
        patch: Dict[str, Any] = {}
        for name, value in given.items():
            arr = np.asarray(value, dtype=np.float64).ravel()
            if arr.shape != (self.num_dofs,):
                raise ValueError(
                    "%s has %d entries but the server reports %d DOF"
                    % (name, arr.size, self.num_dofs)
                )
            patch[name] = arr
        return replace(self, joint_limits_source="config", **patch)

    def to_metadata(self) -> Dict[str, Any]:
        """The subset that belongs in an episode's metadata.json.

        The URDF text is excluded: it is written next to the episode as
        `robot.urdf` instead of being inlined into JSON.

        The joint limits are included even though they are derivable from that
        URDF, because on the lab robot they deliberately are not: an episode
        has to record which limits its safety checks were made against.
        """
        return {
            "backend": self.backend,
            "robot_model": self.robot_model,
            "num_dofs": int(self.num_dofs),
            "control_hz": float(self.control_hz),
            "ee_link_name": self.ee_link_name,
            "home_pose": _list(self.home_pose),
            "default_Kq": _list(self.default_Kq),
            "default_Kqd": _list(self.default_Kqd),
            "default_Kx": _list(self.default_Kx),
            "default_Kxd": _list(self.default_Kxd),
            "joint_pos_min": _list(self.joint_pos_min),
            "joint_pos_max": _list(self.joint_pos_max),
            "joint_vel_max": _list(self.joint_vel_max),
            "joint_limits_source": self.joint_limits_source,
            "polymetis_version": self.polymetis_version,
        }


def _list(a: Optional[np.ndarray]) -> Optional[List[float]]:
    return None if a is None else [float(v) for v in np.asarray(a).ravel()]


# ---------------------------------------------------------------------------
# Logged data
# ---------------------------------------------------------------------------

#: (name, dtype, per-joint?) for every array in a robot log.
#: Mirrors polymetis_pb2.RobotState one-for-one so nothing is silently dropped
#: -- the droid wrapper losing `motor_torques_external` is exactly the failure
#: this list exists to prevent (plan 1.2).
ROBOT_ARRAY_FIELDS: Tuple[Tuple[str, Any, bool], ...] = (
    ("timestamp_ns", np.int64, False),
    ("q", np.float64, True),
    ("dq", np.float64, True),
    ("tau_measured", np.float64, True),
    ("tau_external", np.float64, True),
    ("tau_computed", np.float64, True),
    ("tau_desired", np.float64, True),
    ("tau_computed_prev", np.float64, True),
    ("tau_safened_prev", np.float64, True),
    ("latency_ms", np.float64, False),
    ("command_successful", np.bool_, False),
    ("error_code", np.int32, False),
)

GRIPPER_ARRAY_FIELDS: Tuple[Tuple[str, Any, bool], ...] = (
    ("timestamp_ns", np.int64, False),
    ("width", np.float64, False),
    ("is_grasped", np.bool_, False),
    ("is_moving", np.bool_, False),
    ("error_code", np.int32, False),
)


@dataclass
class EpisodeBuffer:
    """One contiguous run of robot states, as plain arrays.

    Time is int64 nanoseconds since the epoch, which is exactly what the server
    sends (`Timestamp.seconds` + `.nanos`) and is lossless. Absolute epoch
    seconds in float64 would quantise to ~0.5 us at present dates -- small
    against a 1 ms period, but this log exists partly to *measure* that period,
    so the analysis should not carry avoidable noise of its own.

    `timestamp` (float seconds) and `t` (relative seconds) are derived.
    """

    timestamp_ns: np.ndarray  # (N,) int64
    q: np.ndarray  # (N, D)
    dq: np.ndarray
    tau_measured: np.ndarray
    tau_external: np.ndarray
    tau_computed: np.ndarray
    tau_desired: np.ndarray
    tau_computed_prev: np.ndarray
    tau_safened_prev: np.ndarray
    latency_ms: np.ndarray  # (N,)
    command_successful: np.ndarray  # (N,) bool
    error_code: np.ndarray  # (N,) int32

    def __post_init__(self) -> None:
        for name, dtype, per_joint in ROBOT_ARRAY_FIELDS:
            arr = np.ascontiguousarray(getattr(self, name), dtype=dtype)
            want = 2 if per_joint else 1
            if arr.ndim != want:
                raise ValueError(
                    "EpisodeBuffer.%s must be %dD, got shape %r"
                    % (name, want, arr.shape)
                )
            setattr(self, name, arr)
        n = self.timestamp_ns.shape[0]
        for name, _, _ in ROBOT_ARRAY_FIELDS:
            got = getattr(self, name).shape[0]
            if got != n:
                raise ValueError(
                    "EpisodeBuffer length mismatch: timestamp_ns=%d but %s=%d"
                    % (n, name, got)
                )
        dofs = {getattr(self, nm).shape[1] for nm, _, pj in ROBOT_ARRAY_FIELDS if pj}
        if len(dofs) > 1:
            raise ValueError("inconsistent DOF across fields: %s" % sorted(dofs))

    # ---- derived -------------------------------------------------------

    @property
    def n(self) -> int:
        return int(self.timestamp_ns.shape[0])

    @property
    def num_dofs(self) -> int:
        return int(self.q.shape[1])

    @property
    def timestamp(self) -> np.ndarray:
        """Absolute epoch seconds, for lining up against other clocks."""
        return self.timestamp_ns * 1e-9

    @property
    def dt(self) -> np.ndarray:
        """Intervals between consecutive samples, in seconds."""
        return np.diff(self.timestamp_ns) * 1e-9

    @property
    def duration(self) -> float:
        if self.n < 2:
            return 0.0
        return float(self.timestamp_ns[-1] - self.timestamp_ns[0]) * 1e-9

    @property
    def effective_hz(self) -> float:
        """Samples actually delivered per second.

        (N-1)/duration rather than N/duration: N samples span N-1 intervals,
        and the difference matters when comparing against a nominal 1000 Hz.
        """
        d = self.duration
        return 0.0 if d <= 0 else float(self.n - 1) / d

    @property
    def t(self) -> np.ndarray:
        """Timestamps relative to the first sample."""
        if self.n == 0:
            return np.zeros(0)
        return (self.timestamp_ns - self.timestamp_ns[0]) * 1e-9

    # ---- transforms ----------------------------------------------------

    def slice(self, start: int = 0, stop: Optional[int] = None) -> "EpisodeBuffer":
        sl = slice(start, stop)
        return EpisodeBuffer(**{nm: getattr(self, nm)[sl] for nm, _, _ in ROBOT_ARRAY_FIELDS})

    def trim_seconds(self, head: float = 0.0, tail: float = 0.0) -> "EpisodeBuffer":
        if self.n == 0 or (head <= 0 and tail <= 0):
            return self
        t = self.t
        keep = (t >= head) & (t <= max(t[-1] - tail, head))
        idx = np.flatnonzero(keep)
        if idx.size == 0:
            raise ValueError("trim_seconds removed every sample")
        return self.slice(int(idx[0]), int(idx[-1]) + 1)

    # ---- io ------------------------------------------------------------

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {nm: getattr(self, nm) for nm, _, _ in ROBOT_ARRAY_FIELDS}

    @classmethod
    def from_dict(cls, d: Dict[str, np.ndarray]) -> "EpisodeBuffer":
        missing = [nm for nm, _, _ in ROBOT_ARRAY_FIELDS if nm not in d]
        if missing:
            raise ValueError("missing robot log field(s): %s" % missing)
        return cls(**{nm: np.asarray(d[nm]) for nm, _, _ in ROBOT_ARRAY_FIELDS})

    @classmethod
    def empty(cls, num_dofs: int) -> "EpisodeBuffer":
        z1 = np.zeros(0)
        z2 = np.zeros((0, num_dofs))
        return cls(**{
            nm: (z2 if pj else z1).astype(dt)
            for nm, dt, pj in ROBOT_ARRAY_FIELDS
        })


@dataclass
class GripperBuffer:
    """Polled Robotiq states. Independent clock from the arm (plan 2.8)."""

    timestamp_ns: np.ndarray
    width: np.ndarray
    is_grasped: np.ndarray
    is_moving: np.ndarray
    error_code: np.ndarray

    def __post_init__(self) -> None:
        for name, dtype, _ in GRIPPER_ARRAY_FIELDS:
            setattr(self, name, np.ascontiguousarray(getattr(self, name), dtype=dtype))
        n = self.timestamp_ns.shape[0]
        for name, _, _ in GRIPPER_ARRAY_FIELDS:
            if getattr(self, name).shape[0] != n:
                raise ValueError("GripperBuffer length mismatch on %s" % name)

    @property
    def n(self) -> int:
        # timestamp_ns, not the derived `timestamp`: the latter allocates a
        # whole float array just to be asked for its length.
        return int(self.timestamp_ns.shape[0])

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {nm: getattr(self, nm) for nm, _, _ in GRIPPER_ARRAY_FIELDS}

    @classmethod
    def from_dict(cls, d: Dict[str, np.ndarray]) -> "GripperBuffer":
        return cls(**{nm: np.asarray(d[nm]) for nm, _, _ in GRIPPER_ARRAY_FIELDS})

    @property
    def timestamp(self) -> np.ndarray:
        return self.timestamp_ns * 1e-9

    @classmethod
    def empty(cls) -> "GripperBuffer":
        return cls(
            **{nm: np.zeros(0, dtype=dt) for nm, dt, _ in GRIPPER_ARRAY_FIELDS}
        )


@dataclass
class TelemetrySample:
    """A single low-rate sample for display.

    Display only. The 1 kHz record never travels through this path -- it is
    fetched in one batch from the server buffer after the session ends
    (invariant 1, baseline 9).
    """

    timestamp: float
    q: np.ndarray
    dq: np.ndarray
    tau_external: np.ndarray
    error_code: int
    command_successful: bool
    #: Server-reported latency of the previous control step, in ms. The
    #: autonomous tools quote it beside their measurements: a pose sampled
    #: while the control loop was struggling is not a pose we should fit.
    #:
    #: The same quantity the logs call `EpisodeBuffer.latency_ms`. Kept under
    #: two names on purpose: `latency_ms` is an array key inside every
    #: `robot_raw.npz` already written, and those files are immutable
    #: (invariant 2), so renaming it would strand the recorded episodes.
    controller_latency_ms: float = 0.0
    gripper_width: Optional[float] = None
    gripper_is_grasped: Optional[bool] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "timestamp": float(self.timestamp),
            "q": _list(self.q),
            "dq": _list(self.dq),
            "tau_external": _list(self.tau_external),
            "error_code": int(self.error_code),
            "command_successful": bool(self.command_successful),
            "controller_latency_ms": float(self.controller_latency_ms),
            "gripper_width": self.gripper_width,
            "gripper_is_grasped": self.gripper_is_grasped,
        }


# ---------------------------------------------------------------------------
# The protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class RobotBackend(Protocol):
    """What the core requires of a robot.

    Deliberately small, and deliberately shaped around *server-side* work:
    send a policy, let the server's RT loop run it, collect the log afterwards.
    A per-step command API would not survive 1 kHz over gRPC (baseline 9).
    """

    def connect(self) -> None:
        ...

    def close(self) -> None:
        ...

    def spec(self) -> RobotSpec:
        ...

    # ---- read ----
    def get_joint_positions(self) -> np.ndarray:
        ...

    def get_telemetry(self) -> TelemetrySample:
        ...

    # ---- motion ----
    def go_home(
        self, time_to_go: Optional[float] = None, blocking: bool = True
    ) -> None:
        ...

    def expected_move_time_s(
        self, positions: np.ndarray, time_to_go: Optional[float] = None
    ) -> float:
        """How long a move to `positions` should take, in seconds.

        Only ever used as the basis for a watchdog. A blocking move raises if it
        goes wrong; a non-blocking one just never reports finishing, so whoever
        polls it needs a bound to give up at.
        """

    def move_to_joint_positions(
        self,
        positions: np.ndarray,
        time_to_go: Optional[float] = None,
        blocking: bool = True,
    ) -> None:
        """Move to `positions`.

        `blocking=False` returns as soon as the policy is accepted, leaving the
        caller to poll `is_running_policy()`. The WebUI worker needs that: a
        blocking move parks the one thread allowed to touch the robot, and the
        stop button is dead for exactly as long as the arm is moving on its own.
        """

    def start_teaching(self, Kq: np.ndarray, Kqd: np.ndarray) -> None:
        """Send a zero-stiffness joint impedance policy and return immediately."""

    def start_replay(
        self,
        q_traj: np.ndarray,
        dq_traj: np.ndarray,
        Kq: np.ndarray,
        Kqd: np.ndarray,
        Kx: np.ndarray,
        Kxd: np.ndarray,
    ) -> None:
        """Hand a whole trajectory to the server and return immediately."""

    # ---- lifecycle ----
    def is_running_policy(self) -> bool:
        ...

    def terminate_policy(self) -> EpisodeBuffer:
        """Stop the running policy and return its server-side log."""

    # ---- gripper ----
    def get_gripper_sample(self) -> Optional[Tuple[float, float, bool, bool, int]]:
        """(timestamp, width, is_grasped, is_moving, error_code) or None."""
