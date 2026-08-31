"""Configuration objects for the kinesthetic teaching tool.

Nothing in this module imports polymetis or torch, so config can be loaded from
any interpreter (tests, analysis notebooks, the WebUI process).

Config is plain dataclasses with an optional YAML overlay:

    cfg = Config.load("configs/real.yaml")     # file overrides defaults
    cfg = Config.default()
"""
from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "BackendConfig",
    "TeachConfig",
    "ReplayConfig",
    "ProcessConfig",
    "Config",
]


@dataclass
class BackendConfig:
    """Which robot to talk to.

    `mock` runs in this process; `real` dials a polymetis controller server
    (implementation_plan.md 3.1).
    """

    kind: str = "mock"  # mock | real
    ip_address: str = "localhost"
    port: int = 50051
    enforce_version: bool = False

    # The Robotiq gripper is a separate polymetis server on an independent
    # clock, with no streaming log -- it has to be polled (plan 2.8).
    gripper_enabled: bool = False
    gripper_ip: str = "localhost"
    gripper_port: int = 50052
    gripper_poll_hz: float = 50.0

    # Joint limits the server will not tell us.
    #
    # `RobotStateMetadata` carries dof / hz / URDF / default gains but *no*
    # limits, so `robot_model.get_joint_angle_limits()` reads them out of the
    # URDF. On the lab robot that URDF is `franka_panda/panda_arm.urdf` while
    # the hardware is an FR3, and the two disagree -- worst on joint 6, whose
    # real lower limit is 0.5445 rad against the URDF's -0.0175, i.e. 32 deg of
    # range that does not exist. The server's own C++ safety layer does use FR3
    # numbers, but it only pushes back once the arm is already heading into the
    # limit, which is the situation the replay pre-flight check exists to avoid.
    #
    # None keeps whatever the URDF says. Setting these does not relax anything:
    # the replay gate takes them as-is, so a wrong entry here is a real hazard.
    joint_pos_min: Optional[List[float]] = None
    joint_pos_max: Optional[List[float]] = None
    joint_vel_max: Optional[List[float]] = None

    def __post_init__(self) -> None:
        if self.kind not in ("mock", "real"):
            raise ValueError(
                "backend.kind must be mock or real, got %r" % (self.kind,)
            )
        given = {
            name: getattr(self, name)
            for name in ("joint_pos_min", "joint_pos_max", "joint_vel_max")
            if getattr(self, name) is not None
        }
        sizes = {name: len(v) for name, v in given.items()}
        if len(set(sizes.values())) > 1:
            raise ValueError(
                "backend joint limit overrides disagree on length: %s" % (sizes,)
            )
        lo, hi = self.joint_pos_min, self.joint_pos_max
        if (lo is None) != (hi is None):
            raise ValueError(
                "backend.joint_pos_min and joint_pos_max must be set together"
            )
        if lo is not None and hi is not None:
            bad = [j for j, (a, b) in enumerate(zip(lo, hi)) if not a < b]
            if bad:
                raise ValueError(
                    "backend.joint_pos_min must be below joint_pos_max; "
                    "joint(s) %s are not" % bad
                )
        if self.joint_vel_max is not None and any(v <= 0 for v in self.joint_vel_max):
            raise ValueError("backend.joint_vel_max entries must be positive")


@dataclass
class TeachConfig:
    """Near-zero joint stiffness hand guiding (baseline 6, plan 2.3)."""

    Kq: Optional[List[float]] = None  # None -> zeros(dof)
    Kqd: Optional[List[float]] = None  # None -> robot default_Kqd * Kqd_scale
    Kqd_scale: float = 1.0

    # INVARIANT 4 (plan 6). start_joint_impedance(adaptive=True) also applies
    # Kx_default = [750, 750, 750, 15, 15, 15], which fights the human's hand.
    adaptive: bool = False

    # The server ring buffer holds 300 s at 1 kHz and then silently overwrites
    # the head of the episode (plan 2.4). Stop well before that.
    max_duration_s: float = 280.0

    # Discarded from the head of the log: the moments right after the policy is
    # sent, before the operator has hold of the arm.
    settle_s: float = 0.0


@dataclass
class ReplayConfig:
    """Server-side trajectory execution (plan 2.5)."""

    Kq_scale: float = 0.3  # fraction of the robot's default_Kq
    Kqd_scale: float = 1.0

    # JointTrajectoryExecutor takes Kx/Kxd too, so the plan 2.3 caution applies
    # here as well: leave these zero unless deliberately enabled.
    use_cartesian_stiffness: bool = False
    Kx_scale: float = 0.0
    Kxd_scale: float = 0.0

    # Replay duration = teaching duration * time_scale. >1 is slower than the
    # demonstration, which is what baseline 20 asks for.
    time_scale: float = 2.0

    # What the start-pose gate is protecting against is a torque transient: the
    # executor pulls towards its first waypoint with `Kq * error` the instant it
    # takes over. An angle threshold is a poor stand-in because the joints do
    # not share a stiffness -- at Kq_scale 0.5 the same 0.05 rad is 1.25 Nm on
    # joint 3 and 0.25 Nm on joint 7, so the softest joint is held to five times
    # the standard. It also silently changes meaning whenever Kq_scale changes.
    #
    # `start_pose_tol_nm` is the real check; `start_pose_tol_rad` stays as a
    # gross cap for a joint so far off that the wrong episode is the likelier
    # explanation. Set the torque tolerance to None to get the old angle-only
    # behaviour.
    #
    # There is a floor this must clear. The approach move uses the servo's own
    # `Kq_default`, so it settles where joint friction balances it, leaving
    # `friction / Kq_default` of angle -- and the transient the replay then
    # applies is that angle times *its* stiffness, i.e. `friction * Kq_scale`.
    # On the lab FR3 friction peaks near 1.13 Nm (joint 4, ten settled poses),
    # so at Kq_scale 1.0 nothing below ~1.2 Nm is reachable. 0.8 was set from
    # Kq_scale 0.5 alone and made the gate unsatisfiable at 0.8 and above:
    # it refused replays whose arm was 0.02 rad from the waypoint.
    start_pose_tol_nm: Optional[float] = 2.0
    start_pose_tol_rad: float = 0.25
    approach_time_s: float = 4.0
    source: str = "q_filtered"  # which processed array to replay


@dataclass
class ProcessConfig:
    """Offline filtering (baseline 17, plan M2)."""

    filter: str = "butterworth"  # butterworth | savgol
    cutoff_hz: float = 10.0
    order: int = 4
    sweep_cutoffs: List[float] = field(
        default_factory=lambda: [5.0, 10.0, 15.0, 20.0]
    )
    savgol_window_s: float = 0.05
    savgol_polyorder: int = 3
    tau_cutoff_hz: float = 20.0

    def __post_init__(self) -> None:
        if self.filter not in ("butterworth", "savgol"):
            raise ValueError(
                "process.filter must be butterworth|savgol, got %r" % (self.filter,)
            )


@dataclass
class Config:
    data_root: str = "data/episodes"
    backend: BackendConfig = field(default_factory=BackendConfig)
    teach: TeachConfig = field(default_factory=TeachConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    process: ProcessConfig = field(default_factory=ProcessConfig)

    def __post_init__(self) -> None:
        if self.teach.adaptive:
            raise ValueError(
                "teach.adaptive=True applies Kx_default on top of the joint "
                "impedance controller and makes the arm too stiff to hand "
                "guide (implementation_plan.md 2.3, invariant 4). If you are "
                "deliberately characterising that path, do it in a throwaway "
                "script, not through Config."
            )

    # ---- serialisation -------------------------------------------------

    @classmethod
    def default(cls) -> "Config":
        return cls()

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Config":
        return _build(cls, data or {})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def load(cls, path: Optional[str]) -> "Config":
        """Load a YAML file over the defaults. `None` returns the defaults."""
        if path is None:
            return cls.default()
        import yaml  # local: keeps pyyaml optional for pure-default use

        text = Path(path).read_text()
        return cls.from_dict(yaml.safe_load(text))

    def save(self, path: str) -> None:
        import yaml

        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


def _build(cls, data: Dict[str, Any]):
    """Recursively construct a nested dataclass, rejecting unknown keys.

    An unknown key is almost always a typo in a hand-written YAML file, and a
    silently ignored `adaptive: false` would defeat invariant 4.
    """
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(
            "unknown config key(s) for %s: %s" % (cls.__name__, sorted(unknown))
        )
    kwargs: Dict[str, Any] = {}
    for name, value in data.items():
        # `from __future__ import annotations` makes field types strings, so
        # resolve against the dataclass defaults instead of the annotation.
        f = known[name]
        default = f.default if f.default_factory is MISSING else f.default_factory()
        if is_dataclass(default) and isinstance(value, dict):
            kwargs[name] = _build(type(default), value)
        else:
            kwargs[name] = value
    return cls(**kwargs)
