"""The real robot, over polymetis gRPC.

This module is the *only* one in the project that imports polymetis or touches
`polymetis_pb2.RobotState` (invariant 7).

Absolute imports are the default in Python 3, so `import polymetis` below
resolves to the installed package, not to this file.
"""
from __future__ import annotations

import logging
import re
from typing import Any, List, Optional

import numpy as np

from ..config import BackendConfig
from .base import EpisodeBuffer, RobotSpec, TelemetrySample

log = logging.getLogger(__name__)

__all__ = ["PolymetisBackend", "states_to_buffer"]


def _ns(state: Any) -> int:
    """Exact epoch nanoseconds, straight out of the protobuf Timestamp."""
    return state.timestamp.seconds * 1_000_000_000 + state.timestamp.nanos


def _t(state: Any) -> float:
    return _ns(state) * 1e-9


def states_to_buffer(states: List[Any]) -> EpisodeBuffer:
    """Convert `List[polymetis_pb2.RobotState]` to plain numpy.

    This is the boundary named in invariant 7: protobuf goes in, nothing but
    ndarrays comes out. Every field of RobotState is carried across -- dropping
    `motor_torques_external` here is precisely the mistake that made the droid
    wrapper unusable for us (plan 1.2).
    """
    n = len(states)
    if n == 0:
        return EpisodeBuffer.empty(0)
    return EpisodeBuffer(
        timestamp_ns=np.array([_ns(s) for s in states], dtype=np.int64),
        q=np.array([s.joint_positions for s in states], dtype=np.float64),
        dq=np.array([s.joint_velocities for s in states], dtype=np.float64),
        tau_measured=np.array([s.motor_torques_measured for s in states], dtype=np.float64),
        tau_external=np.array([s.motor_torques_external for s in states], dtype=np.float64),
        tau_computed=np.array([s.joint_torques_computed for s in states], dtype=np.float64),
        tau_desired=np.array([s.motor_torques_desired for s in states], dtype=np.float64),
        tau_computed_prev=np.array([s.prev_joint_torques_computed for s in states], dtype=np.float64),
        tau_safened_prev=np.array([s.prev_joint_torques_computed_safened for s in states], dtype=np.float64),
        latency_ms=np.array([s.prev_controller_latency_ms for s in states], dtype=np.float64),
        command_successful=np.array([s.prev_command_successful for s in states], dtype=bool),
        error_code=np.array([s.error_code for s in states], dtype=np.int32),
    )


def _robot_name_from_urdf(urdf_text: Optional[str]) -> str:
    if not urdf_text:
        return "unknown"
    m = re.search(r"<robot[^>]*\bname\s*=\s*[\"']([^\"']+)", urdf_text)
    return m.group(1) if m else "unknown"


class PolymetisBackend:
    """RobotBackend over a polymetis controller server."""

    def __init__(self, cfg: BackendConfig):
        if cfg.kind != "real":
            raise ValueError(
                "PolymetisBackend is for kind=real; use MockBackend for mock"
            )
        self.cfg = cfg
        self.robot = None  # polymetis.RobotInterface
        self.gripper = None  # polymetis.GripperInterface
        self._spec: Optional[RobotSpec] = None

    # ---- lifecycle -----------------------------------------------------

    def connect(self) -> None:
        import polymetis  # imported lazily so the core can be used without it

        self.robot = polymetis.RobotInterface(
            ip_address=self.cfg.ip_address,
            port=self.cfg.port,
            enforce_version=self.cfg.enforce_version,
        )
        if self.cfg.gripper_enabled:
            self.gripper = polymetis.GripperInterface(
                ip_address=self.cfg.gripper_ip, port=self.cfg.gripper_port
            )
        self._spec = None

    def close(self) -> None:
        if self.robot is not None and self.robot.is_running_policy():
            self.terminate_policy()
        self.robot = None
        self.gripper = None

    def _require(self):
        if self.robot is None:
            raise RuntimeError("backend is not connected; call connect() first")
        return self.robot

    # ---- description ---------------------------------------------------

    def spec(self) -> RobotSpec:
        if self._spec is not None:
            return self._spec
        robot = self._require()
        md = robot.metadata
        urdf = md.urdf_file

        lo = hi = vmax = None
        try:
            limits = robot.robot_model.get_joint_angle_limits()
            lo = np.asarray(limits[0], dtype=np.float64)
            hi = np.asarray(limits[1], dtype=np.float64)
            vmax = np.asarray(robot.robot_model.get_joint_velocity_limits(), dtype=np.float64)
        except Exception:
            # A URDF without explicit limits is not fatal; validate.py just
            # skips the limit-proximity check.
            pass

        spec = RobotSpec(
            backend=self.cfg.kind,
            robot_model=_robot_name_from_urdf(urdf),
            num_dofs=int(md.dof),
            control_hz=float(md.hz),
            ee_link_name=md.ee_link_name or None,
            urdf_text=urdf,
            home_pose=np.asarray(md.rest_pose, dtype=np.float64),
            default_Kq=np.asarray(md.default_Kq, dtype=np.float64),
            default_Kqd=np.asarray(md.default_Kqd, dtype=np.float64),
            default_Kx=np.asarray(md.default_Kx, dtype=np.float64),
            default_Kxd=np.asarray(md.default_Kxd, dtype=np.float64),
            joint_pos_min=lo,
            joint_pos_max=hi,
            joint_vel_max=vmax,
            polymetis_version=str(md.polymetis_version),
        )
        # The metadata above is authoritative; the joint limits are not, since
        # the server never sends them and `lo`/`hi` came from its URDF.
        self._spec = spec.with_limit_overrides(
            joint_pos_min=self.cfg.joint_pos_min,
            joint_pos_max=self.cfg.joint_pos_max,
            joint_vel_max=self.cfg.joint_vel_max,
        )
        return self._spec

    # ---- read ----------------------------------------------------------

    def get_joint_positions(self) -> np.ndarray:
        return np.asarray(self._require().get_robot_state().joint_positions, dtype=np.float64)

    def get_telemetry(self) -> TelemetrySample:
        s = self._require().get_robot_state()
        g = self.get_gripper_sample()
        return TelemetrySample(
            timestamp=_t(s),
            q=np.asarray(s.joint_positions, dtype=np.float64),
            dq=np.asarray(s.joint_velocities, dtype=np.float64),
            tau_external=np.asarray(s.motor_torques_external, dtype=np.float64),
            error_code=int(s.error_code),
            command_successful=bool(s.prev_command_successful),
            controller_latency_ms=float(s.prev_controller_latency_ms),
            gripper_width=None if g is None else g[1],
            gripper_is_grasped=None if g is None else g[2],
        )

    def get_gripper_sample(self):
        """(timestamp_ns, width, is_grasped, is_moving, error_code)."""
        if self.gripper is None:
            return None
        s = self.gripper.get_state()
        return (
            s.timestamp.seconds * 1_000_000_000 + s.timestamp.nanos,
            float(s.width),
            bool(s.is_grasped),
            bool(s.is_moving),
            int(s.error_code),
        )

    # ---- motion --------------------------------------------------------

    def go_home(
        self, time_to_go: Optional[float] = None, blocking: bool = True
    ) -> None:
        import torch

        robot = self._require()
        if time_to_go is None:
            robot.go_home(blocking=blocking)
        else:
            robot.move_to_joint_positions(
                torch.Tensor(robot.home_pose), time_to_go=time_to_go, blocking=blocking
            )

    def expected_move_time_s(
        self, positions: np.ndarray, time_to_go: Optional[float] = None
    ) -> float:
        if time_to_go is not None:
            return float(time_to_go)
        import torch

        robot = self._require()
        target = torch.Tensor(np.asarray(positions, dtype=np.float32))
        try:
            # polymetis' own estimate, so the watchdog cannot disagree with the
            # planner about how long the move it just planned should take.
            return float(robot._adaptive_time_to_go(target - robot.get_joint_positions()))
        except Exception:
            # Same rule from public inputs: mean velocity is an eighth of the
            # joint velocity limit, floored at the interface default.
            log.debug("falling back to a locally computed move time")
            spec = self.spec()
            dq = np.abs(np.asarray(positions, dtype=np.float64) - self.get_joint_positions())
            vmax = spec.joint_vel_max
            est = 0.0 if vmax is None else float(np.max(dq / np.asarray(vmax) * 8.0))
            return max(est, float(getattr(robot, "time_to_go_default", 1.0)))

    def move_to_joint_positions(
        self,
        positions: np.ndarray,
        time_to_go: Optional[float] = None,
        blocking: bool = True,
    ) -> None:
        import torch

        # `RobotInterface.move_to_joint_positions` forwards **kwargs straight to
        # `send_torch_policy`, so blocking=False is a supported path, not a
        # trick. Note it plans a JointTrajectoryExecutor with Kx_default: an
        # approach move is not teaching, so invariant 4 does not apply here.
        self._require().move_to_joint_positions(
            torch.Tensor(np.asarray(positions, dtype=np.float32)),
            time_to_go=time_to_go,
            blocking=blocking,
        )

    def start_teaching(self, Kq: np.ndarray, Kqd: np.ndarray) -> None:
        """Zero-stiffness joint impedance, with no Cartesian term.

        `RobotInterface.start_joint_impedance()` is deliberately not used:
        its default `adaptive=True` builds a HybridJointImpedanceControl that
        also applies `Kx_default = [750, 750, 750, 15, 15, 15]`, which the
        operator would feel as a rigid arm (plan 2.3, invariant 4). Building
        `JointImpedanceControl` here makes it structurally impossible for a
        Cartesian stiffness to leak into a teaching session.
        """
        import torch
        import torchcontrol as toco

        robot = self._require()
        policy = toco.policies.JointImpedanceControl(
            joint_pos_current=robot.get_joint_positions(),
            Kp=torch.Tensor(np.asarray(Kq, dtype=np.float32)),
            Kd=torch.Tensor(np.asarray(Kqd, dtype=np.float32)),
            robot_model=robot.robot_model,
            ignore_gravity=robot.use_grav_comp,
        )
        robot.send_torch_policy(policy, blocking=False)

    def start_replay(self, q_traj, dq_traj, Kq, Kqd, Kx, Kxd) -> None:
        """Hand the whole trajectory to the server's RT loop.

        A python `for` loop calling `update_desired_joint_positions()` is one
        RPC per step and cannot hold 1 kHz (baseline 9, plan 2.5).
        """
        import torch
        import torchcontrol as toco

        robot = self._require()
        q_traj = np.asarray(q_traj, dtype=np.float32)
        dq_traj = np.asarray(dq_traj, dtype=np.float32)
        if q_traj.shape != dq_traj.shape:
            raise ValueError(
                "q_traj %r and dq_traj %r must have the same shape"
                % (q_traj.shape, dq_traj.shape)
            )
        policy = toco.policies.JointTrajectoryExecutor(
            joint_pos_trajectory=[torch.Tensor(x) for x in q_traj],
            joint_vel_trajectory=[torch.Tensor(x) for x in dq_traj],
            Kq=torch.Tensor(np.asarray(Kq, dtype=np.float32)),
            Kqd=torch.Tensor(np.asarray(Kqd, dtype=np.float32)),
            Kx=torch.Tensor(np.asarray(Kx, dtype=np.float32)),
            Kxd=torch.Tensor(np.asarray(Kxd, dtype=np.float32)),
            robot_model=robot.robot_model,
            ignore_gravity=robot.use_grav_comp,
        )
        robot.send_torch_policy(policy, blocking=False)

    # ---- lifecycle -----------------------------------------------------

    def is_running_policy(self) -> bool:
        return bool(self._require().is_running_policy())

    def terminate_policy(self) -> EpisodeBuffer:
        robot = self._require()
        if robot.is_running_policy():
            try:
                return states_to_buffer(robot.terminate_current_policy(return_log=True))
            except Exception:
                # TerminateController answers CANCELLED when nothing is running
                # (polymetis_server.cpp TerminateController). The policy can
                # finish inside the gRPC round trip that follows the check
                # above, and a stop request lands in that window by
                # construction. The arm is stopped either way; fall through and
                # collect the log rather than lose the run to a race.
                log.debug("terminate raced the policy ending; reading its log")
        # The server also ends a JointTrajectoryExecutor on its own when the
        # trajectory runs out; its log is still retrievable.
        try:
            return states_to_buffer(robot.get_previous_log())
        except Exception:
            return EpisodeBuffer.empty(self.spec().num_dofs)
