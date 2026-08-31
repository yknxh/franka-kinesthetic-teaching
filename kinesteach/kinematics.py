"""Offline forward kinematics.

`polymetis_pb2.RobotState` carries no end-effector pose (plan 2.1), so
Cartesian data is reconstructed from the logged joint angles after the fact.
This is not a loss: FK on a recorded `q` is exact and costs ~14 us per state,
so a five-minute episode reconstructs in a few seconds and can be regenerated
whenever the model changes.

The pose produced here is the **flange** (`panda_link8`), not the tool centre
point. The real robot's URDF stops at the flange and has no Robotiq on it
(plan 2.2), so turning this into a TCP pose needs `T_flange_gripper` and
`T_gripper_tcp` from the Phase 2 calibration.
"""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

__all__ = ["ForwardKinematics", "EE_FRAME"]

#: What `ForwardKinematics` returns. Recorded into processed episodes so a
#: later consumer cannot mistake it for a tool pose.
EE_FRAME = "flange"


class ForwardKinematics:
    """Batched FK/Jacobian for one URDF.

    Wraps `torchcontrol.models.RobotModelPinocchio`, which is loaded lazily so
    that importing this module does not require polymetis.
    """

    def __init__(
        self,
        urdf_path,
        ee_link_name: Optional[str] = None,
        arm_dofs: Optional[int] = None,
    ):
        # Held on the instance so the hot loops below do not re-import it per
        # call, and so this import is the one that fails when torch is missing.
        import torch
        from torchcontrol.models import RobotModelPinocchio

        self._torch = torch
        self.urdf_path = str(urdf_path)
        self.ee_link_name = ee_link_name
        self.model = RobotModelPinocchio(self.urdf_path, ee_link_name)
        lo, _ = self.model.get_joint_angle_limits()
        self.model_dofs = int(np.asarray(lo).size)
        self.arm_dofs = int(arm_dofs) if arm_dofs is not None else self.model_dofs

    @classmethod
    def from_episode(cls, episode, arm_dofs: Optional[int] = None) -> Optional["ForwardKinematics"]:
        """Build from the URDF stored inside an episode, or None if absent."""
        path = episode.urdf_path()
        if path is None:
            return None
        meta = episode.read_metadata()
        return cls(path, meta.get("ee_link_name"), arm_dofs)

    # ---- helpers -------------------------------------------------------

    def _prepare(self, q: np.ndarray) -> np.ndarray:
        q = np.atleast_2d(np.asarray(q, dtype=np.float64))
        if q.shape[1] == self.model_dofs:
            return q
        if q.shape[1] > self.model_dofs:
            # A log with more joints than the URDF -- a robot whose controller
            # also drives gripper joints, say. Taking the leading columns is
            # right only if the arm joints come first, so say so out loud.
            log.warning(
                "log has %d DOF but the URDF model has %d; using the leading "
                "%d columns as the arm joints",
                q.shape[1], self.model_dofs, self.model_dofs,
            )
            return q[:, : self.model_dofs]
        raise ValueError(
            "log has %d DOF, fewer than the URDF model's %d; wrong URDF?"
            % (q.shape[1], self.model_dofs)
        )

    # ---- computation ---------------------------------------------------

    def fk(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(position (N,3), quaternion (N,4) as xyzw) of the EE link."""
        torch = self._torch
        qq = self._prepare(q)
        n = qq.shape[0]
        pos = np.empty((n, 3), dtype=np.float64)
        quat = np.empty((n, 4), dtype=np.float64)
        for i in range(n):
            p, r = self.model.forward_kinematics(torch.from_numpy(qq[i]).float())
            pos[i] = p.numpy()
            quat[i] = r.numpy()
        return pos, quat

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """(N, 6, model_dofs) geometric Jacobian at each sample."""
        torch = self._torch
        qq = self._prepare(q)
        n = qq.shape[0]
        out = np.empty((n, 6, self.model_dofs), dtype=np.float64)
        for i in range(n):
            out[i] = self.model.compute_jacobian(torch.from_numpy(qq[i]).float()).numpy()
        return out

    def ee_velocity(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """(N, 6) end-effector twist, from J(q) @ dq."""
        J = self.jacobian(q)
        dqq = self._prepare(dq)
        return np.einsum("nij,nj->ni", J, dqq)


def fk_from_urdf_text(urdf_text: str, ee_link_name: Optional[str] = None) -> ForwardKinematics:
    """Build FK from URDF *text* (pinocchio only reads from a path).

    The temporary file is removed as soon as the model is built: pinocchio
    parses the URDF in `RobotModelPinocchio.__init__` and never reads the path
    again. Leaving it behind used to cost one stray file per call, and the
    workstation had accumulated 355 of them -- every `workspace`, every
    `payload-sweep`, every `--home`.

    `urdf_path` on the returned object therefore names a file that no longer
    exists. It is kept for provenance in logs, not for reopening; episodes
    store their URDF next to the episode instead.
    """
    tmp = tempfile.NamedTemporaryFile(
        "w", prefix="kinesteach-", suffix=".urdf", delete=False)
    try:
        tmp.write(urdf_text)
        tmp.close()
        return ForwardKinematics(tmp.name, ee_link_name)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:  # pragma: no cover - already gone
            pass
