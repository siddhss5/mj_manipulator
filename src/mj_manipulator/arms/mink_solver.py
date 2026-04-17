# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Mink-based numerical IK solver for MuJoCo manipulators.

Uses `mink <https://github.com/kevinzakka/mink>`_ (MuJoCo-native
inverse kinematics by Stéphane Caron / Kevin Zakka) as a numerical
alternative to the analytical EAIK solver. Works with any MuJoCo arm
regardless of kinematic structure — no spherical-wrist or known
decomposition required.

Install: ``uv add "mj-manipulator[mink]"``

Usage::

    from mj_manipulator.arms.mink_solver import MinkIKSolver, make_mink_solver

    # From an Arm instance (recommended — zero config, reads ee_site automatically):
    solver = make_mink_solver(arm)

    # Direct construction (ee_frame_name resolved from ee_site_id if omitted):
    solver = MinkIKSolver(model, data, joint_ids, joint_qpos_indices,
                          ee_site_id, base_body_id, joint_limits)

    solutions = solver.solve_valid(target_pose_4x4)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from mj_manipulator.arm import Arm

logger = logging.getLogger(__name__)


class MinkIKSolver:
    """Numerical IK solver backed by mink's QP-based differential IK.

    Implements the :class:`~mj_manipulator.protocols.IKSolver` protocol.

    Each ``solve()`` call runs multiple restarts from random seeds,
    converging each via ``mink.solve_ik`` + ``configuration.integrate``
    until the pose error drops below threshold. Returns the list of
    unique converged solutions (typically 1 per successful restart).

    Uses a **private copy** of model/data so solving doesn't disturb
    live physics state.

    For 7-DOF (redundant) arms, mink's ``PostureTask`` regularizes the
    null space — no joint-locking or discretization needed (unlike
    EAIK). The ``posture_cost`` parameter controls how strongly the
    solver prefers staying near the seed configuration.
    """

    # Default QP solver. daqp ships with mink and is fast.
    _DEFAULT_SOLVER = "daqp"

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        joint_ids: list[int],
        joint_qpos_indices: list[int],
        ee_site_id: int,
        base_body_id: int,
        joint_limits: tuple[np.ndarray, np.ndarray] | None = None,
        *,
        ee_frame_name: str | None = None,
        n_restarts: int = 4,
        max_iters: int = 50,
        convergence_thresh: float = 1e-3,
        orientation_thresh: float = 0.01,
        dt: float = 0.1,
        posture_cost: float = 1e-3,
        position_cost: float = 1.0,
        orientation_cost: float = 1.0,
        damping: float = 1e-3,
        solver: str | None = None,
    ):
        """
        Args:
            model: MuJoCo model (a private copy is made).
            data: MuJoCo data (a private copy is made).
            joint_ids: Joint IDs for the arm.
            joint_qpos_indices: Indices into qpos for the arm joints.
            ee_site_id: Site ID for end-effector.
            base_body_id: Body ID of the arm's base (for frame conversion).
            joint_limits: ``(lower, upper)`` arrays, or None for no filtering.
            ee_frame_name: Name of the EE site (for mink FrameTask). If None,
                resolved automatically from ``ee_site_id`` via the model.
            n_restarts: Number of random seeds per ``solve()`` call.
            max_iters: Max convergence iterations per restart.
            convergence_thresh: Position error threshold (meters).
            orientation_thresh: Orientation error threshold (radians, geodesic).
            dt: Integration timestep for ``solve_ik`` (larger = faster but less stable).
            posture_cost: Weight on PostureTask (null-space regularization).
            position_cost: Weight on position tracking in FrameTask.
            orientation_cost: Weight on orientation tracking in FrameTask.
            damping: QP damping for numerical stability.
            solver: QP solver backend (default: "daqp").
        """
        import mink as _mink

        # Resolve EE site name from ID if not provided.
        if ee_frame_name is None:
            ee_frame_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_id)
            if ee_frame_name is None:
                raise ValueError(f"ee_site_id={ee_site_id} has no name in the model; pass ee_frame_name explicitly.")

        self._mink = _mink
        self._joint_ids = list(joint_ids)
        self._joint_qpos_indices = list(joint_qpos_indices)
        self._ee_site_id = ee_site_id
        self._base_body_id = base_body_id
        self._joint_limits = joint_limits
        self._n_restarts = n_restarts
        self._max_iters = max_iters
        self._convergence_thresh = convergence_thresh
        self._orientation_thresh = orientation_thresh
        self._dt = dt
        self._damping = damping
        self._solver = solver or self._DEFAULT_SOLVER
        self._rng = np.random.default_rng()

        # Private model/data copy for IK solving (don't touch live state).
        self._model = model
        self._data = mujoco.MjData(model)

        # Build mink tasks once (reuse across calls).
        self._config = _mink.Configuration(model)
        self._ee_task = _mink.FrameTask(
            frame_name=ee_frame_name,
            frame_type="site",
            position_cost=position_cost,
            orientation_cost=orientation_cost,
        )
        self._posture_task = _mink.PostureTask(model=model, cost=posture_cost)
        self._config_limit = _mink.ConfigurationLimit(model)

        self._tasks = [self._ee_task, self._posture_task]
        self._limits = [self._config_limit]

        # Cache joint limit arrays for random seed generation.
        if joint_limits is not None:
            self._q_lower = np.array(joint_limits[0])
            self._q_upper = np.array(joint_limits[1])
        else:
            n = len(joint_ids)
            self._q_lower = np.full(n, -np.pi)
            self._q_upper = np.full(n, np.pi)

    def solve(
        self,
        pose: np.ndarray,
        q_init: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """Solve IK for a world-frame target pose via multi-restart.

        Runs ``n_restarts`` convergence attempts from random seeds.
        If ``q_init`` is provided, it is used as the first seed (best
        chance of finding a nearby solution).

        Args:
            pose: 4×4 target pose in world frame (at the EE site).
            q_init: Optional initial configuration hint (used as first seed).

        Returns:
            List of unique joint configurations that converge within
            tolerance. May be empty if no restart succeeds.
        """
        target = self._mink.SE3.from_matrix(pose)
        self._ee_task.set_target(target)
        self._target_pos = pose[:3, 3].copy()
        self._target_R = pose[:3, :3].copy()

        seeds = self._generate_seeds(q_init)
        solutions: list[np.ndarray] = []

        for seed in seeds:
            q = self._solve_single(seed)
            if q is not None and self._is_unique(q, solutions):
                solutions.append(q)

        return solutions

    def solve_valid(
        self,
        pose: np.ndarray,
        q_init: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """Solve IK and return only in-limits solutions.

        Args:
            pose: 4×4 target pose in world frame.
            q_init: Optional initial configuration hint.

        Returns:
            List of valid joint configurations (may be empty).
        """
        solutions = self.solve(pose, q_init=q_init)

        if not solutions or self._joint_limits is None:
            return solutions

        lower, upper = self._q_lower, self._q_upper
        return [q for q in solutions if np.all(q >= lower - 1e-6) and np.all(q <= upper + 1e-6)]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    # Accept a stalled solution if it's within these generous bounds.
    # The QP solver often oscillates around the solution — the stall
    # point IS the best this restart will achieve.
    _STALL_ACCEPT_POS = 0.005  # 5 mm
    _STALL_ACCEPT_ROT = 0.05  # ~3 degrees

    def _solve_single(self, q_seed: np.ndarray) -> np.ndarray | None:
        """Run one convergence attempt from a seed configuration.

        Returns the converged joint config, or None if it didn't converge
        or stalled too far from the target.
        """
        # Reset configuration to seed.
        self._set_config(q_seed)
        self._posture_task.set_target_from_configuration(self._config)

        prev_err = float("inf")
        best_pos_err = float("inf")
        best_rot_err = float("inf")
        best_q: np.ndarray | None = None
        stall_count = 0

        for _ in range(self._max_iters):
            vel = self._mink.solve_ik(
                self._config,
                self._tasks,
                self._dt,
                solver=self._solver,
                damping=self._damping,
                limits=self._limits,
            )
            self._config.integrate_inplace(vel, self._dt)

            pos_err, rot_err = self._compute_error()

            # Track best solution seen so far.
            if pos_err + rot_err < best_pos_err + best_rot_err:
                best_pos_err = pos_err
                best_rot_err = rot_err
                best_q = self._read_arm_qpos()

            # Tight convergence — return immediately.
            if pos_err < self._convergence_thresh and rot_err < self._orientation_thresh:
                return self._read_arm_qpos()

            # Early termination if error has stalled.
            total_err = pos_err + rot_err
            if total_err >= prev_err - 1e-8:
                stall_count += 1
                if stall_count > 20:
                    break
            else:
                stall_count = 0
            prev_err = total_err

        # Accept the best solution if it's close enough (the QP often
        # oscillates around the minimum rather than converging exactly).
        if best_q is not None and best_pos_err < self._STALL_ACCEPT_POS and best_rot_err < self._STALL_ACCEPT_ROT:
            return best_q

        return None

    def _generate_seeds(self, q_init: np.ndarray | None) -> list[np.ndarray]:
        """Generate seed configurations for multi-restart."""
        seeds: list[np.ndarray] = []
        if q_init is not None:
            seeds.append(np.array(q_init, dtype=float))
        while len(seeds) < self._n_restarts:
            q = self._rng.uniform(self._q_lower, self._q_upper)
            seeds.append(q)
        return seeds

    def _set_config(self, q_arm: np.ndarray) -> None:
        """Write arm joint positions into the mink Configuration."""
        q_full = self._config.q.copy()
        for i, idx in enumerate(self._joint_qpos_indices):
            q_full[idx] = q_arm[i]
        self._config.update(q_full)

    def _read_arm_qpos(self) -> np.ndarray:
        """Read arm joint positions from the mink Configuration."""
        return np.array([self._config.q[idx] for idx in self._joint_qpos_indices])

    def _compute_error(self) -> tuple[float, float]:
        """Compute position and orientation error between EE and target."""
        data = self._config.data
        current_pos = data.site_xpos[self._ee_site_id].copy()
        current_R = data.site_xmat[self._ee_site_id].reshape(3, 3)

        pos_err = float(np.linalg.norm(current_pos - self._target_pos))

        # Geodesic rotation error: angle of R_current^T @ R_target.
        R_err = current_R.T @ self._target_R
        cos_angle = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
        rot_err = float(np.arccos(cos_angle))

        return pos_err, rot_err

    @staticmethod
    def _is_unique(q: np.ndarray, existing: list[np.ndarray], tol: float = 0.1) -> bool:
        """Check if q is sufficiently different from all existing solutions."""
        for q_other in existing:
            if np.linalg.norm(q - q_other) < tol:
                return False
        return True


def make_mink_solver(
    arm: Arm,
    **kwargs,
) -> MinkIKSolver:
    """Create a MinkIKSolver from an Arm instance.

    Convenience factory that extracts joint IDs, limits, site ID, and
    base body from the Arm. The EE site name is resolved automatically
    from the arm's ``ee_site_id`` — no configuration needed.

    Args:
        arm: An Arm instance (created without IK, just for index resolution).
        **kwargs: Forwarded to :class:`MinkIKSolver`.

    Returns:
        Configured MinkIKSolver.
    """
    joint_limits = arm.get_joint_limits()
    first_joint_body = arm.env.model.jnt_bodyid[arm.joint_ids[0]]
    base_body_id = int(arm.env.model.body_parentid[first_joint_body])

    return MinkIKSolver(
        model=arm.env.model,
        data=arm.env.data,
        joint_ids=list(arm.joint_ids),
        joint_qpos_indices=arm.joint_qpos_indices,
        ee_site_id=arm.ee_site_id,
        base_body_id=base_body_id,
        joint_limits=joint_limits,
        **kwargs,
    )
