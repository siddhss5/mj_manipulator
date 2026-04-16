# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""EAIK-based IK solver for MuJoCo manipulators.

Extracts joint axes (H) and position offsets (P) directly from a MuJoCo model,
creating an EAIK HPRobot whose FK matches MuJoCo exactly. No DH parameters or
frame calibration needed.

Works with any MuJoCo arm (6-DOF or 7-DOF with joint discretization).

Usage:
    solver = MuJoCoEAIKSolver.from_arm(model, data, joint_ids, ee_site_id)
    solutions = solver.solve(target_pose)
"""

from __future__ import annotations

import logging

import mujoco
import numpy as np

logger = logging.getLogger(__name__)


def _read_body_pose(data: mujoco.MjData, body_id: int) -> np.ndarray:
    """Read a 4x4 pose matrix from a MuJoCo body."""
    T = np.eye(4)
    T[:3, :3] = data.xmat[body_id].reshape(3, 3)
    T[:3, 3] = data.xpos[body_id]
    return T


def _extract_hp(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_ids: list[int],
    joint_qpos_indices: list[int],
    ee_site_id: int,
    base_body_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract H (joint axes) and P (position offsets) from a MuJoCo model.

    Sets joints to zeros, runs FK, and reads axes/positions in the base
    body's frame. The resulting H/P can be passed directly to EAIK's HPRobot.

    Args:
        model: MuJoCo model.
        data: MuJoCo data (not modified — uses a temporary copy).
        joint_ids: MuJoCo joint IDs for the arm.
        joint_qpos_indices: qpos indices for the arm joints.
        ee_site_id: MuJoCo site ID for the end-effector.
        base_body_id: Body ID of the arm base.

    Returns:
        (H, P, ee_rot) where H is (n_joints, 3) joint axes, P is (n_joints+1, 3)
        position offsets, and ee_rot is (3, 3) orientation of the EE site in the
        base frame at q=zeros. All values are in the base body frame.
    """
    n_joints = len(joint_ids)

    # Use a temporary MjData to avoid disturbing live state
    tmp_data = mujoco.MjData(model)
    np.copyto(tmp_data.qpos, data.qpos)
    for idx in joint_qpos_indices:
        tmp_data.qpos[idx] = 0.0
    mujoco.mj_forward(model, tmp_data)

    # Base frame
    base_pos = tmp_data.xpos[base_body_id].copy()
    base_rot_inv = tmp_data.xmat[base_body_id].reshape(3, 3).T

    # Joint axes and positions in world frame
    H_world = np.zeros((n_joints, 3))
    positions_world = np.zeros((n_joints + 1, 3))

    for i, jid in enumerate(joint_ids):
        body_id = model.jnt_bodyid[jid]
        body_rot = tmp_data.xmat[body_id].reshape(3, 3)
        H_world[i] = body_rot @ model.jnt_axis[jid]
        positions_world[i] = tmp_data.xpos[body_id] + body_rot @ model.jnt_pos[jid]

    positions_world[n_joints] = tmp_data.site_xpos[ee_site_id]

    # EE site orientation in base frame at q=zeros.
    # EAIK's FK at q=zeros gives identity rotation (all joint angles are zero),
    # so the actual site orientation at q=zeros IS the fixed offset that EAIK
    # does not model. We store it so solve() can correct IK targets accordingly.
    ee_rot_world = tmp_data.site_xmat[ee_site_id].reshape(3, 3)
    ee_rot_base = base_rot_inv @ ee_rot_world

    # Convert to base frame
    H_base = (base_rot_inv @ H_world.T).T
    P_offsets = np.zeros((n_joints + 1, 3))
    P_offsets[0] = base_rot_inv @ (positions_world[0] - base_pos)
    for i in range(1, n_joints + 1):
        P_offsets[i] = base_rot_inv @ (positions_world[i] - positions_world[i - 1])

    return H_base, P_offsets, ee_rot_base


def find_locked_joint_index(H: np.ndarray, P: np.ndarray) -> int | None:
    """Find which joint to lock for EAIK 7-DOF arm support.

    Iterates over all joints and returns the first index that yields a known
    EAIK decomposition when locked. Call this once (e.g. in a one-off script)
    after extracting H/P from your model to determine ``fixed_joint_index``
    for ``MuJoCoEAIKSolver``.

    Example::

        from mj_manipulator.arms.eaik_solver import _extract_hp, find_locked_joint_index

        # After creating your Arm (arm) and extracting joint/site IDs:
        H, P, _ = _extract_hp(model, data, joint_ids, qpos_indices, ee_site_id, base_id)
        idx = find_locked_joint_index(H, P)
        # → e.g. 4 for Franka. Hardcode this as YOUR_ROBOT_LOCKED_JOINT_INDEX.

    Args:
        H: Joint axes (n_joints, 3) from ``_extract_hp``.
        P: Position offsets (n_joints+1, 3) from ``_extract_hp``.

    Returns:
        Joint index to pass as ``fixed_joint_index``, or None if no joint works
        (the arm may not be supported by EAIK).
    """
    from eaik.IK_HP import HPRobot

    for i in range(len(H)):
        robot = HPRobot(H, P, fixed_axes=[(i, 0.0)])
        if robot.hasKnownDecomposition():
            return i
    return None


class MuJoCoEAIKSolver:
    """EAIK IK solver that extracts kinematics directly from a MuJoCo model.

    Uses EAIK's HPRobot with joint axes (H) and position offsets (P) read
    from MuJoCo at q=zeros. Since H/P come from the model itself, EAIK FK
    matches MuJoCo FK in the base frame — no frame calibration is needed.

    Supports both standard 6-DOF arms and 7-DOF arms with joint
    discretization via ``fixed_joint_index``.

    Assumes the MuJoCo model is static after construction — H, P, and the EE
    orientation offset are extracted once at init. If the model geometry
    changes, create a new solver.

    Implements the IKSolver protocol (solve / solve_valid).

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        joint_ids: MuJoCo joint IDs for the arm.
        joint_qpos_indices: qpos indices for the arm joints.
        ee_site_id: MuJoCo site ID for end-effector.
        base_body_id: Body ID of the arm base.
        joint_limits: (lower, upper) arrays for solution filtering.
        fixed_joint_index: For 7-DOF arms, the joint index to lock and
            discretize. None for standard 6-DOF arms.
        n_discretizations: Number of values to sample for the locked joint.
    """

    # Maximum FK position error for accepting a discretized solution (meters).
    _FK_TOLERANCE = 1e-3

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        joint_ids: list[int],
        joint_qpos_indices: list[int],
        ee_site_id: int,
        base_body_id: int,
        joint_limits: tuple[np.ndarray, np.ndarray] | None = None,
        fixed_joint_index: int | None = None,
        n_discretizations: int = 16,
    ):
        from eaik.IK_HP import HPRobot

        self._model = model
        self._data = data
        self._joint_qpos_indices = joint_qpos_indices
        self._base_body_id = base_body_id
        self._joint_limits = joint_limits
        self._fixed_joint_index = fixed_joint_index
        self._HPRobot = HPRobot

        # Extract H/P and EE orientation offset from MuJoCo model.
        # _ee_rot_offset is the EE site's orientation in the base frame at q=zeros.
        # EAIK's FK at q=zeros gives identity rotation (all joint angles zero),
        # so this offset must be removed from IK targets before passing to EAIK.
        self._H, self._P, self._ee_rot_offset = _extract_hp(
            model,
            data,
            joint_ids,
            joint_qpos_indices,
            ee_site_id,
            base_body_id,
        )

        if fixed_joint_index is None:
            # Standard 6-DOF: create robot once
            self._robot = HPRobot(self._H, self._P)
            if not self._robot.hasKnownDecomposition():
                logger.warning(
                    "EAIK has no known decomposition for this arm: %s",
                    self._robot.getKinematicFamily(),
                )
            self._discretize_values = None
        else:
            # 7-DOF: precompute discretization values for the locked joint
            self._robot = None
            if joint_limits is not None:
                lo = joint_limits[0][fixed_joint_index]
                hi = joint_limits[1][fixed_joint_index]
            else:
                lo, hi = -3.14, 3.14
            self._discretize_values = np.linspace(lo, hi, n_discretizations)

    @property
    def H(self) -> np.ndarray:
        """Joint axes matrix (n_joints, 3) in base frame."""
        return self._H

    @property
    def P(self) -> np.ndarray:
        """Position offsets (n_joints+1, 3) in base frame."""
        return self._P

    @property
    def robot(self):
        """EAIK HPRobot instance, or None for 7-DOF (discretized)."""
        return self._robot

    @property
    def fixed_joint_index(self) -> int | None:
        """Index of the locked joint for 7-DOF discretization, or None."""
        return self._fixed_joint_index

    @property
    def discretize_values(self) -> np.ndarray | None:
        """Discretization values for the locked joint, or None for 6-DOF."""
        return self._discretize_values

    def _get_base_pose(self) -> np.ndarray:
        """Get arm base body pose in world frame (4x4)."""
        return _read_body_pose(self._data, self._base_body_id)

    def _to_base_frame(self, pose_world: np.ndarray) -> np.ndarray:
        """Convert a world-frame pose to the arm's base body frame."""
        T_world_base = self._get_base_pose()
        return np.linalg.inv(T_world_base) @ pose_world

    def solve(
        self,
        pose: np.ndarray,
        q_init: np.ndarray | None = None,
        *,
        discretizations: list[np.ndarray] | None = None,
    ) -> list[np.ndarray]:
        """Solve IK for a world-frame target pose.

        For 6-DOF arms, solves analytically and returns all solutions.

        For redundant arms (7-DOF here, with one locked/discretized
        joint), sweeps a set of values for the locked joint and returns
        every solution whose FK matches within tolerance. The sweep
        set defaults to the evenly-spaced discretization computed at
        init; callers can override via ``discretizations`` — for
        example, to force a single value (the arm's current locked-joint
        angle, for smooth Cartesian paths) or a narrow window around
        some anchor.

        Args:
            pose: 4x4 target pose in world frame (at the EE site).
            q_init: Ignored (EAIK is analytical; present for
                :class:`IKSolver` protocol compatibility).
            discretizations: Optional per-locked-joint value arrays.
                For a 6-DOF arm: ignored. For a single-locked-joint 7-DOF
                arm: a length-1 list whose first array is the set of
                values to try for the locked joint. ``None`` uses the
                precomputed full-range sweep.

        Returns:
            List of joint configurations (may include out-of-limits).
        """
        T_base = self._to_base_frame(pose)

        # EAIK's FK at q=zeros gives identity rotation, but the actual EE site
        # has orientation _ee_rot_offset in the base frame at q=zeros:
        #   actual_site_R(q) = EAIK_FK_R(q) @ _ee_rot_offset
        # To achieve target R_target: EAIK_FK_R must equal R_target @ _ee_rot_offset.T
        T_eaik = T_base.copy()
        T_eaik[:3, :3] = T_base[:3, :3] @ self._ee_rot_offset.T

        if self._robot is not None:
            # 6-DOF: solve directly
            try:
                result = self._robot.IK(T_eaik)
            except RuntimeError:
                return []
            return [result.Q[i].copy() for i in range(result.num_solutions()) if not result.is_LS[i]]

        # 7-DOF: sweep the locked joint.
        if discretizations is None:
            values = self._discretize_values
        else:
            if len(discretizations) != 1:
                raise ValueError(
                    f"solve() expects 1 discretization array for this arm "
                    f"(one locked joint), got {len(discretizations)}."
                )
            values = np.asarray(discretizations[0], dtype=float)

        all_solutions = []
        for theta in values:
            robot = self._HPRobot(
                self._H,
                self._P,
                fixed_axes=[(self._fixed_joint_index, theta)],
            )
            try:
                result = robot.IK(T_eaik)
            except RuntimeError:
                continue
            for i in range(result.num_solutions()):
                q = result.Q[i].copy()
                # EAIK marks all fixed_axes solutions as LS, even exact ones.
                # Verify by FK round-trip instead.
                T_check = robot.fwdKin(q)
                if np.linalg.norm(T_check[:3, 3] - T_eaik[:3, 3]) < self._FK_TOLERANCE:
                    all_solutions.append(q)

        return all_solutions

    def solve_valid(
        self,
        pose: np.ndarray,
        q_init: np.ndarray | None = None,
        *,
        discretizations: list[np.ndarray] | None = None,
    ) -> list[np.ndarray]:
        """Solve IK and return only valid (in-limits) solutions.

        Args:
            pose: 4x4 target pose in world frame (at the EE site).
            q_init: Ignored (EAIK is analytical; present for
                :class:`IKSolver` protocol compatibility).
            discretizations: Optional per-locked-joint value arrays,
                forwarded to :meth:`solve`. See :meth:`solve` for details.

        Returns:
            List of valid joint configurations.
        """
        solutions = self.solve(pose, q_init=q_init, discretizations=discretizations)

        if not solutions or self._joint_limits is None:
            return solutions

        lower, upper = self._joint_limits
        return [q for q in solutions if np.all(q >= lower) and np.all(q <= upper)]
