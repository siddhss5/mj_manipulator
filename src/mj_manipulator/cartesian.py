"""Jacobian-based Cartesian velocity control.

Core function ``twist_to_joint_velocity()`` solves a constrained QP:

    min  (1/2)||J*q_dot - v_d||_W^2 + (λ/2)||q_dot||^2
    s.t. ℓ <= q_dot <= u

where bounds ℓ,u incorporate both velocity limits AND position limits
(converted to per-timestep velocity bounds).

Higher-level functions (``step_twist``, contact detection) take explicit
parameters rather than an ``Arm`` object, so they work with any robot.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from mj_manipulator.contacts import iter_contacts

if TYPE_CHECKING:
    from mj_manipulator.arm import Arm

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class CartesianControlConfig:
    """Configuration for Cartesian velocity control.

    Attributes:
        length_scale: Length scale L for twist weighting (meters).
                     W = diag(1,1,1, 1/L², 1/L², 1/L²)
        damping: Regularization λ for joint velocity.
        joint_margin_deg: Degrees from joint limits to treat as buffer.
        velocity_scale: Fraction of max joint velocity to use (0-1].
        min_progress: Minimum achieved_fraction to continue motion (0-1).
    """

    length_scale: float = 0.1
    damping: float = 1e-4
    joint_margin_deg: float = 5.0
    velocity_scale: float = 1.0
    min_progress: float = 0.5

    def __post_init__(self):
        if self.length_scale <= 0:
            raise ValueError(f"length_scale must be > 0, got {self.length_scale}")
        if self.damping < 0:
            raise ValueError(f"damping must be >= 0, got {self.damping}")
        if self.joint_margin_deg < 0:
            raise ValueError(f"joint_margin_deg must be >= 0, got {self.joint_margin_deg}")
        if not 0 < self.velocity_scale <= 1:
            raise ValueError(f"velocity_scale must be in (0, 1], got {self.velocity_scale}")
        if not 0 <= self.min_progress <= 1:
            raise ValueError(f"min_progress must be in [0, 1], got {self.min_progress}")


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class TwistStepResult:
    """Result of a single twist-to-joint-velocity computation."""

    joint_velocities: np.ndarray
    twist_error: float
    achieved_fraction: float
    limiting_factor: str | None = None


@dataclass
class MoveUntilTouchResult:
    """Result of a move_until_touch operation."""

    success: bool
    terminated_by: Literal["contact", "max_distance", "no_progress"]
    distance_moved: float
    final_force: np.ndarray | None = None
    final_torque: np.ndarray | None = None
    contact_geom: str | None = None


@dataclass
class TwistExecutionResult:
    """Result of a twist execution operation."""

    terminated_by: Literal["duration", "distance", "condition", "no_progress"]
    distance_moved: float
    duration: float
    final_pose: np.ndarray


# =============================================================================
# Core Jacobian Functions
# =============================================================================


def get_ee_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ee_site_id: int,
    joint_vel_indices: list[int],
) -> np.ndarray:
    """Compute the 6xN end-effector Jacobian for an arm.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        ee_site_id: Site ID for end-effector.
        joint_vel_indices: Indices into qvel for arm joints.

    Returns:
        6xN Jacobian: rows 0-2 linear, rows 3-5 angular.
    """
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)

    J_pos = jacp[:, joint_vel_indices]
    J_rot = jacr[:, joint_vel_indices]

    return np.vstack([J_pos, J_rot])


# =============================================================================
# Core Control Function (Constrained QP)
# =============================================================================


def twist_to_joint_velocity(
    J: np.ndarray,
    twist: np.ndarray,
    q_current: np.ndarray,
    q_min: np.ndarray,
    q_max: np.ndarray,
    qd_max: np.ndarray,
    dt: float,
    config: CartesianControlConfig | None = None,
    q_dot_prev: np.ndarray | None = None,
) -> TwistStepResult:
    """Convert a Cartesian twist to joint velocities via constrained QP.

    Solves:
        min  (1/2)||J*q_dot - v_d||_W^2 + (λ/2)||q_dot||^2
        s.t. ℓ <= q_dot <= u

    Args:
        J: 6xN Jacobian matrix.
        twist: 6D desired twist [vx, vy, vz, wx, wy, wz].
        q_current: Current joint positions (rad).
        q_min: Lower joint position limits (rad).
        q_max: Upper joint position limits (rad).
        qd_max: Maximum joint velocities (rad/s), assumed symmetric.
        dt: Controller timestep (seconds).
        config: Control configuration (uses defaults if None).
        q_dot_prev: Previous solution for warm-starting.

    Returns:
        TwistStepResult with joint velocities and diagnostics.
    """
    if config is None:
        config = CartesianControlConfig()

    n_joints = J.shape[1]
    L = config.length_scale
    lam = config.damping
    margin = np.deg2rad(config.joint_margin_deg)

    qd_max_scaled = qd_max * config.velocity_scale

    # Step 1: Convert position limits to velocity bounds
    ell_pos = ((q_min + margin) - q_current) / dt
    u_pos = ((q_max - margin) - q_current) / dt

    ell = np.maximum(-qd_max_scaled, ell_pos)
    u = np.minimum(+qd_max_scaled, u_pos)

    # Relax infeasible bounds
    infeasible = ell > u
    if np.any(infeasible):
        ell[infeasible] = np.minimum(ell[infeasible], 0)
        u[infeasible] = np.maximum(u[infeasible], 0)

    # Step 2: Build QP matrices
    w_diag = np.array([1.0, 1.0, 1.0, 1.0/L**2, 1.0/L**2, 1.0/L**2])
    W = np.diag(w_diag)

    JtW = J.T @ W
    H = JtW @ J + lam * np.eye(n_joints)
    g = -JtW @ twist

    # Step 3: Solve via projected gradient descent
    try:
        cho = cho_factor(H)
        qd_unconstrained = cho_solve(cho, -g)
    except np.linalg.LinAlgError:
        qd_unconstrained = np.linalg.solve(H, -g)

    if np.all(qd_unconstrained >= ell) and np.all(qd_unconstrained <= u):
        q_dot = qd_unconstrained
    else:
        if q_dot_prev is not None:
            q_dot = np.clip(q_dot_prev, ell, u)
        else:
            q_dot = np.clip(qd_unconstrained, ell, u)

        alpha = 1.0 / (np.linalg.norm(H, 2) + 1e-6)

        for _ in range(20):
            grad = H @ q_dot + g
            q_new = np.clip(q_dot - alpha * grad, ell, u)
            if np.linalg.norm(q_new - q_dot) < 1e-8:
                break
            q_dot = q_new

    # Step 4: Diagnostics
    twist_achieved = J @ q_dot
    twist_diff = twist_achieved - twist
    twist_error = float(np.sqrt(twist_diff @ W @ twist_diff))

    twist_norm = float(np.sqrt(twist @ W @ twist))
    if twist_norm > 1e-10:
        achieved_fraction = float(
            np.dot(twist_achieved, W @ twist) / (twist_norm**2)
        )
        achieved_fraction = max(0.0, min(1.0, achieved_fraction))
    else:
        achieved_fraction = 1.0

    limiting_factor = None
    at_lower = np.abs(q_dot - ell) < 1e-6
    at_upper = np.abs(q_dot - u) < 1e-6
    at_bound = at_lower | at_upper

    if np.any(at_bound):
        at_pos_lower = at_lower & (ell > -qd_max_scaled + 1e-6)
        at_pos_upper = at_upper & (u < qd_max_scaled - 1e-6)
        if np.any(at_pos_lower | at_pos_upper):
            limiting_factor = "joint_limit"
        else:
            limiting_factor = "velocity"

    return TwistStepResult(
        joint_velocities=q_dot,
        twist_error=twist_error,
        achieved_fraction=achieved_fraction,
        limiting_factor=limiting_factor,
    )


# =============================================================================
# Step Twist (parameterized — no Arm dependency)
# =============================================================================


def step_twist(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ee_site_id: int,
    joint_qpos_indices: list[int],
    joint_qvel_indices: list[int],
    q_min: np.ndarray,
    q_max: np.ndarray,
    qd_max: np.ndarray,
    twist: np.ndarray,
    frame: str = "world",
    dt: float = 0.004,
    config: CartesianControlConfig | None = None,
    q_dot_prev: np.ndarray | None = None,
) -> tuple[np.ndarray, TwistStepResult]:
    """Execute one timestep of Cartesian velocity control.

    Computes joint velocities that achieve the desired twist, then
    integrates to get new joint positions.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        ee_site_id: Site ID for end-effector.
        joint_qpos_indices: Indices into qpos for arm joints.
        joint_qvel_indices: Indices into qvel for arm joints.
        q_min: Lower joint position limits (rad).
        q_max: Upper joint position limits (rad).
        qd_max: Maximum joint velocities (rad/s).
        twist: 6D twist [vx, vy, vz, wx, wy, wz].
        frame: "world" or "hand" (tool frame).
        dt: Timestep duration in seconds.
        config: Control configuration.
        q_dot_prev: Previous joint velocities for warm-starting.

    Returns:
        Tuple of (new_joint_positions, TwistStepResult).
    """
    J = get_ee_jacobian(model, data, ee_site_id, joint_qvel_indices)

    if frame == "hand":
        R = data.site_xmat[ee_site_id].reshape(3, 3)
        twist_world = np.zeros(6)
        twist_world[:3] = R @ twist[:3]
        twist_world[3:] = R @ twist[3:]
        twist = twist_world

    q_current = np.array([data.qpos[idx] for idx in joint_qpos_indices])

    result = twist_to_joint_velocity(
        J=J, twist=twist, q_current=q_current,
        q_min=q_min, q_max=q_max, qd_max=qd_max,
        dt=dt, config=config, q_dot_prev=q_dot_prev,
    )

    q_new = q_current + result.joint_velocities * dt
    return q_new, result


# =============================================================================
# Contact Detection (parameterized — no Arm dependency)
# =============================================================================


def check_gripper_contact(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    gripper_body_names: list[str],
) -> str | None:
    """Check for gripper contact.

    Returns name of contacted geom, or None if no contact.
    """
    gripper_body_ids = set()
    for name in gripper_body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id != -1:
            gripper_body_ids.add(body_id)

    for b1, b2, contact in iter_contacts(model, data):
        if b1 in gripper_body_ids:
            return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
        if b2 in gripper_body_ids:
            return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)

    return None


def get_arm_body_ids(
    model: mujoco.MjModel,
    joint_names: list[str],
    gripper_body_names: list[str] | None = None,
) -> set[int]:
    """Get all body IDs that are part of an arm.

    Args:
        model: MuJoCo model.
        joint_names: Joint names for the arm.
        gripper_body_names: Optional gripper body names.

    Returns:
        Set of body IDs belonging to the arm.
    """
    body_ids: set[int] = set()

    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        if joint_id != -1:
            body_ids.add(model.jnt_bodyid[joint_id])

    if gripper_body_names:
        for name in gripper_body_names:
            body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, name
            )
            if body_id != -1:
                body_ids.add(body_id)

    return body_ids


def check_arm_contact(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm_body_ids: set[int],
    exclude_self_collision: bool = True,
) -> str | None:
    """Check for arm contact with environment.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        arm_body_ids: Body IDs belonging to the arm.
        exclude_self_collision: Ignore contacts between arm bodies.

    Returns:
        Name of contacted geom, or None if no contact.
    """
    for b1, b2, contact in iter_contacts(model, data):
        b1_is_arm = b1 in arm_body_ids
        b2_is_arm = b2 in arm_body_ids

        if exclude_self_collision and b1_is_arm and b2_is_arm:
            continue

        if b1_is_arm and not b2_is_arm:
            return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
        if b2_is_arm and not b1_is_arm:
            return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)

    return None


# =============================================================================
# CartesianController (stateful — wraps step_twist with warm-start state)
# =============================================================================


def _rotation_error_axis_angle(R_target: np.ndarray, R_current: np.ndarray) -> np.ndarray:
    """Rotation error as an axis-angle vector (magnitude = angle in radians)."""
    R_err = R_target @ R_current.T
    angle = np.arccos(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-8:
        return np.zeros(3)
    skew_factor = angle / (2.0 * np.sin(angle))
    return skew_factor * np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ])


class CartesianController:
    """Stateful Cartesian velocity controller.

    Wraps ``step_twist`` with warm-start state and provides higher-level
    motion primitives for teleop and small Cartesian plans.

    Usage — teleop (call from your control loop)::

        controller = CartesianController.from_arm(arm)
        # in your 125 Hz loop:
        result = controller.step(twist, dt=0.008)

    Usage — scripted Cartesian motion::

        result = controller.move(
            twist=np.array([0, 0, -0.05, 0, 0, 0]),
            dt=0.008, max_distance=0.05,
        )
        result = controller.move_to(target_pose, dt=0.008, speed=0.05)

    See ``docs/cartesian-control.md`` for the underlying QP formulation.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        ee_site_id: int,
        joint_qpos_indices: list[int],
        joint_qvel_indices: list[int],
        q_min: np.ndarray,
        q_max: np.ndarray,
        qd_max: np.ndarray,
        config: CartesianControlConfig | None = None,
    ):
        """
        Args:
            model: MuJoCo model.
            data: Live MuJoCo data — controller reads state and writes qpos.
            ee_site_id: Site ID for end-effector.
            joint_qpos_indices: Indices into qpos for arm joints.
            joint_qvel_indices: Indices into qvel for arm joints.
            q_min: Lower joint position limits (rad).
            q_max: Upper joint position limits (rad).
            qd_max: Maximum joint velocities (rad/s).
            config: Control configuration (uses defaults if None).
        """
        self.model = model
        self.data = data
        self.ee_site_id = ee_site_id
        self.joint_qpos_indices = joint_qpos_indices
        self.joint_qvel_indices = joint_qvel_indices
        self.q_min = q_min
        self.q_max = q_max
        self.qd_max = qd_max
        self.config = config or CartesianControlConfig()
        self._q_dot_prev: np.ndarray | None = None

    @classmethod
    def from_arm(
        cls,
        arm: "Arm",
        config: CartesianControlConfig | None = None,
    ) -> "CartesianController":
        """Create a CartesianController from an Arm instance."""
        q_min, q_max = arm.get_joint_limits()
        return cls(
            model=arm.env.model,
            data=arm.env.data,
            ee_site_id=arm.ee_site_id,
            joint_qpos_indices=list(arm.joint_qpos_indices),
            joint_qvel_indices=list(arm.joint_qvel_indices),
            q_min=q_min,
            q_max=q_max,
            qd_max=arm.config.kinematic_limits.velocity,
            config=config,
        )

    def reset(self) -> None:
        """Clear warm-start state. Call before starting a new motion."""
        self._q_dot_prev = None

    def step(
        self,
        twist: np.ndarray,
        dt: float,
        frame: str = "world",
    ) -> TwistStepResult:
        """Execute one Cartesian control step.

        Computes joint velocities to achieve the desired twist, writes new
        joint positions to ``data.qpos``, and updates warm-start state.

        For teleop: call this once per control cycle from your loop. The
        simulator (``mj_forward`` / ``mj_step``) handles kinematics update.

        Args:
            twist: 6D desired twist [vx, vy, vz, wx, wy, wz].
            dt: Control timestep (seconds).
            frame: ``"world"`` or ``"hand"`` — frame for twist interpretation.

        Returns:
            TwistStepResult with joint velocities and diagnostics.
        """
        q_new, result = step_twist(
            model=self.model,
            data=self.data,
            ee_site_id=self.ee_site_id,
            joint_qpos_indices=self.joint_qpos_indices,
            joint_qvel_indices=self.joint_qvel_indices,
            q_min=self.q_min,
            q_max=self.q_max,
            qd_max=self.qd_max,
            twist=twist,
            frame=frame,
            dt=dt,
            config=self.config,
            q_dot_prev=self._q_dot_prev,
        )
        self._q_dot_prev = result.joint_velocities
        for i, idx in enumerate(self.joint_qpos_indices):
            self.data.qpos[idx] = q_new[i]
        return result

    def move(
        self,
        twist: np.ndarray,
        dt: float,
        *,
        max_duration: float = 5.0,
        max_distance: float | None = None,
        stop_condition: Callable[[], bool] | None = None,
        step_fn: Callable[[], None] | None = None,
    ) -> TwistExecutionResult:
        """Execute a constant twist until a stop condition is met.

        Runs an internal loop calling ``step()`` until ``max_duration``
        elapses, ``max_distance`` is reached, ``stop_condition`` returns
        True, or ``achieved_fraction`` drops below ``config.min_progress``.

        Useful for scripted Cartesian motions: approach, retreat, sweep.

        Args:
            twist: 6D desired twist [vx, vy, vz, wx, wy, wz].
            dt: Control timestep (seconds).
            max_duration: Stop after this many seconds.
            max_distance: Stop after EE moves this far (meters).
            stop_condition: Callable returning True to stop early.
            step_fn: Advance simulation after each step. Defaults to
                ``mj_forward`` for kinematic simulation.

        Returns:
            TwistExecutionResult describing why motion terminated.
        """
        if step_fn is None:
            def step_fn():
                mujoco.mj_forward(self.model, self.data)

        self.reset()

        t = 0.0
        distance = 0.0
        last_pos = self.data.site_xpos[self.ee_site_id].copy()
        terminated_by: Literal["duration", "distance", "condition", "no_progress"] = "duration"

        while t < max_duration:
            if stop_condition is not None and stop_condition():
                terminated_by = "condition"
                break

            result = self.step(twist, dt)
            step_fn()

            current_pos = self.data.site_xpos[self.ee_site_id].copy()
            distance += float(np.linalg.norm(current_pos - last_pos))
            last_pos = current_pos
            t += dt

            if result.achieved_fraction < self.config.min_progress:
                terminated_by = "no_progress"
                break

            if max_distance is not None and distance >= max_distance:
                terminated_by = "distance"
                break

        return TwistExecutionResult(
            terminated_by=terminated_by,
            distance_moved=distance,
            duration=t,
            final_pose=self._get_ee_pose(),
        )

    def move_to(
        self,
        target_pose: np.ndarray,
        dt: float,
        *,
        max_duration: float = 10.0,
        speed: float = 0.05,
        position_tol: float = 0.005,
        rotation_tol: float = 0.05,
        step_fn: Callable[[], None] | None = None,
    ) -> TwistExecutionResult:
        """Move end-effector to a target pose.

        Generates a twist proportional to the pose error and steps until
        within tolerance or ``max_duration`` elapses.

        Args:
            target_pose: 4x4 target pose matrix (world frame).
            dt: Control timestep (seconds).
            max_duration: Stop after this many seconds.
            speed: Maximum linear speed (m/s). Angular speed is bounded
                proportionally at ``speed / config.length_scale`` (rad/s).
            position_tol: Convergence threshold for position (meters).
            rotation_tol: Convergence threshold for rotation (radians).
            step_fn: Advance simulation after each step. Defaults to
                ``mj_forward`` for kinematic simulation.

        Returns:
            TwistExecutionResult. ``terminated_by == "condition"`` means
            the target was reached within tolerance.
        """
        if step_fn is None:
            def step_fn():
                mujoco.mj_forward(self.model, self.data)

        self.reset()

        t = 0.0
        distance = 0.0
        last_pos = self.data.site_xpos[self.ee_site_id].copy()
        max_omega = speed / self.config.length_scale
        terminated_by: Literal["duration", "distance", "condition", "no_progress"] = "duration"

        while t < max_duration:
            current_pos = self.data.site_xpos[self.ee_site_id].copy()
            current_R = self.data.site_xmat[self.ee_site_id].reshape(3, 3)

            pos_err = target_pose[:3, 3] - current_pos
            omega = _rotation_error_axis_angle(target_pose[:3, :3], current_R)

            pos_err_norm = float(np.linalg.norm(pos_err))
            rot_err_norm = float(np.linalg.norm(omega))

            if pos_err_norm < position_tol and rot_err_norm < rotation_tol:
                terminated_by = "condition"
                break

            v = pos_err * min(1.0, speed / (pos_err_norm + 1e-8))
            w = omega * min(1.0, max_omega / (rot_err_norm + 1e-8))
            twist = np.concatenate([v, w])

            result = self.step(twist, dt)
            step_fn()

            current_pos = self.data.site_xpos[self.ee_site_id].copy()
            distance += float(np.linalg.norm(current_pos - last_pos))
            last_pos = current_pos
            t += dt

            if result.achieved_fraction < self.config.min_progress:
                terminated_by = "no_progress"
                break

        return TwistExecutionResult(
            terminated_by=terminated_by,
            distance_moved=distance,
            duration=t,
            final_pose=self._get_ee_pose(),
        )

    def _get_ee_pose(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
        T[:3, 3] = self.data.site_xpos[self.ee_site_id]
        return T


def check_arm_contact_after_move(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm_body_ids: set[int],
    joint_qpos_indices: list[int],
    q_new: np.ndarray,
    exclude_self_collision: bool = True,
) -> str | None:
    """Check if moving to q_new would cause arm collision.

    Temporarily sets arm to proposed position, checks collision, restores.
    """
    q_saved = np.array([data.qpos[idx] for idx in joint_qpos_indices])

    for i, idx in enumerate(joint_qpos_indices):
        data.qpos[idx] = q_new[i]
    mujoco.mj_forward(model, data)

    contact_geom = check_arm_contact(
        model, data, arm_body_ids, exclude_self_collision
    )

    for i, idx in enumerate(joint_qpos_indices):
        data.qpos[idx] = q_saved[i]
    mujoco.mj_forward(model, data)

    return contact_geom
