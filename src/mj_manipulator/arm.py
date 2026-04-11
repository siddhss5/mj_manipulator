# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Generic robot arm abstraction for MuJoCo manipulators.

Wraps an Environment + ArmConfig to provide:
- State queries (joint positions, EE pose, joint limits)
- Forward kinematics (non-destructive, for planning)
- Motion planning via pycbirrt (config-to-config, TSR-based, pose-based)

Robot-specific code (IK solvers, grippers) is injected via protocols.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from pycbirrt import CBiRRT, CBiRRTConfig
from tsr import TSR

from mj_manipulator.collision import CollisionChecker
from mj_manipulator.config import ArmConfig
from mj_manipulator.trajectory import Trajectory

if TYPE_CHECKING:
    from mj_environment import Environment

    from mj_manipulator.grasp_manager import GraspManager
    from mj_manipulator.protocols import Gripper, IKSolver

logger = logging.getLogger(__name__)


# =============================================================================
# pycbirrt RobotModel adapters
# =============================================================================


class ArmRobotModel:
    """Adapts Arm for pycbirrt's RobotModel protocol (single-threaded).

    Uses Arm.forward_kinematics() which creates a temporary MjData copy,
    so it's safe for planning but not thread-safe.
    """

    def __init__(self, arm: Arm):
        self._arm = arm

    @property
    def dof(self) -> int:
        return self._arm.dof

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self._arm.get_joint_limits()

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        return self._arm.forward_kinematics(q)


class ContextRobotModel:
    """Thread-safe RobotModel adapter using isolated MjData.

    Each instance owns a private MjData copy for FK computation.
    Created by Arm.create_planner() for parallel planning.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        joint_qpos_indices: list[int],
        ee_site_id: int,
        joint_limits: tuple[np.ndarray, np.ndarray],
        tcp_offset: np.ndarray | None = None,
    ):
        self._model = model
        self._data = data
        self._joint_qpos_indices = joint_qpos_indices
        self._ee_site_id = ee_site_id
        self._joint_limits = joint_limits
        self._tcp_offset = tcp_offset

    @property
    def dof(self) -> int:
        return len(self._joint_qpos_indices)

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self._joint_limits

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """Compute EE pose on private data (thread-safe)."""
        for i, idx in enumerate(self._joint_qpos_indices):
            self._data.qpos[idx] = q[i]
        mujoco.mj_forward(self._model, self._data)
        return _read_site_pose(self._data, self._ee_site_id, self._tcp_offset)


# =============================================================================
# Helpers
# =============================================================================


def _read_site_pose(
    data: mujoco.MjData,
    site_id: int,
    tcp_offset: np.ndarray | None = None,
) -> np.ndarray:
    """Read a 4x4 pose from a MuJoCo site, optionally applying tcp_offset."""
    pos = data.site_xpos[site_id]
    mat = data.site_xmat[site_id].reshape(3, 3)
    T = np.eye(4)
    T[:3, :3] = mat
    T[:3, 3] = pos
    if tcp_offset is not None:
        T = T @ tcp_offset
    return T


def add_subtree_gravcomp(
    spec: mujoco.MjSpec,
    root_body_name: str,
) -> int:
    """Enable gravity compensation on a body and all its descendants.

    Must be called **before** ``spec.compile()``. MuJoCo optimizes
    gravcomp away at compile time if every body has ``gravcomp=0``;
    runtime writes to ``model.body_gravcomp`` are silently ignored.
    That's why this helper operates on the MjSpec (editable) rather
    than a compiled ``MjModel``.

    Walks the MjSpec body tree rooted at ``root_body_name`` via an
    explicit stack (robust to whatever iteration the MjSpec API
    provides) and sets ``body.gravcomp = 1.0`` on every body it
    finds. This is the primitive that per-arm helpers like
    :func:`mj_manipulator.arms.franka.add_franka_gravcomp` delegate
    to — they just know the right root body name.

    Idempotent: calling twice on the same spec, or on overlapping
    subtrees, is harmless because setting ``gravcomp = 1.0`` twice
    produces the same result.

    Failure modes handled:

    - **root_body_name not found**: raises ``ValueError`` with the
      bad name AND the list of top-level world-body children, so
      typos are easy to diagnose.
    - **Root with no descendants**: still sets gravcomp on the root
      and returns ``count = 1``. Degenerate but valid.

    Not handled (caller's responsibility):

    - Calling on an already-compiled spec. The MjSpec API doesn't
      expose a reliable "was this compiled?" check; the call will
      still "succeed" but have no effect on the existing MjModel.
    - Scoping: if you pass a root that's an ancestor of bodies you
      don't want gravcomp'd (e.g. the world body, or a linear base
      beneath the arm), the walker will touch them too. Per-arm
      helpers sidestep this by passing the arm's kinematic root,
      not a higher ancestor.

    Args:
        spec: MjSpec loaded from a scene XML. Must not have been
            compiled yet (or the gravcomp change will be ignored).
        root_body_name: Name of the root body for the gravcomp
            subtree. Typically the arm's base link (e.g. ``"link0"``
            for Franka, ``"base"`` for UR5e).

    Returns:
        Number of bodies that had ``gravcomp`` set. Useful for
        sanity-checking that the walker touched the expected count
        (11 for Franka, 7 for bare UR5e, etc.).

    Raises:
        ValueError: If ``root_body_name`` is not found in the spec.
            The message includes the bad name and the list of
            top-level children of ``spec.worldbody`` so the caller
            can see what's actually there.
    """
    root = spec.body(root_body_name)
    if root is None:
        available = [b.name for b in spec.worldbody.bodies if b.name]
        raise ValueError(
            f"add_subtree_gravcomp: body '{root_body_name}' not found in spec. "
            f"Top-level worldbody children: {available}"
        )

    count = 0
    stack = [root]
    while stack:
        body = stack.pop()
        body.gravcomp = 1.0
        count += 1
        stack.extend(body.bodies)
    return count


# =============================================================================
# Arm
# =============================================================================


class Arm:
    """Generic robot arm abstraction.

    Provides state queries, forward kinematics, and motion planning for
    any MuJoCo robot arm. Robot-specific capabilities (IK, gripper) are
    injected via protocols.

    Args:
        env: MuJoCo environment (provides model and data).
        config: Arm configuration (joint names, limits, ee_site, etc.).
        gripper: Optional gripper implementation.
        grasp_manager: Optional grasp state tracker.
        ik_solver: Optional IK solver for pose-based planning.
    """

    env: Environment
    config: ArmConfig
    gripper: Gripper | None
    grasp_manager: GraspManager | None
    ik_solver: IKSolver | None
    joint_ids: list[int]
    joint_qpos_indices: list[int]
    joint_qvel_indices: list[int]
    ee_site_id: int
    dof: int

    def __init__(
        self,
        env: Environment,
        config: ArmConfig,
        *,
        gripper: Gripper | None = None,
        grasp_manager: GraspManager | None = None,
        ik_solver: IKSolver | None = None,
    ):
        self.env: Environment = env
        self.config: ArmConfig = config
        self.gripper: Gripper | None = gripper
        self.grasp_manager: GraspManager | None = grasp_manager
        self.ik_solver: IKSolver | None = ik_solver
        self.ft_valid: bool = False  # Set by ExecutionContext when F/T is meaningful
        self._ft_tare_offset: np.ndarray = np.zeros(6)  # Tare baseline

        model = env.model

        # Resolve joint IDs and cache indices
        self.joint_ids: list[int] = []
        self.joint_qpos_indices: list[int] = []
        self.joint_qvel_indices: list[int] = []

        for name in config.joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid == -1:
                raise ValueError(f"Joint '{name}' not found in model")
            self.joint_ids.append(jid)
            self.joint_qpos_indices.append(model.jnt_qposadr[jid])
            self.joint_qvel_indices.append(model.jnt_dofadr[jid])

        # Resolve EE site
        self.ee_site_id: int
        if config.ee_site:
            self.ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, config.ee_site)
            if self.ee_site_id == -1:
                raise ValueError(f"EE site '{config.ee_site}' not found in model")
        else:
            self.ee_site_id = -1

        # Resolve actuator IDs (actuators whose transmission targets our joints)
        # Filter by trntype to exclude tendon/site actuators whose trnid
        # could collide with joint IDs (e.g. Franka gripper tendon actuator).
        self.actuator_ids: list[int] = []
        joint_id_set = set(self.joint_ids)
        for act_id in range(model.nu):
            trntype = model.actuator_trntype[act_id]
            if trntype == mujoco.mjtTrn.mjTRN_JOINT and (model.actuator_trnid[act_id, 0] in joint_id_set):
                self.actuator_ids.append(act_id)

        # Cache DOF and joint limits
        self.dof: int = len(config.joint_names)
        self._joint_limits: tuple[np.ndarray, np.ndarray] | None = None

        # Check whether gravity compensation is active on the arm subtree.
        # MuJoCo optimizes gravcomp away at compile time if every body has
        # gravcomp=0, and runtime changes to body_gravcomp are silently
        # ignored. The per-arm MjSpec helpers (add_franka_gravcomp,
        # add_ur5e_gravcomp) must be called BEFORE spec.compile(). Real
        # robot controllers (Franka FCI, UR RTDE, KUKA FRI, etc.) run
        # gravity compensation internally; without it in sim, the PD loop
        # must fight gravity via steady-state position error, producing
        # sag at rest and tracking lag in motion.
        first_joint_body = model.jnt_bodyid[self.joint_ids[0]]
        base_body_id = model.body_parentid[first_joint_body]
        subtree_has_gravcomp = any(model.body_gravcomp[bid] > 0 for bid in range(base_body_id, model.nbody))
        if not subtree_has_gravcomp:
            logger.warning(
                "Arm '%s' has no gravity compensation on its kinematic "
                "subtree. Call add_%s_gravcomp(spec) BEFORE spec.compile() "
                "(or bake gravcomp='1' into the source XML). Without it, "
                "the PD loop must fight gravity via steady-state position "
                "error, producing sag at rest and tracking lag in motion.",
                config.name,
                config.name,
            )

        # Resolve F/T sensor indices (if configured)
        self._ft_force_adr: int | None = None
        self._ft_torque_adr: int | None = None
        self.ft_site_id: int | None = None
        if config.ft_force_sensor:
            sid = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_SENSOR,
                config.ft_force_sensor,
            )
            if sid == -1:
                raise ValueError(f"Force sensor '{config.ft_force_sensor}' not found")
            self._ft_force_adr = model.sensor_adr[sid]
            self.ft_site_id = model.sensor_objid[sid]
        if config.ft_torque_sensor:
            sid = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_SENSOR,
                config.ft_torque_sensor,
            )
            if sid == -1:
                raise ValueError(f"Torque sensor '{config.ft_torque_sensor}' not found")
            self._ft_torque_adr = model.sensor_adr[sid]

    # -----------------------------------------------------------------
    # State queries
    # -----------------------------------------------------------------

    def get_joint_positions(self) -> np.ndarray:
        """Current joint positions (rad)."""
        return np.array([self.env.data.qpos[idx] for idx in self.joint_qpos_indices])

    def set_joint_positions(self, q: np.ndarray, ctx=None) -> None:
        """Set joint positions directly, sync viewer.

        Simulation only — teleports the arm to the target configuration.
        On real hardware, use plan_to_configuration() instead.

        Args:
            q: Joint positions (rad), length must match DOF.
            ctx: ExecutionContext for syncing. If None, runs mj_forward only.
        """
        q = np.asarray(q, dtype=float)
        if len(q) != self.dof:
            raise ValueError(f"Expected {self.dof} joints, got {len(q)}")
        lower, upper = self.get_joint_limits()
        for i in range(self.dof):
            if q[i] < lower[i] or q[i] > upper[i]:
                raise ValueError(f"Joint {i} value {q[i]:.3f} outside limits [{lower[i]:.3f}, {upper[i]:.3f}]")
        for i, idx in enumerate(self.joint_qpos_indices):
            self.env.data.qpos[idx] = q[i]
        for idx in self.joint_qvel_indices:
            self.env.data.qvel[idx] = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)
        if ctx is not None:
            ctx.sync()

    def get_joint_velocities(self) -> np.ndarray:
        """Current joint velocities (rad/s)."""
        return np.array([self.env.data.qvel[idx] for idx in self.joint_qvel_indices])

    def get_joint_torques(self) -> np.ndarray:
        """Current joint-torque vector (N·m per joint).

        Returns ``qfrc_actuator`` for this arm's joints — the torque each
        actuator applies to the joint. With gravity compensation active,
        this reduces (at rest) largely to whatever external load the arm
        is working against: the weight of a held object, contact forces,
        etc. That makes it a useful load signal for arms whose only load
        sensing is at the joints (e.g. Franka via ``tau_ext``), parallel
        to :meth:`get_ft_wrench` for arms with a wrist F/T sensor.

        Returns NaN when ``ft_valid`` is False — the same validity gate
        as F/T. Kinematic sim doesn't run physics integration, so
        ``qfrc_actuator`` is meaningless there. On real hardware the
        ``HardwareContext`` supplies the driver's external-torque
        estimate and sets ``ft_valid = True``.

        Returns:
            np.ndarray of shape ``(dof,)``: actuator torques in N·m,
            indexed to match ``get_joint_positions()``. All NaN if joint
            torque data is not currently meaningful.
        """
        if not self.ft_valid:
            return np.full(self.dof, np.nan)
        data = self.env.data
        return np.array([data.qfrc_actuator[idx] for idx in self.joint_qvel_indices])

    def get_ft_wrench(self) -> np.ndarray:
        """Current wrist force/torque reading as [fx, fy, fz, tx, ty, tz].

        Returns the 6D wrench from the wrist F/T sensor in the **sensor
        local frame** (not world frame). The sensor reports the force
        exerted on the child body (gripper) by the parent body (wrist).

        Returns NaN when ``ft_valid`` is False (default). The execution
        context sets ``ft_valid = True`` when F/T data is meaningful:
        physics sim after ``mj_step``, or real hardware with a live
        sensor. In kinematic sim, MuJoCo's constraint solver produces
        artifact values (100-300N) that are not physical wrist forces.

        To transform to world frame::

            wrench = arm.get_ft_wrench()
            R = data.site_xmat[arm.ft_site_id].reshape(3, 3)
            force_world = R @ wrench[:3]
            torque_world = R @ wrench[3:]

        Returns:
            np.ndarray of shape (6,): [fx, fy, fz, tx, ty, tz] in sensor frame.
            All NaN if no physics step has been run.

        Raises:
            RuntimeError: If no F/T sensor is configured.
        """
        if self._ft_force_adr is None or self._ft_torque_adr is None:
            raise RuntimeError("No F/T sensor configured. Set ft_force_sensor and ft_torque_sensor in ArmConfig.")
        if not self.ft_valid:
            return np.full(6, np.nan)
        data = self.env.data
        force = data.sensordata[self._ft_force_adr : self._ft_force_adr + 3]
        torque = data.sensordata[self._ft_torque_adr : self._ft_torque_adr + 3]
        return np.concatenate([force, torque]) - self._ft_tare_offset

    def get_ft_wrench_world(self) -> np.ndarray:
        """Current wrist force/torque reading in the **world frame**.

        Convenience wrapper around :meth:`get_ft_wrench` that rotates
        the wrench from the sensor local frame to the world frame.

        Returns:
            np.ndarray of shape (6,): [fx, fy, fz, tx, ty, tz] in world frame.
            All NaN if F/T is not valid (kinematic mode).
        """
        wrench = self.get_ft_wrench()
        if np.isnan(wrench[0]):
            return wrench
        R = self.env.data.site_xmat[self.ft_site_id].reshape(3, 3)
        force_world = R @ wrench[:3]
        torque_world = R @ wrench[3:]
        return np.concatenate([force_world, torque_world])

    def tare_ft(self) -> None:
        """Zero the F/T sensor at the current reading (tare).

        Records the current raw sensor reading as the baseline. All
        subsequent ``get_ft_wrench()`` calls return the delta from
        this baseline. Arm should be stationary when taring.

        Matches UR5e's ``zero_ftsensor()`` URScript command.
        """
        if self._ft_force_adr is None or self._ft_torque_adr is None:
            raise RuntimeError("No F/T sensor configured.")
        if not self.ft_valid:
            raise RuntimeError("F/T not valid (kinematic mode). Use physics mode to tare.")
        data = self.env.data
        force = data.sensordata[self._ft_force_adr : self._ft_force_adr + 3]
        torque = data.sensordata[self._ft_torque_adr : self._ft_torque_adr + 3]
        self._ft_tare_offset = np.concatenate([force, torque]).copy()

    @property
    def has_ft_sensor(self) -> bool:
        """Whether this arm has a wrist F/T sensor configured."""
        return self._ft_force_adr is not None and self._ft_torque_adr is not None

    def get_ee_pose(self) -> np.ndarray:
        """Current end-effector pose as 4x4 homogeneous transform.

        Calls mj_forward to ensure kinematics are up-to-date, then reads
        the EE site pose. Applies tcp_offset if configured.
        """
        if self.ee_site_id == -1:
            raise RuntimeError("No ee_site configured")
        mujoco.mj_forward(self.env.model, self.env.data)
        return _read_site_pose(self.env.data, self.ee_site_id, self.config.tcp_offset)

    def get_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Joint position limits as (lower, upper) arrays."""
        if self._joint_limits is None:
            model = self.env.model
            lower = np.array([model.jnt_range[jid, 0] for jid in self.joint_ids])
            upper = np.array([model.jnt_range[jid, 1] for jid in self.joint_ids])
            self._joint_limits = (lower, upper)
        return self._joint_limits

    # -----------------------------------------------------------------
    # Forward kinematics (non-destructive, for planning)
    # -----------------------------------------------------------------

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """Compute EE pose at configuration q without modifying live state.

        Creates a temporary MjData copy, sets joints to q, runs mj_forward,
        and reads the resulting pose. The live env.data is never touched.
        """
        if self.ee_site_id == -1:
            raise RuntimeError("No ee_site configured")

        tmp_data = mujoco.MjData(self.env.model)
        # Copy current state as baseline
        np.copyto(tmp_data.qpos, self.env.data.qpos)
        # Set arm joints to requested config
        for i, idx in enumerate(self.joint_qpos_indices):
            tmp_data.qpos[idx] = q[i]
        mujoco.mj_forward(self.env.model, tmp_data)
        return _read_site_pose(tmp_data, self.ee_site_id, self.config.tcp_offset)

    # -----------------------------------------------------------------
    # Planning
    # -----------------------------------------------------------------

    def check_collisions(self) -> list[tuple[str, str, float]]:
        """Check current configuration for collisions.

        Uses the same collision checker as the planner, including
        grasp-aware filtering (gripper-to-held-object contacts are OK).

        Returns:
            List of (arm_body, other_body, penetration_mm) tuples.
            Empty list if collision-free. Prints a summary.

        Example::

            robot.right.check_collisions()
            # Right arm: 2 contacts
            #   forearm_link <-> sugar_box_0: 2.3mm
            #   gripper/pad <-> table: 0.8mm
        """
        planner = self.create_planner()
        contacts = planner.collision.get_contacts(self.get_joint_positions())
        if contacts:
            print(f"{self.config.name} arm: {len(contacts)} contact(s)")
            for arm_body, other, depth in contacts:
                # Shorten body names for readability
                arm_short = arm_body.split("/", 1)[-1] if "/" in arm_body else arm_body
                other_short = other.split("/", 1)[-1] if "/" in other else other
                print(f"  {arm_short} <-> {other_short}: {depth:.1f}mm")
        else:
            print(f"{self.config.name} arm: collision-free")
        return contacts

    def create_planner(
        self,
        config: CBiRRTConfig | None = None,
    ) -> CBiRRT:
        """Create a thread-safe planner with isolated state.

        Each planner has its own MjData copy and adapters, so multiple
        planners can run in parallel threads.

        Args:
            config: Planner configuration. Defaults built from
                    self.config.planning_defaults.

        Returns:
            Configured CBiRRT planner ready to call .plan().
        """
        if config is None:
            defaults = self.config.planning_defaults
            config = CBiRRTConfig(
                timeout=defaults.timeout,
                max_iterations=defaults.max_iterations,
                step_size=defaults.step_size,
                goal_bias=defaults.goal_bias,
                smoothing_iterations=defaults.smoothing_iterations,
            )

        # Fork environment for isolated planning state
        planning_env = self.env.fork()
        model = planning_env.model
        data = planning_env.data

        # Build adapters
        robot_model = ContextRobotModel(
            model=model,
            data=data,
            joint_qpos_indices=self.joint_qpos_indices,
            ee_site_id=self.ee_site_id,
            joint_limits=self.get_joint_limits(),
            tcp_offset=self.config.tcp_offset,
        )

        # Collision checker with snapshot of current grasp state.
        # Only include objects grasped by THIS arm — objects held by other
        # arms are static obstacles, not part of this arm's robot model.
        if self.grasp_manager is not None:
            arm_name = self.config.name
            grasped_objects = frozenset(
                (obj, arm) for obj, arm in self.grasp_manager.grasped.items() if arm == arm_name
            )
            attachments = {
                obj: att for obj, att in self.grasp_manager._attachments.items() if obj in dict(grasped_objects)
            }
            collision_checker = CollisionChecker(
                model=model,
                data=data,
                joint_names=self.config.joint_names,
                grasped_objects=grasped_objects,
                attachments=attachments,
            )
        else:
            collision_checker = CollisionChecker(
                model=model,
                data=data,
                joint_names=self.config.joint_names,
            )

        # IK solver — use injected solver or a no-op stub
        ik = self.ik_solver if self.ik_solver is not None else _NoIKSolver()

        return CBiRRT(
            robot=robot_model,
            ik_solver=ik,
            collision_checker=collision_checker,
            config=config,
        )

    def _make_planner_config(
        self,
        timeout: float | None,
        planner_config: CBiRRTConfig | None,
        abort_fn: Callable[[], bool] | None = None,
    ) -> CBiRRTConfig:
        """Build a planner config from planning_defaults, with optional overrides."""
        defaults = self.config.planning_defaults
        if planner_config is not None:
            overrides = {}
            if timeout is not None:
                overrides["timeout"] = timeout
            if abort_fn is not None:
                overrides["abort_fn"] = abort_fn
            return dataclasses.replace(planner_config, **overrides) if overrides else planner_config
        return CBiRRTConfig(
            timeout=timeout if timeout is not None else defaults.timeout,
            max_iterations=defaults.max_iterations,
            step_size=defaults.step_size,
            goal_bias=defaults.goal_bias,
            smoothing_iterations=defaults.smoothing_iterations,
            abort_fn=abort_fn,
        )

    def _make_pose_tsr(self, pose: np.ndarray) -> TSR:
        """Create a point TSR for an EE pose, encoding tcp_offset as Tw_e."""
        Tw_e = self.config.tcp_offset if self.config.tcp_offset is not None else None
        return TSR(T0_w=pose, Tw_e=Tw_e)

    def plan_to_configuration(
        self,
        q_goal: np.ndarray,
        *,
        constraint_tsrs: list[TSR] | None = None,
        timeout: float | None = None,
        seed: int | None = None,
        planner_config: CBiRRTConfig | None = None,
        abort_fn: Callable[[], bool] | None = None,
    ) -> list[np.ndarray] | None:
        """Plan a collision-free path from current config to q_goal.

        Args:
            q_goal: Goal joint configuration.
            constraint_tsrs: Path constraints (all must be satisfied).
            timeout: Planning timeout in seconds (default from planning_defaults).
            seed: RNG seed for reproducibility.
            planner_config: Override planner configuration.
            abort_fn: If provided, called each iteration; return True to abort.

        Returns:
            List of waypoint configurations, or None if planning failed.
        """
        config = self._make_planner_config(timeout, planner_config, abort_fn=abort_fn)
        planner = self.create_planner(config)
        path = planner.plan(
            start=self.get_joint_positions(),
            goal=q_goal,
            constraint_tsrs=constraint_tsrs,
            seed=seed,
        )
        if path is not None:
            logger.info("Plan to configuration succeeded: %d waypoints", len(path))
        else:
            logger.info("Plan to configuration failed")
        return path

    def plan_to_configurations(
        self,
        q_goals: list[np.ndarray],
        *,
        constraint_tsrs: list[TSR] | None = None,
        timeout: float | None = None,
        seed: int | None = None,
        planner_config: CBiRRTConfig | None = None,
        abort_fn: Callable[[], bool] | None = None,
    ) -> list[np.ndarray] | None:
        """Plan to the nearest reachable goal from a set of configurations.

        Args:
            q_goals: List of candidate goal configurations.
            constraint_tsrs: Path constraints (all must be satisfied).
            timeout: Planning timeout in seconds (default from planning_defaults).
            seed: RNG seed for reproducibility.
            planner_config: Override planner configuration.
            abort_fn: If provided, called each iteration; return True to abort.

        Returns:
            Path to nearest reachable goal, or None if all failed.
        """
        config = self._make_planner_config(timeout, planner_config, abort_fn=abort_fn)
        planner = self.create_planner(config)
        return planner.plan(
            start=self.get_joint_positions(),
            goal=q_goals,
            constraint_tsrs=constraint_tsrs,
            seed=seed,
        )

    def plan_to_tsrs(
        self,
        goal_tsrs: list[TSR],
        *,
        constraint_tsrs: list[TSR] | None = None,
        timeout: float | None = None,
        seed: int | None = None,
        planner_config: CBiRRTConfig | None = None,
        return_details: bool = False,
        abort_fn: Callable[[], bool] | None = None,
    ) -> list[np.ndarray] | None:
        """Plan to a TSR-defined goal region.

        The planner samples poses from the goal TSRs, solves IK
        internally, and plans a collision-free path.

        Args:
            goal_tsrs: Goal TSRs (union — any is acceptable).
            constraint_tsrs: Path constraints (all must be satisfied).
            timeout: Planning timeout in seconds (default from planning_defaults).
            seed: RNG seed for reproducibility.
            planner_config: Override planner configuration.
            return_details: If True, return pycbirrt PlanResult with indices
                and stats instead of just the path.
            abort_fn: If provided, called each iteration; return True to abort.

        Returns:
            Path to a goal satisfying the TSRs, or None if failed.
            If return_details=True, returns pycbirrt.PlanResult instead.
        """
        if self.ik_solver is None:
            raise RuntimeError("plan_to_tsrs requires an IK solver. Pass ik_solver= to the Arm constructor.")
        config = self._make_planner_config(timeout, planner_config, abort_fn=abort_fn)
        planner = self.create_planner(config)
        result = planner.plan(
            start=self.get_joint_positions(),
            goal_tsrs=goal_tsrs,
            constraint_tsrs=constraint_tsrs,
            seed=seed,
            return_details=return_details,
        )
        if return_details:
            if result is not None and result.success:
                logger.info(
                    "Plan to TSRs succeeded: %d waypoints in %.1fs",
                    len(result.path),
                    result.planning_time,
                )
            elif result is not None:
                logger.info("Plan to TSRs failed: %s", result.failure_reason or "unknown")
            else:
                logger.info("Plan to TSRs failed: no result")
        elif result is None:
            logger.info("Plan to TSRs failed")
        return result

    def plan_to_pose(
        self,
        pose: np.ndarray,
        *,
        constraint_tsrs: list[TSR] | None = None,
        timeout: float | None = None,
        seed: int | None = None,
        planner_config: CBiRRTConfig | None = None,
    ) -> list[np.ndarray] | None:
        """Plan to an end-effector pose.

        Creates a point TSR from the pose and delegates to plan_to_tsrs.
        The planner handles IK internally.

        If tcp_offset is configured, pose should be the tool center point
        pose (the offset is applied internally via the TSR). Otherwise,
        pose is the EE site pose directly.

        Args:
            pose: 4x4 target pose (tcp frame if tcp_offset set, else EE site).
            constraint_tsrs: Path constraints (all must be satisfied).
            timeout: Planning timeout in seconds (default from planning_defaults).
            seed: RNG seed for reproducibility.
            planner_config: Override planner configuration.

        Returns:
            Path to pose, or None if planning fails.
        """
        return self.plan_to_tsrs(
            goal_tsrs=[self._make_pose_tsr(pose)],
            constraint_tsrs=constraint_tsrs,
            timeout=timeout,
            seed=seed,
            planner_config=planner_config,
        )

    def plan_to_poses(
        self,
        poses: list[np.ndarray],
        *,
        constraint_tsrs: list[TSR] | None = None,
        timeout: float | None = None,
        seed: int | None = None,
        planner_config: CBiRRTConfig | None = None,
    ) -> list[np.ndarray] | None:
        """Plan to any of the given end-effector poses.

        Creates a union of point TSRs and delegates to plan_to_tsrs.

        If tcp_offset is configured, poses should be tool center point
        poses (the offset is applied internally via the TSR). Otherwise,
        poses are EE site poses directly.

        Args:
            poses: List of 4x4 target poses (tcp frame if tcp_offset set, else EE site).
            constraint_tsrs: Path constraints (all must be satisfied).
            timeout: Planning timeout in seconds (default from planning_defaults).
            seed: RNG seed for reproducibility.
            planner_config: Override planner configuration.

        Returns:
            Path to nearest reachable pose, or None if planning fails.
        """
        return self.plan_to_tsrs(
            goal_tsrs=[self._make_pose_tsr(p) for p in poses],
            constraint_tsrs=constraint_tsrs,
            timeout=timeout,
            seed=seed,
            planner_config=planner_config,
        )

    def retime(
        self,
        path: list[np.ndarray],
        *,
        control_dt: float = 0.008,
    ) -> Trajectory:
        """Time-parameterize a path using TOPP-RA.

        Converts a geometric path (from any plan_to_* method) into a
        time-optimal trajectory respecting the arm's kinematic limits.

        Args:
            path: List of waypoint configurations from a planner.
            control_dt: Control timestep for trajectory sampling (default 125 Hz).

        Returns:
            Time-optimal Trajectory with positions, velocities, and accelerations.
        """
        limits = self.config.kinematic_limits
        return Trajectory.from_path(
            path=path,
            vel_limits=limits.velocity,
            acc_limits=limits.acceleration,
            control_dt=control_dt,
            entity=self.config.name,
            joint_names=self.config.joint_names,
        )


class _NoIKSolver:
    """Stub IK solver that returns no solutions.

    Used when no IK solver is injected. Config-to-config planning
    still works; only pose/TSR-based planning requires real IK.
    """

    def solve(self, pose: np.ndarray, q_init: np.ndarray | None = None) -> list[np.ndarray]:
        return []

    def solve_valid(self, pose: np.ndarray, q_init: np.ndarray | None = None) -> list[np.ndarray]:
        return self.solve(pose, q_init)
