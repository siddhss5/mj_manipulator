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
        return _read_site_pose(
            self._data, self._ee_site_id, self._tcp_offset
        )


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

    def __init__(
        self,
        env: Environment,
        config: ArmConfig,
        *,
        gripper: Gripper | None = None,
        grasp_manager: GraspManager | None = None,
        ik_solver: IKSolver | None = None,
    ):
        self.env = env
        self.config = config
        self.gripper = gripper
        self.grasp_manager = grasp_manager
        self.ik_solver = ik_solver

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
        if config.ee_site:
            self.ee_site_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, config.ee_site
            )
            if self.ee_site_id == -1:
                raise ValueError(
                    f"EE site '{config.ee_site}' not found in model"
                )
        else:
            self.ee_site_id = -1

        # Resolve actuator IDs (actuators whose transmission targets our joints)
        # Filter by trntype to exclude tendon/site actuators whose trnid
        # could collide with joint IDs (e.g. Franka gripper tendon actuator).
        self.actuator_ids: list[int] = []
        joint_id_set = set(self.joint_ids)
        for act_id in range(model.nu):
            trntype = model.actuator_trntype[act_id]
            if trntype == mujoco.mjtTrn.mjTRN_JOINT and (
                model.actuator_trnid[act_id, 0] in joint_id_set
            ):
                self.actuator_ids.append(act_id)

        # Cache DOF and joint limits
        self.dof = len(config.joint_names)
        self._joint_limits: tuple[np.ndarray, np.ndarray] | None = None

    # -----------------------------------------------------------------
    # State queries
    # -----------------------------------------------------------------

    def get_joint_positions(self) -> np.ndarray:
        """Current joint positions (rad)."""
        return np.array([
            self.env.data.qpos[idx] for idx in self.joint_qpos_indices
        ])

    def get_joint_velocities(self) -> np.ndarray:
        """Current joint velocities (rad/s)."""
        return np.array([
            self.env.data.qvel[idx] for idx in self.joint_qvel_indices
        ])

    def get_ee_pose(self) -> np.ndarray:
        """Current end-effector pose as 4x4 homogeneous transform.

        Calls mj_forward to ensure kinematics are up-to-date, then reads
        the EE site pose. Applies tcp_offset if configured.
        """
        if self.ee_site_id == -1:
            raise RuntimeError("No ee_site configured")
        mujoco.mj_forward(self.env.model, self.env.data)
        return _read_site_pose(
            self.env.data, self.ee_site_id, self.config.tcp_offset
        )

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

        # Collision checker with snapshot of current grasp state
        if self.grasp_manager is not None:
            grasped_objects = frozenset(self.grasp_manager.grasped.items())
            attachments = dict(self.grasp_manager._attachments)
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
    ) -> CBiRRTConfig:
        """Build a planner config from planning_defaults, with optional overrides."""
        defaults = self.config.planning_defaults
        if planner_config is not None:
            if timeout is not None:
                return dataclasses.replace(planner_config, timeout=timeout)
            return planner_config
        return CBiRRTConfig(
            timeout=timeout if timeout is not None else defaults.timeout,
            max_iterations=defaults.max_iterations,
            step_size=defaults.step_size,
            goal_bias=defaults.goal_bias,
            smoothing_iterations=defaults.smoothing_iterations,
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
    ) -> list[np.ndarray] | None:
        """Plan a collision-free path from current config to q_goal.

        Args:
            q_goal: Goal joint configuration.
            constraint_tsrs: Path constraints (all must be satisfied).
            timeout: Planning timeout in seconds (default from planning_defaults).
            seed: RNG seed for reproducibility.
            planner_config: Override planner configuration.

        Returns:
            List of waypoint configurations, or None if planning failed.
        """
        config = self._make_planner_config(timeout, planner_config)
        planner = self.create_planner(config)
        return planner.plan(
            start=self.get_joint_positions(),
            goal=q_goal,
            constraint_tsrs=constraint_tsrs,
            seed=seed,
        )

    def plan_to_configurations(
        self,
        q_goals: list[np.ndarray],
        *,
        constraint_tsrs: list[TSR] | None = None,
        timeout: float | None = None,
        seed: int | None = None,
        planner_config: CBiRRTConfig | None = None,
    ) -> list[np.ndarray] | None:
        """Plan to the nearest reachable goal from a set of configurations.

        Args:
            q_goals: List of candidate goal configurations.
            constraint_tsrs: Path constraints (all must be satisfied).
            timeout: Planning timeout in seconds (default from planning_defaults).
            seed: RNG seed for reproducibility.
            planner_config: Override planner configuration.

        Returns:
            Path to nearest reachable goal, or None if all failed.
        """
        config = self._make_planner_config(timeout, planner_config)
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

        Returns:
            Path to a goal satisfying the TSRs, or None if failed.
        """
        if self.ik_solver is None:
            raise RuntimeError(
                "plan_to_tsrs requires an IK solver. "
                "Pass ik_solver= to the Arm constructor."
            )
        config = self._make_planner_config(timeout, planner_config)
        planner = self.create_planner(config)
        return planner.plan(
            start=self.get_joint_positions(),
            goal_tsrs=goal_tsrs,
            constraint_tsrs=constraint_tsrs,
            seed=seed,
        )

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

    def solve(
        self, pose: np.ndarray, q_init: np.ndarray | None = None
    ) -> list[np.ndarray]:
        return []

    def solve_valid(
        self, pose: np.ndarray, q_init: np.ndarray | None = None
    ) -> list[np.ndarray]:
        return self.solve(pose, q_init)
