# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Base controller for multi-arm MuJoCo manipulation.

Provides the shared orchestration layer for physics, kinematic, and (future)
hardware controllers. Owns target state management, non-blocking trajectory
runners, blocking trajectory execution, and executor wrappers.

Subclasses implement :meth:`_apply_targets_and_step` (one control cycle),
:meth:`close_gripper`, and :meth:`open_gripper`.

Architecture::

    Controller (ABC)
    ├── PhysicsController       — ctrl writes + mj_step + verifier ticks
    ├── KinematicController     — qpos writes + mj_forward + grasp manager update
    └── (future) HardwareController — RTDE/ROS commands + encoder reads
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import Future
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import mujoco
import numpy as np

if TYPE_CHECKING:
    from mj_manipulator.arm import Arm
    from mj_manipulator.config import ExecutionConfig
    from mj_manipulator.protocols import Gripper
    from mj_manipulator.trajectory import Trajectory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal per-arm / per-gripper state
# ---------------------------------------------------------------------------


@dataclass
class _ArmState:
    """Per-arm state managed by Controller."""

    arm: Arm
    actuator_ids: np.ndarray  # int array for vectorized ctrl writes
    joint_qpos_indices: np.ndarray
    joint_qvel_indices: np.ndarray
    target_position: np.ndarray
    target_velocity: np.ndarray
    lookahead: float | None = None  # None = use controller default


@dataclass
class _EntityState:
    """Per-entity state for non-arm controllable entities (e.g. linear bases)."""

    actuator_ids: np.ndarray
    joint_qpos_indices: np.ndarray
    joint_qvel_indices: np.ndarray
    target_position: np.ndarray
    target_velocity: np.ndarray


@dataclass
class _GripperState:
    """Per-gripper state managed by Controller."""

    gripper: Gripper
    actuator_id: int
    ctrl_open: float
    ctrl_closed: float
    target_ctrl: float


# ---------------------------------------------------------------------------
# TrajectoryRunner
# ---------------------------------------------------------------------------


class TrajectoryRunner:
    """Non-blocking trajectory target provider.

    Advances one waypoint per call to :meth:`advance`, writing targets to
    the arm's ``_ArmState``. Does NOT call ``mj_step`` — the event loop
    owns stepping.

    This is the "update method" pattern from game programming: the game
    loop calls ``runner.advance()`` each frame, then steps once with all
    arms' targets applied.

    Created via :meth:`Controller.start_trajectory`.
    """

    def __init__(
        self,
        controller: Controller,
        entity_name: str,
        trajectory: Trajectory,
        abort_fn: Callable[[], bool] | None = None,
    ):
        self._controller = controller
        self._entity_name = entity_name
        self._trajectory = trajectory
        self._abort_fn = abort_fn
        self._waypoint_index = 0
        self._done = False
        self._converging = False
        self._convergence_steps = 0
        self._future: Future[bool] = Future()
        self._realtime = controller.viewer is not None
        self._t_start = time.time() if self._realtime else 0.0

        # Resolve state from arms or entities
        if entity_name in controller._arms:
            self._state = controller._arms[entity_name]
            self._is_arm = True
        elif entity_name in controller._entities:
            self._state = controller._entities[entity_name]
            self._is_arm = False
        else:
            raise ValueError(f"Unknown arm or entity: {entity_name}")

    @property
    def entity_name(self) -> str:
        """Which arm or entity this runner controls."""
        return self._entity_name

    @property
    def done(self) -> bool:
        """Whether the trajectory has completed (success or abort)."""
        return self._done

    @property
    def future(self) -> Future[bool]:
        """Future that resolves when the trajectory completes."""
        return self._future

    def advance(self) -> None:
        """Write next waypoint target to arm state.

        Call once per tick. After the last waypoint, enters convergence
        mode (checking position/velocity error each tick).
        """
        if self._done:
            return

        if self._abort_fn is not None and self._abort_fn():
            self._state.target_velocity = np.zeros(len(self._state.actuator_ids))
            self._finish(False)
            logger.info(
                "Trajectory aborted at waypoint %d/%d",
                self._waypoint_index,
                self._trajectory.num_waypoints,
            )
            return

        if self._converging:
            self._advance_convergence()
            return

        state = self._state
        traj = self._trajectory

        state.target_position = traj.positions[self._waypoint_index]
        state.target_velocity = traj.velocities[self._waypoint_index]

        self._waypoint_index += 1

        if self._waypoint_index >= traj.num_waypoints:
            state.target_position = traj.positions[-1].copy()
            state.target_velocity = np.zeros(len(state.actuator_ids))
            if self._is_arm:
                # Arms need convergence settling (PD dynamics in physics,
                # instant in kinematic — tolerance check passes immediately)
                self._converging = True
            else:
                # Entities (bases) don't need convergence
                self._finish(True)

    def _advance_convergence(self) -> None:
        """Check convergence one step at a time."""
        cfg = self._controller.config
        state = self._state
        data = self._controller.data

        current_pos = data.qpos[state.joint_qpos_indices]
        pos_error = np.abs(state.target_position - current_pos)
        current_vel = data.qvel[state.joint_qvel_indices]

        if np.all(pos_error < cfg.position_tolerance) and np.all(np.abs(current_vel) < cfg.velocity_tolerance):
            self._finish(True)
            return

        self._convergence_steps += 1
        if self._convergence_steps >= cfg.convergence_timeout_steps:
            logger.warning(
                "Convergence timeout for %s: max_pos_err=%.2f° (limit %.2f°)",
                self._entity_name,
                np.rad2deg(float(np.max(pos_error))),
                np.rad2deg(cfg.position_tolerance),
            )
            self._finish(True)  # timed out but close enough

    def _finish(self, success: bool) -> None:
        self._done = True
        if not self._future.done():
            self._future.set_result(success)


# ---------------------------------------------------------------------------
# Executor wrappers
# ---------------------------------------------------------------------------


class ArmExecutor:
    """Executor interface for one arm, backed by a Controller.

    Provides the standard ``Executor.execute(trajectory)`` interface while
    ensuring all other actuators hold their positions during execution.
    Obtained via :meth:`Controller.get_executor`.
    """

    def __init__(self, controller: Controller, arm_name: str):
        self.controller = controller
        self.arm_name = arm_name

    def execute(self, trajectory: Trajectory) -> bool:
        """Execute trajectory on this arm."""
        return self.controller.execute(self.arm_name, trajectory)


class EntityExecutor:
    """Executor interface for a non-arm entity, backed by a Controller.

    Same pattern as ArmExecutor but routes to execute_entity().
    Obtained via :meth:`Controller.get_entity_executor`.
    """

    def __init__(self, controller: Controller, entity_name: str):
        self.controller = controller
        self.entity_name = entity_name

    def execute(self, trajectory: Trajectory) -> bool:
        """Execute trajectory on this entity."""
        return self.controller.execute_entity(self.entity_name, trajectory)


# ---------------------------------------------------------------------------
# Controller base class
# ---------------------------------------------------------------------------


class Controller(ABC):
    """Base controller for multi-arm MuJoCo manipulation.

    Coordinates all arms, entities, and grippers. Provides target state
    management, non-blocking trajectory runners (driven by the event loop),
    blocking trajectory execution (for no-event-loop usage), and per-arm
    executor wrappers.

    Subclasses override :meth:`_apply_targets_and_step` to define how
    targets become motion:

    - **PhysicsController**: writes ``data.ctrl`` with lookahead feedforward,
      calls ``mj_step``, ticks grasp verifiers, syncs viewer.
    - **KinematicController**: writes ``data.qpos`` directly, calls
      ``mj_forward``, updates grasp manager attached poses, syncs viewer.
    - **(future) HardwareController**: sends to RTDE/ROS, reads encoders.

    Args:
        model: MuJoCo model.
        data: MuJoCo data (modified during execution).
        arms: Dict mapping arm names to Arm instances.
        config: Execution parameters (control_dt, convergence tolerances, etc.).
        gripper_config: Gripper control parameters (physics-mode close steps, etc.).
        viewer: Optional viewer to sync during execution.
        viewer_sync_interval: Minimum seconds between viewer syncs.
        initial_positions: Optional per-arm initial joint positions.
        entities: Optional dict of non-arm controllable entities.
        abort_fn: Optional global abort predicate.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        arms: dict[str, Arm],
        *,
        config: ExecutionConfig | None = None,
        gripper_config: object | None = None,
        viewer=None,
        viewer_sync_interval: float = 0.033,
        initial_positions: dict[str, np.ndarray] | None = None,
        entities: dict[str, object] | None = None,
        abort_fn=None,
    ):
        from mj_manipulator.config import ExecutionConfig as _ExecutionConfig

        self.model = model
        self.data = data
        self.viewer = viewer
        self._abort_fn = abort_fn

        self.config = config or _ExecutionConfig()
        self.gripper_config = gripper_config

        self.control_dt = self.config.control_dt
        self.lookahead_time = self.config.lookahead_time

        self._last_viewer_sync = 0.0
        self._viewer_sync_interval = viewer_sync_interval

        # Build per-arm state
        self._arms: dict[str, _ArmState] = {}
        self._grippers: dict[str, _GripperState] = {}

        for name, arm in arms.items():
            if initial_positions and name in initial_positions:
                target_pos = np.asarray(initial_positions[name]).copy()
            else:
                target_pos = arm.get_joint_positions().copy()

            self._arms[name] = _ArmState(
                arm=arm,
                actuator_ids=np.array(arm.actuator_ids, dtype=np.intp),
                joint_qpos_indices=np.array(arm.joint_qpos_indices, dtype=np.intp),
                joint_qvel_indices=np.array(arm.joint_qvel_indices, dtype=np.intp),
                target_position=target_pos,
                target_velocity=np.zeros(arm.dof),
            )

            gripper = arm.gripper
            if gripper is not None and gripper.actuator_id is not None:
                self._grippers[name] = _GripperState(
                    gripper=gripper,
                    actuator_id=gripper.actuator_id,
                    ctrl_open=gripper.ctrl_open,
                    ctrl_closed=gripper.ctrl_closed,
                    target_ctrl=data.ctrl[gripper.actuator_id],
                )

        # Build per-entity state (linear bases, etc.)
        self._entities: dict[str, _EntityState] = {}
        for name, entity in (entities or {}).items():
            qpos_idx = np.array(entity.joint_qpos_indices, dtype=np.intp)
            target_pos = data.qpos[qpos_idx].copy()
            self._entities[name] = _EntityState(
                actuator_ids=np.array(entity.actuator_ids, dtype=np.intp),
                joint_qpos_indices=qpos_idx,
                joint_qvel_indices=np.array(
                    entity.joint_qvel_indices,
                    dtype=np.intp,
                ),
                target_position=target_pos,
                target_velocity=np.zeros(len(qpos_idx)),
            )

        # Active non-blocking trajectory runners (entity_name → TrajectoryRunner)
        self._runners: dict[str, TrajectoryRunner] = {}

        # Deferred hold: when True, step() calls hold_all() before applying
        # targets. This lets callers modify qpos after reset_state() and have
        # the controller pick up whatever's there on the next tick.
        self._hold_pending: bool = False

        # Initialize qpos/qvel to targets (prevents violent jumps on first step)
        for state in self._arms.values():
            data.qpos[state.joint_qpos_indices] = state.target_position
            data.qvel[state.joint_qvel_indices] = 0.0

        mujoco.mj_forward(model, data)

    # -- Target management --------------------------------------------------

    def hold_all(self) -> None:
        """Update all targets (arms, entities, grippers) to current positions."""
        self._hold_pending = False
        for state in self._arms.values():
            state.target_position = self.data.qpos[state.joint_qpos_indices].copy()
            state.target_velocity = np.zeros(len(state.actuator_ids))

        for state in self._entities.values():
            state.target_position = self.data.qpos[state.joint_qpos_indices].copy()
            state.target_velocity = np.zeros(len(state.target_velocity))

        for name, gs in self._grippers.items():
            gs.target_ctrl = gs.ctrl_open
            self.data.ctrl[gs.actuator_id] = gs.ctrl_open

    def request_hold(self) -> None:
        """Request that the next step() captures current qpos as targets.

        Use instead of :meth:`hold_all` when qpos will be modified after
        the call (e.g. scene setup after a reset). The hold is deferred
        to the next :meth:`step`, so whatever qpos state exists at
        tick-time gets captured — including post-reset modifications.
        """
        self._hold_pending = True

    def set_arm_target(
        self,
        arm_name: str,
        position: np.ndarray,
        velocity: np.ndarray | None = None,
    ) -> None:
        """Set target position and velocity for an arm.

        Args:
            arm_name: Arm identifier.
            position: Target joint positions (rad).
            velocity: Target joint velocities (rad/s), or None for zero.
        """
        if arm_name not in self._arms:
            raise ValueError(f"Unknown arm: {arm_name}")

        state = self._arms[arm_name]
        state.target_position = np.asarray(position).copy()
        state.target_velocity = (
            np.asarray(velocity).copy() if velocity is not None else np.zeros(len(state.actuator_ids))
        )

    # -- Stepping (template method) -----------------------------------------

    def step(self) -> None:
        """Apply current targets and advance one control cycle.

        If a deferred hold is pending (from :meth:`request_hold` or
        :meth:`reset_state`), captures current qpos as targets first.
        Then delegates to :meth:`_apply_targets_and_step`.
        """
        if self._hold_pending:
            self.hold_all()
        self._apply_targets_and_step()

    @abstractmethod
    def _apply_targets_and_step(self) -> None:
        """Apply current arm/entity/gripper targets and advance one control cycle.

        Read from ``_ArmState.target_position/target_velocity``,
        ``_EntityState.target_position/target_velocity``, and
        ``_GripperState.target_ctrl``. Advance the simulation by one
        control cycle, then sync the viewer (throttled).

        Physics: write data.ctrl with lookahead → mj_step × N → tick
            verifiers → viewer sync.
        Kinematic: write data.qpos → mj_forward → update grasp managers
            → viewer sync.
        """

    def step_reactive(
        self,
        arm_name: str,
        position: np.ndarray,
        velocity: np.ndarray | None = None,
    ) -> None:
        """Reactive single-arm step for cartesian streaming control.

        Sets targets and steps. In physics mode, overridden with small
        lookahead and hold-others behavior. In kinematic mode, the default
        (set + step) gives instant tracking.

        Args:
            arm_name: Which arm to control reactively.
            position: Target joint positions.
            velocity: Target joint velocities for feedforward.
        """
        self.set_arm_target(arm_name, position, velocity)
        self.step()

    def _throttled_viewer_sync(self) -> None:
        """Sync viewer if present, throttled to viewer_sync_interval."""
        if self.viewer is not None:
            now = time.time()
            if now - self._last_viewer_sync >= self._viewer_sync_interval:
                self.viewer.sync()
                self._last_viewer_sync = now

    # -- Non-blocking trajectory execution ----------------------------------

    def start_trajectory(
        self,
        entity_name: str,
        trajectory: Trajectory,
        abort_fn: Callable[[], bool] | None = None,
    ) -> Future[bool]:
        """Start a non-blocking trajectory on an arm or entity.

        Returns a Future that resolves to True when the trajectory completes
        (and the arm converges, if applicable), or False if aborted. The
        caller blocks on the Future while :meth:`advance_all` (called by
        the event loop's tick) drives the trajectory forward one waypoint
        per cycle.

        Args:
            entity_name: Which arm or entity to execute on.
            trajectory: Time-parameterized trajectory.
            abort_fn: Per-entity abort check. Runner stops when this returns True.

        Returns:
            Future[bool] — resolves when trajectory finishes.
        """
        # Resolve state for DOF validation
        if entity_name in self._arms:
            state = self._arms[entity_name]
        elif entity_name in self._entities:
            state = self._entities[entity_name]
        else:
            raise ValueError(f"Unknown arm or entity: {entity_name}")

        if trajectory.dof != len(state.joint_qpos_indices):
            raise ValueError(
                f"Trajectory DOF {trajectory.dof} doesn't match joint count {len(state.joint_qpos_indices)}"
            )

        runner = TrajectoryRunner(self, entity_name, trajectory, abort_fn)
        self._runners[entity_name] = runner
        logger.debug(
            "Started trajectory on %s (%d waypoints)",
            entity_name,
            trajectory.num_waypoints,
        )
        return runner.future

    def advance_all(self) -> None:
        """Advance all active trajectory runners one waypoint each.

        Called by the event loop's tick() before :meth:`step`. Runners that
        are done are automatically removed.
        """
        if not self._runners:
            return

        done_arms = []
        for arm_name, runner in self._runners.items():
            runner.advance()
            if runner.done:
                done_arms.append(arm_name)

        for arm_name in done_arms:
            del self._runners[arm_name]

    def has_active_runner(self, arm_name: str | None = None) -> bool:
        """Check if any (or a specific) arm has an active trajectory runner."""
        if arm_name is not None:
            return arm_name in self._runners
        return bool(self._runners)

    # -- Blocking trajectory execution --------------------------------------

    def execute(self, arm_name: str, trajectory: Trajectory) -> bool:
        """Execute trajectory on one arm while others hold position.

        After the trajectory completes, waits for the arm to converge to
        the final target (convergence-based settling in physics mode,
        immediate in kinematic mode).

        Args:
            arm_name: Which arm to execute on.
            trajectory: Time-parameterized trajectory.

        Returns:
            True if execution completed and arm converged.
        """
        if arm_name not in self._arms:
            raise ValueError(f"Unknown arm: {arm_name}")

        state = self._arms[arm_name]

        if trajectory.dof != len(state.joint_qpos_indices):
            raise ValueError(
                f"Trajectory DOF {trajectory.dof} doesn't match arm joint count {len(state.joint_qpos_indices)}"
            )

        # Follow trajectory at real-time rate
        realtime = self.viewer is not None
        t_start = time.time() if realtime else 0.0
        for i in range(trajectory.num_waypoints):
            if self._abort_fn is not None and self._abort_fn():
                logger.info("Trajectory aborted at waypoint %d/%d", i, trajectory.num_waypoints)
                # Zero velocity so arm holds position while other arms move
                state.target_velocity = np.zeros(len(state.actuator_ids))
                return False
            state.target_position = trajectory.positions[i]
            state.target_velocity = trajectory.velocities[i]
            self.step()
            if realtime:
                t_target = t_start + (i + 1) * self.control_dt
                t_remaining = t_target - time.time()
                if t_remaining > 0:
                    time.sleep(t_remaining)

        # Hold final position (zero velocity)
        state.target_position = trajectory.positions[-1].copy()
        state.target_velocity = np.zeros(len(state.actuator_ids))

        return self._wait_for_convergence(arm_name)

    def execute_entity(self, entity_name: str, trajectory: Trajectory) -> bool:
        """Execute trajectory on an entity while arms and other entities hold.

        Args:
            entity_name: Entity identifier (e.g. "left_base").
            trajectory: Time-parameterized trajectory.

        Returns:
            True if execution completed.
        """
        if entity_name not in self._entities:
            raise ValueError(f"Unknown entity: {entity_name}")

        state = self._entities[entity_name]

        if trajectory.dof != len(state.joint_qpos_indices):
            raise ValueError(
                f"Trajectory DOF {trajectory.dof} doesn't match entity joint count {len(state.joint_qpos_indices)}"
            )

        realtime = self.viewer is not None
        t_start = time.time() if realtime else 0.0
        for i in range(trajectory.num_waypoints):
            if self._abort_fn is not None and self._abort_fn():
                logger.info("Entity trajectory aborted at waypoint %d/%d", i, trajectory.num_waypoints)
                return False
            state.target_position = trajectory.positions[i]
            state.target_velocity = trajectory.velocities[i]
            self.step()
            if realtime:
                t_target = t_start + (i + 1) * self.control_dt
                t_remaining = t_target - time.time()
                if t_remaining > 0:
                    time.sleep(t_remaining)

        state.target_position = trajectory.positions[-1].copy()
        state.target_velocity = np.zeros(len(state.actuator_ids))
        return True

    def _wait_for_convergence(
        self,
        arm_name: str,
        position_tolerance: float | None = None,
        velocity_tolerance: float | None = None,
        timeout_steps: int | None = None,
    ) -> bool:
        """Wait for arm to converge to target position.

        In physics mode, the arm's PD controller drives it to the target
        over multiple steps. In kinematic mode, targets are written directly
        to qpos, so convergence succeeds on the first check.

        Returns True if converged, False on timeout.
        """
        cfg = self.config
        if position_tolerance is None:
            position_tolerance = cfg.position_tolerance
        if velocity_tolerance is None:
            velocity_tolerance = cfg.velocity_tolerance
        if timeout_steps is None:
            timeout_steps = cfg.convergence_timeout_steps

        state = self._arms[arm_name]
        pos_error = np.zeros(len(state.joint_qpos_indices))
        current_vel = np.zeros(len(state.joint_qvel_indices))

        for _ in range(timeout_steps):
            self.step()

            current_pos = self.data.qpos[state.joint_qpos_indices]
            pos_error = np.abs(state.target_position - current_pos)

            current_vel = self.data.qvel[state.joint_qvel_indices]

            if np.all(pos_error < position_tolerance) and np.all(np.abs(current_vel) < velocity_tolerance):
                return True

        logger.warning(
            "Convergence timeout for %s: max_pos_err=%.2f° (limit %.2f°), max_vel=%.3f rad/s (limit %.3f)",
            arm_name,
            np.rad2deg(np.max(pos_error)),
            np.rad2deg(position_tolerance),
            np.max(np.abs(current_vel)),
            velocity_tolerance,
        )
        return False

    # -- Gripper control (abstract) -----------------------------------------

    @abstractmethod
    def close_gripper(
        self,
        arm_name: str,
        candidate_objects: list[str] | None = None,
        steps: int | None = None,
    ) -> bool:
        """Close the gripper for the specified arm.

        Args:
            arm_name: Which arm's gripper to close.
            candidate_objects: Optional list of expected grasp targets.
            steps: Number of close steps (mode-specific default if None).

        Returns:
            True if the close sequence completed, False if no gripper.
        """

    @abstractmethod
    def open_gripper(self, arm_name: str, steps: int | None = None) -> None:
        """Open the gripper for the specified arm.

        Args:
            arm_name: Which arm's gripper to open.
            steps: Number of open steps (mode-specific default if None).
        """

    # -- Executor wrappers --------------------------------------------------

    def get_executor(self, arm_name: str) -> ArmExecutor:
        """Get Executor interface for a single arm.

        Returns an object with the standard ``Executor.execute(trajectory)``
        interface that internally delegates to this controller.

        Args:
            arm_name: Which arm to create an executor for.
        """
        if arm_name not in self._arms:
            raise ValueError(f"Unknown arm: {arm_name}")
        return ArmExecutor(self, arm_name)

    def get_entity_executor(self, entity_name: str) -> EntityExecutor:
        """Get Executor interface for an entity (base, etc.)."""
        if entity_name not in self._entities:
            raise ValueError(f"Unknown entity: {entity_name}")
        return EntityExecutor(self, entity_name)
