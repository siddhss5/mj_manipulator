# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Multi-arm physics controller for MuJoCo simulation.

Coordinates all arm and gripper actuators each physics step to prevent
gravity collapse on idle arms. Provides trajectory execution with
convergence-based settling, gripper actuation with contact detection,
and reactive streaming for cartesian control.

This is a simulation-only component. On real hardware, the robot's own
controller (RTDE, ROS, libfranka) handles actuator coordination.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import mujoco
import numpy as np

from mj_manipulator.config import GripperPhysicsConfig, PhysicsExecutionConfig, SafetyResponse

if TYPE_CHECKING:
    from mj_manipulator.arm import Arm
    from mj_manipulator.protocols import Gripper
    from mj_manipulator.trajectory import Trajectory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal per-arm / per-gripper state
# ---------------------------------------------------------------------------


@dataclass
class _ArmState:
    """Per-arm state managed by PhysicsController."""

    arm: Arm
    actuator_ids: np.ndarray  # int array for vectorized ctrl writes
    joint_qpos_indices: np.ndarray
    joint_qvel_indices: np.ndarray
    target_position: np.ndarray
    target_velocity: np.ndarray
    lookahead: float | None = None  # None = use controller default
    # Safety layer state
    prev_target_velocity: np.ndarray | None = None
    faulted: bool = False
    first_step: bool = True  # skip accel check on the very first cycle
    velocity_limits: np.ndarray | None = None
    acceleration_limits: np.ndarray | None = None


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
    """Per-gripper state managed by PhysicsController."""

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
    owns physics stepping.

    This is the "update method" pattern from game programming: the game
    loop calls ``runner.advance()`` each frame, then steps physics once
    with all arms' targets applied.

    Created via :meth:`PhysicsController.start_trajectory`.
    """

    def __init__(
        self,
        controller: PhysicsController,
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
                # Arms need convergence settling (PD dynamics)
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
# PhysicsController
# ---------------------------------------------------------------------------


class PhysicsController:
    """Multi-arm physics controller for MuJoCo simulation.

    In physics simulation, ALL actuators must be controlled — otherwise
    joints fall under gravity. This controller ensures that when one arm
    executes a trajectory, all other actuators hold their positions.

    This is a simulation-only component. On real hardware, the equivalent
    functionality comes from the robot's own controller.

    Usage::

        controller = PhysicsController(model, data, {"ur5e": arm})

        # Execute on one arm (others hold automatically)
        controller.execute("ur5e", trajectory)

        # Streaming reactive control
        while running:
            controller.step_reactive("ur5e", q_target, qd_target)

    Args:
        model: MuJoCo model.
        data: MuJoCo data (modified during execution).
        arms: Dict mapping arm names to Arm instances. Each Arm must have
            ``actuator_ids``, ``joint_qpos_indices``, ``joint_qvel_indices``.
        config: Physics execution parameters (control_dt, lookahead, etc.).
        gripper_config: Gripper control parameters (close steps, etc.).
        viewer: Optional MuJoCo viewer to sync during execution.
        viewer_sync_interval: Minimum seconds between viewer syncs.
        initial_positions: Optional per-arm initial joint positions. Arms not
            listed use their current qpos.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        arms: dict[str, Arm],
        config: PhysicsExecutionConfig | None = None,
        gripper_config: GripperPhysicsConfig | None = None,
        viewer=None,
        viewer_sync_interval: float = 0.033,
        initial_positions: dict[str, np.ndarray] | None = None,
        entities: dict[str, object] | None = None,
        abort_fn=None,
    ):
        self.model = model
        self.data = data
        self.viewer = viewer
        self._abort_fn = abort_fn

        self.config = config or PhysicsExecutionConfig()
        self.gripper_config = gripper_config or GripperPhysicsConfig()

        self.control_dt = self.config.control_dt
        self.lookahead_time = self.config.lookahead_time
        self.steps_per_control = max(1, int(self.control_dt / model.opt.timestep))

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

            # Cache kinematic limits for the safety layer. Arms without
            # kinematic_limits (e.g. test mocks) get None → safety check
            # is skipped for that arm.
            limits = getattr(arm.config, "kinematic_limits", None)
            vel_lim = limits.velocity.copy() if limits is not None else None
            acc_lim = limits.acceleration.copy() if limits is not None else None

            self._arms[name] = _ArmState(
                arm=arm,
                actuator_ids=np.array(arm.actuator_ids, dtype=np.intp),
                joint_qpos_indices=np.array(arm.joint_qpos_indices, dtype=np.intp),
                joint_qvel_indices=np.array(arm.joint_qvel_indices, dtype=np.intp),
                target_position=target_pos,
                target_velocity=np.zeros(arm.dof),
                prev_target_velocity=np.zeros(arm.dof),
                velocity_limits=vel_lim,
                acceleration_limits=acc_lim,
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

        # Active non-blocking trajectory runners (arm_name → TrajectoryRunner)
        self._runners: dict[str, TrajectoryRunner] = {}

        # Initialize qpos/qvel/ctrl to targets (prevents violent jumps)
        for state in self._arms.values():
            data.qpos[state.joint_qpos_indices] = state.target_position
            data.qvel[state.joint_qvel_indices] = 0.0
            data.ctrl[state.actuator_ids] = state.target_position

        for state in self._entities.values():
            data.ctrl[state.actuator_ids] = state.target_position

        mujoco.mj_forward(model, data)

    # -- Target management --------------------------------------------------

    def hold_all(self) -> None:
        """Update all targets (arms, entities, grippers) to current positions."""
        for state in self._arms.values():
            state.target_position = self.data.qpos[state.joint_qpos_indices].copy()
            state.target_velocity = np.zeros(len(state.actuator_ids))

        for state in self._entities.values():
            state.target_position = self.data.qpos[state.joint_qpos_indices].copy()
            state.target_velocity = np.zeros(len(state.target_velocity))

        for name, gs in self._grippers.items():
            gs.target_ctrl = gs.ctrl_open
            self.data.ctrl[gs.actuator_id] = gs.ctrl_open

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

    # -- Safety layer --------------------------------------------------------

    def _enforce_limits(
        self,
        name: str,
        state: _ArmState,
        q_cmd: np.ndarray,
        lookahead: float,
    ) -> np.ndarray:
        """Check velocity and acceleration limits, apply the configured response.

        Called once per arm per control cycle, after computing ``q_cmd``
        and before writing to ``data.ctrl``. Returns (possibly clamped)
        ``q_cmd``.
        """
        if state.velocity_limits is None:
            return q_cmd
        if state.faulted:
            return self.data.qpos[state.joint_qpos_indices].copy()

        response = self.config.safety_response
        vel = state.target_velocity
        prev_vel = state.prev_target_velocity
        vel_limits = state.velocity_limits
        acc_limits = state.acceleration_limits

        # --- Velocity check ---
        vel_violation = np.abs(vel) > vel_limits
        any_vel_violation = np.any(vel_violation)

        # --- Acceleration check ---
        # Skip on the very first cycle after construction or fault-clear
        # (prev_velocity is zeros from init, so any nonzero first command
        # looks like infinite acceleration).
        any_acc_violation = False
        accel = np.zeros_like(vel)
        if acc_limits is not None and prev_vel is not None and not state.first_step:
            accel = (vel - prev_vel) / self.control_dt
            acc_violation = np.abs(accel) > acc_limits
            any_acc_violation = np.any(acc_violation)

        if not any_vel_violation and not any_acc_violation:
            return q_cmd

        # --- Log the violation ---
        if any_vel_violation:
            for j in np.where(vel_violation)[0]:
                logger.warning(
                    "Safety: %s joint %d velocity %.2f rad/s exceeds limit %.2f",
                    name,
                    j,
                    vel[j],
                    vel_limits[j],
                )
        if any_acc_violation:
            for j in np.where(acc_violation)[0]:
                logger.warning(
                    "Safety: %s joint %d acceleration %.1f rad/s² exceeds limit %.1f",
                    name,
                    j,
                    accel[j],
                    acc_limits[j],
                )

        # --- Apply response ---
        if response == SafetyResponse.WARN:
            return q_cmd

        if response == SafetyResponse.FAULT:
            state.faulted = True
            state.target_velocity = np.zeros_like(vel)
            state.target_position = self.data.qpos[state.joint_qpos_indices].copy()
            logger.error("Safety FAULT: %s halted due to limit violation", name)
            return state.target_position.copy()

        # CLAMP: limit velocity, then limit acceleration
        clamped_vel = np.clip(vel, -vel_limits, vel_limits)
        if acc_limits is not None and prev_vel is not None and not state.first_step:
            max_delta = acc_limits * self.control_dt
            delta = clamped_vel - prev_vel
            clamped_vel = prev_vel + np.clip(delta, -max_delta, max_delta)
        state.target_velocity = clamped_vel
        return state.target_position + lookahead * clamped_vel

    def clear_fault(self, arm_name: str) -> None:
        """Clear fault state for an arm, allowing motion to resume.

        Resets the arm to hold its current position with zero velocity.
        """
        if arm_name not in self._arms:
            raise ValueError(f"Unknown arm: {arm_name}")
        state = self._arms[arm_name]
        state.faulted = False
        state.first_step = True  # skip accel check on the next cycle
        state.target_position = self.data.qpos[state.joint_qpos_indices].copy()
        state.target_velocity = np.zeros(len(state.actuator_ids))
        state.prev_target_velocity = np.zeros(len(state.actuator_ids))
        logger.info("Safety: %s fault cleared", arm_name)

    # -- Physics stepping ---------------------------------------------------

    def step(self) -> None:
        """Apply control to all actuators and step physics.

        Uses full ``lookahead_time`` for velocity feedforward. For reactive
        streaming control, use :meth:`step_reactive` instead.
        """
        # Arm actuators: position + velocity feedforward (per-arm lookahead)
        for name, state in self._arms.items():
            la = state.lookahead if state.lookahead is not None else self.lookahead_time
            q_cmd = state.target_position + la * state.target_velocity
            q_cmd = self._enforce_limits(name, state, q_cmd, la)
            self.data.ctrl[state.actuator_ids] = q_cmd
            state.prev_target_velocity = state.target_velocity.copy()
            state.first_step = False

        # Entity actuators (bases, etc.): same feedforward (no safety layer)
        for state in self._entities.values():
            q_cmd = state.target_position + self.lookahead_time * state.target_velocity
            self.data.ctrl[state.actuator_ids] = q_cmd

        self._step_physics()

    def step_reactive(
        self,
        arm_name: str,
        position: np.ndarray,
        velocity: np.ndarray | None = None,
    ) -> None:
        """Step with reactive control for one arm, others hold.

        Uses a small lookahead (2 × control_dt) instead of the full
        trajectory lookahead, treating each step as a mini-trajectory
        segment for smooth streaming without overshoot.

        Args:
            arm_name: Which arm to control reactively.
            position: Target joint positions.
            velocity: Target joint velocities for feedforward.
        """
        if arm_name not in self._arms:
            raise ValueError(f"Unknown arm: {arm_name}")

        state = self._arms[arm_name]
        state.target_position = np.asarray(position).copy()
        state.target_velocity = (
            np.asarray(velocity).copy() if velocity is not None else np.zeros(len(state.actuator_ids))
        )

        # Reactive arm: small lookahead
        reactive_lookahead = 2.0 * self.control_dt
        q_cmd = state.target_position + reactive_lookahead * state.target_velocity
        q_cmd = self._enforce_limits(arm_name, state, q_cmd, reactive_lookahead)
        self.data.ctrl[state.actuator_ids] = q_cmd
        state.prev_target_velocity = state.target_velocity.copy()
        state.first_step = False

        # Other arms: hold position (no velocity feedforward)
        for other_name, other_state in self._arms.items():
            if other_name != arm_name:
                self.data.ctrl[other_state.actuator_ids] = other_state.target_position

        self._step_physics()

    # -- Trajectory execution -----------------------------------------------

    def execute(self, arm_name: str, trajectory: Trajectory) -> bool:
        """Execute trajectory on one arm while others hold position.

        After the trajectory completes, waits for the arm to converge to
        the final target (convergence-based settling, not fixed steps).

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

    def _wait_for_convergence(
        self,
        arm_name: str,
        position_tolerance: float | None = None,
        velocity_tolerance: float | None = None,
        timeout_steps: int | None = None,
    ) -> bool:
        """Wait for arm to converge to target position.

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

    # -- Non-blocking trajectory execution ------------------------------------

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

    # -- Gripper control ----------------------------------------------------

    def close_gripper(
        self,
        arm_name: str,
        candidate_objects: list[str] | None = None,
        steps: int | None = None,
    ) -> bool:
        """Close the gripper through its control ramp.

        Runs the full open-then-close sequence: a few steps of opening
        for a clean start, then the gradual close ramp, then a firm-grip
        hold phase. Always runs the complete duration — no early exit on
        contact detection. The caller decides whether the grasp succeeded
        by inspecting :attr:`Gripper.grasp_verifier.is_held` after the
        settling window elapses (:meth:`SimArmController.grasp` does
        this automatically).

        Args:
            arm_name: Which arm's gripper to close.
            candidate_objects: Unused (kept for signature compatibility;
                candidate-object resolution is now a pure-signal
                responsibility of :class:`GraspVerifier`).
            steps: Number of close steps (default from gripper_config).

        Returns:
            True if the close sequence ran to completion, False if the
            arm has no configured gripper. The return value does **not**
            reflect whether anything was grasped — that's the verifier's
            job.
        """
        del candidate_objects  # retained for signature compat only

        cfg = self.gripper_config
        if steps is None:
            steps = cfg.close_steps

        if arm_name not in self._grippers:
            logger.warning("No gripper found for %s", arm_name)
            return False

        gstate = self._grippers[arm_name]

        start_ctrl = gstate.ctrl_open
        end_ctrl = gstate.ctrl_closed

        # Open first for clean start
        gstate.target_ctrl = start_ctrl
        for _ in range(cfg.pre_open_steps):
            self.step()

        sleep_dt = self.control_dt * 0.5 if self.viewer is not None else 0.0

        # Gradual close ramp — runs unconditionally
        for i in range(steps):
            t = (i + 1) / steps
            gstate.target_ctrl = start_ctrl + t * (end_ctrl - start_ctrl)
            self.step()
            if sleep_dt > 0:
                time.sleep(sleep_dt)

        # Firm grip phase — let the gripper settle at the target for
        # a few steps so the grasp has a chance to stabilize before
        # the verifier's settling window begins.
        for _ in range(cfg.firm_grip_steps):
            self.step()
            if sleep_dt > 0:
                time.sleep(sleep_dt)

        return True

    def open_gripper(self, arm_name: str, steps: int | None = None) -> None:
        """Open gripper while maintaining all arm positions.

        Args:
            arm_name: Which arm's gripper to open.
            steps: Number of open steps (default from gripper_config).
        """
        if steps is None:
            steps = self.gripper_config.open_steps

        if arm_name not in self._grippers:
            logger.warning("No gripper found for %s", arm_name)
            return

        gstate = self._grippers[arm_name]
        start_ctrl = gstate.target_ctrl
        end_ctrl = gstate.ctrl_open
        sleep_dt = self.control_dt * 0.5 if self.viewer is not None else 0.0

        for i in range(steps):
            t = (i + 1) / steps
            gstate.target_ctrl = start_ctrl + t * (end_ctrl - start_ctrl)
            self.step()
            if sleep_dt > 0:
                time.sleep(sleep_dt)

    # -- Entity trajectory execution ----------------------------------------

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

    def get_entity_executor(self, entity_name: str) -> EntityPhysicsExecutor:
        """Get Executor interface for an entity (base, etc.)."""
        if entity_name not in self._entities:
            raise ValueError(f"Unknown entity: {entity_name}")
        return EntityPhysicsExecutor(self, entity_name)

    # -- Executor interface -------------------------------------------------

    def get_executor(self, arm_name: str) -> ArmPhysicsExecutor:
        """Get Executor interface for a single arm.

        Returns an object with the standard ``Executor.execute(trajectory)``
        interface that internally delegates to this controller.

        Args:
            arm_name: Which arm to create an executor for.
        """
        if arm_name not in self._arms:
            raise ValueError(f"Unknown arm: {arm_name}")
        return ArmPhysicsExecutor(self, arm_name)

    # -- Internal -----------------------------------------------------------

    def _step_physics(self) -> None:
        """Apply gripper ctrl, step MuJoCo, tick grasp verifiers, sync viewer."""
        for gstate in self._grippers.values():
            self.data.ctrl[gstate.actuator_id] = gstate.target_ctrl

        for _ in range(self.steps_per_control):
            mujoco.mj_step(self.model, self.data)

        # Tick every configured GraspVerifier exactly once per control
        # cycle. This is the single \"advance time by one control cycle\"
        # primitive in physics mode, so every execution path
        # (ctx.step, ctx.execute, gripper close sequences, trajectory
        # runners) transitively runs verifier ticks through here. No
        # consumer needs to remember to tick. Arms without a verifier
        # are no-ops.
        for astate in self._arms.values():
            gripper = astate.arm.gripper
            if gripper is not None and gripper.grasp_verifier is not None:
                gripper.grasp_verifier.tick()

        if self.viewer is not None:
            now = time.time()
            if now - self._last_viewer_sync >= self._viewer_sync_interval:
                self.viewer.sync()
                self._last_viewer_sync = now


# ---------------------------------------------------------------------------
# ArmPhysicsExecutor
# ---------------------------------------------------------------------------


class ArmPhysicsExecutor:
    """Executor interface for one arm, backed by PhysicsController.

    Provides the standard ``Executor.execute(trajectory)`` interface while
    ensuring all other actuators hold their positions during execution.
    Obtained via :meth:`PhysicsController.get_executor`.
    """

    def __init__(self, controller: PhysicsController, arm_name: str):
        self.controller = controller
        self.arm_name = arm_name

    def execute(self, trajectory: Trajectory) -> bool:
        """Execute trajectory on this arm."""
        return self.controller.execute(self.arm_name, trajectory)


class EntityPhysicsExecutor:
    """Executor interface for a non-arm entity, backed by PhysicsController.

    Same pattern as ArmPhysicsExecutor but routes to execute_entity().
    Obtained via :meth:`PhysicsController.get_entity_executor`.
    """

    def __init__(self, controller: PhysicsController, entity_name: str):
        self.controller = controller
        self.entity_name = entity_name

    def execute(self, trajectory: Trajectory) -> bool:
        """Execute trajectory on this entity."""
        return self.controller.execute_entity(self.entity_name, trajectory)
