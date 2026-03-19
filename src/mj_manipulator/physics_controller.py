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
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from mj_manipulator.config import GripperPhysicsConfig, PhysicsExecutionConfig
from mj_manipulator.grasp_manager import detect_grasped_object

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


@dataclass
class _GripperState:
    """Per-gripper state managed by PhysicsController."""

    gripper: Gripper
    actuator_id: int
    ctrl_open: float
    ctrl_closed: float
    target_ctrl: float


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
    ):
        self.model = model
        self.data = data
        self.viewer = viewer

        self.config = config or PhysicsExecutionConfig()
        self.gripper_config = gripper_config or GripperPhysicsConfig()

        self.control_dt = self.config.control_dt
        self.lookahead_time = self.config.lookahead_time
        self.steps_per_control = max(
            1, int(self.control_dt / model.opt.timestep)
        )

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

        # Initialize qpos/qvel/ctrl to targets (prevents violent jumps)
        for state in self._arms.values():
            data.qpos[state.joint_qpos_indices] = state.target_position
            data.qvel[state.joint_qvel_indices] = 0.0
            data.ctrl[state.actuator_ids] = state.target_position

        mujoco.mj_forward(model, data)

    # -- Target management --------------------------------------------------

    def hold_all(self) -> None:
        """Update all arm targets to current positions."""
        for state in self._arms.values():
            state.target_position = self.data.qpos[state.joint_qpos_indices].copy()
            state.target_velocity = np.zeros(len(state.actuator_ids))

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
            np.asarray(velocity).copy() if velocity is not None
            else np.zeros(len(state.actuator_ids))
        )

    # -- Physics stepping ---------------------------------------------------

    def step(self) -> None:
        """Apply control to all actuators and step physics.

        Uses full ``lookahead_time`` for velocity feedforward. For reactive
        streaming control, use :meth:`step_reactive` instead.
        """
        # Arm actuators: position + velocity feedforward
        for state in self._arms.values():
            q_cmd = (
                state.target_position
                + self.lookahead_time * state.target_velocity
            )
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
            np.asarray(velocity).copy() if velocity is not None
            else np.zeros(len(state.actuator_ids))
        )

        # Reactive arm: small lookahead
        reactive_lookahead = 2.0 * self.control_dt
        q_cmd = (
            state.target_position
            + reactive_lookahead * state.target_velocity
        )
        self.data.ctrl[state.actuator_ids] = q_cmd

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
                f"Trajectory DOF {trajectory.dof} doesn't match "
                f"arm joint count {len(state.joint_qpos_indices)}"
            )

        # Follow trajectory
        sleep_dt = self.control_dt if self.viewer is not None else 0.0
        for i in range(trajectory.num_waypoints):
            state.target_position = trajectory.positions[i]
            state.target_velocity = trajectory.velocities[i]
            self.step()
            if sleep_dt > 0:
                time.sleep(sleep_dt)

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

            if (np.all(pos_error < position_tolerance)
                    and np.all(np.abs(current_vel) < velocity_tolerance)):
                return True

        logger.warning(
            "Convergence timeout for %s: max_pos_err=%.2f° (limit %.2f°), "
            "max_vel=%.3f rad/s (limit %.3f)",
            arm_name,
            np.rad2deg(np.max(pos_error)),
            np.rad2deg(position_tolerance),
            np.max(np.abs(current_vel)),
            velocity_tolerance,
        )
        return False

    # -- Gripper control ----------------------------------------------------

    def close_gripper(
        self,
        arm_name: str,
        candidate_objects: list[str] | None = None,
        steps: int | None = None,
    ) -> str | None:
        """Close gripper with contact detection.

        Gradually closes the gripper while monitoring for object contact.
        After contact is detected, continues closing for a firm grip, then
        performs final bilateral contact verification.

        Args:
            arm_name: Which arm's gripper to close.
            candidate_objects: Objects to consider for grasp detection.
                If None, considers all objects in contact.
            steps: Number of close steps (default from gripper_config).

        Returns:
            Name of grasped object, or None if nothing grasped.
        """
        cfg = self.gripper_config
        if steps is None:
            steps = cfg.close_steps

        if arm_name not in self._grippers:
            logger.warning("No gripper found for %s", arm_name)
            return None

        gstate = self._grippers[arm_name]
        gripper = gstate.gripper

        start_ctrl = gstate.ctrl_open
        end_ctrl = gstate.ctrl_closed

        # Open first for clean start
        gstate.target_ctrl = start_ctrl
        for _ in range(cfg.pre_open_steps):
            self.step()

        contacts_detected = False
        grasped = None
        sleep_dt = self.control_dt * 0.5 if self.viewer is not None else 0.0

        # Gradually close
        for i in range(steps):
            t = (i + 1) / steps
            gstate.target_ctrl = start_ctrl + t * (end_ctrl - start_ctrl)
            self.step()

            # Check contacts periodically (unilateral during closing)
            if i % cfg.contact_check_interval == 0 and not contacts_detected:
                grasped = detect_grasped_object(
                    self.model, self.data,
                    gripper.gripper_body_names,
                    candidate_objects,
                    require_bilateral=False,
                    debug=cfg.debug,
                )
                if grasped:
                    contacts_detected = True

            if sleep_dt > 0:
                time.sleep(sleep_dt)

        # Firm grip if contact was detected during closing
        if contacts_detected:
            for _ in range(cfg.firm_grip_steps):
                self.step()
                if sleep_dt > 0:
                    time.sleep(sleep_dt)

        # Final bilateral detection (robust)
        grasped = detect_grasped_object(
            self.model, self.data,
            gripper.gripper_body_names,
            candidate_objects,
            require_bilateral=True,
            debug=cfg.debug,
        )

        if not grasped:
            # Fallback to unilateral
            grasped = detect_grasped_object(
                self.model, self.data,
                gripper.gripper_body_names,
                candidate_objects,
                require_bilateral=False,
                debug=cfg.debug,
            )
            if grasped:
                logger.warning(
                    "Gripper %s: only unilateral contact with %s "
                    "— grasp may be unstable",
                    arm_name, grasped,
                )

        # Warn if fully closed with no contacts (missed object)
        if not grasped:
            gripper_pos = gripper.get_actual_position()
            if gripper_pos > cfg.fully_closed_threshold:
                logger.warning(
                    "Gripper %s: fully closed (pos=%.3f) with no contacts",
                    arm_name, gripper_pos,
                )

        return grasped

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
        """Apply gripper ctrl, step MuJoCo, sync viewer."""
        for gstate in self._grippers.values():
            self.data.ctrl[gstate.actuator_id] = gstate.target_ctrl

        for _ in range(self.steps_per_control):
            mujoco.mj_step(self.model, self.data)

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
