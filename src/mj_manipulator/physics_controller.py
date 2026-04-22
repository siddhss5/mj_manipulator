# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Physics-mode controller for MuJoCo simulation.

Subclass of :class:`Controller` that steps MuJoCo physics each control
cycle. Coordinates all arm and gripper actuators to prevent gravity
collapse on idle arms. Provides velocity feedforward, convergence-based
settling, gradual gripper control ramps, and reactive streaming.

This is a simulation-only component. On real hardware, the robot's own
controller (RTDE, ROS, libfranka) handles actuator coordination.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from mj_manipulator.config import GripperPhysicsConfig, PhysicsExecutionConfig
from mj_manipulator.controller import (
    ArmExecutor,
    Controller,
    EntityExecutor,
)

if TYPE_CHECKING:
    from mj_manipulator.arm import Arm

logger = logging.getLogger(__name__)

# Backwards compatibility aliases — these classes moved to controller.py
ArmPhysicsExecutor = ArmExecutor
EntityPhysicsExecutor = EntityExecutor


class PhysicsController(Controller):
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
        gripper_config = gripper_config or GripperPhysicsConfig()
        super().__init__(
            model,
            data,
            arms,
            config=config,
            gripper_config=gripper_config,
            viewer=viewer,
            viewer_sync_interval=viewer_sync_interval,
            initial_positions=initial_positions,
            entities=entities,
            abort_fn=abort_fn,
        )

        self.gripper_config: GripperPhysicsConfig  # narrow the type
        self.steps_per_control = max(1, int(self.control_dt / model.opt.timestep))

        # Physics-specific: initialize ctrl to targets (prevents gravity collapse)
        for state in self._arms.values():
            data.ctrl[state.actuator_ids] = state.target_position

        for state in self._entities.values():
            data.ctrl[state.actuator_ids] = state.target_position

        mujoco.mj_forward(model, data)

    # -- Stepping (physics-specific) ----------------------------------------

    def _apply_targets_and_step(self) -> None:
        """Apply control to all actuators and step physics.

        Uses full ``lookahead_time`` for velocity feedforward, applies
        gripper ctrl, calls ``mj_step``, ticks grasp verifiers, and
        syncs the viewer (throttled).
        """
        # Arm actuators: position + velocity feedforward (per-arm lookahead)
        for state in self._arms.values():
            la = state.lookahead if state.lookahead is not None else self.lookahead_time
            q_cmd = state.target_position + la * state.target_velocity
            self.data.ctrl[state.actuator_ids] = q_cmd

        # Entity actuators (bases, etc.): same feedforward
        for state in self._entities.values():
            q_cmd = state.target_position + self.lookahead_time * state.target_velocity
            self.data.ctrl[state.actuator_ids] = q_cmd

        # Gripper ctrl
        for gstate in self._grippers.values():
            self.data.ctrl[gstate.actuator_id] = gstate.target_ctrl

        # Step MuJoCo physics
        for _ in range(self.steps_per_control):
            mujoco.mj_step(self.model, self.data)

        # Tick every configured GraspVerifier exactly once per control
        # cycle. This is the single "advance time by one control cycle"
        # primitive in physics mode, so every execution path
        # (ctx.step, ctx.execute, gripper close sequences, trajectory
        # runners) transitively runs verifier ticks through here. No
        # consumer needs to remember to tick. Arms without a verifier
        # are no-ops.
        for astate in self._arms.values():
            gripper = astate.arm.gripper
            if gripper is not None and gripper.grasp_verifier is not None:
                gripper.grasp_verifier.tick()

        self._throttled_viewer_sync()

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
        self.data.ctrl[state.actuator_ids] = q_cmd

        # Other arms: hold position (no velocity feedforward)
        for other_name, other_state in self._arms.items():
            if other_name != arm_name:
                self.data.ctrl[other_state.actuator_ids] = other_state.target_position

        # Gripper ctrl + mj_step + verifiers + viewer sync
        for gstate in self._grippers.values():
            self.data.ctrl[gstate.actuator_id] = gstate.target_ctrl

        for _ in range(self.steps_per_control):
            mujoco.mj_step(self.model, self.data)

        for astate in self._arms.values():
            gripper = astate.arm.gripper
            if gripper is not None and gripper.grasp_verifier is not None:
                gripper.grasp_verifier.tick()

        self._throttled_viewer_sync()

    # -- Gripper control (physics-specific) ---------------------------------

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
