# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Kinematic-mode controller for MuJoCo simulation.

Subclass of :class:`Controller` that writes joint positions directly
to ``data.qpos`` and calls ``mj_forward`` (no physics dynamics). Used
for fast planning visualization, testing, and as the fallback on the
real robot when perception data is unavailable.

Provides the same orchestration as PhysicsController (non-blocking
trajectory runners, per-arm ownership, concurrent multi-arm control)
but with instant, exact tracking instead of PD convergence.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from mj_manipulator.config import ExecutionConfig
from mj_manipulator.controller import Controller

if TYPE_CHECKING:
    from mj_manipulator.arm import Arm

logger = logging.getLogger(__name__)

# Default kinematic control_dt: 250 Hz (faster than physics 125 Hz since
# there's no simulation cost, just mj_forward)
_KINEMATIC_CONTROL_DT = 0.004


class KinematicController(Controller):
    """Kinematic controller — direct qpos writes with mj_forward.

    Targets are written directly to ``data.qpos`` each step, giving
    instant and exact tracking. No physics dynamics, no actuator ctrl
    writes, no convergence settling needed.

    Attached objects (kinematic welds via GraspManager) are updated
    each step since there's no physics solver to maintain them.

    Args:
        model: MuJoCo model.
        data: MuJoCo data (modified during execution).
        arms: Dict mapping arm names to Arm instances.
        config: Execution parameters. Defaults to 250 Hz control_dt.
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
        viewer=None,
        viewer_sync_interval: float = 0.033,
        initial_positions: dict[str, np.ndarray] | None = None,
        entities: dict[str, object] | None = None,
        abort_fn=None,
    ):
        if config is None:
            config = ExecutionConfig(control_dt=_KINEMATIC_CONTROL_DT)
        super().__init__(
            model,
            data,
            arms,
            config=config,
            gripper_config=None,
            viewer=viewer,
            viewer_sync_interval=viewer_sync_interval,
            initial_positions=initial_positions,
            entities=entities,
            abort_fn=abort_fn,
        )

    # -- Stepping (kinematic-specific) --------------------------------------

    def _apply_targets_and_step(self) -> None:
        """Write targets to qpos, run mj_forward, update grasp attachments.

        No actuator ctrl writes, no mj_step, no verifier ticks. Just
        direct position setting followed by forward kinematics and
        grasp manager updates for kinematic welds.
        """
        # Write arm targets directly to qpos
        for state in self._arms.values():
            self.data.qpos[state.joint_qpos_indices] = state.target_position
            self.data.qvel[state.joint_qvel_indices] = 0.0

        # Write entity targets directly to qpos
        for state in self._entities.values():
            self.data.qpos[state.joint_qpos_indices] = state.target_position
            self.data.qvel[state.joint_qvel_indices] = 0.0

        # Forward kinematics (computes site/body poses from qpos)
        mujoco.mj_forward(self.model, self.data)

        # Update attached object poses (kinematic welds). In physics mode,
        # MuJoCo contacts/constraints maintain the grasp. In kinematic mode
        # there's no solver, so we must manually update attached poses.
        has_grasp_managers = False
        for astate in self._arms.values():
            gm = astate.arm.grasp_manager
            if gm is not None:
                gm.update_attached_poses()
                has_grasp_managers = True

        # Second mj_forward needed if any grasp manager modified object qpos
        if has_grasp_managers:
            mujoco.mj_forward(self.model, self.data)

        self._throttled_viewer_sync()

    # -- Gripper control (kinematic-specific) -------------------------------

    def close_gripper(
        self,
        arm_name: str,
        candidate_objects: list[str] | None = None,
        steps: int | None = None,
    ) -> bool:
        """Close the gripper kinematically.

        Delegates to the gripper's ``kinematic_close()`` method which
        interpolates joint positions and scans contacts to detect when
        to stop. No physics stepping, no ctrl ramp.

        Args:
            arm_name: Which arm's gripper to close.
            candidate_objects: Objects the gripper should try to grasp.
            steps: Unused (kinematic close has its own step count).

        Returns:
            True if the close sequence completed, False if no gripper.
        """
        del steps  # kinematic_close manages its own steps

        if arm_name not in self._grippers:
            logger.warning("No gripper found for %s", arm_name)
            return False

        gripper = self._arms[arm_name].arm.gripper
        gripper.set_candidate_objects(candidate_objects)
        gripper.kinematic_close()
        self._apply_targets_and_step()  # sync state after gripper move
        return True

    def open_gripper(self, arm_name: str, steps: int | None = None) -> None:
        """Open the gripper kinematically.

        Delegates to the gripper's ``kinematic_open()`` method which
        directly sets joint positions to the open configuration.

        Args:
            arm_name: Which arm's gripper to open.
            steps: Unused (kinematic open is instant).
        """
        del steps  # kinematic_open is instant

        if arm_name not in self._grippers:
            logger.warning("No gripper found for %s", arm_name)
            return

        gripper = self._arms[arm_name].arm.gripper
        gripper.kinematic_open()
        self._apply_targets_and_step()  # sync state after gripper move
