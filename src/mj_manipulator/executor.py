"""Trajectory execution for simulation and real robot.

Provides two executors:
- KinematicExecutor: Direct joint position setting (no physics)
- PhysicsExecutor: Physics stepping with velocity feedforward

Both implement the Executor protocol (execute(trajectory) -> bool).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Protocol

import mujoco
import numpy as np

if TYPE_CHECKING:
    from mj_manipulator.grasp_manager import GraspManager
    from mj_manipulator.trajectory import Trajectory

logger = logging.getLogger(__name__)


class Executor(Protocol):
    """Protocol for trajectory execution."""

    def execute(self, trajectory: Trajectory) -> bool:
        """Execute trajectory. Returns True if successful."""
        ...


class KinematicExecutor:
    """Kinematic trajectory execution with perfect tracking.

    Directly sets joint positions and velocities without physics simulation.
    Uses mj_forward for forward kinematics only.

    For manipulation tasks, set a GraspManager to automatically update
    attached object poses when the gripper moves.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        joint_qpos_indices: list[int],
        control_dt: float = 0.008,
        viewer=None,
        grasp_manager: GraspManager | None = None,
        viewer_sync_interval: float = 0.033,
        abort_fn=None,
    ):
        self.model = model
        self.data = data
        self.joint_qpos_indices = joint_qpos_indices
        self.control_dt = control_dt
        self.viewer = viewer
        self.grasp_manager = grasp_manager
        self._abort_fn = abort_fn

        self._last_viewer_sync = 0.0
        self._viewer_sync_interval = viewer_sync_interval

        self._target_position = np.array([
            data.qpos[idx] for idx in joint_qpos_indices
        ])

    def execute(self, trajectory: Trajectory) -> bool:
        """Execute trajectory kinematically with perfect tracking."""
        if trajectory.dof != len(self.joint_qpos_indices):
            raise ValueError(
                f"Trajectory DOF {trajectory.dof} doesn't match "
                f"joint count {len(self.joint_qpos_indices)}"
            )

        t_start = time.time()
        for i in range(trajectory.num_waypoints):
            if self._abort_fn is not None and self._abort_fn():
                logger.info("Trajectory aborted at waypoint %d/%d", i, trajectory.num_waypoints)
                return False
            for joint_idx, qpos_idx in enumerate(self.joint_qpos_indices):
                self.data.qpos[qpos_idx] = trajectory.positions[i, joint_idx]
                self.data.qvel[qpos_idx] = trajectory.velocities[i, joint_idx]

            mujoco.mj_forward(self.model, self.data)

            if self.grasp_manager is not None:
                self.grasp_manager.update_attached_poses()
                mujoco.mj_forward(self.model, self.data)

            if self.viewer is not None:
                now = time.time()
                if now - self._last_viewer_sync >= self._viewer_sync_interval:
                    self.viewer.sync()
                    self._last_viewer_sync = now

            t_target = t_start + (i + 1) * self.control_dt
            t_remaining = t_target - time.time()
            if t_remaining > 0:
                time.sleep(t_remaining)

        # Final state with zero velocity
        for joint_idx, qpos_idx in enumerate(self.joint_qpos_indices):
            self.data.qpos[qpos_idx] = trajectory.positions[-1, joint_idx]
            self.data.qvel[qpos_idx] = 0.0

        mujoco.mj_forward(self.model, self.data)

        if self.grasp_manager is not None:
            self.grasp_manager.update_attached_poses()
            mujoco.mj_forward(self.model, self.data)

        if self.viewer is not None:
            self.viewer.sync()

        return True

    def set_position(self, q: np.ndarray) -> None:
        """Set joint positions directly (kinematic)."""
        self._target_position = np.asarray(q).copy()
        for joint_idx, qpos_idx in enumerate(self.joint_qpos_indices):
            self.data.qpos[qpos_idx] = q[joint_idx]
            self.data.qvel[qpos_idx] = 0.0
        mujoco.mj_forward(self.model, self.data)

        if self.grasp_manager is not None:
            self.grasp_manager.update_attached_poses()
            mujoco.mj_forward(self.model, self.data)

    def step(self) -> None:
        """Apply current target position and sync viewer."""
        for joint_idx, qpos_idx in enumerate(self.joint_qpos_indices):
            self.data.qpos[qpos_idx] = self._target_position[joint_idx]
            self.data.qvel[qpos_idx] = 0.0
        mujoco.mj_forward(self.model, self.data)

        if self.grasp_manager is not None:
            self.grasp_manager.update_attached_poses()
            mujoco.mj_forward(self.model, self.data)

        if self.viewer is not None:
            now = time.time()
            if now - self._last_viewer_sync >= self._viewer_sync_interval:
                self.viewer.sync()
                self._last_viewer_sync = now


class PhysicsExecutor:
    """Execute trajectories with physics simulation and velocity feedforward.

    Uses position-controlled actuators with velocity feedforward:
        cmd = q_desired + lookahead_time * qd_desired

    MuJoCo's actuator PD (defined in XML) handles low-level servo control.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        joint_qpos_indices: list[int],
        actuator_ids: list[int],
        control_dt: float = 0.008,
        lookahead_time: float = 0.1,
        viewer=None,
        joint_qvel_indices: list[int] | None = None,
    ):
        self.model = model
        self.data = data
        self.joint_qpos_indices = joint_qpos_indices
        self.joint_qvel_indices = (
            joint_qvel_indices if joint_qvel_indices is not None
            else joint_qpos_indices
        )
        self.actuator_ids = actuator_ids
        self.control_dt = control_dt
        self.lookahead_time = lookahead_time
        self.viewer = viewer

        self.steps_per_control = max(1, int(control_dt / model.opt.timestep))

        self._target_position = np.array([
            data.qpos[idx] for idx in joint_qpos_indices
        ])
        self._target_velocity = np.zeros(len(joint_qpos_indices))

        self._last_viewer_sync = 0.0
        self._viewer_sync_interval = 0.033

    @property
    def target_position(self) -> np.ndarray:
        return self._target_position.copy()

    def set_target(
        self,
        position: np.ndarray,
        velocity: np.ndarray | None = None,
    ) -> None:
        """Set target position (and optionally velocity) for position hold."""
        self._target_position = np.asarray(position).copy()
        if velocity is not None:
            self._target_velocity = np.asarray(velocity).copy()
        else:
            self._target_velocity = np.zeros(len(self.joint_qpos_indices))

    def step(self) -> None:
        """Apply control and step physics once."""
        q_command = (
            self._target_position
            + self.lookahead_time * self._target_velocity
        )
        for joint_idx, actuator_id in enumerate(self.actuator_ids):
            self.data.ctrl[actuator_id] = q_command[joint_idx]

        for _ in range(self.steps_per_control):
            mujoco.mj_step(self.model, self.data)

        if self.viewer is not None:
            now = time.time()
            if now - self._last_viewer_sync >= self._viewer_sync_interval:
                self.viewer.sync()
                self._last_viewer_sync = now

    def hold(self) -> None:
        """Capture current position and hold it."""
        self._target_position = np.array([
            self.data.qpos[idx] for idx in self.joint_qpos_indices
        ])
        self._target_velocity = np.zeros(len(self.joint_qpos_indices))

    def execute(self, trajectory: Trajectory, abort_fn=None) -> bool:
        """Execute trajectory with velocity feedforward."""
        if trajectory.dof != len(self.joint_qpos_indices):
            raise ValueError(
                f"Trajectory DOF {trajectory.dof} doesn't match "
                f"joint count {len(self.joint_qpos_indices)}"
            )

        for i in range(trajectory.num_waypoints):
            if abort_fn is not None and abort_fn():
                logger.info("Trajectory aborted at waypoint %d/%d", i, trajectory.num_waypoints)
                return False
            self._target_position = trajectory.positions[i]
            self._target_velocity = trajectory.velocities[i]
            self.step()
            time.sleep(self.control_dt)

        # Hold final position (zero velocity)
        self._target_position = trajectory.positions[-1].copy()
        self._target_velocity = np.zeros(len(self.joint_qpos_indices))

        # Settling period
        for _ in range(self.steps_per_control * 20):
            q_command = self._target_position
            for joint_idx, actuator_id in enumerate(self.actuator_ids):
                self.data.ctrl[actuator_id] = q_command[joint_idx]
            mujoco.mj_step(self.model, self.data)

        if self.viewer is not None:
            self.viewer.sync()

        return True

    def get_position(self) -> np.ndarray:
        """Get current actual joint positions."""
        return np.array([
            self.data.qpos[idx] for idx in self.joint_qpos_indices
        ])

    def get_velocity(self) -> np.ndarray:
        """Get current actual joint velocities."""
        return np.array([
            self.data.qvel[idx] for idx in self.joint_qvel_indices
        ])

    def get_tracking_error(self) -> np.ndarray:
        """Get target - actual position error."""
        return self._target_position - self.get_position()
