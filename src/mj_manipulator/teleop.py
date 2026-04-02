"""Teleop controller for interactive arm control.

Accepts either SE(3) poses (from a gizmo, VR controller, mouse) or 6D twists
(from a joystick, SpaceMouse, keyboard) and streams control to any arm via
the ExecutionContext protocol.

Works identically with SimContext (kinematic or physics) and HardwareContext
(real robot via ROS/RTDE).

Usage — pose input (Viser gizmo)::

    controller = TeleopController(arm, ctx)
    controller.activate()
    # In Viser on_update callback:
    controller.set_target_pose(gizmo_pose)
    # In on_sync callback:
    controller.step()

Usage — twist input (joystick)::

    controller = TeleopController(arm, ctx)
    controller.activate()
    # In device callback:
    controller.set_target_twist(joystick_twist)
    # In control loop:
    controller.step()
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mj_manipulator.arm import Arm

logger = logging.getLogger(__name__)


class TeleopState(Enum):
    """Teleop controller state."""

    IDLE = "idle"
    TRACKING = "tracking"
    UNREACHABLE = "unreachable"


@dataclass
class TeleopConfig:
    """Configuration for TeleopController."""

    max_joint_delta: float = 0.15
    """Maximum joint-space step per control cycle (rad). Limits jumps
    when IK solutions change between cycles."""

    twist_dt: float = 0.008
    """Timestep for CartesianController twist integration."""

    idle_timeout: float = 0.5
    """Seconds without input before transitioning to IDLE."""


@dataclass
class TeleopFrame:
    """Single frame of a recorded teleop trajectory."""

    timestamp: float
    joint_positions: np.ndarray
    ee_pose: np.ndarray
    gripper_position: float


class TeleopController:
    """Unified teleop controller with pose and twist inputs.

    Two input paths, one output::

        Pose input (gizmo/VR)  → IK (closest to current q) → step_cartesian
        Twist input (joystick) → CartesianController.step   → step_cartesian

    Thread-safe: input methods can be called from any thread (e.g., Viser
    callbacks). The ``step()`` method must be called from the thread that
    owns the MuJoCo data (typically the main thread or Viser's sync loop).

    Args:
        arm: Arm instance with IK solver.
        context: ExecutionContext for streaming control.
        config: Optional configuration overrides.
    """

    def __init__(
        self,
        arm: Arm,
        context: object,
        config: TeleopConfig | None = None,
    ):
        self._arm = arm
        self._ctx = context
        self._config = config or TeleopConfig()

        self._state = TeleopState.IDLE
        self._active = False

        # Thread-safe input buffer
        self._lock = threading.Lock()
        self._target_pose: np.ndarray | None = None
        self._target_twist: np.ndarray | None = None
        self._input_mode: str | None = None  # "pose" or "twist"
        self._last_input_time: float = 0.0

        # Twist path: CartesianController (created lazily on first twist input)
        self._cart_ctrl = None

        # Recording
        self._recording = False
        self._frames: list[TeleopFrame] = []

    @property
    def state(self) -> TeleopState:
        """Current teleop state."""
        return self._state

    @property
    def is_active(self) -> bool:
        """Whether teleop is currently activated."""
        return self._active

    # -- Input methods (thread-safe) ------------------------------------------

    def set_target_pose(self, pose_4x4: np.ndarray) -> None:
        """Set desired EE pose. Called from device callback thread.

        Args:
            pose_4x4: 4x4 homogeneous transform for the end-effector.
        """
        with self._lock:
            self._target_pose = np.array(pose_4x4, dtype=np.float64)
            self._target_twist = None
            self._input_mode = "pose"
            self._last_input_time = time.monotonic()

    def set_target_twist(self, twist_6d: np.ndarray) -> None:
        """Set desired EE twist. Called from device callback thread.

        Args:
            twist_6d: [vx, vy, vz, wx, wy, wz] in world frame.
        """
        with self._lock:
            self._target_twist = np.array(twist_6d, dtype=np.float64)
            self._target_pose = None
            self._input_mode = "twist"
            self._last_input_time = time.monotonic()

    # -- Control loop ---------------------------------------------------------

    def step(self) -> TeleopState:
        """Execute one teleop control cycle.

        Reads the latest input, computes joint targets, and steps the arm
        via the ExecutionContext. Call this from the main control loop
        (e.g., Viser's ``on_sync`` at ~30 Hz).

        Returns:
            Current TeleopState after this step.
        """
        if not self._active:
            self._state = TeleopState.IDLE
            return self._state

        # Read and clear input atomically
        with self._lock:
            pose = self._target_pose
            twist = self._target_twist
            mode = self._input_mode
            last_input = self._last_input_time
            self._target_pose = None
            self._target_twist = None

        # Idle timeout
        if mode is None or (time.monotonic() - last_input > self._config.idle_timeout):
            self._state = TeleopState.IDLE
            return self._state

        if mode == "pose" and pose is not None:
            self._state = self._step_pose(pose)
        elif mode == "twist" and twist is not None:
            self._state = self._step_twist(twist)
        else:
            self._state = TeleopState.IDLE

        # Record frame if recording
        if self._recording and self._state == TeleopState.TRACKING:
            self._record_frame()

        return self._state

    # -- Lifecycle ------------------------------------------------------------

    def activate(self) -> np.ndarray:
        """Enter teleop mode.

        Returns:
            Current EE pose (4x4) for initializing the gizmo position.
        """
        self._active = True
        self._state = TeleopState.IDLE
        with self._lock:
            self._target_pose = None
            self._target_twist = None
            self._input_mode = None
        # Reset CartesianController warm-start
        if self._cart_ctrl is not None:
            self._cart_ctrl.reset()
        logger.info("Teleop activated for %s arm", self._arm.config.name)
        return self._arm.get_ee_pose()

    def deactivate(self) -> None:
        """Exit teleop mode. Arm holds current position."""
        self._active = False
        self._state = TeleopState.IDLE
        with self._lock:
            self._target_pose = None
            self._target_twist = None
            self._input_mode = None
        if self._recording:
            self.stop_recording()
        logger.info("Teleop deactivated for %s arm", self._arm.config.name)

    # -- Recording ------------------------------------------------------------

    def start_recording(self) -> None:
        """Begin recording teleop frames."""
        self._frames = []
        self._recording = True
        self._record_start = time.monotonic()
        logger.info("Teleop recording started")

    def stop_recording(self) -> list[TeleopFrame]:
        """Stop recording and return captured frames.

        Returns:
            List of TeleopFrame captured during the recording.
        """
        self._recording = False
        frames = self._frames
        self._frames = []
        logger.info("Teleop recording stopped: %d frames", len(frames))
        return frames

    @property
    def is_recording(self) -> bool:
        """Whether recording is active."""
        return self._recording

    # -- Internal: pose path --------------------------------------------------

    def _step_pose(self, pose: np.ndarray) -> TeleopState:
        """IK-based pose tracking."""
        ik = self._arm.ik_solver
        if ik is None:
            return TeleopState.UNREACHABLE

        solutions = ik.solve(pose)
        if not solutions:
            return TeleopState.UNREACHABLE

        q_current = self._arm.get_joint_positions()
        q_best = self._pick_closest(solutions, q_current)
        if q_best is None:
            return TeleopState.UNREACHABLE

        # Compute velocity feedforward for smooth tracking
        dt = self._config.twist_dt
        qd = (q_best - q_current) / max(dt, 1e-6)

        arm_name = self._arm.config.name
        self._ctx.step_cartesian(arm_name, q_best, qd)
        return TeleopState.TRACKING

    def _pick_closest(
        self, solutions: list[np.ndarray], q_current: np.ndarray,
    ) -> np.ndarray | None:
        """Pick the IK solution closest to current config.

        Rejects solutions that require a joint-space jump larger than
        max_joint_delta (prevents sudden large motions).
        """
        best = None
        best_dist = float("inf")
        max_delta = self._config.max_joint_delta

        for q in solutions:
            if isinstance(q, list):
                q = np.array(q)
            if q.size == 0:
                continue
            delta = np.abs(q - q_current)
            if np.max(delta) > max_delta:
                continue
            dist = float(np.linalg.norm(delta))
            if dist < best_dist:
                best_dist = dist
                best = q

        return best

    # -- Internal: twist path -------------------------------------------------

    def _step_twist(self, twist: np.ndarray) -> TeleopState:
        """CartesianController-based twist tracking."""
        ctrl = self._get_cart_ctrl()
        result = ctrl.step(twist, dt=self._config.twist_dt)

        # Read the new joint positions from the arm (CartesianController
        # already wrote to data.qpos)
        q_new = self._arm.get_joint_positions()
        arm_name = self._arm.config.name
        self._ctx.step_cartesian(arm_name, q_new)

        if result.achieved_fraction < 0.1:
            return TeleopState.UNREACHABLE
        return TeleopState.TRACKING

    def _get_cart_ctrl(self):
        """Lazily create CartesianController for twist input."""
        if self._cart_ctrl is None:
            from mj_manipulator.cartesian import CartesianController

            arm_name = self._arm.config.name

            def step_fn(q, qd):
                self._ctx.step_cartesian(arm_name, q, qd)

            self._cart_ctrl = CartesianController.from_arm(
                self._arm, step_fn=step_fn,
            )
        return self._cart_ctrl

    # -- Internal: recording --------------------------------------------------

    def _record_frame(self) -> None:
        """Capture current state as a TeleopFrame."""
        gripper = self._arm.gripper
        gripper_pos = gripper.get_actual_position() if gripper is not None else 0.0
        self._frames.append(TeleopFrame(
            timestamp=time.monotonic() - self._record_start,
            joint_positions=self._arm.get_joint_positions().copy(),
            ee_pose=self._arm.get_ee_pose().copy(),
            gripper_position=float(gripper_pos),
        ))
