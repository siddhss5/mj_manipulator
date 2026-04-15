# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

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
from dataclasses import dataclass
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
    TRACKING_COLLISION = "tracking_collision"
    UNREACHABLE = "unreachable"


class SafetyMode(Enum):
    """Collision safety mode for teleop.

    ALLOW:  Move and flag collisions in status. For rescue, contact-rich
            demos, and general teleop. Default.
    REJECT: Don't move to colliding configs. For real robot safety.
    """

    ALLOW = "allow"
    REJECT = "reject"


@dataclass
class TeleopConfig:
    """Configuration for TeleopController."""

    max_joint_step: float = 0.05
    """Maximum per-joint step (rad) per control cycle. Larger deltas are
    scaled down uniformly. Prevents IK solution flips (elbow up/down)
    while always making progress toward the target. At 30Hz this is
    ~1.5 rad/s (~85°/s) — comfortable teleop speed."""

    twist_dt: float = 0.008
    """Timestep for CartesianController twist integration."""

    idle_timeout: float = 0.5
    """Seconds without input before transitioning to IDLE."""

    safety_mode: SafetyMode = SafetyMode.ALLOW
    """Collision safety mode. ALLOW for sim, REJECT for real robot."""


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

        # Safety: collision checker (created lazily)
        self._collision_checker = None

        # Gripper toggle request (set from callback, executed in step)
        self._gripper_toggle_requested = False

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

    @property
    def safety_mode(self) -> SafetyMode:
        """Current collision safety mode."""
        return self._config.safety_mode

    @safety_mode.setter
    def safety_mode(self, mode: SafetyMode) -> None:
        """Change collision safety mode (can be toggled at runtime)."""
        self._config.safety_mode = mode
        logger.info("Teleop safety mode: %s", mode.value)

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

        # Read latest input (keep it for next step — gizmo only fires on drag)
        with self._lock:
            pose = self._target_pose
            twist = self._target_twist
            mode = self._input_mode
            last_input = self._last_input_time
            gripper_toggle = self._gripper_toggle_requested
            # Only clear twist (continuous input — stale twist = no motion).
            # Keep pose (gizmo position persists between drags).
            self._target_twist = None
            self._gripper_toggle_requested = False

        # Execute gripper toggle if requested (must happen in this thread)
        if gripper_toggle:
            self._execute_gripper_toggle()

        # No input yet, or twist timed out → idle
        if mode is None or (mode == "twist" and time.monotonic() - last_input > self._config.idle_timeout):
            self._state = TeleopState.IDLE
            return self._state

        prev_state = self._state

        if mode == "pose" and pose is not None:
            self._state = self._step_pose(pose)
        elif mode == "twist" and twist is not None:
            self._state = self._step_twist(twist)
        else:
            self._state = TeleopState.IDLE

        # Log state transitions (not every frame)
        if self._state != prev_state:
            arm_name = self._arm.config.name
            if self._state == TeleopState.TRACKING_COLLISION:
                logger.warning("Teleop %s: collision detected", arm_name)
            elif self._state == TeleopState.UNREACHABLE:
                logger.warning("Teleop %s: target unreachable", arm_name)
            elif self._state == TeleopState.TRACKING and prev_state != TeleopState.IDLE:
                logger.info("Teleop %s: tracking resumed", arm_name)

        # Record frame if recording (both clean tracking and collision tracking)
        if self._recording and self._state in (
            TeleopState.TRACKING,
            TeleopState.TRACKING_COLLISION,
        ):
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

    # -- Gripper control ------------------------------------------------------

    def toggle_gripper(self) -> None:
        """Request gripper toggle. Thread-safe — executed in step().

        The actual grasp/release happens in the teleop loop thread
        which owns the MuJoCo data, avoiding data races.
        Ignored if teleop is not active.
        """
        if not self._active:
            return
        with self._lock:
            self._gripper_toggle_requested = True

    def _execute_gripper_toggle(self) -> None:
        """Execute gripper toggle. Called from step() in the teleop thread.

        Temporarily clears the pose target so the IK tracking doesn't
        fight the physics controller during gripper actuation.
        """
        arm_name = self._arm.config.name
        gripper = self._arm.gripper
        if gripper is None:
            return

        # Pause pose tracking during gripper actuation
        with self._lock:
            self._target_pose = None
            self._input_mode = None

        pos = gripper.get_actual_position()
        if pos > 0.5:
            self._ctx.arm(arm_name).release()
        else:
            self._ctx.arm(arm_name).grasp()
        self._ctx.sync()

        # Restore tracking at current EE pose (not the old target)
        ee_pose = self._arm.get_ee_pose()
        with self._lock:
            self._target_pose = ee_pose
            self._input_mode = "pose"
            self._last_input_time = time.monotonic()

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

    # -- Internal: safety check -----------------------------------------------

    def _check_and_commit(self, q_target: np.ndarray) -> TeleopState:
        """Check collision safety and commit joint targets.

        Shared by both pose and twist paths. Applies the safety mode:
        - ALLOW: commit and return TRACKING_COLLISION if in collision
        - REJECT: don't commit if in collision, return UNREACHABLE

        Checks both per-arm collisions (forked env) and arm-arm
        collisions (live data via is_arm_in_collision).
        """
        mode = self._config.safety_mode
        in_collision = False

        # Check per-arm collisions (arm vs environment/objects)
        cc = self._get_collision_checker()
        if cc is not None:
            in_collision = not cc.is_valid(q_target)

        # Check arm-arm collisions on live data.
        # The forked collision checker doesn't see the other arm's position.
        # Use the Arm's live collision method which reads from live MjData.
        if not in_collision:
            in_collision = self._check_live_collisions(q_target)

        if in_collision and mode == SafetyMode.REJECT:
            return TeleopState.UNREACHABLE

        # Clamp per-joint position step to the arm's velocity limits.
        # This ensures the position target and velocity feedforward are
        # consistent — both respect the same limits. The max step per
        # joint per tick is vel_limit[j] * dt.
        q_current = self._arm.get_joint_positions()
        delta = q_target - q_current
        dt = self._config.twist_dt
        limits = getattr(self._arm.config, "kinematic_limits", None)
        if limits is not None:
            max_step = limits.velocity * dt  # per-joint, rad/tick
            delta = np.clip(delta, -max_step, max_step)
        else:
            # Fallback: uniform step limit (for arms without kinematic_limits)
            max_step = self._config.max_joint_step
            max_component = float(np.max(np.abs(delta)))
            if max_component > max_step:
                delta = delta * (max_step / max_component)
        q_target = q_current + delta

        # Velocity feedforward — now guaranteed within limits because
        # delta was already clamped to vel_limit * dt above.
        qd = delta / max(dt, 1e-6)

        arm_name = self._arm.config.name
        self._ctx.step_cartesian(arm_name, q_target, qd)

        if in_collision:
            return TeleopState.TRACKING_COLLISION
        return TeleopState.TRACKING

    def _check_live_collisions(self, q_target: np.ndarray) -> bool:
        """Check for arm-arm collisions using live MuJoCo data.

        The per-arm collision checker uses a forked env and can't see the
        other arm. This method checks the live scene by temporarily setting
        candidate positions, running mj_forward, and scanning contacts.

        Must be called while holding the sim lock.
        """
        if not hasattr(self._arm, "env"):
            return False
        import mujoco

        model = self._arm.env.model
        data = self._arm.env.data

        # Build body ID set for THIS arm (reuse CollisionChecker's logic)
        my_body_ids: set[int] = set()
        for jname in self._arm.config.joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            bid = model.jnt_bodyid[jid]
            my_body_ids.add(bid)
        # Add all descendant bodies
        changed = True
        while changed:
            changed = False
            for i in range(model.nbody):
                if i not in my_body_ids and model.body_parentid[i] in my_body_ids:
                    my_body_ids.add(i)
                    changed = True

        # Save and set candidate positions
        indices = self._arm.joint_qpos_indices
        q_saved = np.array([data.qpos[i] for i in indices])
        for i, idx in enumerate(indices):
            data.qpos[idx] = q_target[i]
        mujoco.mj_forward(model, data)

        # Scan contacts for arm-arm collisions
        in_collision = False
        for i in range(data.ncon):
            c = data.contact[i]
            if c.dist >= 0:
                continue  # no penetration
            b1 = model.geom_bodyid[c.geom1]
            b2 = model.geom_bodyid[c.geom2]
            b1_mine = b1 in my_body_ids
            b2_mine = b2 in my_body_ids
            if not (b1_mine or b2_mine):
                continue  # neither body is ours
            if b1_mine and b2_mine:
                continue  # self-collision (handled by exclude tags)
            # One is ours, one isn't — check if the other is another arm
            other = b2 if b1_mine else b1
            other_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other) or ""
            if "ur5e" in other_name or "gripper" in other_name:
                in_collision = True
                break

        # Restore
        for i, idx in enumerate(indices):
            data.qpos[idx] = q_saved[i]
        mujoco.mj_forward(model, data)

        return in_collision

    def _get_collision_checker(self):
        """Lazily create a collision checker from the arm's planner."""
        if self._collision_checker is None:
            try:
                planner = self._arm.create_planner()
                self._collision_checker = planner.collision
            except Exception:
                return None
        return self._collision_checker

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

        return self._check_and_commit(q_best)

    def _pick_closest(
        self,
        solutions: list[np.ndarray],
        q_current: np.ndarray,
    ) -> np.ndarray | None:
        """Pick the IK solution closest to current config.

        In physics mode and on real hardware, the PD controller / servo
        handles velocity limiting naturally. No clamping needed.
        """
        best = None
        best_dist = float("inf")

        for q in solutions:
            if isinstance(q, list):
                q = np.array(q)
            if q.size == 0:
                continue
            dist = float(np.linalg.norm(q - q_current))
            if dist < best_dist:
                best_dist = dist
                best = q

        return best

    # -- Internal: twist path -------------------------------------------------

    def _step_twist(self, twist: np.ndarray) -> TeleopState:
        """CartesianController-based twist tracking."""
        ctrl = self._get_cart_ctrl()
        result = ctrl.step(twist, dt=self._config.twist_dt)

        if result.achieved_fraction < 0.1:
            return TeleopState.UNREACHABLE

        # CartesianController wrote to data.qpos — read the new positions
        q_new = self._arm.get_joint_positions()
        return self._check_and_commit(q_new)

    def _get_cart_ctrl(self):
        """Lazily create CartesianController for twist input."""
        if self._cart_ctrl is None:
            from mj_manipulator.cartesian import CartesianController

            arm_name = self._arm.config.name

            def step_fn(q, qd):
                self._ctx.step_cartesian(arm_name, q, qd)

            self._cart_ctrl = CartesianController.from_arm(
                self._arm,
                step_fn=step_fn,
            )
        return self._cart_ctrl

    # -- Internal: recording --------------------------------------------------

    def _record_frame(self) -> None:
        """Capture current state as a TeleopFrame."""
        gripper = self._arm.gripper
        gripper_pos = gripper.get_actual_position() if gripper is not None else 0.0
        self._frames.append(
            TeleopFrame(
                timestamp=time.monotonic() - self._record_start,
                joint_positions=self._arm.get_joint_positions().copy(),
                ee_pose=self._arm.get_ee_pose().copy(),
                gripper_position=float(gripper_pos),
            )
        )
