# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Protocols defining the contracts between mj_manipulator and robot-specific packages.

Robot-specific packages (geodude, franka_control, xarm_control) implement
these protocols to plug into the generic manipulation framework.

The central abstraction is ExecutionContext — the sim-to-real bridge. All
primitives, cartesian control, and policies interact with the robot exclusively
through this interface. Whether the underlying implementation talks to MuJoCo
or to real hardware is invisible to the caller.

Protocols:
    ExecutionContext: Unified interface for robot execution (sim or real)
    ArmController: Per-arm grasp/release operations within a context
    Gripper: Low-level gripper hardware abstraction
    IKSolver: Inverse kinematics solver
    GraspSource: Provides grasps/placements for manipulation primitives
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from mj_manipulator.planning import PlanResult
    from mj_manipulator.trajectory import Trajectory


# ---------------------------------------------------------------------------
# Execution context — the sim-to-real bridge
# ---------------------------------------------------------------------------


@runtime_checkable
class ArmController(Protocol):
    """Per-arm control interface within an execution context.

    Combines gripper actuation with grasp-manager bookkeeping (weld creation,
    collision-group updates, attached-object tracking). The context creates
    one of these per arm.

    Implementations:
        SimArmController (mj_manipulator) — MuJoCo kinematic or physics mode
        HardwareArmController (robot-specific) — real gripper via ROS/RTDE/etc.
    """

    def grasp(self, object_name: str) -> str | None:
        """Close gripper and attempt to grasp an object.

        Handles the full grasp pipeline:
        1. Actuate gripper (physics or kinematic close)
        2. Detect contact / confirm grasp
        3. Update grasp manager (weld, collision groups)

        Args:
            object_name: Name of the object to grasp.

        Returns:
            Name of grasped object if successful, None otherwise.
        """
        ...

    def release(self, object_name: str | None = None) -> None:
        """Open gripper and release held object(s).

        Handles the full release pipeline:
        1. Detach object (remove weld, restore collision groups)
        2. Actuate gripper open
        3. Sync state

        Args:
            object_name: Specific object to release, or None for all.
        """
        ...


@runtime_checkable
class ExecutionContext(Protocol):
    """Unified interface for executing robot motions — the sim-to-real bridge.

    All planning, primitives, cartesian control, and policies interact with the
    robot exclusively through this interface. Whether the implementation talks
    to MuJoCo or to real hardware is invisible to the caller.

    This is the central abstraction of mj_manipulator. The same user code:

        with context as ctx:
            result = arm.plan_to_pose(target)
            ctx.execute(result)
            ctx.arm("franka").grasp("mug")

    works whether ``context`` is a SimContext (kinematic or physics MuJoCo) or a
    HardwareContext (real robot via ROS, RTDE, gRPC, etc.).

    Three execution patterns:

    1. **Batch trajectory execution** (plan then execute):
           result = arm.plan_to_pose(target)
           ctx.execute(result)

    2. **Streaming joint control** (joint-level policies):
           while ctx.is_running():
               q_target = policy(arm.get_joint_positions())
               ctx.step({"franka": q_target})

    3. **Streaming cartesian control** (teleop, force-guided motion):
           while ctx.is_running():
               ctx.step_cartesian("franka", q_new, qd_new)

    Implementations:
        SimContext (mj_manipulator) — kinematic or physics MuJoCo execution
        HardwareContext (robot-specific) — real robot control
    """

    def execute(self, item: "Trajectory | PlanResult") -> bool:
        """Execute a trajectory or plan result.

        For PlanResult, executes all trajectories in sequence (base first,
        then arm). Each trajectory is routed to the appropriate entity.

        Args:
            item: Single trajectory or PlanResult with ordered trajectories.

        Returns:
            True if execution completed successfully.
        """
        ...

    def step(self, targets: dict[str, np.ndarray] | None = None) -> None:
        """Advance one control cycle with optional joint targets.

        Used for streaming joint-level control (policies, teleoperation).
        Each key is an arm name, each value is the target joint positions.
        Arms not in ``targets`` hold their current position.

        In simulation: steps MuJoCo physics (or sets qpos for kinematic mode).
        On hardware: sends joint commands and waits for one control cycle.

        Args:
            targets: Dict mapping arm names to target joint positions.
                    None means hold all arms at current positions.
        """
        ...

    def step_cartesian(
        self,
        arm_name: str,
        position: np.ndarray,
        velocity: np.ndarray | None = None,
    ) -> None:
        """Advance one control cycle with a cartesian-space joint target.

        Used for reactive cartesian control (teleop, move_until_touch,
        force-guided motion). Passes both position and velocity for smooth
        feedforward tracking.

        In simulation: uses reactive lookahead (2x control_dt) for smooth PD
        control (physics) or direct qpos setting (kinematic).
        On hardware: streams to low-level joint controller.

        Args:
            arm_name: Which arm to control.
            position: Target joint positions (from IK / QP solver output).
            velocity: Target joint velocities for feedforward (optional).
        """
        ...

    def sync(self) -> None:
        """Synchronize state with the environment.

        In simulation: calls mj_forward and viewer.sync().
        On hardware: reads joint encoders, F/T sensors, gripper state.
        """
        ...

    def is_running(self) -> bool:
        """Check if the context is still active and safe to command.

        In simulation: True while the viewer window is open (or always
        True in headless mode).
        On hardware: True while no e-stop is triggered and comms are live.
        """
        ...

    def arm(self, name: str) -> ArmController:
        """Get per-arm controller for grasp/release operations.

        Args:
            name: Arm identifier (e.g. "left", "franka").

        Returns:
            ArmController for the specified arm.
        """
        ...

    @property
    def control_dt(self) -> float:
        """Control timestep in seconds.

        Determines the rate of step() and step_cartesian() calls.
        In simulation: MuJoCo timestep (typically 0.002-0.008s).
        On hardware: control loop period (e.g. 0.002s for UR RTDE).
        """
        ...


# ---------------------------------------------------------------------------
# Robot hardware protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class Gripper(Protocol):
    """Protocol for gripper implementations.

    Each robot provides its own concrete gripper:
    - geodude: Robotiq2F140 (parallel jaw, 4-bar linkage)
    - franka_control: FrankaHand (parallel jaw, prismatic)
    - etc.
    """

    @property
    def arm_name(self) -> str:
        """Which arm this gripper belongs to."""
        ...

    @property
    def gripper_body_names(self) -> list[str]:
        """MuJoCo body names for contact detection and collision filtering."""
        ...

    @property
    def attachment_body(self) -> str:
        """MuJoCo body name that objects weld to during kinematic grasping.

        This is the body that makes contact with the object. For a parallel
        jaw gripper, typically one of the finger pads or follower links.
        """
        ...

    @property
    def actuator_id(self) -> int | None:
        """MuJoCo actuator ID for gripper control, or None if no actuator."""
        ...

    @property
    def ctrl_open(self) -> float:
        """Actuator control value for fully open position."""
        ...

    @property
    def ctrl_closed(self) -> float:
        """Actuator control value for fully closed position."""
        ...

    @property
    def is_holding(self) -> bool:
        """Whether the gripper is currently holding an object."""
        ...

    @property
    def held_object(self) -> str | None:
        """Name of the held object, or None."""
        ...

    def set_candidate_objects(self, objects: list[str] | None) -> None:
        """Set which objects the gripper should try to grasp.

        Args:
            objects: List of object names to consider, or None for all.
        """
        ...

    def kinematic_close(self, steps: int = 50) -> str | None:
        """Close gripper in kinematic mode (no physics).

        Detects contact geometrically and returns the grasped object name.

        Args:
            steps: Number of interpolation steps for closing motion.

        Returns:
            Name of grasped object, or None if no object detected.
        """
        ...

    def kinematic_open(self) -> None:
        """Open gripper in kinematic mode."""
        ...

    def get_actual_position(self) -> float:
        """Get current gripper position (0.0 = fully open, 1.0 = fully closed)."""
        ...


@runtime_checkable
class IKSolver(Protocol):
    """Protocol for inverse kinematics solvers.

    Mirrors pycbirrt's IKSolver protocol. Each robot provides its own
    solver (analytical for UR5e/Franka, numerical for others).
    """

    def solve(self, pose: np.ndarray, q_init: np.ndarray | None = None) -> list[np.ndarray]:
        """Solve IK for a target pose (raw, may include invalid solutions).

        Args:
            pose: 4x4 homogeneous transform of desired end-effector pose.
            q_init: Optional initial configuration hint.

        Returns:
            List of joint configurations (may include out-of-limits or colliding).
        """
        ...

    def solve_valid(self, pose: np.ndarray, q_init: np.ndarray | None = None) -> list[np.ndarray]:
        """Solve IK and return only collision-free, in-limits solutions.

        Args:
            pose: 4x4 homogeneous transform of desired end-effector pose.
            q_init: Optional initial configuration hint.

        Returns:
            List of valid joint configurations (may be empty).
        """
        ...


@runtime_checkable
class GraspSource(Protocol):
    """Protocol for providing grasps and placements for objects.

    Robot-specific packages implement this to tell the manipulation
    primitives how to grasp and place objects. For example:
    - geodude: AffordanceRegistry (loads TSR templates from YAML)
    - A learning-based system could implement this with a neural grasp predictor
    """

    def get_grasps(self, object_name: str, hand_type: str) -> list:
        """Get grasp TSRs for an object.

        Args:
            object_name: Name of the object to grasp.
            hand_type: Gripper type string for affordance matching.

        Returns:
            List of TSR objects representing valid grasps.
        """
        ...

    def get_placements(self, destination: str, object_name: str) -> list:
        """Get placement TSRs for an object at a destination.

        Args:
            destination: Where to place the object.
            object_name: What object is being placed.

        Returns:
            List of TSR objects representing valid placements.
        """
        ...

    def get_graspable_objects(self) -> list[str]:
        """Get list of objects that can be grasped."""
        ...

    def get_place_destinations(self, object_name: str) -> list[str]:
        """Get valid placement destinations for an object."""
        ...
