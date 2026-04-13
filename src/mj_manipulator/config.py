# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Generic configuration for MuJoCo manipulators.

Robot-specific configs (UR5e joint names, Robotiq body lists, etc.)
belong in the robot-specific package, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class SafetyResponse(Enum):
    """How the controller responds to velocity/acceleration limit violations.

    Applied per control cycle inside :meth:`PhysicsController.step`.
    """

    WARN = "warn"
    """Log the violation but don't modify the command. Use for debugging."""

    CLAMP = "clamp"
    """Clamp the command to within limits and log. Default for sim."""

    FAULT = "fault"
    """Halt the arm (hold current position), set a fault flag, log at
    ERROR. Matches UR5e protective stop / Franka reflex behavior.
    Call :meth:`PhysicsController.clear_fault` to resume."""


@dataclass
class KinematicLimits:
    """Velocity and acceleration limits for trajectory planning.

    These are used by TOPP-RA for trajectory retiming. They are NOT physics
    parameters — MuJoCo doesn't enforce them, but planners must respect them
    to ensure trajectories are executable on real hardware.

    Position limits come from the MuJoCo model XML (joint range attributes).
    """

    velocity: np.ndarray  # rad/s per joint (shape: dof,)
    acceleration: np.ndarray  # rad/s² per joint (shape: dof,)


@dataclass
class PlanningDefaults:
    """Default parameters for CBiRRT motion planning."""

    timeout: float = 30.0
    max_iterations: int = 5000
    step_size: float = 0.1
    goal_bias: float = 0.1
    smoothing_iterations: int = 100

    @classmethod
    def fast(cls) -> "PlanningDefaults":
        """Fast planning with shorter timeout."""
        return cls(timeout=10.0, max_iterations=2000, smoothing_iterations=50)

    @classmethod
    def thorough(cls) -> "PlanningDefaults":
        """Thorough planning for difficult problems."""
        return cls(timeout=60.0, max_iterations=10000, smoothing_iterations=200)


@dataclass
class EntityConfig:
    """Base configuration for a controllable entity (arm, base, gripper)."""

    name: str  # Unique identifier: "left_arm", "franka", etc.
    entity_type: str  # Category: "arm", "base", "gripper"
    joint_names: list[str]  # MuJoCo joint names this entity controls


@dataclass
class ArmConfig(EntityConfig):
    """Configuration for a single arm.

    Gripper-specific fields (body names, actuator, hand type) belong on the
    Gripper protocol, not here. ArmConfig defines the kinematic chain only.

    kinematic_limits is required and robot-specific. Use your robot's
    datasheet values (e.g., KinematicLimits for UR5e, Franka, Xarm).
    """

    kinematic_limits: KinematicLimits  # required: robot-specific velocity/acceleration limits
    ee_site: str = ""  # MuJoCo site name for Jacobian / FK
    tcp_offset: np.ndarray | None = None  # 4x4 SE3 from ee_site to tool center point
    ft_force_sensor: str | None = None  # MuJoCo force sensor name (3-axis)
    ft_torque_sensor: str | None = None  # MuJoCo torque sensor name (3-axis)
    planning_defaults: PlanningDefaults = field(default_factory=PlanningDefaults)

    def __post_init__(self):
        """Set entity_type to arm."""
        object.__setattr__(self, "entity_type", "arm")


@dataclass
class PhysicsExecutionConfig:
    """Parameters for physics-based trajectory execution."""

    control_dt: float = 0.008  # 125 Hz
    lookahead_time: float = 0.1  # Velocity feedforward gain (seconds)
    position_tolerance: float = 0.1  # Convergence threshold (rad)
    velocity_tolerance: float = 0.1  # rad/s
    convergence_timeout_steps: int = 500
    base_settling_steps: int = 50
    safety_response: SafetyResponse = SafetyResponse.CLAMP

    @classmethod
    def tight(cls) -> "PhysicsExecutionConfig":
        """Tighter convergence for precision tasks."""
        return cls(
            position_tolerance=0.02,
            velocity_tolerance=0.05,
            convergence_timeout_steps=1000,
        )


@dataclass
class GripperPhysicsConfig:
    """Parameters for physics-based gripper control."""

    close_steps: int = 200
    open_steps: int = 100
    firm_grip_steps: int = 50
    pre_open_steps: int = 20
    contact_check_interval: int = 10
    fully_closed_threshold: float = 0.98
    debug: bool = False


@dataclass
class RecoveryConfig:
    """Parameters for failure recovery motions."""

    retract_height: float = 0.15  # meters
    interpolation_steps: int = 1000


@dataclass
class PhysicsConfig:
    """Combined physics simulation configuration."""

    execution: PhysicsExecutionConfig = field(default_factory=PhysicsExecutionConfig)
    gripper: GripperPhysicsConfig = field(default_factory=GripperPhysicsConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
