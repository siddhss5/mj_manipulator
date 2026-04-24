# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Generic MuJoCo manipulator control framework.

Provides planning, execution, grasping, and cartesian control for any robot arm.

The central abstraction is ExecutionContext — a protocol that unifies simulation
and hardware execution. The same code works whether you're running in MuJoCo
or on a real robot:

    with context as ctx:
        result = arm.plan_to_pose(target)
        ctx.execute(result)
        ctx.arm("franka").grasp("mug")
"""

from mj_manipulator.arm import Arm, ArmRobotModel, ContextRobotModel
from mj_manipulator.cartesian import (
    CartesianControlConfig,
    CartesianController,
    MoveUntilTouchResult,
    TwistExecutionResult,
    TwistStepResult,
)
from mj_manipulator.collision import CollisionChecker
from mj_manipulator.config import (
    ArmConfig,
    EntityConfig,
    ExecutionConfig,
    GripperPhysicsConfig,
    KinematicLimits,
    PhysicsConfig,
    PhysicsExecutionConfig,
    PlanningDefaults,
    RecoveryConfig,
)
from mj_manipulator.controller import ArmExecutor, Controller, EntityExecutor
from mj_manipulator.executor import KinematicExecutor, PhysicsExecutor
from mj_manipulator.force_control import ForceThresholds, SpeedProfile
from mj_manipulator.grasp_manager import GraspManager
from mj_manipulator.grasp_verifier import (
    GraspState,
    GraspVerifier,
    VerifierFacts,
    VerifierParams,
)
from mj_manipulator.grippers import FrankaGripper, RobotiqGripper
from mj_manipulator.kinematic_controller import KinematicController
from mj_manipulator.load_signals import (
    GripperPositionSignal,
    JointTorqueSignal,
    LoadSignal,
    WristFTSignal,
)
from mj_manipulator.outcome import FailureKind, Outcome, failure, success
from mj_manipulator.ownership import OwnerKind, OwnershipRegistry
from mj_manipulator.perception import SimPerceptionService
from mj_manipulator.physics_controller import ArmPhysicsExecutor, PhysicsController
from mj_manipulator.planning import PlanResult
from mj_manipulator.protocols import (
    ArmController,
    ExecutionContext,
    GraspSource,
    Gripper,
    IKSolver,
    PerceptionService,
)
from mj_manipulator.robot import ManipulationRobot, RobotBase
from mj_manipulator.safe_retract import safe_retract
from mj_manipulator.servo import ft_guarded_move, servo_to_pose
from mj_manipulator.sim_context import SimArmController, SimContext
from mj_manipulator.status_hud import StatusHud
from mj_manipulator.teleop import (
    SafetyMode,
    TeleopConfig,
    TeleopController,
    TeleopFrame,
    TeleopState,
)
from mj_manipulator.trajectory import Trajectory, create_linear_trajectory

__all__ = [
    # Arm
    "Arm",
    "ArmRobotModel",
    "ContextRobotModel",
    # Protocols (the core contracts)
    "ExecutionContext",
    "ArmController",
    "Gripper",
    "IKSolver",
    "GraspSource",
    "PerceptionService",
    "LoadSignal",
    # Perception
    "SimPerceptionService",
    # Load signals
    "GripperPositionSignal",
    "WristFTSignal",
    "JointTorqueSignal",
    # Grasp verification
    "GraspVerifier",
    "GraspState",
    "VerifierParams",
    "VerifierFacts",
    # Ownership (concurrent multi-arm control)
    "OwnershipRegistry",
    "OwnerKind",
    # Trajectory
    "Trajectory",
    "create_linear_trajectory",
    # Planning
    "PlanResult",
    # Grasp management
    "GraspManager",
    # Grippers
    "RobotiqGripper",
    "FrankaGripper",
    # Collision checking
    "CollisionChecker",
    # Executors (standalone, no-event-loop usage)
    "KinematicExecutor",
    "PhysicsExecutor",
    # Controller hierarchy
    "Controller",
    "PhysicsController",
    "KinematicController",
    "ArmExecutor",
    "EntityExecutor",
    "ArmPhysicsExecutor",  # backwards compat alias for ArmExecutor
    # Robot protocol and base class
    "ManipulationRobot",
    "RobotBase",
    # Simulation context
    "SimContext",
    "SimArmController",
    # Cartesian control
    "CartesianController",
    "CartesianControlConfig",
    "TwistStepResult",
    "MoveUntilTouchResult",
    "TwistExecutionResult",
    # Outcome (structured behavior returns)
    "Outcome",
    "FailureKind",
    "success",
    "failure",
    # Force control
    "ForceThresholds",
    "SpeedProfile",
    # Servo primitives (contact-rich manipulation)
    "servo_to_pose",
    "ft_guarded_move",
    # Safe retract
    "safe_retract",
    # Teleop
    "TeleopController",
    "TeleopConfig",
    "TeleopState",
    "TeleopFrame",
    "SafetyMode",
    # Status HUD
    "StatusHud",
    # Config
    "ArmConfig",
    "EntityConfig",
    "KinematicLimits",
    "PlanningDefaults",
    "PhysicsConfig",
    "ExecutionConfig",
    "PhysicsExecutionConfig",
    "GripperPhysicsConfig",
    "RecoveryConfig",
]
