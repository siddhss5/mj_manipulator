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

from mj_manipulator.config import (
    ArmConfig,
    EntityConfig,
    GripperPhysicsConfig,
    KinematicLimits,
    PhysicsConfig,
    PhysicsExecutionConfig,
    PlanningDefaults,
    RecoveryConfig,
)
from mj_manipulator.planning import PlanResult
from mj_manipulator.protocols import (
    ArmController,
    ExecutionContext,
    GraspSource,
    Gripper,
    IKSolver,
)
from mj_manipulator.collision import CollisionChecker
from mj_manipulator.grasp_manager import GraspManager, detect_grasped_object
from mj_manipulator.trajectory import Trajectory, create_linear_trajectory

__all__ = [
    # Protocols (the core contracts)
    "ExecutionContext",
    "ArmController",
    "Gripper",
    "IKSolver",
    "GraspSource",
    # Trajectory
    "Trajectory",
    "create_linear_trajectory",
    # Planning
    "PlanResult",
    # Grasp management
    "GraspManager",
    "detect_grasped_object",
    # Collision checking
    "CollisionChecker",
    # Config
    "ArmConfig",
    "EntityConfig",
    "KinematicLimits",
    "PlanningDefaults",
    "PhysicsConfig",
    "PhysicsExecutionConfig",
    "GripperPhysicsConfig",
    "RecoveryConfig",
]
