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
from mj_manipulator.arm import Arm, ArmRobotModel, ContextRobotModel
from mj_manipulator.planning import PlanResult
from mj_manipulator.protocols import (
    ArmController,
    ExecutionContext,
    GraspSource,
    Gripper,
    IKSolver,
)
from mj_manipulator.cartesian import (
    CartesianControlConfig,
    MoveUntilTouchResult,
    TwistExecutionResult,
    TwistStepResult,
    get_ee_jacobian,
    step_twist,
    twist_to_joint_velocity,
)
from mj_manipulator.collision import CollisionChecker
from mj_manipulator.executor import KinematicExecutor, PhysicsExecutor
from mj_manipulator.grasp_manager import GraspManager, detect_grasped_object
from mj_manipulator.grippers import FrankaGripper, RobotiqGripper
from mj_manipulator.physics_controller import ArmPhysicsExecutor, PhysicsController
from mj_manipulator.sim_context import SimArmController, SimContext
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
    # Trajectory
    "Trajectory",
    "create_linear_trajectory",
    # Planning
    "PlanResult",
    # Grasp management
    "GraspManager",
    "detect_grasped_object",
    # Grippers
    "RobotiqGripper",
    "FrankaGripper",
    # Collision checking
    "CollisionChecker",
    # Executors
    "KinematicExecutor",
    "PhysicsExecutor",
    # Physics controller (simulation)
    "PhysicsController",
    "ArmPhysicsExecutor",
    # Simulation context
    "SimContext",
    "SimArmController",
    # Cartesian control
    "CartesianControlConfig",
    "TwistStepResult",
    "MoveUntilTouchResult",
    "TwistExecutionResult",
    "get_ee_jacobian",
    "twist_to_joint_velocity",
    "step_twist",
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
