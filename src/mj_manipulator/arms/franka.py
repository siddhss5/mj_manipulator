"""Franka Emika Panda arm definition with EAIK analytical IK.

Provides constants and a factory function for creating a fully configured
Arm instance for the Franka Panda. Since the Panda is 7-DOF and EAIK only
supports 6-DOF, joint 5 is discretized and 6-DOF IK is solved for each value.

The menagerie Franka model has no EE site. Use add_franka_ee_site() with MjSpec
before creating the Environment, or pass the ee_site name if your model has one.

Usage:
    import mujoco
    from mj_environment import Environment
    from mj_manipulator.arms.franka import create_franka_arm, add_franka_ee_site

    spec = mujoco.MjSpec.from_file("path/to/franka/scene.xml")
    add_franka_ee_site(spec)
    model = spec.compile()
    # ... create Environment from model, or save XML and load ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

from mj_manipulator.arm import Arm
from mj_manipulator.arms.eaik_solver import MuJoCoEAIKSolver
from mj_manipulator.config import ArmConfig, KinematicLimits

if TYPE_CHECKING:
    from mj_environment import Environment

    from mj_manipulator.grasp_manager import GraspManager
    from mj_manipulator.protocols import Gripper

# ---------------------------------------------------------------------------
# Franka constants
# ---------------------------------------------------------------------------

FRANKA_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]

FRANKA_HOME = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

# From Franka documentation, halved for conservative planning
FRANKA_VELOCITY_LIMITS = (
    np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]) * 0.5
)
FRANKA_ACCELERATION_LIMITS = (
    np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0]) * 0.5
)

# Joint 5 (index 4) is the only joint whose locking yields an EAIK
# decomposition with exact analytical solutions (SPHERICAL_SECOND_TWO_PARALLEL).
_FRANKA_LOCKED_JOINT_INDEX = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def add_franka_ee_site(
    spec: mujoco.MjSpec,
    site_name: str = "ee_site",
    pos: list[float] | None = None,
) -> None:
    """Add an end-effector site to the Franka hand body in an MjSpec.

    Call this before compiling the spec if the Franka model doesn't have
    an EE site (the menagerie model doesn't include one).

    Args:
        spec: MjSpec loaded from a Franka scene XML.
        site_name: Name for the new site.
        pos: Position relative to hand body. Defaults to [0, 0, 0.1034]
             (roughly at fingertip center).
    """
    if pos is None:
        pos = [0.0, 0.0, 0.1034]
    hand = spec.worldbody.find_child("hand")
    site = hand.add_site()
    site.name = site_name
    site.pos = pos


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_franka_arm(
    env: Environment,
    *,
    ee_site: str = "ee_site",
    with_ik: bool = True,
    n_discretizations: int = 16,
    tcp_offset: np.ndarray | None = None,
    gripper: Gripper | None = None,
    grasp_manager: GraspManager | None = None,
) -> Arm:
    """Create a fully configured Arm for the Franka Panda.

    Args:
        env: MuJoCo environment containing a Franka model with an EE site.
        ee_site: Name of the end-effector site in the model.
        with_ik: If True, configure EAIK IK solver with joint-5 discretization.
        n_discretizations: Number of joint-5 values to sample for IK.
        tcp_offset: Optional 4x4 transform from ee_site to tool center point.
        gripper: Optional gripper implementation (e.g., FrankaGripper).
        grasp_manager: Optional grasp state tracker.

    Returns:
        Arm instance with IK solver, planning, and state queries ready to use.
    """
    config = ArmConfig(
        name="franka",
        entity_type="arm",
        joint_names=list(FRANKA_JOINT_NAMES),
        kinematic_limits=KinematicLimits(
            velocity=FRANKA_VELOCITY_LIMITS.copy(),
            acceleration=FRANKA_ACCELERATION_LIMITS.copy(),
        ),
        ee_site=ee_site,
        tcp_offset=tcp_offset,
    )

    ik_solver = None
    if with_ik:
        # Create Arm first to resolve joint indices
        arm = Arm(env, config)
        joint_limits = arm.get_joint_limits()

        # Base body = parent of first joint's body
        first_joint_body = env.model.jnt_bodyid[arm.joint_ids[0]]
        base_body_id = env.model.body_parentid[first_joint_body]

        ik_solver = MuJoCoEAIKSolver(
            model=env.model,
            data=env.data,
            joint_ids=list(arm.joint_ids),
            joint_qpos_indices=arm.joint_qpos_indices,
            ee_site_id=arm.ee_site_id,
            base_body_id=base_body_id,
            joint_limits=joint_limits,
            fixed_joint_index=_FRANKA_LOCKED_JOINT_INDEX,
            n_discretizations=n_discretizations,
        )

        return Arm(env, config, ik_solver=ik_solver, gripper=gripper, grasp_manager=grasp_manager)

    return Arm(env, config, gripper=gripper, grasp_manager=grasp_manager)
