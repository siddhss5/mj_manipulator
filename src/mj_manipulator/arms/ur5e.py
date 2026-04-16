# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""UR5e arm definition with EAIK analytical IK.

Provides constants and a factory function for creating a fully configured
Arm instance for the Universal Robots UR5e.

Usage:
    from mj_environment import Environment
    from mj_manipulator.arms.ur5e import create_ur5e_arm

    env = Environment("path/to/ur5e/scene.xml")
    arm = create_ur5e_arm(env)
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
# UR5e constants
# ---------------------------------------------------------------------------

UR5E_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

UR5E_EE_SITE = "attachment_site"
# When Robotiq is attached with prefix "gripper/", use its grasp_site instead.
UR5E_ROBOTIQ_EE_SITE = "gripper/grasp_site"

UR5E_HOME = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])

# UR5e max axis velocities; halved for conservative planning.
# Source: Universal Robots UR5e tech specs — base/shoulder/elbow 180°/s,
# wrist 1/2/3 360°/s.
# https://www.universal-robots.com/manuals/EN/HTML/SW5_19/Content/prod-usr-man/complianceUR5e/H_g5_sections/appendix_g5/tech_spec_sheet.htm
UR5E_VELOCITY_LIMITS = np.array([3.14, 3.14, 3.14, 6.28, 6.28, 6.28]) * 0.5

# UR's recommended joint acceleration is ~5 rad/s² (300°/s² URScript default).
# Halved for planning. The UR5e can push higher (up to ~6 rad/s² per some
# research) but the URScript default is the operational sweet spot.
UR5E_ACCELERATION_LIMITS = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0]) * 0.5


# ---------------------------------------------------------------------------
# MjSpec helpers (must run before spec.compile())
# ---------------------------------------------------------------------------


def add_ur5e_gravcomp(spec: mujoco.MjSpec) -> None:
    """Enable gravity compensation on every UR5e body in an MjSpec.

    Must be called **before** ``spec.compile()``. Real UR5e controllers
    (URScript / RTDE / URCap) run gravity compensation internally, so
    enabling it in sim matches hardware behavior — otherwise the PD
    loop must fight gravity via steady-state position error, producing
    sag at rest and tracking lag in motion. Call this on every UR5e
    MjSpec loaded from the menagerie. The geodude_assets UR5e already
    has ``gravcomp=1`` baked into its source XML, so this helper is a
    no-op there (idempotent).

    Delegates to :func:`mj_manipulator.arm.add_subtree_gravcomp` with
    the UR5e kinematic root ``"base"``. The subtree walker visits
    every descendant body, which for the bare menagerie UR5e is the
    7 arm links (``base`` through ``wrist_3_link``). If a gripper is
    attached under ``wrist_3_link`` before this helper runs, its
    bodies will also be included, which is usually what you want.

    Args:
        spec: MjSpec loaded from a UR5e scene XML.
    """
    from mj_manipulator.arm import add_subtree_gravcomp

    add_subtree_gravcomp(spec, "base")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_ur5e_arm(
    env: Environment,
    *,
    ee_site: str = UR5E_EE_SITE,
    with_ik: bool = True,
    tcp_offset: np.ndarray | None = None,
    gripper: Gripper | None = None,
    grasp_manager: GraspManager | None = None,
) -> Arm:
    """Create a fully configured Arm for the UR5e.

    Args:
        env: MuJoCo environment containing a UR5e model.
        ee_site: Name of the end-effector site in the model.
        with_ik: If True, configure EAIK analytical IK solver.
        tcp_offset: Optional 4x4 transform from ee_site to tool center point.
        gripper: Optional gripper implementation (e.g., RobotiqGripper).
        grasp_manager: Optional grasp state tracker.

    Returns:
        Arm instance with IK solver, planning, and state queries ready to use.
    """
    config = ArmConfig(
        name="ur5e",
        entity_type="arm",
        joint_names=list(UR5E_JOINT_NAMES),
        kinematic_limits=KinematicLimits(
            velocity=UR5E_VELOCITY_LIMITS.copy(),
            acceleration=UR5E_ACCELERATION_LIMITS.copy(),
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
        )

        return Arm(env, config, ik_solver=ik_solver, gripper=gripper, grasp_manager=grasp_manager)

    return Arm(env, config, gripper=gripper, grasp_manager=grasp_manager)
