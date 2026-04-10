# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

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
FRANKA_VELOCITY_LIMITS = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]) * 0.5
FRANKA_ACCELERATION_LIMITS = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0]) * 0.5

# Joint 5 (index 4) is the only joint whose locking yields a known EAIK
# decomposition (SPHERICAL_SECOND_TWO_PARALLEL). Determined via:
#   find_locked_joint_index(H, P)  →  4
# For a new arm, call find_locked_joint_index() to discover the right index.
_FRANKA_LOCKED_JOINT_INDEX = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fix_franka_grip_force(model: mujoco.MjModel, target_force: float = 140.0) -> None:
    """Scale Franka gripper actuator to match real hardware grip force.

    The menagerie model's actuator8 under-produces grip force. The real
    Franka hand grips at 140N (70N per finger). We scale both gain and
    bias proportionally so the actuator reaches target_force at full close
    while ctrl=255 can still hold the fingers open.

    Args:
        model: Compiled MjModel (modified in place).
        target_force: Desired grip force at full close [N]. Default 140N
            from the Franka Emika datasheet.
    """
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
    if aid < 0:
        return

    # Actuator force model: force = gain * ctrl + bias[1] * length + bias[2] * velocity
    # To grip at target_force when ctrl=0 and grasping a typical object:
    #   target_force = |bias[1]| * length_grasp
    # To hold open when ctrl=255:
    #   gain * 255 = |bias[1]| * length_open  (zero net force)
    length_open = 0.08  # 2 × finger_joint open position (0.04m each)
    length_grasp = 0.066  # typical grasp (e.g., soda can r=33mm)

    new_bias1 = -target_force / length_grasp
    new_gain = abs(new_bias1) * length_open / 255.0

    # Scale damping proportionally to bias spring
    old_bias1 = model.actuator_biasprm[aid, 1]
    scale = new_bias1 / old_bias1 if old_bias1 != 0 else 1.0

    model.actuator_biasprm[aid, 1] = new_bias1
    model.actuator_biasprm[aid, 2] *= scale
    model.actuator_gainprm[aid, 0] = new_gain


def add_franka_ee_site(
    spec: mujoco.MjSpec,
    site_name: str = "grasp_site",
    pos: list[float] | None = None,
) -> None:
    """Add a grasp_site to the Franka hand body in an MjSpec.

    The site is placed at the canonical TSR EE frame (z=approach toward
    fingertips, y=finger-opening). The Franka hand frame already has this
    orientation, so no rotation is needed — only a position offset.

    The default position [0, 0, 0.0584] is the finger-joint origin (palm),
    matching the TSR convention used by FrankaHand: EE at the palm so that
    finger_length (44.5 mm) gives the correct fingertip standoff.

    Call this before compiling the spec if the Franka model doesn't have
    an EE site (the menagerie model doesn't include one).

    Args:
        spec: MjSpec loaded from a Franka scene XML.
        site_name: Name for the new site (default: "grasp_site").
        pos: Position relative to hand body. Defaults to [0, 0, 0.0584]
             (finger-joint origin = palm, 44.5 mm from fingertip contact).
    """
    if pos is None:
        pos = [0.0, 0.0, 0.0584]
    hand = spec.worldbody.find_child("hand")
    site = hand.add_site()
    site.name = site_name
    site.pos = pos


def add_franka_gravcomp(spec: mujoco.MjSpec) -> None:
    """Enable gravity compensation on every Franka body in an MjSpec.

    Must be called **before** ``spec.compile()``. MuJoCo optimizes gravcomp
    away at compile time if every body has ``gravcomp=0``; runtime changes
    to ``model.body_gravcomp`` are silently ignored.

    The menagerie Franka model ships without gravcomp. Real Franka FCI runs
    gravity compensation internally at 1 kHz, so enabling it in sim matches
    hardware behavior — otherwise the PD loop must fight gravity via
    steady-state position error, producing sag at rest and tracking lag in
    motion. Call this on every Franka MjSpec loaded from the menagerie,
    analogous to ``add_franka_ee_site``.

    Args:
        spec: MjSpec loaded from a Franka scene XML.
    """
    _FRANKA_BODIES = [
        "link0", "link1", "link2", "link3", "link4", "link5", "link6",
        "link7", "hand", "left_finger", "right_finger",
    ]
    for name in _FRANKA_BODIES:
        body = spec.body(name)
        if body is not None:
            body.gravcomp = 1.0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_franka_arm(
    env: Environment,
    *,
    ee_site: str = "grasp_site",
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
