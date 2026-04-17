# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""KUKA LBR iiwa 14 arm definition with EAIK analytical IK.

7-DOF arm from mujoco_menagerie's ``kuka_iiwa_14`` model. Ships as a
bare arm (no gripper); the demo in ``mj_manipulator.demos.iiwa14_setup``
attaches a Robotiq 2F-85 from menagerie at runtime via MjSpec.

Since the iiwa is 7-DOF and EAIK only supports 6-DOF, one joint is
discretized. The iiwa's joint 1 lies on the spherical base (joints 1, 2,
3 intersect), so locking it yields a
``SPHERICAL_FIRST_TWO_INTERSECTING`` family that EAIK can close-form.

Usage::

    import mujoco
    from mj_environment import Environment
    from mj_manipulator.arms.iiwa14 import create_iiwa14_arm, add_iiwa14_ee_site

    spec = mujoco.MjSpec.from_file("path/to/kuka_iiwa_14/scene.xml")
    add_iiwa14_ee_site(spec)
    env = Environment.from_model(spec.compile())
    arm = create_iiwa14_arm(env)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from mj_manipulator.arm import Arm
from mj_manipulator.config import ArmConfig, KinematicLimits

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from mj_environment import Environment

    from mj_manipulator.grasp_manager import GraspManager
    from mj_manipulator.protocols import Gripper


# ---------------------------------------------------------------------------
# iiwa 14 constants
# ---------------------------------------------------------------------------

IIWA14_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]

# Menagerie "home" keyframe: all joints zero. Slightly bent configuration
# for the demo so the arm doesn't start at a stretched-out singularity.
IIWA14_HOME = np.array([0.0, 0.3, 0.0, -1.5, 0.0, 1.2, 0.0])

# KUKA iiwa 14 R820 max axis velocities (rad/s); halved for planning.
# Source: KUKA LBR iiwa 14 R820 datasheet (J1–J7 in °/s):
#   85, 85, 100, 75, 130, 135, 135
# https://assets.robots.com/robots/KUKA/Collaborative/KUKA_LBR_IIWA_14_R820_Datasheet.pdf
IIWA14_VELOCITY_LIMITS = np.array([1.484, 1.484, 1.745, 1.309, 2.269, 2.356, 2.356]) * 0.5

# KUKA doesn't publish acceleration limits. We derive them from a
# "reach max velocity in 100 ms" rule of thumb (typical high-performance
# cobot spec). α = v_max / 0.1 s. Halved for planning, matching the
# velocity convention. The resulting numbers are within the same range
# as Franka's libfranka-published limits [15, 7.5, 10, 12.5, 15, 20, 20] —
# reassuring given iiwa 14 and Franka are similar-mass 7-DOF cobots.
IIWA14_ACCELERATION_LIMITS = np.array([14.8, 14.8, 17.4, 13.1, 22.7, 23.6, 23.6]) * 0.5

# Which joint to lock for 7-DOF EAIK decomposition. Discovered via
# ``find_locked_joint_index(H, P)`` at ``q = zeros`` — see
# ``scripts/find_iiwa14_locked_joint.py`` or the one-off shown in
# README.md "Adding a New Arm" step 2. Result: joint 0 gives
# SPHERICAL_FIRST_TWO_INTERSECTING (iiwa's joints 1, 2, 3 all meet at
# the shoulder, forming a spherical base).
_IIWA14_LOCKED_JOINT_INDEX = 0


# ---------------------------------------------------------------------------
# MjSpec helpers (must run before spec.compile())
# ---------------------------------------------------------------------------


def add_iiwa14_ee_site(
    spec: mujoco.MjSpec,
    site_name: str = "grasp_site",
    pos: list[float] | None = None,
) -> None:
    """Add an end-effector site on the iiwa 14's link7 (flange).

    The menagerie iiwa doesn't ship with a named EE site — the arm
    ends at ``link7``, the flange where a gripper or tool attaches.
    Call this before :meth:`mujoco.MjSpec.compile` to plant a named
    site at the flange that the IK solver and TSR machinery target.

    Args:
        spec: MjSpec loaded from an iiwa 14 scene XML.
        site_name: Name for the new site (default: ``"grasp_site"``).
        pos: Position relative to link7. Defaults to ``[0, 0, 0]``
            (the flange origin).
    """
    if pos is None:
        pos = [0.0, 0.0, 0.0]
    # ``spec.body(name)`` walks the tree; the iiwa's link7 is nested
    # 6 levels under worldbody so ``worldbody.find_child`` won't find it.
    link7 = spec.body("link7")
    if link7 is None:
        raise RuntimeError("add_iiwa14_ee_site: no body named 'link7' in this spec")
    site = link7.add_site()
    site.name = site_name
    site.pos = pos


def add_iiwa14_gravcomp(spec: mujoco.MjSpec) -> None:
    """Enable gravity compensation on every iiwa 14 body in an MjSpec.

    Must be called **before** ``spec.compile()``. Real iiwa controllers
    (KUKA's Sunrise.Connectivity layer) run gravity compensation
    internally, so enabling it in sim matches hardware behavior —
    the PD loop doesn't have to fight gravity via steady-state position
    error.

    Delegates to :func:`mj_manipulator.arm.add_subtree_gravcomp` with
    the iiwa's kinematic root ``"base"``.
    """
    from mj_manipulator.arm import add_subtree_gravcomp

    add_subtree_gravcomp(spec, "base")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_iiwa14_arm(
    env: Environment,
    *,
    ee_site: str = "grasp_site",
    with_ik="auto",
    tcp_offset: np.ndarray | None = None,
    gripper: Gripper | None = None,
    grasp_manager: GraspManager | None = None,
) -> Arm:
    """Create a fully configured Arm for the KUKA iiwa 14.

    Args:
        env: MuJoCo environment containing an iiwa 14 model.
        ee_site: Name of the end-effector site. Use the default when
            the arm has a bare ``grasp_site`` on link7; override when
            a gripper has been attached and brings its own grasp site.
        with_ik: IK solver mode — ``"auto"`` (EAIK, mink fallback),
            ``"eaik"``, ``"mink"``, ``"none"``, or bool for backward
            compat (``True`` → ``"auto"``).
        tcp_offset: Optional 4×4 transform from ``ee_site`` to the TCP.
        gripper: Optional gripper implementation.
        grasp_manager: Optional grasp state tracker.

    Returns:
        Arm ready for planning, execution, and state queries.
    """
    from mj_manipulator.arms._ik_factory import resolve_ik_solver

    config = ArmConfig(
        name="iiwa14",
        entity_type="arm",
        joint_names=list(IIWA14_JOINT_NAMES),
        kinematic_limits=KinematicLimits(
            velocity=IIWA14_VELOCITY_LIMITS.copy(),
            acceleration=IIWA14_ACCELERATION_LIMITS.copy(),
        ),
        ee_site=ee_site,
        tcp_offset=tcp_offset,
    )

    arm = Arm(env, config)
    ik_solver = resolve_ik_solver(
        arm,
        with_ik=with_ik,
        fixed_joint_index=_IIWA14_LOCKED_JOINT_INDEX,
    )
    return Arm(env, config, ik_solver=ik_solver, gripper=gripper, grasp_manager=grasp_manager)
