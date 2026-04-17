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
# Franka constants
# ---------------------------------------------------------------------------

FRANKA_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]

FRANKA_HOME = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

# From libfranka's operational limits; halved for conservative planning.
# These are the stricter runtime limits the controller enforces (below
# the datasheet peak of 150 °/s = 2.618 rad/s for J1-J4).
# Source: https://github.com/frankaemika/libfranka include/franka/rate_limiting.h
#   kMaxJointVelocity     = {2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61}  rad/s
#   kMaxJointAcceleration = {15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0}       rad/s²
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


def fix_franka_grip_force(model: mujoco.MjModel, target_force: float = 70.0) -> None:
    """Replace Franka gripper actuator with a **constant-force** grip.

    The menagerie model ships actuator8 as a *position-spring*
    (``bias = bias[0] + bias[1] * length``), which means grip force
    scales linearly with finger gap: full at 66 mm, halved at 33 mm,
    tiny when fingers are almost closed. That's the opposite of how
    the real Franka hand behaves (constant force regardless of gap)
    and the reason cans slip out during transport — any inertial
    perturbation that closes the fingers slightly *reduces* grip, so
    the slip feeds on itself until the object falls.

    This helper rewrites the actuator so that:

    - ``ctrl = 0``   → net force = ``-target_force``  (closing, constant)
    - ``ctrl = 255`` → net force = ``+target_force``  (opening, constant)
    - Force is independent of gap (``biasprm[1]`` set to 0)

    The menagerie's default ``forcerange`` of ``[-100, 100]`` N is
    left untouched; target_force should stay within it (70 N default
    leaves plenty of headroom).

    Args:
        model: Compiled MjModel (modified in place).
        target_force: Total grip force at full close [N]. Default 70 N
            (the lower end of the Franka Emika continuous-grip spec,
            35 N per finger). The grip-force sweep in tests/grip_sweep.py
            showed 50–140 N all achieve 6/6 on 3-can recycling; 70 N is
            a comfortable middle.

    See ``mj_manipulator/docs/grippers.md`` §2 for the full derivation
    of the actuator-rewrite recipe; this function is the Franka-specific
    sibling of :func:`mj_manipulator.grippers.robotiq.fix_robotiq_grip_force`.
    """
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
    if aid < 0:
        return

    limit = float(model.actuator_forcerange[aid, 1])
    if target_force > limit:
        logger.warning(
            "fix_franka_grip_force: target_force=%.1fN exceeds the "
            "menagerie actuator's forcerange (%.1fN). The actuator will "
            "clamp to %.1fN at full close.",
            target_force,
            limit,
            limit,
        )

    # Affine actuator force model:
    #   force = gain[0] * ctrl + bias[0] + bias[1] * length + bias[2] * vel
    #
    # For constant force, we set bias[1] = 0. We want a linear map
    # from ctrl ∈ [0, 255] to force ∈ [-target, +target]:
    #   force(ctrl) = gain * ctrl + bias[0]
    #   force(0)   = bias[0]          = -target   → bias[0] = -target
    #   force(255) = 255*gain + bias[0] = +target → gain = 2*target / 255
    new_bias0 = -target_force
    new_gain = 2.0 * target_force / 255.0

    # Keep damping (bias[2]) proportional to the grip force change — it
    # stabilizes the finger motion against its own inertia. Scale from
    # the old bias[1] magnitude to preserve the damper's ratio.
    old_bias1 = model.actuator_biasprm[aid, 1]
    old_bias2 = model.actuator_biasprm[aid, 2]
    damping_scale = abs(new_bias0 / old_bias1) if old_bias1 != 0 else 1.0

    model.actuator_biasprm[aid, 0] = new_bias0
    model.actuator_biasprm[aid, 1] = 0.0  # kill position coupling
    model.actuator_biasprm[aid, 2] = old_bias2 * damping_scale
    model.actuator_gainprm[aid, 0] = new_gain


def add_franka_ee_site(
    spec: mujoco.MjSpec,
    site_name: str = "grasp_site",
    pos: list[float] | None = None,
) -> None:
    """Add a grasp_site to the Franka hand body in an MjSpec.

    The site is placed at the canonical TSR EE frame (z=approach toward
    fingertips, y=finger-opening). The Franka hand frame already has
    this orientation, so no rotation is needed — only a position offset.

    The default position is ``[0, 0, FrankaHand.PALM_OFFSET_FROM_HAND]``
    = ``[0, 0, 0.0753]`` — the forward edge of the ``hand`` body's
    collision mesh (the metal collar around the finger mounts). That
    matches the TSR convention: EE at the palm where the finger
    mechanism attaches, nothing except fingers forward of it.

    Before this was aligned with the 2F-85 convention (#129), the
    default was ``[0, 0, 0.0584]`` (the finger-joint origin), which
    buried the grasp_site inside the 17 mm-deep collar and caused
    deep-grasp TSR templates to drive the collar into the object. See
    ``docs/grippers.md`` §1 for the palm–housing story.

    Call this before compiling the spec if the Franka model doesn't
    have an EE site (the menagerie model doesn't include one).

    Args:
        spec: MjSpec loaded from a Franka scene XML.
        site_name: Name for the new site (default: ``"grasp_site"``).
        pos: Position relative to hand body. Defaults to the new
            palm convention ``[0, 0, 0.0753]`` (collar forward edge).
    """
    if pos is None:
        from tsr.hands import FrankaHand

        pos = [0.0, 0.0, FrankaHand.PALM_OFFSET_FROM_HAND]
    hand = spec.worldbody.find_child("hand")
    site = hand.add_site()
    site.name = site_name
    site.pos = pos


def add_franka_pad_friction(
    spec: mujoco.MjSpec,
    *,
    sliding_friction: float = 1.5,
    torsional_friction: float = 0.05,
    rolling_friction: float = 0.0002,
    solref: tuple[float, float] = (0.01, 1.0),
    solimp: tuple[float, float, float, float, float] = (0.9, 0.95, 0.001, 0.5, 2.0),
) -> None:
    """Boost fingertip-pad grip to mimic compliant silicone contact.

    Must be called **before** ``spec.compile()``.

    The real Franka hand has **moulded silicone pads** on the fingertips: a
    ~12 × 22 mm flat face with a shallow cylindrical groove, ~3 mm thick,
    that deforms a couple of millimeters under 70 N of normal force. The
    compliance turns line contact against a cylinder into a strip contact
    ~8 mm wide, dramatically increasing grip against transverse load.

    The menagerie Panda model approximates each pad with **five small
    rigid boxes** and no friction override. Against a can the contact
    area collapses to almost a point, the silicone grip is lost, and
    even a modest acceleration during the lift slides the object out of
    the gripper. This helper applies two compensations to the pad
    collision geoms:

    1. **High priority friction**: ``friction=(sliding, torsional,
       rolling)`` with ``priority=1`` so the pad wins over the held
       object's friction (normally MuJoCo takes per-parameter max).
       Default ``sliding=1.5`` is within the physically plausible
       range for silicone-on-aluminum. Bumping higher (tried 3.0–4.0
       in the grip sweep) was counterproductive — rigid-contact
       friction spikes seem to eject compliant objects during grasp
       close.

    2. **Soft contact**: smaller ``solref[0]`` (contact time constant)
       and tighter ``solimp`` (constraint impedance). This lets the
       constraint solver produce a slight penetration (~1 mm) that
       visually matches a deformed silicone pad and enlarges the
       effective contact area.

    Together these mimic silicone compliance and grip without changing
    the pad geometry, at the cost of modeling rigid-body contact that
    behaves "as if" it were compliant. This is the standard sim
    tradeoff for parallel-jaw grasping tasks — adding compliant-contact
    primitives to MuJoCo is out of scope.

    Args:
        spec: MjSpec loaded from a Franka scene XML.
        sliding_friction: Coulomb friction coefficient (1st friction
            parameter). Default 1.5 (silicone-on-aluminum range).
        torsional_friction: Rotational friction about contact normal
            (2nd friction parameter). MuJoCo default 0.005 is too low;
            default 0.05 here resists in-hand twist during transport.
        rolling_friction: Resistance to rolling (3rd friction parameter).
            MuJoCo default 0.0001 is fine but bumped slightly.
        solref: ``(timeconst, dampratio)`` contact solver reference.
            Smaller timeconst = softer contact, more penetration.
            Default 0.01 s gives ~1 mm of "squish" against a can.
        solimp: ``(dmin, dmax, width, midpoint, power)`` contact solver
            impedance. Tighter ``dmin``/``dmax`` produce stiffer contact;
            ``width`` controls how quickly the solver transitions between
            them. MuJoCo defaults are ``(0.9, 0.95, 0.001, 0.5, 2.0)``.
    """
    friction = [sliding_friction, torsional_friction, rolling_friction]
    solref_list = list(solref)
    solimp_list = list(solimp)

    for finger_name in ("left_finger", "right_finger"):
        body = spec.body(finger_name)
        if body is None:
            continue
        for geom in body.geoms:
            # Only touch the pad collision boxes — skip visual meshes and
            # the larger finger collision mesh. The pads are the five
            # small boxes defined via fingertip_pad_collision_* classes.
            if geom.type != mujoco.mjtGeom.mjGEOM_BOX:
                continue
            if geom.contype == 0:  # visual group
                continue
            geom.friction = friction
            geom.solref = solref_list
            geom.solimp = solimp_list
            geom.priority = 1


def add_franka_finger_exclude(spec: mujoco.MjSpec) -> None:
    """Exclude the ``left_finger`` ↔ ``right_finger`` contact pair.

    Must be called **before** ``spec.compile()``.

    The menagerie Franka models the two fingers as independent rigid
    bodies with their collision boxes meeting at the palm. When the
    gripper closes on nothing (empty grasp), the solver can't keep the
    fingers from interpenetrating — they push into each other by up to
    ~20 mm at rest. That persistent penetration looks like a real
    collision to the motion planner, so every plan from an
    empty-closed state fails with "start configuration in collision".

    On the real Franka, a parallel-jaw linkage prevents finger-finger
    contact entirely (mechanical hard stop before the fingertips
    touch). Excluding the pair matches that hardware behavior and
    frees the planner to treat an empty-closed gripper as a normal
    start state.
    """
    exclude = spec.add_exclude()
    exclude.bodyname1 = "left_finger"
    exclude.bodyname2 = "right_finger"


def add_franka_gravcomp(spec: mujoco.MjSpec) -> None:
    """Enable gravity compensation on every Franka body in an MjSpec.

    Must be called **before** ``spec.compile()``. Real Franka FCI runs
    gravity compensation internally at 1 kHz, so enabling it in sim
    matches hardware behavior — otherwise the PD loop must fight
    gravity via steady-state position error, producing sag at rest
    and tracking lag in motion. Call this on every Franka MjSpec
    loaded from the menagerie, analogous to ``add_franka_ee_site``.

    Delegates to :func:`mj_manipulator.arm.add_subtree_gravcomp` with
    the Franka kinematic root ``"link0"``. The subtree walker visits
    every descendant body, which for the menagerie Franka is the 7
    arm links plus ``hand``, ``left_finger``, ``right_finger`` (11
    bodies total).

    Args:
        spec: MjSpec loaded from a Franka scene XML.
    """
    from mj_manipulator.arm import add_subtree_gravcomp

    add_subtree_gravcomp(spec, "link0")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_franka_arm(
    env: Environment,
    *,
    ee_site: str = "grasp_site",
    with_ik="auto",
    n_discretizations: int = 16,
    tcp_offset: np.ndarray | None = None,
    gripper: Gripper | None = None,
    grasp_manager: GraspManager | None = None,
) -> Arm:
    """Create a fully configured Arm for the Franka Panda.

    Args:
        env: MuJoCo environment containing a Franka model with an EE site.
        ee_site: Name of the end-effector site in the model.
        with_ik: IK solver mode — ``"auto"`` (EAIK, mink fallback),
            ``"eaik"``, ``"mink"``, ``"none"``, or bool for backward
            compat (``True`` → ``"auto"``).
        n_discretizations: Number of joint-5 values to sample for EAIK.
        tcp_offset: Optional 4x4 transform from ee_site to tool center point.
        gripper: Optional gripper implementation (e.g., FrankaGripper).
        grasp_manager: Optional grasp state tracker.

    Returns:
        Arm instance with IK solver, planning, and state queries ready to use.
    """
    from mj_manipulator.arms._ik_factory import resolve_ik_solver

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

    arm = Arm(env, config)
    ik_solver = resolve_ik_solver(
        arm,
        with_ik=with_ik,
        fixed_joint_index=_FRANKA_LOCKED_JOINT_INDEX,
        n_discretizations=n_discretizations,
    )
    return Arm(env, config, ik_solver=ik_solver, gripper=gripper, grasp_manager=grasp_manager)
