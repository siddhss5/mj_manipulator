# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Shared logic for resolving IK solvers in arm factories.

Handles the ``with_ik`` parameter (``"auto"`` / ``"eaik"`` / ``"mink"``
/ ``"none"`` / bool) and the EAIK → mink fallback when EAIK has no
known decomposition for the arm's kinematics.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from mj_manipulator.arm import Arm
    from mj_manipulator.protocols import IKSolver

logger = logging.getLogger(__name__)

# The union type accepted by all arm factories.
IKMode = Literal["auto", "eaik", "mink", "none"] | bool


def resolve_ik_solver(
    arm: "Arm",
    with_ik: IKMode,
    *,
    fixed_joint_index: int | None = None,
    n_discretizations: int = 16,
) -> "IKSolver | None":
    """Build the right IK solver for an arm based on ``with_ik``.

    The EE site name (for mink's FrameTask) is resolved automatically
    from ``arm.ee_site_id`` — no configuration needed.

    Args:
        arm: A bare Arm (no IK yet) — used to read joint indices/limits.
        with_ik: ``"auto"`` (default), ``"eaik"``, ``"mink"``, ``"none"``,
            or bool for backward compat (``True`` → ``"auto"``).
        fixed_joint_index: For EAIK on 7-DOF arms, which joint to lock.
        n_discretizations: Discretization count for EAIK on 7-DOF arms.

    Returns:
        An IKSolver, or None if ``with_ik="none"`` or both solvers fail.
    """
    # Normalize bool → string.
    if with_ik is True:
        with_ik = "auto"
    elif with_ik is False:
        with_ik = "none"

    if with_ik == "none":
        return None

    if with_ik == "eaik":
        return _make_eaik(arm, fixed_joint_index, n_discretizations)

    if with_ik == "mink":
        return _make_mink(arm)

    # "auto": try EAIK, fall back to mink.
    eaik = _try_eaik(arm, fixed_joint_index, n_discretizations)
    if eaik is not None:
        return eaik

    logger.info(
        "EAIK has no known decomposition for '%s'; falling back to mink numerical IK.",
        arm.config.name,
    )
    return _make_mink(arm)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _make_eaik(arm, fixed_joint_index, n_discretizations):
    from mj_manipulator.arms.eaik_solver import MuJoCoEAIKSolver

    env = arm.env
    joint_limits = arm.get_joint_limits()
    first_joint_body = env.model.jnt_bodyid[arm.joint_ids[0]]
    base_body_id = int(env.model.body_parentid[first_joint_body])

    kwargs = dict(
        model=env.model,
        data=env.data,
        joint_ids=list(arm.joint_ids),
        joint_qpos_indices=arm.joint_qpos_indices,
        ee_site_id=arm.ee_site_id,
        base_body_id=base_body_id,
        joint_limits=joint_limits,
    )
    if fixed_joint_index is not None:
        kwargs["fixed_joint_index"] = fixed_joint_index
        kwargs["n_discretizations"] = n_discretizations

    return MuJoCoEAIKSolver(**kwargs)


def _try_eaik(arm, fixed_joint_index, n_discretizations):
    """Try EAIK — return solver if it has a known decomposition, else None."""
    try:
        solver = _make_eaik(arm, fixed_joint_index, n_discretizations)
    except Exception as e:
        logger.debug("EAIK construction failed for '%s': %s", arm.config.name, e)
        return None

    # For 6-DOF (no fixed joint): check if EAIK recognizes the kinematics.
    if fixed_joint_index is None and solver.robot is not None:
        if not solver.robot.hasKnownDecomposition():
            logger.debug(
                "EAIK has no known decomposition for '%s' (family: %s).",
                arm.config.name,
                solver.robot.getKinematicFamily(),
            )
            return None

    return solver


def _make_mink(arm):
    from mj_manipulator.arms.mink_solver import make_mink_solver

    return make_mink_solver(arm)
