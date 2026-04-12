# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Pre-built subtree patterns for common manipulation tasks.

Each function returns a py_trees composite that can be used standalone
or composed into larger trees. All subtrees use namespaced blackboard
keys for multi-arm support.

These are **action sequences**, not pipelines with built-in recovery.
If a node fails, the sequence returns FAILURE and the primitives
layer (``robot.pickup()``, ``robot.place()``) handles cleanup via
:func:`~mj_manipulator.primitives._recover`. Users composing custom
BTs handle failure however they like.

Usage::

    from mj_manipulator.bt import pickup, full_pickup

    # Minimal: just the pickup action sequence
    tree = pickup("/right")

    # With TSR generation:
    tree = full_pickup("/right")
"""

from __future__ import annotations

import numpy as np
import py_trees

from mj_manipulator.bt.nodes import (
    Execute,
    GenerateGrasps,
    GeneratePlaceTSRs,
    Grasp,
    PlanToTSRs,
    Release,
    Retime,
    SafeRetract,
    Sync,
)


def plan_and_execute(ns: str, tsrs_key: str = "tsrs") -> py_trees.composites.Sequence:
    """Plan to TSRs, retime, execute.

    Requires on blackboard: ``{ns}/arm``, ``{ns}/{tsrs_key}``, ``{ns}/timeout``
    """
    return py_trees.composites.Sequence(
        name="plan_and_execute",
        memory=True,
        children=[
            PlanToTSRs(ns=ns, tsrs_key=tsrs_key),
            Retime(ns=ns),
            Execute(ns=ns),
        ],
    )


def pickup(ns: str, *, with_lift: bool = True) -> py_trees.composites.Sequence:
    """Full pickup: plan → execute → grasp → (optional cartesian lift).

    Requires on blackboard:
        ``{ns}/arm``, ``{ns}/grasp_tsrs``, ``{ns}/timeout``,
        ``{ns}/arm_name``, ``{ns}/object_name``

    Args:
        ns: Blackboard namespace for this arm.
        with_lift: If True (default), append a 15cm cartesian ``SafeRetract``
            after the grasp — appropriate for fixed-base arms (Franka) where
            the arm itself must clear the grasp. Set to False for arms mounted
            on a linear base (e.g. geodude's Vention gantry) where the base
            handles the post-grasp clearance and a cartesian arm lift would
            be redundant or harmful.

    Sets ``{ns}/twist`` and ``{ns}/distance`` when ``with_lift=True``.
    """
    children: list[py_trees.behaviour.Behaviour] = [
        plan_and_execute(ns, tsrs_key="grasp_tsrs"),
        Sync(ns=ns),
        Grasp(ns=ns),
        Sync(ns=ns),
    ]

    if with_lift:
        set_twist = py_trees.behaviours.SetBlackboardVariable(
            name="set_lift_twist",
            variable_name=f"{ns}/twist",
            variable_value=np.array([0.0, 0.0, 0.10, 0.0, 0.0, 0.0]),
            overwrite=True,
        )
        set_distance = py_trees.behaviours.SetBlackboardVariable(
            name="set_lift_distance",
            variable_name=f"{ns}/distance",
            variable_value=0.15,
            overwrite=True,
        )
        children.extend([set_twist, set_distance, SafeRetract(ns=ns), Sync(ns=ns)])

    return py_trees.composites.Sequence(name="pickup", memory=True, children=children)


def place(ns: str) -> py_trees.composites.Sequence:
    """Place: plan to place TSRs → execute → release.

    Requires: ``{ns}/arm``, ``{ns}/place_tsrs``, ``{ns}/timeout``,
    ``{ns}/arm_name``, ``/context``
    """
    return py_trees.composites.Sequence(
        name="place",
        memory=True,
        children=[
            plan_and_execute(ns, tsrs_key="place_tsrs"),
            Sync(ns=ns),
            Release(ns=ns),
            Sync(ns=ns),
        ],
    )


# ---------------------------------------------------------------------------
# Full primitives with TSR generation (requires GraspSource on blackboard)
# ---------------------------------------------------------------------------


def full_pickup(ns: str) -> py_trees.composites.Sequence:
    """Generate grasp TSRs then pickup.

    Requires on blackboard:
        ``{ns}/grasp_source``, ``{ns}/hand_type``, ``{ns}/object_name``,
        ``{ns}/arm``, ``{ns}/arm_name``, ``{ns}/timeout``, ``/context``
    """
    return py_trees.composites.Sequence(
        name="full_pickup",
        memory=True,
        children=[
            GenerateGrasps(ns=ns),
            pickup(ns),
        ],
    )


def full_place(ns: str) -> py_trees.composites.Sequence:
    """Generate placement TSRs then place.

    Requires on blackboard:
        ``{ns}/grasp_source``, ``{ns}/destination``, ``{ns}/object_name``,
        ``{ns}/arm``, ``{ns}/arm_name``, ``{ns}/timeout``, ``/context``
    """
    return py_trees.composites.Sequence(
        name="full_place",
        memory=True,
        children=[
            GeneratePlaceTSRs(ns=ns),
            place(ns),
        ],
    )
