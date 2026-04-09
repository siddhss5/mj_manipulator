# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Pre-built subtree patterns for common manipulation tasks.

Each function returns a py_trees composite that can be used standalone
or composed into larger trees. All subtrees use namespaced blackboard
keys for multi-arm support.

Usage::

    from mj_manipulator.bt import pickup_with_recovery

    tree = pickup_with_recovery("/ur5e")
"""

from __future__ import annotations

import numpy as np
import py_trees

from mj_manipulator.bt.nodes import (
    CartesianMove,
    CheckNotNearConfig,
    Execute,
    GenerateGrasps,
    GeneratePlaceTSRs,
    Grasp,
    PlanToConfig,
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


def pickup(ns: str) -> py_trees.composites.Sequence:
    """Full pickup: plan → execute → grasp → lift.

    Requires on blackboard:
        ``{ns}/arm``, ``{ns}/grasp_tsrs``, ``{ns}/timeout``,
        ``{ns}/arm_name``, ``{ns}/object_name``

    Sets ``{ns}/twist`` and ``{ns}/distance`` for the lift.
    """
    # SetBlackboardVariable for lift parameters
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

    return py_trees.composites.Sequence(
        name="pickup",
        memory=True,
        children=[
            plan_and_execute(ns, tsrs_key="grasp_tsrs"),
            Sync(ns=ns),
            Grasp(ns=ns),
            Sync(ns=ns),
            set_twist,
            set_distance,
            SafeRetract(ns=ns),
            Sync(ns=ns),
        ],
    )


def recover(ns: str) -> py_trees.composites.Sequence:
    """Recovery: release → retract up (if needed) → plan home → execute.

    Skips the retract if the arm is already near home — avoids the
    visible "shake" when recovery runs on an arm that never moved.

    Requires: ``{ns}/arm``, ``{ns}/arm_name``, ``{ns}/goal_config``,
    ``{ns}/timeout``, ``/context``
    """
    set_retract_twist = py_trees.behaviours.SetBlackboardVariable(
        name="set_retract_twist",
        variable_name=f"{ns}/twist",
        variable_value=np.array([0.0, 0.0, 0.10, 0.0, 0.0, 0.0]),
        overwrite=True,
    )
    set_retract_distance = py_trees.behaviours.SetBlackboardVariable(
        name="set_retract_distance",
        variable_name=f"{ns}/distance",
        variable_value=0.10,
        overwrite=True,
    )

    # Only retract if arm has moved away from home
    guarded_retract = py_trees.composites.Sequence(
        name="retract_if_needed",
        memory=True,
        children=[
            CheckNotNearConfig(ns=ns),
            set_retract_twist,
            set_retract_distance,
            CartesianMove(ns=ns),
        ],
    )

    return py_trees.composites.Sequence(
        name="recover",
        memory=True,
        children=[
            Release(ns=ns),
            py_trees.decorators.FailureIsSuccess(
                name="optional_retract",
                child=guarded_retract,
            ),
            PlanToConfig(ns=ns),
            Retime(ns=ns),
            Execute(ns=ns),
            Sync(ns=ns),
        ],
    )


def pickup_with_recovery(ns: str) -> py_trees.composites.Selector:
    """Pickup with fallback recovery on failure.

    If pickup fails at any stage, releases, retracts, and returns home.
    The Selector still returns FAILURE because recovery wraps with
    SuccessIsFailure — cleanup succeeded but the task did not.
    """
    return py_trees.composites.Selector(
        name="pickup_or_recover",
        memory=True,
        children=[
            pickup(ns),
            py_trees.decorators.SuccessIsFailure(
                name="recover_then_fail",
                child=recover(ns),
            ),
        ],
    )


def recover_keep_grasp(ns: str) -> py_trees.composites.Sequence:
    """Recovery without releasing: retract up (if needed) → plan home → execute.

    Same as recover() but keeps the grasp. Used for place failures where
    dropping the object is worse than keeping it.
    """
    set_retract_twist = py_trees.behaviours.SetBlackboardVariable(
        name="set_retract_twist",
        variable_name=f"{ns}/twist",
        variable_value=np.array([0.0, 0.0, 0.10, 0.0, 0.0, 0.0]),
        overwrite=True,
    )
    set_retract_distance = py_trees.behaviours.SetBlackboardVariable(
        name="set_retract_distance",
        variable_name=f"{ns}/distance",
        variable_value=0.10,
        overwrite=True,
    )

    guarded_retract = py_trees.composites.Sequence(
        name="retract_if_needed",
        memory=True,
        children=[
            CheckNotNearConfig(ns=ns),
            set_retract_twist,
            set_retract_distance,
            CartesianMove(ns=ns),
        ],
    )

    return py_trees.composites.Sequence(
        name="recover_keep_grasp",
        memory=True,
        children=[
            py_trees.decorators.FailureIsSuccess(
                name="optional_retract",
                child=guarded_retract,
            ),
            PlanToConfig(ns=ns),
            Retime(ns=ns),
            Execute(ns=ns),
            Sync(ns=ns),
        ],
    )


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


def place_with_recovery(ns: str) -> py_trees.composites.Selector:
    """Place with fallback recovery on failure.

    If place planning/execution fails, keeps the grasp and returns home.
    Does NOT release the object — the caller can retry or release manually.
    """
    return py_trees.composites.Selector(
        name="place_or_recover",
        memory=True,
        children=[
            place(ns),
            py_trees.decorators.SuccessIsFailure(
                name="recover_then_fail",
                child=recover_keep_grasp(ns),
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Full primitives with TSR generation (requires GraspSource on blackboard)
# ---------------------------------------------------------------------------


def full_pickup(ns: str) -> py_trees.composites.Sequence:
    """Generate grasp TSRs then pickup with recovery.

    Requires on blackboard:
        ``{ns}/grasp_source``, ``{ns}/hand_type``, ``{ns}/object_name``,
        ``{ns}/arm``, ``{ns}/arm_name``, ``{ns}/timeout``, ``/context``
    """
    return py_trees.composites.Sequence(
        name="full_pickup",
        memory=True,
        children=[
            GenerateGrasps(ns=ns),
            pickup_with_recovery(ns),
        ],
    )


def full_place(ns: str) -> py_trees.composites.Sequence:
    """Generate placement TSRs then place with recovery.

    Requires on blackboard:
        ``{ns}/grasp_source``, ``{ns}/destination``, ``{ns}/object_name``,
        ``{ns}/arm``, ``{ns}/arm_name``, ``{ns}/timeout``, ``/context``
    """
    return py_trees.composites.Sequence(
        name="full_place",
        memory=True,
        children=[
            GeneratePlaceTSRs(ns=ns),
            place_with_recovery(ns),
        ],
    )
