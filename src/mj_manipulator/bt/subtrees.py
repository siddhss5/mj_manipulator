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
    Execute,
    Grasp,
    PlanToConfig,
    PlanToTSRs,
    Release,
    Retime,
    Sync,
)


def plan_and_execute(ns: str) -> py_trees.composites.Sequence:
    """Plan to TSRs, retime, execute.

    Requires on blackboard: ``{ns}/arm``, ``{ns}/tsrs``, ``{ns}/timeout``
    """
    return py_trees.composites.Sequence(
        name="plan_and_execute",
        memory=True,
        children=[
            PlanToTSRs(ns=ns),
            Retime(ns=ns),
            Execute(ns=ns),
        ],
    )


def pickup(ns: str) -> py_trees.composites.Sequence:
    """Full pickup: plan → execute → grasp → lift.

    Requires on blackboard:
        ``{ns}/arm``, ``{ns}/tsrs``, ``{ns}/timeout``,
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
        variable_value=0.05,
        overwrite=True,
    )

    return py_trees.composites.Sequence(
        name="pickup",
        memory=True,
        children=[
            plan_and_execute(ns),
            Sync(ns=ns),
            Grasp(ns=ns),
            Sync(ns=ns),
            set_twist,
            set_distance,
            CartesianMove(ns=ns),
            Sync(ns=ns),
        ],
    )


def recover(ns: str) -> py_trees.composites.Sequence:
    """Recovery: release → retract up → plan home → execute.

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

    return py_trees.composites.Sequence(
        name="recover",
        memory=True,
        children=[
            Release(ns=ns),
            set_retract_twist,
            set_retract_distance,
            CartesianMove(ns=ns),
            PlanToConfig(ns=ns),
            Retime(ns=ns),
            Execute(ns=ns),
            Sync(ns=ns),
        ],
    )


def pickup_with_recovery(ns: str) -> py_trees.composites.Selector:
    """Pickup with fallback recovery on failure.

    If pickup fails at any stage, releases, retracts, and returns home.
    """
    return py_trees.composites.Selector(
        name="pickup_or_recover",
        memory=True,
        children=[
            pickup(ns),
            recover(ns),
        ],
    )


def place(ns: str) -> py_trees.composites.Sequence:
    """Place: plan to place TSRs → execute → release.

    Requires: ``{ns}/arm``, ``{ns}/tsrs``, ``{ns}/timeout``,
    ``{ns}/arm_name``, ``/context``
    """
    return py_trees.composites.Sequence(
        name="place",
        memory=True,
        children=[
            plan_and_execute(ns),
            Sync(ns=ns),
            Release(ns=ns),
            Sync(ns=ns),
        ],
    )


def place_with_recovery(ns: str) -> py_trees.composites.Selector:
    """Place with fallback recovery on failure.

    If place planning/execution fails, releases, retracts, returns home.
    """
    return py_trees.composites.Selector(
        name="place_or_recover",
        memory=True,
        children=[
            place(ns),
            recover(ns),
        ],
    )
