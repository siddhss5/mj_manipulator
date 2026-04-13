# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Pre-built behavior tree subtrees for manipulation.

Two functions: :func:`pickup` and :func:`place`. Each builds a flat
py_trees Sequence of leaf nodes that does one complete task. No
recovery — if any step fails, the sequence returns FAILURE and the
primitives layer handles cleanup.

Usage::

    from mj_manipulator.bt import pickup, place

    tree = pickup("/right")
    tree = place("/left")

See ``docs/behavior-trees.md`` for the full composition guide.
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
)


def pickup(ns: str, *, with_lift: bool = True) -> py_trees.composites.Sequence:
    """Find grasps, plan, move, and close the gripper.

    Builds a flat sequence::

        Pick up object
         ├── Find grasps for object
         ├── Plan arm path
         ├── Smooth trajectory
         ├── Move arm
         ├── Close gripper
         └── Lift arm off table     (only if with_lift=True)

    Requires on blackboard:
        ``{ns}/arm``, ``{ns}/arm_name``, ``{ns}/timeout``,
        ``{ns}/object_name``, ``{ns}/grasp_source``,
        ``{ns}/hand_type``, ``/context``

    Args:
        ns: Blackboard namespace for this arm (e.g. "/left", "/right").
        with_lift: If True (default), append a 15cm upward retraction
            after the grasp. Set to False for arms on a linear base
            (e.g. geodude's Vention) where the base handles clearance.
    """
    children: list[py_trees.behaviour.Behaviour] = [
        GenerateGrasps(ns=ns, name="Find grasps for object"),
        PlanToTSRs(ns=ns, tsrs_key="grasp_tsrs", name="Plan arm path"),
        Retime(ns=ns, name="Smooth trajectory"),
        Execute(ns=ns, name="Move arm"),
        Grasp(ns=ns, name="Close gripper"),
    ]

    if with_lift:
        children.append(
            SafeRetract(
                ns=ns,
                name="Lift arm off table",
                twist=np.array([0.0, 0.0, 0.10, 0.0, 0.0, 0.0]),
                distance=0.15,
            ),
        )

    return py_trees.composites.Sequence(
        name="Pick up object",
        memory=True,
        children=children,
    )


def place(ns: str) -> py_trees.composites.Sequence:
    """Find placements, plan, move, and open the gripper.

    Builds a flat sequence::

        Place object
         ├── Find placement poses
         ├── Plan arm path
         ├── Smooth trajectory
         ├── Move arm
         └── Open gripper

    Requires on blackboard:
        ``{ns}/arm``, ``{ns}/arm_name``, ``{ns}/timeout``,
        ``{ns}/destination``, ``{ns}/object_name``,
        ``{ns}/grasp_source``, ``/context``
    """
    return py_trees.composites.Sequence(
        name="Place object",
        memory=True,
        children=[
            GeneratePlaceTSRs(ns=ns, name="Find placement poses"),
            PlanToTSRs(ns=ns, tsrs_key="place_tsrs", name="Plan arm path"),
            Retime(ns=ns, name="Smooth trajectory"),
            Execute(ns=ns, name="Move arm"),
            Release(ns=ns, name="Open gripper"),
        ],
    )
