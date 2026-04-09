# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Behavior tree nodes and subtree builders for manipulation.

Provides py_trees leaf nodes wrapping mj_manipulator operations, plus
convenience functions for common subtree patterns (pickup, place, recover).

All nodes use blackboard namespaces (``ns``) for multi-arm support::

    from mj_manipulator.bt import pickup_with_recovery

    tree = pickup_with_recovery("/right")

Requires: ``pip install mj_manipulator[bt]`` (py_trees >= 2.2)
"""

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
from mj_manipulator.bt.subtrees import (
    full_pickup,
    full_place,
    pickup,
    pickup_with_recovery,
    place,
    place_with_recovery,
    plan_and_execute,
    recover,
)

__all__ = [
    # Leaf nodes
    "PlanToTSRs",
    "PlanToConfig",
    "Retime",
    "Execute",
    "Grasp",
    "Release",
    "CartesianMove",
    "SafeRetract",
    "CheckNotNearConfig",
    "Sync",
    "GenerateGrasps",
    "GeneratePlaceTSRs",
    # Subtree builders
    "plan_and_execute",
    "pickup",
    "pickup_with_recovery",
    "place",
    "place_with_recovery",
    "recover",
    "full_pickup",
    "full_place",
]
