# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Behavior tree nodes and subtrees for manipulation.

Two layers:

- **Nodes** (Layer 1): Pure building blocks — ``PlanToTSRs``, ``Execute``,
  ``Grasp``, ``Release``, etc. Each does one thing. Compose them into
  custom trees for full control.
- **Subtrees** (Layer 2): ``pickup(ns)`` and ``place(ns)``. Flat sequences
  that do one complete task. No recovery — if any step fails, the
  sequence returns FAILURE.

Usage::

    from mj_manipulator.bt import pickup, place

    tree = pickup("/right")
    tree = place("/left")

See ``docs/behavior-trees.md`` for the full composition guide.
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
    pickup,
    place,
)

__all__ = [
    # Nodes (Layer 1 building blocks)
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
    # Subtrees (Layer 2)
    "pickup",
    "place",
]
