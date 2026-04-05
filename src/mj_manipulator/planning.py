# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Planning result types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mj_manipulator.trajectory import Trajectory


@dataclass
class PlanResult:
    """Result of a planning operation with optional base motion.

    When planning with base_heights, the result includes both an arm
    trajectory and potentially a base trajectory. The base trajectory
    should be executed first.

    Attributes:
        arm_name: Name of the arm that will execute the motion
        arm_trajectory: Trajectory for the arm joints
        base_trajectory: Optional trajectory for base height adjustment
        base_height: The base height used for planning (if applicable)
    """

    arm_name: str
    arm_trajectory: "Trajectory"
    base_trajectory: "Trajectory | None" = None
    base_height: float | None = None

    @property
    def trajectories(self) -> list["Trajectory"]:
        """All trajectories in execution order (base first, then arm)."""
        result = []
        if self.base_trajectory is not None:
            result.append(self.base_trajectory)
        result.append(self.arm_trajectory)
        return result

    @property
    def success(self) -> bool:
        """Whether planning succeeded."""
        return self.arm_trajectory is not None

    @property
    def total_duration(self) -> float:
        """Total duration of all trajectories in seconds."""
        return sum(t.duration for t in self.trajectories)
