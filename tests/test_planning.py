# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for PlanResult."""

import numpy as np

from mj_manipulator.planning import PlanResult
from mj_manipulator.trajectory import Trajectory


def _make_trajectory(duration: float = 1.0, dof: int = 6, entity: str | None = None):
    """Create a simple test trajectory."""
    n = max(2, int(duration / 0.008))
    timestamps = np.linspace(0, duration, n)
    return Trajectory(
        timestamps=timestamps,
        positions=np.zeros((n, dof)),
        velocities=np.zeros((n, dof)),
        accelerations=np.zeros((n, dof)),
        entity=entity,
    )


class TestPlanResult:
    """Tests for PlanResult."""

    def test_arm_only(self):
        """Plan with arm trajectory only."""
        arm_traj = _make_trajectory(2.0, dof=6, entity="ur5e_arm")
        result = PlanResult(arm_name="left", arm_trajectory=arm_traj)

        assert result.success
        assert result.total_duration == 2.0
        assert len(result.trajectories) == 1
        assert result.base_trajectory is None
        assert result.base_height is None

    def test_arm_plus_base(self):
        """Plan with both arm and base trajectories."""
        arm_traj = _make_trajectory(2.0, dof=6, entity="left_arm")
        base_traj = _make_trajectory(1.0, dof=1, entity="left_base")

        result = PlanResult(
            arm_name="left",
            arm_trajectory=arm_traj,
            base_trajectory=base_traj,
            base_height=0.3,
        )

        assert result.success
        assert result.total_duration == 3.0
        assert len(result.trajectories) == 2
        # Base trajectory should come first
        assert result.trajectories[0].entity == "left_base"
        assert result.trajectories[1].entity == "left_arm"

    def test_7dof_arm(self):
        """PlanResult works with 7-DOF (Franka) trajectories."""
        arm_traj = _make_trajectory(1.5, dof=7, entity="franka_arm")
        result = PlanResult(arm_name="franka", arm_trajectory=arm_traj)

        assert result.success
        assert result.arm_trajectory.dof == 7
