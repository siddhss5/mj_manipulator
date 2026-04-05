# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for trajectory representation and retiming."""

import numpy as np
import pytest

from mj_manipulator.trajectory import Trajectory, create_linear_trajectory


class TestTrajectory:
    """Tests for the Trajectory dataclass."""

    def test_basic_construction(self):
        """Trajectory with valid arrays constructs without error."""
        traj = Trajectory(
            timestamps=np.array([0.0, 0.5, 1.0]),
            positions=np.array([[0, 0], [0.5, 0.5], [1, 1]]),
            velocities=np.array([[1, 1], [1, 1], [0, 0]]),
            accelerations=np.array([[0, 0], [0, 0], [-2, -2]]),
        )
        assert traj.duration == 1.0
        assert traj.dof == 2
        assert traj.num_waypoints == 3

    def test_dimension_mismatch_raises(self):
        """Mismatched array dimensions raise ValueError."""
        with pytest.raises(ValueError, match="Position shape"):
            Trajectory(
                timestamps=np.array([0.0, 1.0]),
                positions=np.array([[0, 0], [1, 1], [2, 2]]),  # 3 rows, not 2
                velocities=np.array([[1, 1], [0, 0]]),
                accelerations=np.array([[0, 0], [0, 0]]),
            )

    def test_joint_names_validation(self):
        """Joint names length must match DOF."""
        with pytest.raises(ValueError, match="joint_names"):
            Trajectory(
                timestamps=np.array([0.0]),
                positions=np.array([[0, 0]]),
                velocities=np.array([[0, 0]]),
                accelerations=np.array([[0, 0]]),
                joint_names=["j1", "j2", "j3"],  # 3 names, 2 DOF
            )

    def test_sample_interpolation(self):
        """Sampling interpolates linearly between waypoints."""
        traj = Trajectory(
            timestamps=np.array([0.0, 1.0]),
            positions=np.array([[0.0], [2.0]]),
            velocities=np.array([[2.0], [2.0]]),
            accelerations=np.array([[0.0], [0.0]]),
        )
        pos, vel, acc = traj.sample(0.5)
        assert pos[0] == pytest.approx(1.0)
        assert vel[0] == pytest.approx(2.0)

    def test_sample_clamps_to_bounds(self):
        """Sampling clamps to trajectory bounds."""
        traj = Trajectory(
            timestamps=np.array([0.0, 1.0]),
            positions=np.array([[0.0], [1.0]]),
            velocities=np.array([[1.0], [0.0]]),
            accelerations=np.array([[0.0], [0.0]]),
        )
        pos_before, _, _ = traj.sample(-1.0)
        pos_after, _, _ = traj.sample(5.0)
        assert pos_before[0] == pytest.approx(0.0)
        assert pos_after[0] == pytest.approx(1.0)

    def test_entity_metadata(self):
        """Entity and joint_names metadata is stored."""
        traj = Trajectory(
            timestamps=np.array([0.0]),
            positions=np.array([[0.0, 0.0]]),
            velocities=np.array([[0.0, 0.0]]),
            accelerations=np.array([[0.0, 0.0]]),
            entity="franka_arm",
            joint_names=["joint1", "joint2"],
        )
        assert traj.entity == "franka_arm"
        assert traj.joint_names == ["joint1", "joint2"]


class TestFromPath:
    """Tests for TOPP-RA trajectory creation."""

    def test_basic_retiming(self):
        """from_path creates a valid time-optimal trajectory."""
        path = [np.array([0.0, 0.0]), np.array([1.0, 1.0])]
        vel_limits = np.array([2.0, 2.0])
        acc_limits = np.array([4.0, 4.0])

        traj = Trajectory.from_path(path, vel_limits, acc_limits)

        assert traj.dof == 2
        assert traj.duration > 0
        assert traj.num_waypoints > 2
        # Start and end positions should match path
        assert traj.positions[0] == pytest.approx(path[0], abs=0.05)
        assert traj.positions[-1] == pytest.approx(path[-1], abs=0.05)

    def test_6dof_path(self):
        """Works with 6-DOF paths (UR5e)."""
        rng = np.random.default_rng(42)
        path = [rng.uniform(-np.pi, np.pi, 6) for _ in range(5)]
        vel_limits = np.full(6, 3.14)
        acc_limits = np.full(6, 2.5)

        traj = Trajectory.from_path(path, vel_limits, acc_limits)
        assert traj.dof == 6
        assert traj.duration > 0

    def test_7dof_path(self):
        """Works with 7-DOF paths (Franka)."""
        rng = np.random.default_rng(42)
        path = [rng.uniform(-np.pi, np.pi, 7) for _ in range(5)]
        vel_limits = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
        acc_limits = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])

        traj = Trajectory.from_path(path, vel_limits, acc_limits)
        assert traj.dof == 7
        assert traj.duration > 0

    def test_empty_path_raises(self):
        """Empty path raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            Trajectory.from_path([], np.array([1.0]), np.array([1.0]))

    def test_single_waypoint_returns_trivial(self):
        """Single waypoint returns zero-duration trajectory."""
        path = [np.array([1.0, 2.0])]
        traj = Trajectory.from_path(path, np.array([1.0, 1.0]), np.array([1.0, 1.0]))
        assert traj.duration == 0.0
        assert traj.num_waypoints == 1

    def test_duplicate_waypoints_filtered(self):
        """Consecutive duplicate waypoints are filtered out."""
        path = [np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([1.0, 1.0])]
        traj = Trajectory.from_path(path, np.array([2.0, 2.0]), np.array([4.0, 4.0]))
        assert traj.duration > 0


class TestLinearTrajectory:
    """Tests for 1D trapezoidal trajectory generation."""

    def test_trapezoidal_profile(self):
        """Long distances produce trapezoidal velocity profile."""
        traj = create_linear_trajectory(0.0, 1.0, vel_limit=0.5, acc_limit=1.0)
        assert traj.dof == 1
        assert traj.duration > 0
        assert traj.positions[0, 0] == pytest.approx(0.0)
        assert traj.positions[-1, 0] == pytest.approx(1.0, abs=1e-6)

    def test_triangular_profile(self):
        """Short distances produce triangular profile (no cruise phase)."""
        traj = create_linear_trajectory(0.0, 0.01, vel_limit=1.0, acc_limit=1.0)
        assert traj.dof == 1
        assert traj.positions[-1, 0] == pytest.approx(0.01, abs=1e-4)

    def test_zero_distance(self):
        """Zero distance returns single-waypoint trajectory."""
        traj = create_linear_trajectory(0.5, 0.5, vel_limit=0.1, acc_limit=0.2)
        assert traj.num_waypoints == 1
        assert traj.positions[0, 0] == pytest.approx(0.5)

    def test_negative_direction(self):
        """Negative direction (end < start) works correctly."""
        traj = create_linear_trajectory(1.0, 0.0, vel_limit=0.5, acc_limit=1.0)
        assert traj.positions[0, 0] == pytest.approx(1.0)
        assert traj.positions[-1, 0] == pytest.approx(0.0, abs=1e-6)

    def test_entity_metadata(self):
        """Entity metadata is passed through."""
        traj = create_linear_trajectory(0.0, 0.5, 0.1, 0.2, entity="left_base", joint_names=["base_joint"])
        assert traj.entity == "left_base"
        assert traj.joint_names == ["base_joint"]
