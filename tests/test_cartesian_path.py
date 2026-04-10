# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for :mod:`mj_manipulator.cartesian_path`.

Covers:
- ``translational_waypoints`` edge cases (zero/short/long distances,
  non-unit direction, orientation preservation)
- ``plan_cartesian_path`` happy path and error handling (empty list,
  bad shape, no-IK-solution, branch-jump detection)

The integration tests require the Franka + menagerie; they skip
automatically if either is missing. Pure-logic tests (waypoint
generation, input validation) do not need menagerie.
"""

from __future__ import annotations

import numpy as np
import pytest

from mj_manipulator.cartesian_path import plan_cartesian_path, translational_waypoints

# ---------------------------------------------------------------------------
# translational_waypoints — pure logic, no menagerie
# ---------------------------------------------------------------------------


class TestTranslationalWaypoints:
    """Unit tests for the SE(3) waypoint generator."""

    def test_basic_lift_generates_expected_count(self):
        """15 cm lift with 5 mm spacing → 30 waypoints."""
        start = np.eye(4)
        wps = translational_waypoints(start, np.array([0.0, 0.0, 1.0]), distance=0.15, segment_length=0.005)
        assert len(wps) == 30

    def test_final_waypoint_at_exact_distance(self):
        """Last waypoint is exactly at start + direction * distance."""
        start = np.eye(4)
        wps = translational_waypoints(start, np.array([0.0, 0.0, 1.0]), distance=0.15, segment_length=0.005)
        np.testing.assert_allclose(wps[-1][:3, 3], [0.0, 0.0, 0.15], atol=1e-12)

    def test_waypoints_preserve_orientation(self):
        """Every waypoint has the same rotation as the start pose."""
        start = np.eye(4)
        # 45° rotation about Y
        theta = np.pi / 4
        start[:3, :3] = np.array(
            [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ]
        )
        wps = translational_waypoints(start, np.array([0.0, 0.0, 1.0]), distance=0.1, segment_length=0.01)
        for w in wps:
            np.testing.assert_allclose(w[:3, :3], start[:3, :3], atol=1e-12)

    def test_non_unit_direction_is_normalized(self):
        """Passing a non-unit direction vector still produces correct geometry."""
        start = np.eye(4)
        # Direction magnitude = 2, but distance is still 0.15 m
        wps = translational_waypoints(start, np.array([0.0, 0.0, 2.0]), distance=0.15, segment_length=0.005)
        np.testing.assert_allclose(wps[-1][:3, 3], [0.0, 0.0, 0.15], atol=1e-12)

    def test_arbitrary_direction(self):
        """Diagonal direction places the final waypoint at the right spot."""
        start = np.eye(4)
        direction = np.array([1.0, 1.0, 1.0])  # unit = 1/sqrt(3) * [1,1,1]
        wps = translational_waypoints(start, direction, distance=0.3, segment_length=0.05)
        expected = (direction / np.linalg.norm(direction)) * 0.3
        np.testing.assert_allclose(wps[-1][:3, 3], expected, atol=1e-12)

    def test_short_distance_produces_one_waypoint(self):
        """distance < segment_length → exactly one waypoint at the full distance."""
        start = np.eye(4)
        wps = translational_waypoints(start, np.array([0.0, 0.0, 1.0]), distance=0.002, segment_length=0.005)
        assert len(wps) == 1
        np.testing.assert_allclose(wps[0][:3, 3], [0.0, 0.0, 0.002], atol=1e-12)

    def test_zero_direction_returns_empty(self):
        """Zero-magnitude direction vector → empty waypoint list."""
        start = np.eye(4)
        wps = translational_waypoints(start, np.zeros(3), distance=0.15, segment_length=0.005)
        assert wps == []

    def test_start_pose_translation_preserved(self):
        """Waypoints are offset from the start pose's translation, not origin."""
        start = np.eye(4)
        start[:3, 3] = [0.5, -0.2, 0.3]
        wps = translational_waypoints(start, np.array([0.0, 0.0, 1.0]), distance=0.1, segment_length=0.05)
        np.testing.assert_allclose(wps[-1][:3, 3], [0.5, -0.2, 0.4], atol=1e-12)


# ---------------------------------------------------------------------------
# plan_cartesian_path — input validation (no menagerie needed for most)
# ---------------------------------------------------------------------------


class TestPlanCartesianPathValidation:
    """Tests that exercise plan_cartesian_path's error-handling paths."""

    def test_empty_waypoints_raises(self, franka_arm_at_home):
        with pytest.raises(ValueError, match="waypoints must be non-empty"):
            plan_cartesian_path(franka_arm_at_home, [])

    def test_bad_shape_raises(self, franka_arm_at_home):
        bad = np.zeros((3, 4))  # not 4x4
        with pytest.raises(ValueError, match=r"shape \(3, 4\)"):
            plan_cartesian_path(franka_arm_at_home, [bad])

    def test_no_arm_ik_solver_raises(self, franka_env_with_gravcomp):
        """Arm built without an IK solver should refuse Cartesian planning."""
        from mj_manipulator.arms.franka import FRANKA_HOME, create_franka_arm

        env = franka_env_with_gravcomp
        arm = create_franka_arm(env, with_ik=False)
        for i, idx in enumerate(arm.joint_qpos_indices):
            env.data.qpos[idx] = FRANKA_HOME[i]
        import mujoco

        mujoco.mj_forward(env.model, env.data)

        # One valid waypoint; should still raise because of missing solver
        wp = arm.get_ee_pose()
        wp[2, 3] += 0.05
        with pytest.raises(RuntimeError, match="requires an arm with an IK solver"):
            plan_cartesian_path(arm, [wp])

    def test_unreachable_pose_raises(self, franka_arm_at_home):
        """A pose far outside the reachable workspace raises ValueError."""
        # 5 meters above the base — way outside Franka's ~0.855 m reach.
        wp = np.eye(4)
        wp[:3, 3] = [0.0, 0.0, 5.0]
        with pytest.raises(ValueError, match="no valid IK solution"):
            plan_cartesian_path(franka_arm_at_home, [wp])


# ---------------------------------------------------------------------------
# plan_cartesian_path — happy-path integration
# ---------------------------------------------------------------------------


class TestPlanCartesianPathHappy:
    """Integration tests that actually plan a trajectory."""

    def test_lift_produces_executable_trajectory(self, franka_arm_at_home):
        """A 10 cm +Z lift from home yields a valid, non-empty trajectory."""
        arm = franka_arm_at_home
        start = arm.get_ee_pose()
        wps = translational_waypoints(start, np.array([0.0, 0.0, 1.0]), distance=0.1, segment_length=0.005)
        traj = plan_cartesian_path(arm, wps)

        assert traj.dof == 7
        assert traj.num_waypoints > 0
        assert traj.duration > 0.0
        assert traj.entity == arm.config.name

    def test_trajectory_reaches_target_kinematically(self, franka_arm_at_home):
        """Final joint config in the trajectory produces the target EE z."""
        arm = franka_arm_at_home
        start = arm.get_ee_pose()
        target_z = start[2, 3] + 0.1
        wps = translational_waypoints(start, np.array([0.0, 0.0, 1.0]), distance=0.1, segment_length=0.005)
        traj = plan_cartesian_path(arm, wps)

        # Forward-kinematics the last waypoint and check the EE z.
        final_q = traj.positions[-1]
        pose = arm.forward_kinematics(final_q)
        assert abs(pose[2, 3] - target_z) < 1e-3
        # Translation in x/y should be tiny (straight-line lift).
        np.testing.assert_allclose(pose[:2, 3], start[:2, 3], atol=1e-3)

    def test_trajectory_preserves_orientation(self, franka_arm_at_home):
        """Waypoints in the planned trajectory approximately preserve EE orientation.

        Exact orientation is only guaranteed at the IK anchor waypoints. TOPP-RA
        densifies the joint-space path with linear interpolation between anchors,
        which introduces mild (~1-2°) rotation drift at interior samples — bounded
        by the curvature of the orientation manifold along a joint-space straight
        line. 0.05 (~3°) is the tolerance that catches actual bugs without
        flagging this expected interpolation error.
        """
        arm = franka_arm_at_home
        start = arm.get_ee_pose()
        wps = translational_waypoints(start, np.array([0.0, 0.0, 1.0]), distance=0.1, segment_length=0.005)
        traj = plan_cartesian_path(arm, wps)

        # Check first, middle, and last waypoint
        for idx in [0, traj.num_waypoints // 2, traj.num_waypoints - 1]:
            q = traj.positions[idx]
            pose = arm.forward_kinematics(q)
            rot_err = np.linalg.norm(pose[:3, :3] @ start[:3, :3].T - np.eye(3), ord="fro")
            assert rot_err < 0.05, f"Waypoint {idx} rotation drift: {rot_err}"

    def test_partial_ok_returns_feasible_prefix(self, franka_arm_at_home):
        """partial_ok=True returns a trajectory for the longest feasible prefix.

        Build a lift from home that starts feasibly but extends far enough
        above the workspace that IK fails at some mid-path waypoint. With
        partial_ok, we get back a shorter trajectory covering only the
        feasible portion — not a raise, and not an empty plan.
        """
        arm = franka_arm_at_home
        start = arm.get_ee_pose()
        # 80 cm lift — Franka reach is ~85 cm, so the very top will be
        # unreachable but the first ~30 cm should plan fine.
        wps = translational_waypoints(start, np.array([0.0, 0.0, 1.0]), distance=0.8, segment_length=0.01)

        # Without partial_ok, should raise.
        with pytest.raises(ValueError, match="no valid IK solution"):
            plan_cartesian_path(arm, wps)

        # With partial_ok, returns a non-empty partial trajectory.
        traj = plan_cartesian_path(arm, wps, partial_ok=True)
        assert traj.dof == 7
        assert traj.num_waypoints > 0
        assert traj.duration > 0.0

        # Final EE z of the partial trajectory should be somewhere between
        # the start and the commanded endpoint (not all 80 cm, not zero).
        final_q = traj.positions[-1]
        final_pose = arm.forward_kinematics(final_q)
        lifted = final_pose[2, 3] - start[2, 3]
        assert 0.05 < lifted < 0.8, (
            f"Partial lift should be a positive fraction of the commanded distance, got {lifted * 1000:.1f} mm"
        )

    def test_partial_ok_first_waypoint_infeasible_raises(self, franka_arm_at_home):
        """partial_ok only rescues mid-path failures, not first-waypoint ones.

        If the very first waypoint has no IK solution, there's no feasible
        prefix at all and we still raise — partial_ok doesn't silently
        hide "nothing is reachable" failures.
        """
        arm = franka_arm_at_home
        # Single unreachable waypoint
        unreachable = np.eye(4)
        unreachable[:3, 3] = [0.0, 0.0, 5.0]
        with pytest.raises(ValueError, match="no valid IK solution"):
            plan_cartesian_path(arm, [unreachable], partial_ok=True)

    def test_q_start_override(self, franka_arm_at_home):
        """Passing q_start uses it as the initial configuration, not qpos."""
        from mj_manipulator.arms.franka import FRANKA_HOME

        arm = franka_arm_at_home
        # Move the arm physically to a different pose, but plan from FRANKA_HOME
        import mujoco

        for i, idx in enumerate(arm.joint_qpos_indices):
            arm.env.data.qpos[idx] = FRANKA_HOME[i] + 0.1
        mujoco.mj_forward(arm.env.model, arm.env.data)

        # FK from FRANKA_HOME — not the current (perturbed) pose
        start_pose_at_home = arm.forward_kinematics(FRANKA_HOME)
        wps = translational_waypoints(start_pose_at_home, np.array([0.0, 0.0, 1.0]), 0.05, segment_length=0.005)
        traj = plan_cartesian_path(arm, wps, q_start=FRANKA_HOME)

        # First trajectory waypoint should be FRANKA_HOME (the retimer seeds
        # the path at q_start), not the perturbed current qpos.
        np.testing.assert_allclose(traj.positions[0], FRANKA_HOME, atol=1e-6)
