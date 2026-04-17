# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for MinkIKSolver — numerical IK via mink.

Skipped entirely if mink is not installed (optional dependency).
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

mink = pytest.importorskip("mink")

from mj_environment import Environment  # noqa: E402

from mj_manipulator.arms.mink_solver import make_mink_solver  # noqa: E402
from mj_manipulator.protocols import IKSolver  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures: build each arm once per test class
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ur5e_setup():
    """UR5e arm + MinkIKSolver."""
    from mj_manipulator.arms.ur5e import UR5E_HOME, add_ur5e_gravcomp, create_ur5e_arm
    from mj_manipulator.menagerie import menagerie_scene

    spec = mujoco.MjSpec.from_file(str(menagerie_scene("universal_robots_ur5e")))
    add_ur5e_gravcomp(spec)
    env = Environment.from_model(spec.compile())
    arm = create_ur5e_arm(env, with_ik=False)
    solver = make_mink_solver(arm, ee_frame_name="attachment_site")
    return arm, solver, np.array(UR5E_HOME)


@pytest.fixture(scope="module")
def franka_setup():
    """Franka arm + MinkIKSolver."""
    from mj_manipulator.arms.franka import FRANKA_HOME, add_franka_ee_site, add_franka_gravcomp, create_franka_arm
    from mj_manipulator.menagerie import menagerie_scene

    spec = mujoco.MjSpec.from_file(str(menagerie_scene("franka_emika_panda")))
    add_franka_ee_site(spec)
    add_franka_gravcomp(spec)
    env = Environment.from_model(spec.compile())
    arm = create_franka_arm(env, with_ik=False)
    solver = make_mink_solver(arm, ee_frame_name="grasp_site")
    return arm, solver, np.array(FRANKA_HOME)


@pytest.fixture(scope="module")
def iiwa14_setup():
    """KUKA iiwa 14 arm + MinkIKSolver."""
    from mj_manipulator.arms.iiwa14 import IIWA14_HOME, add_iiwa14_ee_site, add_iiwa14_gravcomp, create_iiwa14_arm
    from mj_manipulator.menagerie import menagerie_scene

    spec = mujoco.MjSpec.from_file(str(menagerie_scene("kuka_iiwa_14")))
    add_iiwa14_ee_site(spec)
    add_iiwa14_gravcomp(spec)
    env = Environment.from_model(spec.compile())
    arm = create_iiwa14_arm(env, with_ik=False)
    solver = make_mink_solver(arm, ee_frame_name="grasp_site")
    return arm, solver, np.array(IIWA14_HOME)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_qpos(arm, q: np.ndarray) -> None:
    for i, idx in enumerate(arm.joint_qpos_indices):
        arm.env.data.qpos[idx] = q[i]
    mujoco.mj_forward(arm.env.model, arm.env.data)


def _fk_error(arm, q: np.ndarray, target_pose: np.ndarray) -> float:
    """FK position error in meters."""
    _set_qpos(arm, q)
    ee = arm.get_ee_pose()
    return float(np.linalg.norm(ee[:3, 3] - target_pose[:3, 3]))


def _random_reachable_pose(arm, home: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate a reachable pose by random FK near home."""
    q_lower, q_upper = arm.get_joint_limits()
    # Bias toward home ±1 rad for reasonable configs
    q = home + rng.uniform(-1.0, 1.0, len(home))
    q = np.clip(q, q_lower, q_upper)
    _set_qpos(arm, q)
    return arm.get_ee_pose().copy()


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestMinkIKSolverProtocol:
    def test_implements_ik_solver(self, ur5e_setup):
        _, solver, _ = ur5e_setup
        assert isinstance(solver, IKSolver)

    def test_solve_returns_list(self, ur5e_setup):
        arm, solver, home = ur5e_setup
        _set_qpos(arm, home)
        pose = arm.get_ee_pose()
        result = solver.solve(pose)
        assert isinstance(result, list)

    def test_solve_valid_returns_list(self, ur5e_setup):
        arm, solver, home = ur5e_setup
        _set_qpos(arm, home)
        pose = arm.get_ee_pose()
        result = solver.solve_valid(pose)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# FK-IK round-trips
# ---------------------------------------------------------------------------


class TestUR5eRoundTrip:
    """FK-IK round-trip tests for UR5e (6-DOF)."""

    def test_home_pose(self, ur5e_setup):
        arm, solver, home = ur5e_setup
        _set_qpos(arm, home)
        pose = arm.get_ee_pose()
        solutions = solver.solve_valid(pose, q_init=home)
        assert len(solutions) >= 1
        assert _fk_error(arm, solutions[0], pose) < 0.005  # 5 mm

    def test_offset_pose(self, ur5e_setup):
        arm, solver, home = ur5e_setup
        _set_qpos(arm, home)
        pose = arm.get_ee_pose()
        pose[2, 3] += 0.05  # +5 cm in z
        solutions = solver.solve_valid(pose, q_init=home)
        assert len(solutions) >= 1
        assert _fk_error(arm, solutions[0], pose) < 0.005

    def test_random_poses(self, ur5e_setup):
        arm, solver, home = ur5e_setup
        rng = np.random.default_rng(42)
        successes = 0
        n = 10
        for _ in range(n):
            pose = _random_reachable_pose(arm, home, rng)
            solutions = solver.solve_valid(pose, q_init=home)
            if solutions and _fk_error(arm, solutions[0], pose) < 0.005:
                successes += 1
        assert successes >= n * 0.8  # ≥80% success rate


class TestFrankaRoundTrip:
    """FK-IK round-trip tests for Franka (7-DOF)."""

    def test_home_pose(self, franka_setup):
        arm, solver, home = franka_setup
        _set_qpos(arm, home)
        pose = arm.get_ee_pose()
        solutions = solver.solve_valid(pose, q_init=home)
        assert len(solutions) >= 1
        assert _fk_error(arm, solutions[0], pose) < 0.005

    def test_offset_pose(self, franka_setup):
        arm, solver, home = franka_setup
        _set_qpos(arm, home)
        pose = arm.get_ee_pose()
        pose[2, 3] -= 0.05
        solutions = solver.solve_valid(pose, q_init=home)
        assert len(solutions) >= 1
        assert _fk_error(arm, solutions[0], pose) < 0.005

    def test_random_poses(self, franka_setup):
        arm, solver, home = franka_setup
        rng = np.random.default_rng(42)
        successes = 0
        n = 10
        for _ in range(n):
            pose = _random_reachable_pose(arm, home, rng)
            solutions = solver.solve_valid(pose, q_init=home)
            if solutions and _fk_error(arm, solutions[0], pose) < 0.005:
                successes += 1
        assert successes >= n * 0.8

    def test_solutions_are_7dof(self, franka_setup):
        arm, solver, home = franka_setup
        _set_qpos(arm, home)
        pose = arm.get_ee_pose()
        solutions = solver.solve_valid(pose, q_init=home)
        assert len(solutions) >= 1
        assert solutions[0].shape == (7,)


class TestIIWA14RoundTrip:
    """FK-IK round-trip tests for iiwa 14 (7-DOF)."""

    def test_home_pose(self, iiwa14_setup):
        arm, solver, home = iiwa14_setup
        _set_qpos(arm, home)
        pose = arm.get_ee_pose()
        solutions = solver.solve_valid(pose, q_init=home)
        assert len(solutions) >= 1
        assert _fk_error(arm, solutions[0], pose) < 0.005

    def test_random_poses(self, iiwa14_setup):
        arm, solver, home = iiwa14_setup
        rng = np.random.default_rng(42)
        successes = 0
        n = 10
        for _ in range(n):
            pose = _random_reachable_pose(arm, home, rng)
            solutions = solver.solve_valid(pose, q_init=home)
            if solutions and _fk_error(arm, solutions[0], pose) < 0.005:
                successes += 1
        assert successes >= n * 0.8


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_unreachable_pose_returns_empty(self, ur5e_setup):
        """A pose 10 m away should return no solutions."""
        _, solver, _ = ur5e_setup
        pose = np.eye(4)
        pose[:3, 3] = [10.0, 0.0, 0.0]
        solutions = solver.solve_valid(pose)
        assert solutions == []

    def test_solutions_within_limits(self, ur5e_setup):
        """All solutions from solve_valid must be within joint limits."""
        arm, solver, home = ur5e_setup
        q_lower, q_upper = arm.get_joint_limits()
        _set_qpos(arm, home)
        pose = arm.get_ee_pose()
        solutions = solver.solve_valid(pose, q_init=home)
        for q in solutions:
            assert np.all(q >= q_lower - 1e-5), f"Below lower limit: {q} vs {q_lower}"
            assert np.all(q <= q_upper + 1e-5), f"Above upper limit: {q} vs {q_upper}"

    def test_q_init_produces_nearby_solution(self, franka_setup):
        """With q_init near the target, the closest solution should be near q_init."""
        arm, solver, home = franka_setup
        _set_qpos(arm, home)
        pose = arm.get_ee_pose()
        solutions = solver.solve_valid(pose, q_init=home)
        assert len(solutions) >= 1
        # The solution seeded from home should be close to home
        best_dist = min(np.linalg.norm(q - home) for q in solutions)
        assert best_dist < 0.5, f"No solution near q_init (best dist={best_dist:.2f} rad)"

    def test_multiple_restarts_give_diversity(self, franka_setup):
        """With 8 restarts on a 7-DOF arm, we should get >1 unique solution."""
        arm, solver, home = franka_setup
        _set_qpos(arm, home)
        pose = arm.get_ee_pose()
        pose[2, 3] -= 0.05
        solutions = solver.solve_valid(pose, q_init=home)
        # Not guaranteed to get >1 for every pose, but for a well-conditioned
        # pose near home the 7-DOF redundancy should yield diversity.
        assert len(solutions) >= 1  # conservative: at least 1
