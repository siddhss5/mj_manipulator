# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for the arms/ module — UR5e and Franka arm factories + IK solvers.

Uses real menagerie robot models to verify:
- Factory functions create valid Arm instances
- EAIK HPRobot IK with MuJoCo-extracted kinematics
- FK ↔ IK round-trip consistency
- Franka discretized joint-5 IK
"""

import mujoco
import numpy as np
import pytest
from mj_environment import Environment

from mj_manipulator.arms.ur5e import (
    UR5E_HOME,
    UR5E_JOINT_NAMES,
    create_ur5e_arm,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ur5e_env():
    try:
        from mj_manipulator.menagerie import menagerie_scene

        scene = menagerie_scene("universal_robots_ur5e")
    except FileNotFoundError:
        pytest.skip("mujoco_menagerie not found")
    return Environment(str(scene))


@pytest.fixture
def ur5e_arm(ur5e_env):
    arm = create_ur5e_arm(ur5e_env)
    # Set to home
    for i, idx in enumerate(arm.joint_qpos_indices):
        ur5e_env.data.qpos[idx] = UR5E_HOME[i]
    mujoco.mj_forward(ur5e_env.model, ur5e_env.data)
    return arm


@pytest.fixture
def franka_env():
    try:
        from mj_manipulator.menagerie import menagerie_scene

        franka_scene = menagerie_scene("franka_emika_panda")
    except FileNotFoundError:
        pytest.skip("mujoco_menagerie not found")
    # Franka needs an EE site added via MjSpec
    from mj_manipulator.arms.franka import add_franka_ee_site

    spec = mujoco.MjSpec.from_file(str(franka_scene))
    add_franka_ee_site(spec)

    # Write modified XML to the menagerie directory so mesh paths resolve
    franka_dir = franka_scene.parent
    tmp_path = franka_dir / "_test_scene_with_ee.xml"
    try:
        tmp_path.write_text(spec.to_xml())
        env = Environment(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)

    return env


@pytest.fixture
def franka_arm(franka_env):
    from mj_manipulator.arms.franka import FRANKA_HOME, create_franka_arm

    arm = create_franka_arm(franka_env)
    for i, idx in enumerate(arm.joint_qpos_indices):
        franka_env.data.qpos[idx] = FRANKA_HOME[i]
    mujoco.mj_forward(franka_env.model, franka_env.data)
    return arm


@pytest.fixture
def iiwa14_env():
    try:
        from mj_manipulator.menagerie import menagerie_scene

        iiwa_scene = menagerie_scene("kuka_iiwa_14")
    except FileNotFoundError:
        pytest.skip("mujoco_menagerie not found")

    from mj_manipulator.arms.iiwa14 import add_iiwa14_ee_site, add_iiwa14_gravcomp

    spec = mujoco.MjSpec.from_file(str(iiwa_scene))
    add_iiwa14_ee_site(spec)
    add_iiwa14_gravcomp(spec)

    # Write modified XML to the menagerie directory so mesh paths resolve
    iiwa_dir = iiwa_scene.parent
    tmp_path = iiwa_dir / "_test_scene_with_ee.xml"
    try:
        tmp_path.write_text(spec.to_xml())
        env = Environment(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)

    return env


@pytest.fixture
def iiwa14_arm(iiwa14_env):
    from mj_manipulator.arms.iiwa14 import IIWA14_HOME, create_iiwa14_arm

    arm = create_iiwa14_arm(iiwa14_env)
    for i, idx in enumerate(arm.joint_qpos_indices):
        iiwa14_env.data.qpos[idx] = IIWA14_HOME[i]
    mujoco.mj_forward(iiwa14_env.model, iiwa14_env.data)
    return arm


# ---------------------------------------------------------------------------
# UR5e factory tests
# ---------------------------------------------------------------------------


class TestUR5eFactory:
    """Tests for create_ur5e_arm()."""

    def test_creates_arm(self, ur5e_arm):
        """Factory creates a valid Arm with 6 DOF."""
        assert ur5e_arm.dof == 6
        assert ur5e_arm.ik_solver is not None

    def test_joint_names(self, ur5e_arm):
        """Arm has correct joint names."""
        assert ur5e_arm.config.joint_names == list(UR5E_JOINT_NAMES)

    def test_joint_positions(self, ur5e_arm):
        """Joint positions match home config after setup."""
        q = ur5e_arm.get_joint_positions()
        np.testing.assert_allclose(q, UR5E_HOME, atol=1e-6)

    def test_ee_pose_shape(self, ur5e_arm):
        """EE pose is a valid 4x4 homogeneous transform."""
        pose = ur5e_arm.get_ee_pose()
        assert pose.shape == (4, 4)
        np.testing.assert_allclose(np.linalg.det(pose[:3, :3]), 1.0, atol=1e-6)

    def test_without_ik(self, ur5e_env):
        """Factory works without IK solver."""
        arm = create_ur5e_arm(ur5e_env, with_ik=False)
        assert arm.ik_solver is None
        assert arm.dof == 6


# ---------------------------------------------------------------------------
# UR5e IK tests
# ---------------------------------------------------------------------------


class TestUR5eIK:
    """Tests for EAIK IK solver with MuJoCo-extracted kinematics."""

    def test_fk_ik_roundtrip_at_home(self, ur5e_arm):
        """IK at home FK pose recovers home config (or equivalent)."""
        pose = ur5e_arm.get_ee_pose()
        solutions = ur5e_arm.ik_solver.solve_valid(pose)

        assert len(solutions) > 0, "No IK solutions found at home pose"

        # At least one solution should produce the same FK
        found_match = False
        for q in solutions:
            fk_pose = ur5e_arm.forward_kinematics(q)
            if np.allclose(fk_pose[:3, 3], pose[:3, 3], atol=1e-3):
                found_match = True
                break
        assert found_match, "No IK solution matches home FK pose"

    def test_fk_ik_roundtrip_at_other_config(self, ur5e_arm):
        """IK round-trip at a non-home configuration."""
        q_test = UR5E_HOME + np.array([0.2, -0.1, 0.15, -0.1, 0.2, 0.1])
        pose = ur5e_arm.forward_kinematics(q_test)
        solutions = ur5e_arm.ik_solver.solve_valid(pose)

        assert len(solutions) > 0, "No IK solutions found"

        # Verify at least one solution gives matching FK
        best_error = float("inf")
        for q in solutions:
            fk_pose = ur5e_arm.forward_kinematics(q)
            pos_error = np.linalg.norm(fk_pose[:3, 3] - pose[:3, 3])
            best_error = min(best_error, pos_error)
        assert best_error < 2e-3, f"Best IK position error: {best_error:.6f}"

    def test_solve_returns_multiple_solutions(self, ur5e_arm):
        """EAIK typically returns multiple solutions for 6-DOF arms."""
        pose = ur5e_arm.get_ee_pose()
        solutions = ur5e_arm.ik_solver.solve(pose)
        assert len(solutions) >= 1

    def test_unreachable_pose_returns_empty(self, ur5e_arm):
        """IK for unreachable pose returns no valid solutions."""
        # Place target 10m away — clearly unreachable
        pose = np.eye(4)
        pose[:3, 3] = [10.0, 0.0, 0.0]
        solutions = ur5e_arm.ik_solver.solve_valid(pose)
        assert len(solutions) == 0


# ---------------------------------------------------------------------------
# Franka factory tests
# ---------------------------------------------------------------------------


class TestFrankaFactory:
    """Tests for create_franka_arm()."""

    def test_creates_arm(self, franka_arm):
        """Factory creates a valid Arm with 7 DOF."""
        assert franka_arm.dof == 7
        assert franka_arm.ik_solver is not None

    def test_ee_pose_shape(self, franka_arm):
        """EE pose is a valid 4x4 homogeneous transform."""
        pose = franka_arm.get_ee_pose()
        assert pose.shape == (4, 4)
        np.testing.assert_allclose(np.linalg.det(pose[:3, :3]), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Franka IK tests
# ---------------------------------------------------------------------------


class TestFrankaIK:
    """Tests for Franka 7-DOF IK via joint-5 discretization."""

    def test_fk_ik_roundtrip(self, franka_arm):
        """IK at home FK pose finds a solution with matching position."""
        pose = franka_arm.get_ee_pose()
        solutions = franka_arm.ik_solver.solve_valid(pose)

        assert len(solutions) > 0, "No IK solutions found at home pose"

        # Verify position match
        best_error = float("inf")
        for q in solutions:
            fk_pose = franka_arm.forward_kinematics(q)
            pos_error = np.linalg.norm(fk_pose[:3, 3] - pose[:3, 3])
            best_error = min(best_error, pos_error)
        assert best_error < 5e-3, f"Best IK position error: {best_error:.6f}"

    def test_solutions_are_7dof(self, franka_arm):
        """All solutions have 7 joint values."""
        pose = franka_arm.get_ee_pose()
        solutions = franka_arm.ik_solver.solve(pose)
        for q in solutions:
            assert len(q) == 7

    def test_solutions_within_limits(self, franka_arm):
        """solve_valid only returns in-limits solutions."""
        pose = franka_arm.get_ee_pose()
        solutions = franka_arm.ik_solver.solve_valid(pose)
        lower, upper = franka_arm.get_joint_limits()
        for q in solutions:
            assert np.all(q >= lower - 1e-10), f"Below lower limit: {q}"
            assert np.all(q <= upper + 1e-10), f"Above upper limit: {q}"


# ---------------------------------------------------------------------------
# add_franka_ee_site tests
# ---------------------------------------------------------------------------


class TestAddFrankaEeSite:
    """Tests for the MjSpec helper."""

    def test_site_added(self):
        """add_franka_ee_site adds a site to the hand body."""
        try:
            from mj_manipulator.menagerie import menagerie_scene

            franka_scene = menagerie_scene("franka_emika_panda")
        except FileNotFoundError:
            pytest.skip("mujoco_menagerie not found")

        from mj_manipulator.arms.franka import add_franka_ee_site

        spec = mujoco.MjSpec.from_file(str(franka_scene))
        add_franka_ee_site(spec, site_name="test_ee")
        model = spec.compile()

        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "test_ee")
        assert site_id != -1

    def test_custom_position(self):
        """Custom position is applied to the site."""
        try:
            from mj_manipulator.menagerie import menagerie_scene

            franka_scene = menagerie_scene("franka_emika_panda")
        except FileNotFoundError:
            pytest.skip("mujoco_menagerie not found")

        from mj_manipulator.arms.franka import add_franka_ee_site

        spec = mujoco.MjSpec.from_file(str(franka_scene))
        add_franka_ee_site(spec, pos=[0, 0, 0.2])
        model = spec.compile()

        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
        assert site_id != -1


# ---------------------------------------------------------------------------
# iiwa 14 factory tests
# ---------------------------------------------------------------------------


class TestIIWA14Factory:
    """Tests for create_iiwa14_arm()."""

    def test_creates_arm(self, iiwa14_arm):
        """Factory creates a valid Arm with 7 DOF."""
        assert iiwa14_arm.dof == 7
        assert iiwa14_arm.ik_solver is not None

    def test_joint_names(self, iiwa14_arm):
        """Arm has correct joint names."""
        from mj_manipulator.arms.iiwa14 import IIWA14_JOINT_NAMES

        assert iiwa14_arm.config.joint_names == list(IIWA14_JOINT_NAMES)

    def test_joint_positions(self, iiwa14_arm):
        """Joint positions match home config after setup."""
        from mj_manipulator.arms.iiwa14 import IIWA14_HOME

        q = iiwa14_arm.get_joint_positions()
        np.testing.assert_allclose(q, IIWA14_HOME, atol=1e-6)

    def test_ee_pose_shape(self, iiwa14_arm):
        """get_ee_pose returns a 4x4 homogeneous transform."""
        pose = iiwa14_arm.get_ee_pose()
        assert pose.shape == (4, 4)
        assert pose[3, 3] == 1.0
        assert np.allclose(pose[3, :3], 0.0)

    def test_without_ik(self, iiwa14_env):
        from mj_manipulator.arms.iiwa14 import create_iiwa14_arm

        arm = create_iiwa14_arm(iiwa14_env, with_ik=False)
        assert arm.dof == 7
        assert arm.ik_solver is None


class TestIIWA14IK:
    """Tests for the iiwa 14's EAIK solver (7-DOF with joint-1 discretization)."""

    def test_fk_ik_roundtrip(self, iiwa14_arm):
        """FK(q_home) → IK returns a solution whose FK matches to mm."""
        target = iiwa14_arm.get_ee_pose()
        sols = iiwa14_arm.ik_solver.solve_valid(target)
        assert len(sols) > 0, "IK should return at least one solution at home"

        # Verify FK of any solution matches target
        for q in sols[:3]:  # check a few
            for i, idx in enumerate(iiwa14_arm.joint_qpos_indices):
                iiwa14_arm.env.data.qpos[idx] = q[i]
            mujoco.mj_forward(iiwa14_arm.env.model, iiwa14_arm.env.data)
            pose = iiwa14_arm.get_ee_pose()
            pos_err = np.linalg.norm(pose[:3, 3] - target[:3, 3])
            assert pos_err < 2e-3, f"FK-IK position error {pos_err * 1000:.2f}mm > 2mm"

    def test_solutions_are_7dof(self, iiwa14_arm):
        """IK solutions have 7 joint values (full iiwa DOF)."""
        pose = iiwa14_arm.get_ee_pose()
        sols = iiwa14_arm.ik_solver.solve_valid(pose)
        for q in sols:
            assert q.shape == (7,)

    def test_solutions_within_limits(self, iiwa14_arm):
        """solve_valid only returns in-limits solutions."""
        pose = iiwa14_arm.get_ee_pose()
        sols = iiwa14_arm.ik_solver.solve_valid(pose)
        lower, upper = iiwa14_arm.get_joint_limits()
        for q in sols:
            assert np.all(q >= lower - 1e-10), f"Below lower limit: {q}"
            assert np.all(q <= upper + 1e-10), f"Above upper limit: {q}"

    def test_unreachable_pose_returns_empty(self, iiwa14_arm):
        """A pose 5m away is unreachable and solve_valid returns []."""
        pose = np.eye(4)
        pose[:3, 3] = [5.0, 5.0, 5.0]
        sols = iiwa14_arm.ik_solver.solve_valid(pose)
        assert sols == []


class TestAddIIWA14EeSite:
    """Tests for the MjSpec helper."""

    def test_site_added(self):
        """add_iiwa14_ee_site adds a site to link7."""
        try:
            from mj_manipulator.menagerie import menagerie_scene

            iiwa_scene = menagerie_scene("kuka_iiwa_14")
        except FileNotFoundError:
            pytest.skip("mujoco_menagerie not found")

        from mj_manipulator.arms.iiwa14 import add_iiwa14_ee_site

        spec = mujoco.MjSpec.from_file(str(iiwa_scene))
        add_iiwa14_ee_site(spec, site_name="test_ee")
        model = spec.compile()

        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "test_ee")
        assert site_id != -1
