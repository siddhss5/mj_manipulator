# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for FrankaGripper.

Uses the mujoco_menagerie Franka Panda scene to verify:
- Factory construction resolves actuators, bodies, and joints correctly
- Gripper protocol satisfaction (isinstance check)
- Kinematic open/close set correct joint positions
- get_actual_position returns normalized [0, 1] values
- Integration with the create_franka_arm factory

The Robotiq 2F-140 gripper counterpart is tested in
``geodude/tests/test_robotiq_gripper.py``, where ``geodude_assets``
(which owns the Robotiq scene XML) is a declared dependency. See
personalrobotics/mj_manipulator#87 for the split rationale.
"""

import mujoco
import pytest
from mj_environment import Environment

from mj_manipulator.grasp_manager import GraspManager
from mj_manipulator.grippers.franka import FrankaGripper
from mj_manipulator.protocols import Gripper

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def franka_env():
    try:
        from mj_manipulator.menagerie import menagerie_scene

        franka_scene = menagerie_scene("franka_emika_panda")
    except FileNotFoundError:
        pytest.skip("mujoco_menagerie not found")
    from mj_manipulator.arms.franka import add_franka_ee_site

    spec = mujoco.MjSpec.from_file(str(franka_scene))
    add_franka_ee_site(spec)

    franka_dir = franka_scene.parent
    tmp_path = franka_dir / "_test_gripper_franka.xml"
    try:
        tmp_path.write_text(spec.to_xml())
        env = Environment(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)

    return env


@pytest.fixture
def franka_gripper(franka_env):
    return FrankaGripper(
        franka_env.model,
        franka_env.data,
        "franka",
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestFrankaGripperConstruction:
    def test_constructs(self, franka_gripper):
        assert franka_gripper is not None

    def test_satisfies_protocol(self, franka_gripper):
        assert isinstance(franka_gripper, Gripper)

    def test_arm_name(self, franka_gripper):
        assert franka_gripper.arm_name == "franka"

    def test_actuator_id(self, franka_gripper):
        assert franka_gripper.actuator_id is not None
        assert franka_gripper.actuator_id >= 0

    def test_ctrl_range(self, franka_gripper):
        assert franka_gripper.ctrl_open == 255.0
        assert franka_gripper.ctrl_closed == 0.0

    def test_body_names(self, franka_gripper):
        names = franka_gripper.gripper_body_names
        assert "hand" in names
        assert "left_finger" in names
        assert "right_finger" in names

    def test_attachment_body(self, franka_gripper):
        assert franka_gripper.attachment_body == "hand"

    def test_initially_not_holding(self, franka_gripper):
        assert not franka_gripper.is_holding
        assert franka_gripper.held_object is None

    def test_invalid_actuator_raises(self, franka_env):
        with pytest.raises(ValueError, match="not found"):
            FrankaGripper(
                franka_env.model,
                franka_env.data,
                "franka",
                prefix="nonexistent/",
            )


# ---------------------------------------------------------------------------
# Kinematic control
# ---------------------------------------------------------------------------


class TestFrankaGripperKinematic:
    def test_kinematic_open(self, franka_gripper):
        franka_gripper.kinematic_open()
        pos = franka_gripper.get_actual_position()
        assert pos == pytest.approx(0.0, abs=0.05)

    def test_kinematic_close_moves_fingers(self, franka_gripper):
        franka_gripper.kinematic_open()
        open_pos = franka_gripper.get_actual_position()

        # Empty candidates: no contact stops the close, gripper fully closes.
        franka_gripper.set_candidate_objects([])
        franka_gripper.kinematic_close(steps=10)
        closed_pos = franka_gripper.get_actual_position()

        # Should have moved toward closed
        assert closed_pos > open_pos

    def test_get_actual_position_range(self, franka_gripper, franka_env):
        # Open
        franka_gripper.kinematic_open()
        pos_open = franka_gripper.get_actual_position()
        assert 0.0 <= pos_open <= 0.1

        # Manually set fingers to closed
        for idx in franka_gripper._finger_qpos_indices:
            franka_env.data.qpos[idx] = 0.0
        mujoco.mj_forward(franka_env.model, franka_env.data)

        pos_closed = franka_gripper.get_actual_position()
        assert 0.9 <= pos_closed <= 1.0


# ---------------------------------------------------------------------------
# GraspManager integration
# ---------------------------------------------------------------------------


class TestFrankaGripperGraspManager:
    def test_is_holding_with_grasp_manager(self, franka_env):
        gm = GraspManager(franka_env.model, franka_env.data)
        gripper = FrankaGripper(
            franka_env.model,
            franka_env.data,
            "franka",
            grasp_manager=gm,
        )
        assert not gripper.is_holding

        gm.mark_grasped("box", "franka")
        assert gripper.is_holding
        assert gripper.held_object == "box"

        gm.mark_released("box")
        assert not gripper.is_holding
        assert gripper.held_object is None


# ---------------------------------------------------------------------------
# Integration with arm factory
# ---------------------------------------------------------------------------


class TestArmFactoryIntegration:
    def test_franka_with_gripper(self, franka_env):
        """Franka arm factory accepts gripper parameter."""
        from mj_manipulator.arms.franka import create_franka_arm

        gm = GraspManager(franka_env.model, franka_env.data)
        gripper = FrankaGripper(
            franka_env.model,
            franka_env.data,
            "franka",
            grasp_manager=gm,
        )
        arm = create_franka_arm(
            franka_env,
            gripper=gripper,
            grasp_manager=gm,
        )
        assert arm.gripper is gripper
        assert arm.grasp_manager is gm
        assert isinstance(arm.gripper, Gripper)


# ---------------------------------------------------------------------------
# GraspVerifier integration: real FrankaGripper + fake load signals
#
# This verifies that _BaseGripper.is_holding / held_object correctly route
# through the verifier when one is attached, exercising the routing change
# from personalrobotics/mj_manipulator#93 on a real gripper class (not a
# stub). The signal values themselves are fake so the test is fast and
# deterministic; physics-loop grasp validation happens end-to-end in the
# recycling demo on geodude.
# ---------------------------------------------------------------------------


class _FakeSignal:
    """Minimal LoadSignal stub for integration tests."""

    def __init__(self, name: str, value: float | None):
        self.name = name
        self.value = value

    def read(self) -> float | None:
        return self.value


class TestFrankaGripperWithGraspVerifier:
    def _attach_verifier(self, gripper: FrankaGripper) -> tuple:
        """Wire up a GraspVerifier with one fake signal and ensure the
        gripper is open so the ``empty_at_fully_closed=True`` branch
        doesn't immediately short-circuit is_held to False.

        Returns (verifier, signal).
        """
        from mj_manipulator.grasp_verifier import GraspVerifier

        gripper.kinematic_open()
        signal = _FakeSignal("wrist_ft_force", value=10.0)
        verifier = GraspVerifier(gripper=gripper, signals=[signal])
        gripper.grasp_verifier = verifier
        return verifier, signal

    def test_is_holding_routes_through_verifier(self, franka_gripper):
        """When a verifier is attached, is_holding should reflect
        verifier.is_held, not GraspManager bookkeeping."""
        verifier, _ = self._attach_verifier(franka_gripper)
        assert franka_gripper.is_holding is False
        verifier.mark_grasped("fake_cube")
        assert franka_gripper.is_holding is True

    def test_held_object_routes_through_verifier(self, franka_gripper):
        """held_object should return what the verifier says, not what
        GraspManager thinks."""
        verifier, _ = self._attach_verifier(franka_gripper)
        assert franka_gripper.held_object is None
        verifier.mark_grasped("fake_cube")
        assert franka_gripper.held_object == "fake_cube"

    def test_signal_collapse_flips_held_object_to_none(self, franka_gripper):
        """Regression for geodude#173: if the load signal collapses,
        held_object should go to None even though mark_grasped was called.
        This is the whole point of the verifier vs. bookkeeping."""
        verifier, signal = self._attach_verifier(franka_gripper)
        verifier.mark_grasped("fake_cube")
        assert franka_gripper.held_object == "fake_cube"
        signal.value = 0.5  # load collapsed
        assert franka_gripper.held_object is None
        assert franka_gripper.is_holding is False

    def test_mark_released_clears_routing(self, franka_gripper):
        """After release, the verifier should report empty."""
        verifier, _ = self._attach_verifier(franka_gripper)
        verifier.mark_grasped("fake_cube")
        verifier.mark_released()
        assert franka_gripper.is_holding is False
        assert franka_gripper.held_object is None

    def test_verifier_falls_back_without_attachment(self, franka_env):
        """Sanity: the default path (no verifier) is unchanged — the
        legacy GraspManager-based is_holding still works."""
        gm = GraspManager(franka_env.model, franka_env.data)
        gripper = FrankaGripper(
            franka_env.model,
            franka_env.data,
            "franka",
            grasp_manager=gm,
        )
        assert gripper.grasp_verifier is None
        assert gripper.is_holding is False
        # Simulate legacy bookkeeping path
        gm.mark_grasped("legacy_cube", "franka")
        assert gripper.is_holding is True
        assert gripper.held_object == "legacy_cube"
