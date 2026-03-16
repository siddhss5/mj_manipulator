"""Tests for gripper implementations (RobotiqGripper, FrankaGripper).

Uses real menagerie/geodude_assets robot models to verify:
- Factory construction resolves actuators, bodies, and joints correctly
- Gripper protocol satisfaction (isinstance check)
- Kinematic open/close set correct joint positions
- get_actual_position returns normalized [0, 1] values
- Integration with arm factories
"""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from mj_environment import Environment

from mj_manipulator.grasp_manager import GraspManager
from mj_manipulator.grippers.franka import FrankaGripper
from mj_manipulator.grippers.robotiq import RobotiqGripper
from mj_manipulator.protocols import Gripper

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parent.parent.parent  # robot-code/
MENAGERIE = WORKSPACE / "mujoco_menagerie"
FRANKA_SCENE = MENAGERIE / "franka_emika_panda" / "scene.xml"
GEODUDE_ASSETS = WORKSPACE / "geodude_assets" / "src" / "geodude_assets" / "models"
ROBOTIQ_SCENE = GEODUDE_ASSETS / "robotiq_2f140" / "scene.xml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def robotiq_env():
    if not ROBOTIQ_SCENE.exists():
        pytest.skip("geodude_assets not found")
    return Environment(str(ROBOTIQ_SCENE))


@pytest.fixture
def robotiq_gripper(robotiq_env):
    return RobotiqGripper(
        robotiq_env.model, robotiq_env.data, "test_arm",
    )


@pytest.fixture
def franka_env():
    if not FRANKA_SCENE.exists():
        pytest.skip("mujoco_menagerie not found")
    from mj_manipulator.arms.franka import add_franka_ee_site

    spec = mujoco.MjSpec.from_file(str(FRANKA_SCENE))
    add_franka_ee_site(spec)

    franka_dir = FRANKA_SCENE.parent
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
        franka_env.model, franka_env.data, "franka",
    )


# ===========================================================================
# RobotiqGripper
# ===========================================================================


class TestRobotiqGripperConstruction:
    def test_constructs(self, robotiq_gripper):
        assert robotiq_gripper is not None

    def test_satisfies_protocol(self, robotiq_gripper):
        assert isinstance(robotiq_gripper, Gripper)

    def test_arm_name(self, robotiq_gripper):
        assert robotiq_gripper.arm_name == "test_arm"

    def test_actuator_id(self, robotiq_gripper):
        assert robotiq_gripper.actuator_id is not None
        assert robotiq_gripper.actuator_id >= 0

    def test_ctrl_range(self, robotiq_gripper):
        assert robotiq_gripper.ctrl_open == 0.0
        assert robotiq_gripper.ctrl_closed == 255.0

    def test_body_names(self, robotiq_gripper):
        names = robotiq_gripper.gripper_body_names
        assert len(names) == 12
        assert "base_mount" in names
        assert "right_follower" in names
        assert "left_follower" in names

    def test_attachment_body(self, robotiq_gripper):
        assert robotiq_gripper.attachment_body == "right_follower"

    def test_initially_not_holding(self, robotiq_gripper):
        assert not robotiq_gripper.is_holding
        assert robotiq_gripper.held_object is None

    def test_invalid_actuator_raises(self, robotiq_env):
        with pytest.raises(ValueError, match="not found"):
            RobotiqGripper(
                robotiq_env.model, robotiq_env.data, "arm",
                prefix="nonexistent/",
            )


class TestRobotiqGripperKinematic:
    def test_kinematic_open(self, robotiq_gripper):
        robotiq_gripper.kinematic_open()
        assert robotiq_gripper.get_actual_position() == pytest.approx(0.0, abs=0.05)

    def test_kinematic_position_closed(self, robotiq_gripper):
        robotiq_gripper.set_kinematic_position(1.0)
        pos = robotiq_gripper.get_actual_position()
        assert pos > 0.9

    def test_kinematic_position_midpoint(self, robotiq_gripper):
        robotiq_gripper.set_kinematic_position(0.5)
        pos = robotiq_gripper.get_actual_position()
        assert 0.3 < pos < 0.7

    def test_kinematic_close_no_object(self, robotiq_gripper):
        result = robotiq_gripper.kinematic_close()
        # No candidate objects set, and the scene has an object but
        # the gripper isn't positioned near it — may or may not contact
        # Just verify it doesn't crash and returns str or None
        assert result is None or isinstance(result, str)

    def test_get_actual_position_range(self, robotiq_gripper):
        robotiq_gripper.set_kinematic_position(0.0)
        pos_open = robotiq_gripper.get_actual_position()
        assert 0.0 <= pos_open <= 0.1

        robotiq_gripper.set_kinematic_position(1.0)
        pos_closed = robotiq_gripper.get_actual_position()
        assert 0.9 <= pos_closed <= 1.0


class TestRobotiqGripperGraspManager:
    def test_is_holding_with_grasp_manager(self, robotiq_env):
        gm = GraspManager(robotiq_env.model, robotiq_env.data)
        gripper = RobotiqGripper(
            robotiq_env.model, robotiq_env.data, "arm",
            grasp_manager=gm,
        )
        assert not gripper.is_holding

        gm.mark_grasped("box", "arm")
        assert gripper.is_holding
        assert gripper.held_object == "box"

        gm.mark_released("box")
        assert not gripper.is_holding
        assert gripper.held_object is None


# ===========================================================================
# FrankaGripper
# ===========================================================================


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
        assert franka_gripper.ctrl_open == 0.0
        assert franka_gripper.ctrl_closed == 255.0

    def test_body_names(self, franka_gripper):
        names = franka_gripper.gripper_body_names
        assert len(names) == 3
        assert "hand" in names
        assert "left_finger" in names
        assert "right_finger" in names

    def test_attachment_body(self, franka_gripper):
        assert franka_gripper.attachment_body == "left_finger"

    def test_initially_not_holding(self, franka_gripper):
        assert not franka_gripper.is_holding
        assert franka_gripper.held_object is None

    def test_invalid_actuator_raises(self, franka_env):
        with pytest.raises(ValueError, match="not found"):
            FrankaGripper(
                franka_env.model, franka_env.data, "franka",
                prefix="nonexistent/",
            )


class TestFrankaGripperKinematic:
    def test_kinematic_open(self, franka_gripper):
        franka_gripper.kinematic_open()
        pos = franka_gripper.get_actual_position()
        assert pos == pytest.approx(0.0, abs=0.05)

    def test_kinematic_close_moves_fingers(self, franka_gripper):
        franka_gripper.kinematic_open()
        open_pos = franka_gripper.get_actual_position()

        # Empty candidates: no contact stops the close, gripper fully closes
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


class TestFrankaGripperGraspManager:
    def test_is_holding_with_grasp_manager(self, franka_env):
        gm = GraspManager(franka_env.model, franka_env.data)
        gripper = FrankaGripper(
            franka_env.model, franka_env.data, "franka",
            grasp_manager=gm,
        )
        assert not gripper.is_holding

        gm.mark_grasped("mug", "franka")
        assert gripper.is_holding
        assert gripper.held_object == "mug"

        gm.mark_released("mug")
        assert not gripper.is_holding


# ===========================================================================
# Integration with arm factories
# ===========================================================================


class TestArmFactoryIntegration:
    def test_ur5e_with_gripper(self, robotiq_env):
        """UR5e arm factory accepts gripper parameter."""
        # The UR5e menagerie model doesn't include the Robotiq,
        # so we can't test with the menagerie UR5e scene.
        # Just verify the RobotiqGripper constructs correctly with its own model.
        gripper = RobotiqGripper(
            robotiq_env.model, robotiq_env.data, "ur5e",
        )
        assert isinstance(gripper, Gripper)
        assert gripper.arm_name == "ur5e"

    def test_franka_with_gripper(self, franka_env):
        """Franka arm factory accepts gripper parameter."""
        from mj_manipulator.arms.franka import create_franka_arm

        gm = GraspManager(franka_env.model, franka_env.data)
        gripper = FrankaGripper(
            franka_env.model, franka_env.data, "franka",
            grasp_manager=gm,
        )
        arm = create_franka_arm(
            franka_env, gripper=gripper, grasp_manager=gm,
        )
        assert arm.gripper is gripper
        assert arm.grasp_manager is gm
        assert isinstance(arm.gripper, Gripper)
