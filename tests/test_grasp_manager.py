# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for GraspManager.

Uses a minimal inline MuJoCo model with a 2-DOF arm and a freejoint box
so tests run without external robot assets.
"""

import mujoco
import numpy as np
import pytest

from mj_manipulator.grasp_manager import GraspManager, find_contacted_object

# Minimal MuJoCo model: 2-joint arm + freejoint box
_MINIMAL_XML = """
<mujoco model="test_grasp">
  <worldbody>
    <body name="link1" pos="0 0 0.5">
      <joint name="joint1" type="hinge" axis="0 0 1"/>
      <geom name="link1_geom" type="capsule" size="0.04" fromto="0 0 0 0.3 0 0"/>
      <body name="link2" pos="0.3 0 0">
        <joint name="joint2" type="hinge" axis="0 0 1"/>
        <geom name="link2_geom" type="capsule" size="0.04" fromto="0 0 0 0.3 0 0"/>
        <body name="gripper_base" pos="0.3 0 0">
          <geom name="gripper_base_geom" type="box" size="0.02 0.04 0.02"/>
          <body name="left_finger" pos="0 0.03 0">
            <geom name="left_finger_geom" type="box" size="0.01 0.01 0.02"/>
          </body>
          <body name="right_finger" pos="0 -0.03 0">
            <geom name="right_finger_geom" type="box" size="0.01 0.01 0.02"/>
          </body>
        </body>
      </body>
    </body>
    <body name="box1" pos="0.5 0 0.5">
      <joint name="box1_free" type="free"/>
      <geom name="box1_geom" type="box" size="0.03 0.03 0.03"/>
    </body>
    <body name="box2" pos="0.8 0 0.5">
      <joint name="box2_free" type="free"/>
      <geom name="box2_geom" type="box" size="0.03 0.03 0.03"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def model_and_data():
    """Create a minimal MuJoCo model + data for testing."""
    model = mujoco.MjModel.from_xml_string(_MINIMAL_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


class TestGraspManager:
    """Tests for GraspManager grasp state tracking."""

    def test_init_empty(self, model_and_data):
        """Initializes with empty grasp state."""
        model, data = model_and_data
        gm = GraspManager(model, data)
        assert gm.grasped == {}
        assert gm._attachments == {}

    def test_mark_grasped(self, model_and_data):
        """mark_grasped tracks state correctly."""
        model, data = model_and_data
        gm = GraspManager(model, data)

        gm.mark_grasped("box1", "right")
        assert gm.is_grasped("box1")
        assert gm.get_holder("box1") == "right"
        assert not gm.is_grasped("box2")

    def test_mark_released(self, model_and_data):
        """mark_released clears state."""
        model, data = model_and_data
        gm = GraspManager(model, data)

        gm.mark_grasped("box1", "right")
        gm.mark_released("box1")
        assert not gm.is_grasped("box1")
        assert gm.get_holder("box1") is None

    def test_mark_grasped_idempotent(self, model_and_data):
        """Marking same object twice is a no-op."""
        model, data = model_and_data
        gm = GraspManager(model, data)

        gm.mark_grasped("box1", "right")
        gm.mark_grasped("box1", "right")
        assert len(gm.grasped) == 1

    def test_mark_released_when_not_grasped(self, model_and_data):
        """Releasing non-grasped object is a no-op."""
        model, data = model_and_data
        gm = GraspManager(model, data)
        gm.mark_released("box1")  # Should not raise

    def test_get_grasped_by(self, model_and_data):
        """get_grasped_by filters by arm."""
        model, data = model_and_data
        gm = GraspManager(model, data)

        gm.mark_grasped("box1", "right")
        gm.mark_grasped("box2", "left")

        assert gm.get_grasped_by("right") == ["box1"]
        assert gm.get_grasped_by("left") == ["box2"]
        assert gm.get_grasped_by("franka") == []

    def test_attach_detach(self, model_and_data):
        """attach_object / detach_object lifecycle."""
        model, data = model_and_data
        gm = GraspManager(model, data)

        gm.attach_object("box1", "right_finger")
        assert gm.is_attached("box1")
        assert gm.get_attachment_body("box1") == "right_finger"
        assert "box1" in gm.get_attached_objects()

        gm.detach_object("box1")
        assert not gm.is_attached("box1")
        assert gm.get_attachment_body("box1") is None

    def test_attach_computes_relative_transform(self, model_and_data):
        """attach_object stores correct relative transform."""
        model, data = model_and_data
        gm = GraspManager(model, data)

        gm.attach_object("box1", "right_finger")
        _, T_rel = gm._attachments["box1"]

        # Verify it's a valid 4x4 transform
        assert T_rel.shape == (4, 4)
        assert T_rel[3, 3] == pytest.approx(1.0)
        np.testing.assert_array_almost_equal(T_rel[3, :3], [0, 0, 0])

    def test_update_attached_poses_moves_object(self, model_and_data):
        """update_attached_poses moves object with gripper."""
        model, data = model_and_data
        gm = GraspManager(model, data)

        # Record initial object position
        box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box1")
        pos_before = data.xpos[box_id].copy()

        # Attach object to gripper
        gm.attach_object("box1", "right_finger")

        # Move arm (change joint angles)
        j1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint1")
        data.qpos[model.jnt_qposadr[j1_id]] = 0.5
        mujoco.mj_forward(model, data)

        # Update attached poses
        gm.update_attached_poses()
        mujoco.mj_forward(model, data)

        pos_after = data.xpos[box_id].copy()

        # Object should have moved
        assert not np.allclose(pos_before, pos_after, atol=0.01)

    def test_update_attached_poses_temp_data(self, model_and_data):
        """update_attached_poses works with temporary MjData."""
        model, data = model_and_data
        gm = GraspManager(model, data)

        gm.attach_object("box1", "right_finger")

        # Create temp data and verify it doesn't modify live data
        temp_data = mujoco.MjData(model)
        temp_data.qpos[:] = data.qpos
        mujoco.mj_forward(model, temp_data)

        box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box1")
        live_pos = data.xpos[box_id].copy()

        # Modify joint in temp data and update
        j1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint1")
        temp_data.qpos[model.jnt_qposadr[j1_id]] = 1.0
        mujoco.mj_forward(model, temp_data)
        gm.update_attached_poses(temp_data)

        # Live data should be unchanged
        np.testing.assert_array_equal(data.xpos[box_id], live_pos)

    def test_set_body_pose_requires_freejoint(self, model_and_data):
        """Setting pose on non-freejoint body raises ValueError."""
        model, data = model_and_data
        gm = GraspManager(model, data)

        with pytest.raises(ValueError, match="freejoint"):
            gm._set_body_pose_in_data("link1", np.eye(4), data)

    def test_body_not_found_raises(self, model_and_data):
        """Referencing non-existent body raises ValueError."""
        model, data = model_and_data
        gm = GraspManager(model, data)

        with pytest.raises(ValueError, match="not found"):
            gm._get_body_pose("nonexistent")


class TestFindContactedObject:
    """Tests for find_contacted_object function."""

    def test_empty_gripper_bodies_returns_none(self, model_and_data):
        """Empty gripper body list returns None."""
        model, data = model_and_data
        assert find_contacted_object(model, data, []) is None

    def test_no_contacts_returns_none(self, model_and_data):
        """No contacts returns None."""
        model, data = model_and_data
        j1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint1")
        data.qpos[model.jnt_qposadr[j1_id]] = 3.14
        mujoco.mj_forward(model, data)
        result = find_contacted_object(model, data, ["left_finger", "right_finger"])
        assert result is None

    def test_candidate_filter(self, model_and_data):
        """candidate_objects filters to specific objects."""
        model, data = model_and_data
        result = find_contacted_object(model, data, ["left_finger", "right_finger"], candidate_objects=["nonexistent"])
        assert result is None

    def test_returns_string_or_none(self, model_and_data):
        """Result is a body name string or None."""
        model, data = model_and_data
        result = find_contacted_object(model, data, ["left_finger", "right_finger"])
        assert result is None or isinstance(result, str)
