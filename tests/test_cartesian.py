"""Tests for Cartesian velocity control.

Tests the QP solver, Jacobian computation, step_twist, and contact detection
using real menagerie arm models (UR5e, Franka Panda).
"""

import mujoco
import numpy as np
import pytest

from mj_manipulator.cartesian import (
    CartesianControlConfig,
    CartesianController,
    TwistStepResult,
    check_arm_contact,
    check_arm_contact_after_move,
    check_gripper_contact,
    get_arm_body_ids,
    get_ee_jacobian,
    step_twist,
    twist_to_joint_velocity,
)

# Minimal inline model for unit tests
_CART_XML = """
<mujoco model="test_cartesian">
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05"/>
    <body name="link1" pos="0 0 0.5">
      <joint name="joint1" type="hinge" axis="0 0 1"
             range="-3.14 3.14"/>
      <geom name="link1_geom" type="capsule" size="0.04"
            fromto="0 0 0 0.3 0 0"/>
      <body name="link2" pos="0.3 0 0">
        <joint name="joint2" type="hinge" axis="0 1 0"
               range="-3.14 3.14"/>
        <geom name="link2_geom" type="capsule" size="0.04"
              fromto="0 0 0 0.3 0 0"/>
        <site name="ee_site" pos="0.3 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

JOINT_NAMES = ["joint1", "joint2"]


@pytest.fixture
def model_and_data():
    model = mujoco.MjModel.from_xml_string(_CART_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


@pytest.fixture
def arm_indices(model_and_data):
    """Return (joint_qpos_indices, joint_qvel_indices, ee_site_id)."""
    model, _ = model_and_data
    qpos_idx = []
    qvel_idx = []
    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_idx.append(model.jnt_qposadr[jid])
        qvel_idx.append(model.jnt_dofadr[jid])
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    return qpos_idx, qvel_idx, ee_id


class TestCartesianControlConfig:
    def test_defaults(self):
        cfg = CartesianControlConfig()
        assert cfg.length_scale == 0.1
        assert cfg.damping == 1e-4
        assert cfg.velocity_scale == 1.0

    def test_invalid_length_scale(self):
        with pytest.raises(ValueError, match="length_scale"):
            CartesianControlConfig(length_scale=-1)

    def test_invalid_damping(self):
        with pytest.raises(ValueError, match="damping"):
            CartesianControlConfig(damping=-1)

    def test_invalid_velocity_scale(self):
        with pytest.raises(ValueError, match="velocity_scale"):
            CartesianControlConfig(velocity_scale=0)
        with pytest.raises(ValueError, match="velocity_scale"):
            CartesianControlConfig(velocity_scale=1.5)


class TestGetEEJacobian:
    def test_shape(self, model_and_data, arm_indices):
        model, data = model_and_data
        _, qvel_idx, ee_id = arm_indices
        J = get_ee_jacobian(model, data, ee_id, qvel_idx)
        assert J.shape == (6, 2)

    def test_nonzero_at_home(self, model_and_data, arm_indices):
        model, data = model_and_data
        _, qvel_idx, ee_id = arm_indices
        J = get_ee_jacobian(model, data, ee_id, qvel_idx)
        # At home (zeros), the 2-DOF arm should have some nonzero Jacobian
        assert np.linalg.norm(J) > 0

    def test_changes_with_config(self, model_and_data, arm_indices):
        model, data = model_and_data
        qpos_idx, qvel_idx, ee_id = arm_indices
        J1 = get_ee_jacobian(model, data, ee_id, qvel_idx)
        # Change config
        data.qpos[qpos_idx[0]] = 1.0
        mujoco.mj_forward(model, data)
        J2 = get_ee_jacobian(model, data, ee_id, qvel_idx)
        assert not np.allclose(J1, J2)


class TestTwistToJointVelocity:
    def test_zero_twist_gives_zero_velocity(self):
        J = np.eye(6, 2)  # Simple Jacobian
        result = twist_to_joint_velocity(
            J=J,
            twist=np.zeros(6),
            q_current=np.zeros(2),
            q_min=-np.ones(2) * 3,
            q_max=np.ones(2) * 3,
            qd_max=np.ones(2) * 2,
            dt=0.01,
        )
        np.testing.assert_allclose(result.joint_velocities, 0, atol=1e-8)
        assert result.achieved_fraction == 1.0

    def test_linear_twist_produces_motion(self):
        # 2-joint arm with identity-like Jacobian
        J = np.zeros((6, 2))
        J[0, 0] = 1.0  # joint1 controls x velocity
        J[1, 1] = 1.0  # joint2 controls y velocity

        twist = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = twist_to_joint_velocity(
            J=J, twist=twist,
            q_current=np.zeros(2),
            q_min=-np.ones(2) * 3,
            q_max=np.ones(2) * 3,
            qd_max=np.ones(2) * 2,
            dt=0.01,
        )
        # Joint 1 should move, joint 2 shouldn't (much)
        assert abs(result.joint_velocities[0]) > 0.01
        assert abs(result.joint_velocities[1]) < abs(result.joint_velocities[0])
        assert result.achieved_fraction > 0.5

    def test_respects_velocity_limits(self):
        J = np.eye(6, 2)
        twist = np.array([100.0, 0, 0, 0, 0, 0])  # Very large twist
        result = twist_to_joint_velocity(
            J=J, twist=twist,
            q_current=np.zeros(2),
            q_min=-np.ones(2) * 3,
            q_max=np.ones(2) * 3,
            qd_max=np.ones(2) * 2,  # Max 2 rad/s
            dt=0.01,
        )
        assert np.all(np.abs(result.joint_velocities) <= 2.0 + 1e-6)
        assert result.limiting_factor == "velocity"

    def test_respects_position_limits(self):
        J = np.zeros((6, 2))
        J[0, 0] = 1.0

        # Joint 0 is very close to upper limit
        twist = np.array([1.0, 0, 0, 0, 0, 0])
        result = twist_to_joint_velocity(
            J=J, twist=twist,
            q_current=np.array([2.9, 0.0]),
            q_min=-np.ones(2) * 3,
            q_max=np.ones(2) * 3,
            qd_max=np.ones(2) * 10,
            dt=0.01,
        )
        # Joint 0 velocity should be limited by position constraint
        q_new = 2.9 + result.joint_velocities[0] * 0.01
        margin = np.deg2rad(5.0)
        assert q_new <= 3.0 - margin + 1e-6

    def test_warm_start(self):
        J = np.eye(6, 2)
        twist = np.array([0.1, 0, 0, 0, 0, 0])
        kwargs = dict(
            J=J, twist=twist,
            q_current=np.zeros(2),
            q_min=-np.ones(2) * 3,
            q_max=np.ones(2) * 3,
            qd_max=np.ones(2) * 2,
            dt=0.01,
        )
        r1 = twist_to_joint_velocity(**kwargs)
        r2 = twist_to_joint_velocity(**kwargs, q_dot_prev=r1.joint_velocities)
        # Warm start should give same result
        np.testing.assert_allclose(
            r1.joint_velocities, r2.joint_velocities, atol=1e-6
        )

    def test_result_type(self):
        J = np.eye(6, 2)
        result = twist_to_joint_velocity(
            J=J, twist=np.zeros(6),
            q_current=np.zeros(2),
            q_min=-np.ones(2) * 3,
            q_max=np.ones(2) * 3,
            qd_max=np.ones(2) * 2,
            dt=0.01,
        )
        assert isinstance(result, TwistStepResult)
        assert result.joint_velocities.shape == (2,)
        assert isinstance(result.twist_error, float)
        assert isinstance(result.achieved_fraction, float)


class TestStepTwist:
    def test_returns_new_positions(self, model_and_data, arm_indices):
        model, data = model_and_data
        qpos_idx, qvel_idx, ee_id = arm_indices

        q_new, result = step_twist(
            model, data, ee_id, qpos_idx, qvel_idx,
            q_min=-np.ones(2) * 3,
            q_max=np.ones(2) * 3,
            qd_max=np.ones(2) * 2,
            twist=np.array([0.01, 0, 0, 0, 0, 0]),
            dt=0.01,
        )
        assert q_new.shape == (2,)
        assert isinstance(result, TwistStepResult)

    def test_zero_twist_gives_same_position(self, model_and_data, arm_indices):
        model, data = model_and_data
        qpos_idx, qvel_idx, ee_id = arm_indices
        q_before = np.array([data.qpos[i] for i in qpos_idx])

        q_new, result = step_twist(
            model, data, ee_id, qpos_idx, qvel_idx,
            q_min=-np.ones(2) * 3,
            q_max=np.ones(2) * 3,
            qd_max=np.ones(2) * 2,
            twist=np.zeros(6),
            dt=0.01,
        )
        np.testing.assert_allclose(q_new, q_before, atol=1e-8)

    def test_hand_frame(self, model_and_data, arm_indices):
        model, data = model_and_data
        qpos_idx, qvel_idx, ee_id = arm_indices

        # Should not crash with hand frame
        q_new, result = step_twist(
            model, data, ee_id, qpos_idx, qvel_idx,
            q_min=-np.ones(2) * 3,
            q_max=np.ones(2) * 3,
            qd_max=np.ones(2) * 2,
            twist=np.array([0.01, 0, 0, 0, 0, 0]),
            frame="hand",
            dt=0.01,
        )
        assert q_new.shape == (2,)


class TestContactDetection:
    def test_check_gripper_contact_no_contact(self, model_and_data):
        model, data = model_and_data
        result = check_gripper_contact(model, data, ["link2"])
        # At home config, link2 shouldn't be touching floor
        # (it's at 0.5m height)
        assert result is None

    def test_get_arm_body_ids(self, model_and_data):
        model, _ = model_and_data
        ids = get_arm_body_ids(model, JOINT_NAMES)
        assert len(ids) == 2  # link1 and link2

    def test_get_arm_body_ids_with_gripper(self, model_and_data):
        model, _ = model_and_data
        ids = get_arm_body_ids(model, JOINT_NAMES, gripper_body_names=["link2"])
        assert len(ids) == 2  # link2 was already included

    def test_check_arm_contact_no_collision(self, model_and_data):
        model, data = model_and_data
        arm_ids = get_arm_body_ids(model, JOINT_NAMES)
        result = check_arm_contact(model, data, arm_ids)
        assert result is None

    def test_check_arm_contact_after_move(self, model_and_data, arm_indices):
        model, data = model_and_data
        qpos_idx, _, _ = arm_indices
        arm_ids = get_arm_body_ids(model, JOINT_NAMES)

        # Moving to home should be safe
        result = check_arm_contact_after_move(
            model, data, arm_ids, qpos_idx, np.zeros(2)
        )
        assert result is None

    def test_check_arm_contact_after_move_restores_state(
        self, model_and_data, arm_indices
    ):
        model, data = model_and_data
        qpos_idx, _, _ = arm_indices
        arm_ids = get_arm_body_ids(model, JOINT_NAMES)

        q_before = np.array([data.qpos[i] for i in qpos_idx])
        check_arm_contact_after_move(
            model, data, arm_ids, qpos_idx, np.array([1.0, 1.0])
        )
        q_after = np.array([data.qpos[i] for i in qpos_idx])
        np.testing.assert_allclose(q_before, q_after)


class TestWithRealModels:
    """Test QP solver and Jacobian with menagerie models if available."""

    @pytest.fixture
    def ur5e(self):
        try:
            from mj_manipulator.menagerie import menagerie_scene
            scene = menagerie_scene("universal_robots_ur5e")
        except FileNotFoundError:
            pytest.skip("mujoco_menagerie not available")
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        # Set home
        home = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0]
        joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                   "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        qpos_idx, qvel_idx = [], []
        for i, name in enumerate(joints):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_idx.append(model.jnt_qposadr[jid])
            qvel_idx.append(model.jnt_dofadr[jid])
            data.qpos[model.jnt_qposadr[jid]] = home[i]
        ee_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )
        mujoco.mj_forward(model, data)
        return model, data, qpos_idx, qvel_idx, ee_id

    @pytest.fixture
    def franka(self):
        try:
            from mj_manipulator.menagerie import menagerie_scene
            scene = menagerie_scene("franka_emika_panda")
        except FileNotFoundError:
            pytest.skip("mujoco_menagerie not available")
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        home = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]
        joints = [f"joint{i}" for i in range(1, 8)]
        qpos_idx, qvel_idx = [], []
        for i, name in enumerate(joints):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_idx.append(model.jnt_qposadr[jid])
            qvel_idx.append(model.jnt_dofadr[jid])
            data.qpos[model.jnt_qposadr[jid]] = home[i]
        # Franka doesn't have an attachment_site, use a body-based approach
        # We'll find the hand body site or just skip if not available
        # Actually let's use the wrist site approach
        mujoco.mj_forward(model, data)
        return model, data, qpos_idx, qvel_idx

    def test_ur5e_jacobian(self, ur5e):
        model, data, qpos_idx, qvel_idx, ee_id = ur5e
        J = get_ee_jacobian(model, data, ee_id, qvel_idx)
        assert J.shape == (6, 6)
        # UR5e at home should have full rank Jacobian
        rank = np.linalg.matrix_rank(J, tol=1e-6)
        assert rank == 6

    def test_ur5e_step_twist(self, ur5e):
        model, data, qpos_idx, qvel_idx, ee_id = ur5e
        q_new, result = step_twist(
            model, data, ee_id, qpos_idx, qvel_idx,
            q_min=-np.ones(6) * 6.28,
            q_max=np.ones(6) * 6.28,
            qd_max=np.ones(6) * 2.0,
            twist=np.array([0.05, 0, 0, 0, 0, 0]),  # 5cm/s in X
            dt=0.004,
        )
        assert q_new.shape == (6,)
        assert result.achieved_fraction > 0.8

    def test_ur5e_twist_hand_frame(self, ur5e):
        model, data, qpos_idx, qvel_idx, ee_id = ur5e
        # Use Y-direction twist — the UR5e attachment_site has a rotated
        # frame (quat="-1 1 0 0"), so Y in hand frame maps to a different
        # world direction than Y in world frame.
        twist_y = np.array([0, 0.05, 0, 0, 0, 0])
        q_world, _ = step_twist(
            model, data, ee_id, qpos_idx, qvel_idx,
            q_min=-np.ones(6) * 6.28,
            q_max=np.ones(6) * 6.28,
            qd_max=np.ones(6) * 2.0,
            twist=twist_y,
            frame="world",
            dt=0.004,
        )
        q_hand, _ = step_twist(
            model, data, ee_id, qpos_idx, qvel_idx,
            q_min=-np.ones(6) * 6.28,
            q_max=np.ones(6) * 6.28,
            qd_max=np.ones(6) * 2.0,
            twist=twist_y,
            frame="hand",
            dt=0.004,
        )
        # World and hand frame should give different results for Y-twist
        # because the attachment_site is rotated relative to world
        assert not np.allclose(q_world, q_hand, atol=1e-6)

    def test_franka_jacobian(self, franka):
        model, data, qpos_idx, qvel_idx = franka
        # Use a body site — add one dynamically isn't easy, use mj_jac instead
        # For test, just verify Jacobian with a specific body
        # Actually, let's compute body Jacobian directly
        hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, hand_id)
        J = np.vstack([jacp[:, qvel_idx], jacr[:, qvel_idx]])
        assert J.shape == (6, 7)
        rank = np.linalg.matrix_rank(J, tol=1e-6)
        assert rank == 6  # 7-DOF arm has rank-6 Jacobian (redundant)

    def test_franka_qp_solver(self, franka):
        model, data, qpos_idx, qvel_idx = franka
        # Get Jacobian for hand
        hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, hand_id)
        J = np.vstack([jacp[:, qvel_idx], jacr[:, qvel_idx]])

        q_current = np.array([data.qpos[i] for i in qpos_idx])
        result = twist_to_joint_velocity(
            J=J,
            twist=np.array([0.0, 0.0, -0.05, 0, 0, 0]),  # Move down
            q_current=q_current,
            q_min=-np.ones(7) * 2.9,
            q_max=np.ones(7) * 2.9,
            qd_max=np.ones(7) * 2.0,
            dt=0.004,
        )
        assert result.joint_velocities.shape == (7,)
        assert result.achieved_fraction > 0.5


class TestCartesianController:
    """Tests for CartesianController class."""

    @pytest.fixture
    def controller(self, model_and_data, arm_indices):
        model, data = model_and_data
        qpos_idx, qvel_idx, ee_id = arm_indices
        q_min = np.array([-3.14, -3.14])
        q_max = np.array([3.14, 3.14])
        qd_max = np.array([2.0, 2.0])
        return CartesianController(
            model, data, ee_id, qpos_idx, qvel_idx, q_min, q_max, qd_max
        )

    def test_step_returns_result(self, controller):
        result = controller.step(twist=np.array([0.05, 0, 0, 0, 0, 0]), dt=0.004)
        assert isinstance(result, TwistStepResult)
        assert result.joint_velocities.shape == (2,)
        assert 0.0 <= result.achieved_fraction <= 1.0

    def test_step_applies_to_qpos(self, model_and_data, controller):
        # joint1 (z-axis) can achieve y linear velocity; x is unreachable
        model, data = model_and_data
        q_before = data.qpos.copy()
        controller.step(twist=np.array([0, 0.05, 0, 0, 0, 0]), dt=0.004)
        assert not np.allclose(data.qpos, q_before)

    def test_reset_clears_warm_start(self, controller):
        controller.step(twist=np.array([0.05, 0, 0, 0, 0, 0]), dt=0.004)
        assert controller._q_dot_prev is not None
        controller.reset()
        assert controller._q_dot_prev is None

    def test_move_terminates_by_distance(self, model_and_data, arm_indices):
        # Large L reduces angular cost so joint1 can achieve y-velocity
        model, data = model_and_data
        qpos_idx, qvel_idx, ee_id = arm_indices
        controller = CartesianController(
            model, data, ee_id, qpos_idx, qvel_idx,
            q_min=np.array([-3.14, -3.14]), q_max=np.array([3.14, 3.14]),
            qd_max=np.array([2.0, 2.0]),
            config=CartesianControlConfig(length_scale=10.0, min_progress=0.0),
        )
        result = controller.move(
            twist=np.array([0, 0.05, 0, 0, 0, 0]),
            dt=0.004,
            max_distance=0.01,
        )
        assert result.terminated_by == "distance"
        assert result.distance_moved >= 0.01

    def test_move_terminates_by_duration(self, controller):
        result = controller.move(
            twist=np.array([0.0, 0, 0, 0, 0, 0]),  # zero twist = no progress
            dt=0.004,
            max_duration=0.02,
        )
        # Zero twist hits min_progress threshold or duration
        assert result.terminated_by in ("duration", "no_progress")

    def test_move_terminates_by_condition(self, model_and_data, arm_indices):
        model, data = model_and_data
        qpos_idx, qvel_idx, ee_id = arm_indices
        controller = CartesianController(
            model, data, ee_id, qpos_idx, qvel_idx,
            q_min=np.array([-3.14, -3.14]), q_max=np.array([3.14, 3.14]),
            qd_max=np.array([2.0, 2.0]),
            config=CartesianControlConfig(min_progress=0.0),
        )
        called = [0]

        def stop():
            called[0] += 1
            return called[0] >= 3

        result = controller.move(
            twist=np.array([0, 0.05, 0, 0, 0, 0]),
            dt=0.004,
            max_duration=5.0,
            stop_condition=stop,
        )
        assert result.terminated_by == "condition"

    def test_move_to_converges(self, model_and_data, arm_indices):
        """move_to should reach a nearby target pose."""
        model, data = model_and_data
        qpos_idx, qvel_idx, ee_id = arm_indices
        # Large L reduces angular penalty so joint1 can achieve y-translation
        controller = CartesianController(
            model, data, ee_id, qpos_idx, qvel_idx,
            q_min=np.array([-3.14, -3.14]), q_max=np.array([3.14, 3.14]),
            qd_max=np.array([2.0, 2.0]),
            config=CartesianControlConfig(length_scale=10.0, min_progress=0.0),
        )

        mujoco.mj_forward(model, data)
        target = np.eye(4)
        target[:3, :3] = data.site_xmat[ee_id].reshape(3, 3)
        target[:3, 3] = data.site_xpos[ee_id] + np.array([0, 0.02, 0])

        result = controller.move_to(
            target, dt=0.004, max_duration=10.0, speed=0.05,
            position_tol=0.005, rotation_tol=0.1,
        )
        assert result.terminated_by == "condition"
        pos_err = np.linalg.norm(target[:3, 3] - data.site_xpos[ee_id])
        assert pos_err < 0.01

    def test_move_until_contact_terminates_by_max_distance(self, model_and_data, arm_indices):
        """Without contact, move_until_contact stops at max_distance."""
        from unittest.mock import patch

        model, data = model_and_data
        qpos_idx, qvel_idx, ee_id = arm_indices
        controller = CartesianController(
            model, data, ee_id, qpos_idx, qvel_idx,
            q_min=np.array([-3.14, -3.14]), q_max=np.array([3.14, 3.14]),
            qd_max=np.array([2.0, 2.0]),
            config=CartesianControlConfig(length_scale=10.0, min_progress=0.0),
        )

        # Patch check_gripper_contact to always return None (no contact)
        with patch("mj_manipulator.cartesian.check_gripper_contact", return_value=None):
            result = controller.move_until_contact(
                twist=np.array([0, 0.05, 0, 0, 0, 0]),
                dt=0.004,
                gripper_body_names=["link2"],
                max_distance=0.01,
            )

        assert not result.success
        assert result.terminated_by == "max_distance"
        assert result.distance_moved >= 0.01
        assert result.contact_geom is None

    def test_move_until_contact_stops_on_contact(self, model_and_data, arm_indices):
        """move_until_contact returns success=True when contact is detected."""
        from unittest.mock import patch

        model, data = model_and_data
        qpos_idx, qvel_idx, ee_id = arm_indices
        controller = CartesianController(
            model, data, ee_id, qpos_idx, qvel_idx,
            q_min=np.array([-3.14, -3.14]), q_max=np.array([3.14, 3.14]),
            qd_max=np.array([2.0, 2.0]),
            config=CartesianControlConfig(length_scale=10.0, min_progress=0.0),
        )

        call_count = [0]

        def fake_contact(model, data, body_names):
            call_count[0] += 1
            # Return a contact after 3 steps so the controller has moved first
            return "obstacle" if call_count[0] >= 3 else None

        with patch("mj_manipulator.cartesian.check_gripper_contact", side_effect=fake_contact):
            result = controller.move_until_contact(
                twist=np.array([0, 0.05, 0, 0, 0, 0]),
                dt=0.004,
                gripper_body_names=["link2"],
                max_distance=1.0,
            )

        assert result.success
        assert result.terminated_by == "contact"
        assert result.contact_geom == "obstacle"
