# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for PhysicsController.

Uses shared test fixtures from conftest.py (MockArm, model_and_data).
"""

import numpy as np
import pytest
from conftest import MockArm, make_trajectory

from mj_manipulator.config import PhysicsExecutionConfig
from mj_manipulator.physics_controller import (
    ArmPhysicsExecutor,
    PhysicsController,
)
from mj_manipulator.trajectory import Trajectory


@pytest.fixture
def controller(model_and_data, mock_arm):
    model, data = model_and_data
    return PhysicsController(
        model,
        data,
        {"test_arm": mock_arm},
        config=PhysicsExecutionConfig(control_dt=0.002),
    )


class TestPhysicsControllerConstruction:
    def test_constructs(self, model_and_data, mock_arm):
        model, data = model_and_data
        ctrl = PhysicsController(model, data, {"arm": mock_arm})
        assert ctrl is not None
        assert ctrl.steps_per_control == 4  # 0.008 / 0.002

    def test_constructs_with_initial_positions(self, model_and_data, mock_arm):
        model, data = model_and_data
        init_pos = {"arm": np.array([0.5, -0.5])}
        PhysicsController(model, data, {"arm": mock_arm}, initial_positions=init_pos)
        # qpos should be set to initial positions
        for i, idx in enumerate(mock_arm.joint_qpos_indices):
            assert abs(data.qpos[idx] - init_pos["arm"][i]) < 1e-6

    def test_initializes_ctrl_to_targets(self, model_and_data, mock_arm):
        model, data = model_and_data
        init_pos = {"arm": np.array([1.0, -1.0])}
        PhysicsController(model, data, {"arm": mock_arm}, initial_positions=init_pos)
        # Actuator ctrl should match initial positions
        np.testing.assert_allclose(
            [data.ctrl[aid] for aid in mock_arm.actuator_ids],
            [1.0, -1.0],
        )


class TestPhysicsControllerStepping:
    def test_step_applies_control(self, controller, model_and_data):
        _, data = model_and_data
        controller.set_arm_target("test_arm", np.array([0.5, -0.5]))
        controller.step()
        # Actuators should have been commanded
        assert data.ctrl[controller._arms["test_arm"].actuator_ids[0]] != 0.0

    def test_hold_all(self, controller, model_and_data):
        _, data = model_and_data
        # Set some position
        for idx in controller._arms["test_arm"].joint_qpos_indices:
            data.qpos[idx] = 0.42
        controller.hold_all()
        np.testing.assert_allclose(
            controller._arms["test_arm"].target_position,
            [0.42, 0.42],
        )

    def test_set_arm_target(self, controller):
        controller.set_arm_target("test_arm", np.array([1.0, 2.0]))
        np.testing.assert_allclose(
            controller._arms["test_arm"].target_position,
            [1.0, 2.0],
        )

    def test_set_arm_target_with_velocity(self, controller):
        controller.set_arm_target(
            "test_arm",
            np.array([1.0, 2.0]),
            velocity=np.array([0.1, 0.2]),
        )
        np.testing.assert_allclose(
            controller._arms["test_arm"].target_velocity,
            [0.1, 0.2],
        )

    def test_set_arm_target_unknown_raises(self, controller):
        with pytest.raises(ValueError, match="Unknown arm"):
            controller.set_arm_target("nonexistent", np.array([0.0, 0.0]))

    def test_step_reactive(self, controller, model_and_data):
        _, data = model_and_data
        controller.step_reactive(
            "test_arm",
            np.array([0.5, -0.5]),
            np.array([0.1, 0.1]),
        )
        # Actuators should have been commanded with reactive lookahead
        # cmd = 0.5 + 2*0.002*0.1 = 0.5004
        expected = 0.5 + 2.0 * controller.control_dt * 0.1
        assert abs(data.ctrl[controller._arms["test_arm"].actuator_ids[0]] - expected) < 1e-6

    def test_step_reactive_unknown_raises(self, controller):
        with pytest.raises(ValueError, match="Unknown arm"):
            controller.step_reactive("nonexistent", np.array([0.0, 0.0]))


class TestPhysicsControllerExecution:
    @pytest.fixture
    def lenient_controller(self, model_and_data, mock_arm):
        """Controller with relaxed convergence for the simple test model.

        The minimal 2-link arm with kp=100 oscillates more than a real
        robot, so we use wider tolerances here.
        """
        model, data = model_and_data
        return PhysicsController(
            model,
            data,
            {"test_arm": mock_arm},
            config=PhysicsExecutionConfig(
                control_dt=0.002,
                position_tolerance=0.3,
                velocity_tolerance=1.0,
                convergence_timeout_steps=2000,
            ),
        )

    def test_execute_trajectory(self, lenient_controller):
        positions = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.1],
                [0.2, 0.2],
            ]
        )
        traj = make_trajectory(positions, entity="test_arm")
        result = lenient_controller.execute("test_arm", traj)
        assert result is True

    def test_execute_unknown_arm_raises(self, controller):
        positions = np.array([[0.0, 0.0]])
        traj = make_trajectory(positions, entity="test_arm")
        with pytest.raises(ValueError, match="Unknown arm"):
            controller.execute("nonexistent", traj)

    def test_execute_dof_mismatch_raises(self, controller):
        positions = np.array([[0.0, 0.0, 0.0]])  # 3 DOF, arm is 2
        traj = Trajectory(
            timestamps=np.array([0.0]),
            positions=positions,
            velocities=np.zeros_like(positions),
            accelerations=np.zeros_like(positions),
            joint_names=["j1", "j2", "j3"],
        )
        with pytest.raises(ValueError, match="DOF"):
            controller.execute("test_arm", traj)

    def test_get_executor(self, controller):
        executor = controller.get_executor("test_arm")
        assert isinstance(executor, ArmPhysicsExecutor)

    def test_get_executor_unknown_raises(self, controller):
        with pytest.raises(ValueError, match="Unknown arm"):
            controller.get_executor("nonexistent")

    def test_executor_delegates_to_controller(self, lenient_controller):
        executor = lenient_controller.get_executor("test_arm")
        positions = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.1],
            ]
        )
        traj = make_trajectory(positions, entity="test_arm")
        result = executor.execute(traj)
        assert result is True


class TestPhysicsControllerMultiArm:
    """Test with two arms to verify hold-while-execute behavior."""

    def test_two_arms_hold(self, model_and_data):
        """Second arm holds position while first executes."""
        model, data = model_and_data
        # Both "arms" share the same joints for simplicity,
        # but the test verifies the controller manages both.
        arm1 = MockArm("arm1", model, data)
        arm2 = MockArm("arm2", model, data)

        ctrl = PhysicsController(
            model,
            data,
            {"arm1": arm1, "arm2": arm2},
            config=PhysicsExecutionConfig(control_dt=0.002),
        )

        ctrl.set_arm_target("arm1", np.array([0.5, 0.5]))
        ctrl.set_arm_target("arm2", np.array([0.0, 0.0]))
        ctrl.step()

        # Both arms should have actuator commands set
        assert ctrl._arms["arm1"].target_position[0] == 0.5
        assert ctrl._arms["arm2"].target_position[0] == 0.0
