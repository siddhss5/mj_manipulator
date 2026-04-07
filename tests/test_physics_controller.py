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


class TestTrajectoryRunner:
    """Tests for non-blocking TrajectoryRunner."""

    @pytest.fixture
    def lenient_controller(self, model_and_data, mock_arm):
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

    def test_advance_writes_targets(self, controller):
        positions = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
        traj = make_trajectory(positions, entity="test_arm")
        future = controller.start_trajectory("test_arm", traj)

        assert controller.has_active_runner("test_arm")
        assert not future.done()

        # Advance first waypoint
        controller.advance_all()
        state = controller._arms["test_arm"]
        np.testing.assert_allclose(state.target_position, [0.0, 0.0])

        # Advance second
        controller.advance_all()
        np.testing.assert_allclose(state.target_position, [0.1, 0.1])

    def test_runner_completes(self, lenient_controller):
        positions = np.array([[0.0, 0.0], [0.1, 0.1]])
        traj = make_trajectory(positions, entity="test_arm")
        future = lenient_controller.start_trajectory("test_arm", traj)

        # Advance through all waypoints
        lenient_controller.advance_all()  # waypoint 0
        lenient_controller.advance_all()  # waypoint 1 → enters convergence

        # Step physics to let arm converge (convergence checks need mj_step)
        for _ in range(600):
            lenient_controller.advance_all()
            lenient_controller.step()

        assert future.done()
        assert not lenient_controller.has_active_runner("test_arm")

    def test_runner_abort(self, controller):
        positions = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
        traj = make_trajectory(positions, entity="test_arm")
        aborted = False

        def abort_fn():
            return aborted

        future = controller.start_trajectory("test_arm", traj, abort_fn=abort_fn)

        controller.advance_all()  # waypoint 0 — not aborted
        assert not future.done()

        aborted = True
        controller.advance_all()  # detects abort

        assert future.done()
        assert future.result() is False
        assert not controller.has_active_runner("test_arm")

    def test_start_trajectory_validates(self, controller):
        # Unknown arm
        positions = np.array([[0.0, 0.0]])
        traj = make_trajectory(positions)
        with pytest.raises(ValueError, match="Unknown arm"):
            controller.start_trajectory("nonexistent", traj)

        # DOF mismatch
        bad_traj = Trajectory(
            timestamps=np.array([0.0]),
            positions=np.array([[0.0, 0.0, 0.0]]),
            velocities=np.zeros((1, 3)),
            accelerations=np.zeros((1, 3)),
            joint_names=["j1", "j2", "j3"],
        )
        with pytest.raises(ValueError, match="DOF"):
            controller.start_trajectory("test_arm", bad_traj)

    def test_advance_all_no_runners_is_noop(self, controller):
        assert not controller.has_active_runner()
        controller.advance_all()  # should not raise

    def test_per_arm_lookahead(self, controller, model_and_data):
        """Per-arm lookahead is used in step()."""
        _, data = model_and_data
        state = controller._arms["test_arm"]

        # Set a custom lookahead
        state.lookahead = 0.05
        state.target_position = np.array([1.0, 1.0])
        state.target_velocity = np.array([2.0, 2.0])
        controller.step()

        # ctrl should be pos + 0.05 * vel = 1.0 + 0.1 = 1.1
        expected = 1.0 + 0.05 * 2.0
        np.testing.assert_allclose(
            [data.ctrl[aid] for aid in state.actuator_ids],
            [expected, expected],
        )

        # Reset to default (None) — should use controller.lookahead_time
        state.lookahead = None
        controller.step()
        default_la = controller.lookahead_time
        expected_default = 1.0 + default_la * 2.0
        np.testing.assert_allclose(
            [data.ctrl[aid] for aid in state.actuator_ids],
            [expected_default, expected_default],
        )
