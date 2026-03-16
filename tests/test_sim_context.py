"""Tests for SimContext and SimArmController.

Uses shared test fixtures from conftest.py (MockArm, model_and_data).
Tests run headless (no viewer).
"""

import numpy as np
import pytest

from mj_manipulator.config import PhysicsConfig, PhysicsExecutionConfig
from mj_manipulator.protocols import ArmController, ExecutionContext
from mj_manipulator.sim_context import SimArmController, SimContext

from conftest import make_trajectory


class TestSimContextLifecycle:
    def test_enter_exit_headless_physics(self, model_and_data, mock_arm):
        model, data = model_and_data
        ctx = SimContext(
            model, data, {"test_arm": mock_arm},
            headless=True,
            physics_config=PhysicsConfig(
                execution=PhysicsExecutionConfig(control_dt=0.002),
            ),
        )
        with ctx as c:
            assert c is ctx
            # Should be able to execute after entering
            traj = make_trajectory(
                np.array([[0.0, 0.0]]), entity="test_arm",
            )
            result = c.execute(traj)
            assert result is True

        # After exit, execution should fail
        with pytest.raises(ValueError):
            ctx.execute(traj)

    def test_enter_exit_headless_kinematic(self, model_and_data, mock_arm):
        model, data = model_and_data
        ctx = SimContext(
            model, data, {"test_arm": mock_arm},
            physics=False, headless=True,
        )
        with ctx as c:
            # Should be able to execute
            traj = make_trajectory(
                np.array([[0.0, 0.0]]), entity="test_arm",
            )
            result = c.execute(traj)
            assert result is True

        # After exit, execution should fail
        with pytest.raises(ValueError):
            ctx.execute(traj)


class TestSimContextProtocol:
    def test_satisfies_execution_context(self, model_and_data, mock_arm):
        model, data = model_and_data
        ctx = SimContext(
            model, data, {"test_arm": mock_arm}, headless=True,
        )
        assert isinstance(ctx, ExecutionContext)

    def test_is_running_headless(self, model_and_data, mock_arm):
        model, data = model_and_data
        with SimContext(
            model, data, {"test_arm": mock_arm}, headless=True,
        ) as ctx:
            assert ctx.is_running() is True

    def test_control_dt_physics(self, model_and_data, mock_arm):
        model, data = model_and_data
        with SimContext(
            model, data, {"test_arm": mock_arm},
            headless=True,
            physics_config=PhysicsConfig(
                execution=PhysicsExecutionConfig(control_dt=0.004),
            ),
        ) as ctx:
            assert ctx.control_dt == 0.004

    def test_control_dt_kinematic(self, model_and_data, mock_arm):
        model, data = model_and_data
        with SimContext(
            model, data, {"test_arm": mock_arm},
            physics=False, headless=True,
        ) as ctx:
            assert ctx.control_dt == 0.004


class TestSimContextExecution:
    def test_execute_trajectory_physics(self, model_and_data, mock_arm):
        model, data = model_and_data
        with SimContext(
            model, data, {"test_arm": mock_arm},
            headless=True,
            physics_config=PhysicsConfig(
                execution=PhysicsExecutionConfig(
                    control_dt=0.002,
                    position_tolerance=0.3,
                    velocity_tolerance=1.0,
                    convergence_timeout_steps=2000,
                ),
            ),
        ) as ctx:
            positions = np.array([
                [0.0, 0.0],
                [0.1, 0.1],
                [0.2, 0.2],
            ])
            traj = make_trajectory(positions, entity="test_arm")
            result = ctx.execute(traj)
            assert result is True

    def test_execute_trajectory_kinematic(self, model_and_data, mock_arm):
        model, data = model_and_data
        with SimContext(
            model, data, {"test_arm": mock_arm},
            physics=False, headless=True,
        ) as ctx:
            positions = np.array([
                [0.0, 0.0],
                [0.1, 0.1],
                [0.3, 0.3],
            ])
            traj = make_trajectory(positions, entity="test_arm")
            result = ctx.execute(traj)
            assert result is True

            # Kinematic: final position should be exact
            for i, idx in enumerate(mock_arm.joint_qpos_indices):
                assert abs(data.qpos[idx] - 0.3) < 1e-6

    def test_execute_no_entity_raises(self, model_and_data, mock_arm):
        model, data = model_and_data
        with SimContext(
            model, data, {"test_arm": mock_arm},
            physics=False, headless=True,
        ) as ctx:
            traj = make_trajectory(
                np.array([[0.0, 0.0]]), entity=None,
            )
            with pytest.raises(ValueError, match="no entity"):
                ctx.execute(traj)

    def test_execute_unknown_entity_raises(self, model_and_data, mock_arm):
        model, data = model_and_data
        with SimContext(
            model, data, {"test_arm": mock_arm},
            physics=False, headless=True,
        ) as ctx:
            traj = make_trajectory(
                np.array([[0.0, 0.0]]), entity="unknown",
            )
            with pytest.raises(ValueError, match="No executor"):
                ctx.execute(traj)


class TestSimContextStep:
    def test_step_physics_with_targets(self, model_and_data, mock_arm):
        model, data = model_and_data
        with SimContext(
            model, data, {"test_arm": mock_arm},
            headless=True,
            physics_config=PhysicsConfig(
                execution=PhysicsExecutionConfig(control_dt=0.002),
            ),
        ) as ctx:
            ctx.step({"test_arm": np.array([0.5, -0.5])})
            # Actuators should have been commanded
            assert data.ctrl[mock_arm.actuator_ids[0]] != 0.0

    def test_step_physics_no_targets(self, model_and_data, mock_arm):
        model, data = model_and_data
        with SimContext(
            model, data, {"test_arm": mock_arm},
            headless=True,
            physics_config=PhysicsConfig(
                execution=PhysicsExecutionConfig(control_dt=0.002),
            ),
        ) as ctx:
            # Should not raise — holds all positions
            ctx.step()

    def test_step_kinematic_with_targets(self, model_and_data, mock_arm):
        model, data = model_and_data
        with SimContext(
            model, data, {"test_arm": mock_arm},
            physics=False, headless=True,
        ) as ctx:
            ctx.step({"test_arm": np.array([0.7, -0.3])})
            # Kinematic: qpos should match target exactly
            for i, idx in enumerate(mock_arm.joint_qpos_indices):
                expected = [0.7, -0.3][i]
                assert abs(data.qpos[idx] - expected) < 1e-6

    def test_step_kinematic_no_targets(self, model_and_data, mock_arm):
        model, data = model_and_data
        with SimContext(
            model, data, {"test_arm": mock_arm},
            physics=False, headless=True,
        ) as ctx:
            # Should not raise
            ctx.step()

    def test_step_cartesian_physics(self, model_and_data, mock_arm):
        model, data = model_and_data
        with SimContext(
            model, data, {"test_arm": mock_arm},
            headless=True,
            physics_config=PhysicsConfig(
                execution=PhysicsExecutionConfig(control_dt=0.002),
            ),
        ) as ctx:
            ctx.step_cartesian("test_arm", np.array([0.3, 0.3]))
            # Physics should have stepped
            assert data.ctrl[mock_arm.actuator_ids[0]] != 0.0

    def test_step_cartesian_kinematic(self, model_and_data, mock_arm):
        model, data = model_and_data
        with SimContext(
            model, data, {"test_arm": mock_arm},
            physics=False, headless=True,
        ) as ctx:
            ctx.step_cartesian("test_arm", np.array([0.4, -0.4]))
            # Kinematic: position should be set
            for i, idx in enumerate(mock_arm.joint_qpos_indices):
                expected = [0.4, -0.4][i]
                assert abs(data.qpos[idx] - expected) < 1e-6


class TestSimContextArmController:
    def test_arm_returns_controller(self, model_and_data, mock_arm):
        model, data = model_and_data
        with SimContext(
            model, data, {"test_arm": mock_arm},
            physics=False, headless=True,
        ) as ctx:
            arm_ctrl = ctx.arm("test_arm")
            assert isinstance(arm_ctrl, SimArmController)

    def test_arm_satisfies_protocol(self, model_and_data, mock_arm):
        model, data = model_and_data
        with SimContext(
            model, data, {"test_arm": mock_arm},
            physics=False, headless=True,
        ) as ctx:
            arm_ctrl = ctx.arm("test_arm")
            assert isinstance(arm_ctrl, ArmController)

    def test_arm_cached(self, model_and_data, mock_arm):
        model, data = model_and_data
        with SimContext(
            model, data, {"test_arm": mock_arm},
            physics=False, headless=True,
        ) as ctx:
            ctrl1 = ctx.arm("test_arm")
            ctrl2 = ctx.arm("test_arm")
            assert ctrl1 is ctrl2

    def test_arm_unknown_raises(self, model_and_data, mock_arm):
        model, data = model_and_data
        with SimContext(
            model, data, {"test_arm": mock_arm},
            physics=False, headless=True,
        ) as ctx:
            with pytest.raises(ValueError, match="Unknown arm"):
                ctx.arm("nonexistent")

    def test_grasp_no_gripper(self, model_and_data, mock_arm):
        """Grasp returns None when arm has no gripper."""
        model, data = model_and_data
        with SimContext(
            model, data, {"test_arm": mock_arm},
            physics=False, headless=True,
        ) as ctx:
            result = ctx.arm("test_arm").grasp("some_object")
            assert result is None

    def test_release_no_gripper(self, model_and_data, mock_arm):
        """Release is a no-op when arm has no gripper."""
        model, data = model_and_data
        with SimContext(
            model, data, {"test_arm": mock_arm},
            physics=False, headless=True,
        ) as ctx:
            # Should not raise
            ctx.arm("test_arm").release("some_object")
