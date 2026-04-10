# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for SimContext and SimArmController.

Uses shared test fixtures from conftest.py (MockArm, model_and_data).
Tests run headless (no viewer).
"""

import numpy as np
import pytest
from conftest import make_trajectory

from mj_manipulator.config import PhysicsConfig, PhysicsExecutionConfig
from mj_manipulator.protocols import ArmController, ExecutionContext
from mj_manipulator.sim_context import SimArmController, SimContext


class TestSimContextLifecycle:
    def test_enter_exit_headless_physics(self, model_and_data, mock_arm):
        model, data = model_and_data
        ctx = SimContext(
            model,
            data,
            {"test_arm": mock_arm},
            headless=True,
            physics_config=PhysicsConfig(
                execution=PhysicsExecutionConfig(control_dt=0.002),
            ),
        )
        with ctx as c:
            assert c is ctx
            # Should be able to execute after entering
            traj = make_trajectory(
                np.array([[0.0, 0.0]]),
                entity="test_arm",
            )
            result = c.execute(traj)
            assert result is True

        # After exit, execution should fail
        with pytest.raises(ValueError):
            ctx.execute(traj)

    def test_enter_exit_headless_kinematic(self, model_and_data, mock_arm):
        model, data = model_and_data
        ctx = SimContext(
            model,
            data,
            {"test_arm": mock_arm},
            physics=False,
            headless=True,
        )
        with ctx as c:
            # Should be able to execute
            traj = make_trajectory(
                np.array([[0.0, 0.0]]),
                entity="test_arm",
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
            model,
            data,
            {"test_arm": mock_arm},
            headless=True,
        )
        assert isinstance(ctx, ExecutionContext)

    def test_is_running_headless(self, model_and_data, mock_arm):
        model, data = model_and_data
        with SimContext(
            model,
            data,
            {"test_arm": mock_arm},
            headless=True,
        ) as ctx:
            assert ctx.is_running() is True

    def test_control_dt_physics(self, model_and_data, mock_arm):
        model, data = model_and_data
        with SimContext(
            model,
            data,
            {"test_arm": mock_arm},
            headless=True,
            physics_config=PhysicsConfig(
                execution=PhysicsExecutionConfig(control_dt=0.004),
            ),
        ) as ctx:
            assert ctx.control_dt == 0.004

    def test_control_dt_kinematic(self, model_and_data, mock_arm):
        model, data = model_and_data
        with SimContext(
            model,
            data,
            {"test_arm": mock_arm},
            physics=False,
            headless=True,
        ) as ctx:
            assert ctx.control_dt == 0.004


class TestSimContextExecution:
    def test_execute_trajectory_physics(self, model_and_data, mock_arm):
        model, data = model_and_data
        with SimContext(
            model,
            data,
            {"test_arm": mock_arm},
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
            positions = np.array(
                [
                    [0.0, 0.0],
                    [0.1, 0.1],
                    [0.2, 0.2],
                ]
            )
            traj = make_trajectory(positions, entity="test_arm")
            result = ctx.execute(traj)
            assert result is True

    def test_execute_trajectory_kinematic(self, model_and_data, mock_arm):
        model, data = model_and_data
        with SimContext(
            model,
            data,
            {"test_arm": mock_arm},
            physics=False,
            headless=True,
        ) as ctx:
            positions = np.array(
                [
                    [0.0, 0.0],
                    [0.1, 0.1],
                    [0.3, 0.3],
                ]
            )
            traj = make_trajectory(positions, entity="test_arm")
            result = ctx.execute(traj)
            assert result is True

            # Kinematic: final position should be exact
            for i, idx in enumerate(mock_arm.joint_qpos_indices):
                assert abs(data.qpos[idx] - 0.3) < 1e-6

    def test_execute_no_entity_raises(self, model_and_data, mock_arm):
        model, data = model_and_data
        with SimContext(
            model,
            data,
            {"test_arm": mock_arm},
            physics=False,
            headless=True,
        ) as ctx:
            traj = make_trajectory(
                np.array([[0.0, 0.0]]),
                entity=None,
            )
            with pytest.raises(ValueError, match="no entity"):
                ctx.execute(traj)

    def test_execute_unknown_entity_raises(self, model_and_data, mock_arm):
        model, data = model_and_data
        with SimContext(
            model,
            data,
            {"test_arm": mock_arm},
            physics=False,
            headless=True,
        ) as ctx:
            traj = make_trajectory(
                np.array([[0.0, 0.0]]),
                entity="unknown",
            )
            with pytest.raises(ValueError, match="No executor"):
                ctx.execute(traj)

    # --- abort_fn parameter ------------------------------------------------
    # The abort_fn parameter lets callers halt a trajectory mid-execution
    # based on real-time signals (e.g. a new collision contact). It composes
    # with (does not replace) the ownership-registry abort and the
    # context-level abort: any of the three returning True stops execution.

    def test_execute_accepts_abort_fn_none_backwards_compat(self, model_and_data, mock_arm):
        """Calling ctx.execute(traj) without abort_fn still works (default=None)."""
        model, data = model_and_data
        with SimContext(
            model,
            data,
            {"test_arm": mock_arm},
            physics=False,
            headless=True,
        ) as ctx:
            traj = make_trajectory(
                np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]]),
                entity="test_arm",
            )
            # No abort_fn passed — should run to completion.
            assert ctx.execute(traj) is True

    def test_execute_caller_abort_fn_runs_to_completion_when_false(self, model_and_data, mock_arm):
        """A caller abort_fn that always returns False doesn't interfere."""
        model, data = model_and_data
        with SimContext(
            model,
            data,
            {"test_arm": mock_arm},
            physics=False,
            headless=True,
        ) as ctx:
            traj = make_trajectory(
                np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]]),
                entity="test_arm",
            )
            call_count = {"n": 0}

            def never_abort():
                call_count["n"] += 1
                return False

            assert ctx.execute(traj, abort_fn=never_abort) is True
            # abort_fn was invoked at least once (pre-flight check in
            # kinematic mode) — proves the parameter was plumbed through.
            assert call_count["n"] >= 1

    def test_execute_caller_abort_fn_short_circuits_when_true(self, model_and_data, mock_arm):
        """A caller abort_fn that returns True halts execution."""
        model, data = model_and_data
        with SimContext(
            model,
            data,
            {"test_arm": mock_arm},
            physics=False,
            headless=True,
        ) as ctx:
            traj = make_trajectory(
                np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]]),
                entity="test_arm",
            )
            # abort_fn fires immediately — the kinematic path should
            # short-circuit before running the trajectory at all.
            assert ctx.execute(traj, abort_fn=lambda: True) is False
            # And qpos should still be at the starting zero, not the
            # final [0.2, 0.2].
            for idx in mock_arm.joint_qpos_indices:
                assert abs(data.qpos[idx]) < 1e-6

    def test_execute_context_abort_fn_still_composes(self, model_and_data, mock_arm):
        """The context-level abort_fn (set at construction) still fires
        alongside the caller-provided abort_fn."""
        model, data = model_and_data
        with SimContext(
            model,
            data,
            {"test_arm": mock_arm},
            physics=False,
            headless=True,
            abort_fn=lambda: True,  # always-abort context-level
        ) as ctx:
            traj = make_trajectory(
                np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]]),
                entity="test_arm",
            )
            # Caller says "never abort", but context says "always abort"
            # — the OR should fire, execution halts.
            assert ctx.execute(traj, abort_fn=lambda: False) is False


class TestSimContextStep:
    def test_step_physics_with_targets(self, model_and_data, mock_arm):
        model, data = model_and_data
        with SimContext(
            model,
            data,
            {"test_arm": mock_arm},
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
            model,
            data,
            {"test_arm": mock_arm},
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
            model,
            data,
            {"test_arm": mock_arm},
            physics=False,
            headless=True,
        ) as ctx:
            ctx.step({"test_arm": np.array([0.7, -0.3])})
            # Kinematic: qpos should match target exactly
            for i, idx in enumerate(mock_arm.joint_qpos_indices):
                expected = [0.7, -0.3][i]
                assert abs(data.qpos[idx] - expected) < 1e-6

    def test_step_kinematic_no_targets(self, model_and_data, mock_arm):
        model, data = model_and_data
        with SimContext(
            model,
            data,
            {"test_arm": mock_arm},
            physics=False,
            headless=True,
        ) as ctx:
            # Should not raise
            ctx.step()

    def test_step_cartesian_physics(self, model_and_data, mock_arm):
        model, data = model_and_data
        with SimContext(
            model,
            data,
            {"test_arm": mock_arm},
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
            model,
            data,
            {"test_arm": mock_arm},
            physics=False,
            headless=True,
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
            model,
            data,
            {"test_arm": mock_arm},
            physics=False,
            headless=True,
        ) as ctx:
            arm_ctrl = ctx.arm("test_arm")
            assert isinstance(arm_ctrl, SimArmController)

    def test_arm_satisfies_protocol(self, model_and_data, mock_arm):
        model, data = model_and_data
        with SimContext(
            model,
            data,
            {"test_arm": mock_arm},
            physics=False,
            headless=True,
        ) as ctx:
            arm_ctrl = ctx.arm("test_arm")
            assert isinstance(arm_ctrl, ArmController)

    def test_arm_cached(self, model_and_data, mock_arm):
        model, data = model_and_data
        with SimContext(
            model,
            data,
            {"test_arm": mock_arm},
            physics=False,
            headless=True,
        ) as ctx:
            ctrl1 = ctx.arm("test_arm")
            ctrl2 = ctx.arm("test_arm")
            assert ctrl1 is ctrl2

    def test_arm_unknown_raises(self, model_and_data, mock_arm):
        model, data = model_and_data
        with SimContext(
            model,
            data,
            {"test_arm": mock_arm},
            physics=False,
            headless=True,
        ) as ctx:
            with pytest.raises(ValueError, match="Unknown arm"):
                ctx.arm("nonexistent")

    def test_grasp_no_gripper(self, model_and_data, mock_arm):
        """Grasp returns None when arm has no gripper."""
        model, data = model_and_data
        with SimContext(
            model,
            data,
            {"test_arm": mock_arm},
            physics=False,
            headless=True,
        ) as ctx:
            result = ctx.arm("test_arm").grasp("some_object")
            assert result is None

    def test_release_no_gripper(self, model_and_data, mock_arm):
        """Release is a no-op when arm has no gripper."""
        model, data = model_and_data
        with SimContext(
            model,
            data,
            {"test_arm": mock_arm},
            physics=False,
            headless=True,
        ) as ctx:
            # Should not raise
            ctx.arm("test_arm").release("some_object")
