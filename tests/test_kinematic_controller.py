# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for KinematicController and tick-driven kinematic execution.

Mirrors key tests from test_physics_controller.py and test_event_loop.py
but with KinematicController, verifying that kinematic mode gets the
same capabilities: non-blocking runners, per-arm abort, concurrent
multi-arm control, and event-loop integration.
"""

import threading

import numpy as np
import pytest
from conftest import MockArm, make_trajectory

from mj_manipulator.config import ExecutionConfig
from mj_manipulator.event_loop import PhysicsEventLoop
from mj_manipulator.kinematic_controller import KinematicController
from mj_manipulator.ownership import OwnerKind, OwnershipRegistry


@pytest.fixture
def controller(model_and_data, mock_arm):
    model, data = model_and_data
    return KinematicController(
        model,
        data,
        {"test_arm": mock_arm},
    )


@pytest.fixture
def loop():
    return PhysicsEventLoop()


class TestKinematicControllerConstruction:
    def test_constructs(self, model_and_data, mock_arm):
        model, data = model_and_data
        ctrl = KinematicController(model, data, {"arm": mock_arm})
        assert ctrl is not None
        assert ctrl.control_dt == 0.004  # kinematic default

    def test_constructs_with_initial_positions(self, model_and_data, mock_arm):
        model, data = model_and_data
        init_pos = {"arm": np.array([0.5, -0.5])}
        KinematicController(model, data, {"arm": mock_arm}, initial_positions=init_pos)
        for i, idx in enumerate(mock_arm.joint_qpos_indices):
            assert abs(data.qpos[idx] - init_pos["arm"][i]) < 1e-6

    def test_custom_control_dt(self, model_and_data, mock_arm):
        model, data = model_and_data
        ctrl = KinematicController(
            model,
            data,
            {"arm": mock_arm},
            config=ExecutionConfig(control_dt=0.01),
        )
        assert ctrl.control_dt == 0.01


class TestKinematicStepping:
    def test_step_writes_qpos_exactly(self, controller, model_and_data, mock_arm):
        """Kinematic step writes targets directly to qpos — exact tracking."""
        _, data = model_and_data
        controller.set_arm_target("test_arm", np.array([0.5, -0.3]))
        controller.step()

        for i, idx in enumerate(mock_arm.joint_qpos_indices):
            expected = [0.5, -0.3][i]
            assert abs(data.qpos[idx] - expected) < 1e-10

    def test_step_zeros_velocity(self, controller, model_and_data, mock_arm):
        """Kinematic step always zeros velocity (no dynamics)."""
        _, data = model_and_data
        controller.set_arm_target("test_arm", np.array([0.5, -0.3]))
        controller.step()

        for idx in mock_arm.joint_qvel_indices:
            assert data.qvel[idx] == 0.0

    def test_step_reactive_is_exact(self, controller, model_and_data, mock_arm):
        """Kinematic step_reactive is the same as set + step (instant tracking)."""
        _, data = model_and_data
        controller.step_reactive("test_arm", np.array([0.7, 0.2]))

        for i, idx in enumerate(mock_arm.joint_qpos_indices):
            expected = [0.7, 0.2][i]
            assert abs(data.qpos[idx] - expected) < 1e-10


class TestKinematicExecution:
    def test_execute_trajectory(self, controller, model_and_data, mock_arm):
        _, data = model_and_data
        positions = np.array([[0.0, 0.0], [0.1, 0.1], [0.3, 0.3]])
        traj = make_trajectory(positions, entity="test_arm")
        result = controller.execute("test_arm", traj)
        assert result is True

        # Final position should be exact
        for i, idx in enumerate(mock_arm.joint_qpos_indices):
            assert abs(data.qpos[idx] - 0.3) < 1e-6

    def test_execute_aborts(self, model_and_data, mock_arm):
        model, data = model_and_data
        ctrl = KinematicController(model, data, {"arm": mock_arm}, abort_fn=lambda: True)
        positions = np.array([[0.0, 0.0], [0.1, 0.1]])
        traj = make_trajectory(positions, entity="arm")
        result = ctrl.execute("arm", traj)
        assert result is False


class TestTickDrivenKinematic:
    """Test that the event loop drives kinematic trajectory runners."""

    def test_tick_advances_runner(self, loop, controller):
        loop.set_controller(controller)

        positions = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
        traj = make_trajectory(positions, entity="test_arm")
        future = controller.start_trajectory("test_arm", traj)

        loop.tick()
        # After first tick: waypoint 0 written + step applied + convergence checked
        assert not future.done() or future.result() is True

    def test_tick_completes_trajectory(self, loop, controller):
        """Kinematic convergence is immediate — trajectory completes quickly."""
        loop.set_controller(controller)

        positions = np.array([[0.0, 0.0], [0.1, 0.1]])
        traj = make_trajectory(positions, entity="test_arm")
        future = controller.start_trajectory("test_arm", traj)

        # Should complete in a few ticks (waypoints + 1 convergence check)
        for _ in range(10):
            loop.tick()
            if future.done():
                break

        assert future.done()
        assert future.result() is True

    def test_queued_commands_processed(self, loop, controller):
        loop.set_controller(controller)

        results = []

        def cmd():
            results.append("ran")
            return 42

        future = loop.submit(cmd)
        loop.tick()

        assert results == ["ran"]
        assert future.result() == 42


class TestTickDrivenBimanualKinematic:
    """Test concurrent trajectory + teleop target writing in kinematic mode."""

    @pytest.fixture
    def two_arm_controller(self, model_and_data):
        model, data = model_and_data
        arm1 = MockArm("arm1", model, data)
        arm2 = MockArm("arm2", model, data)
        return KinematicController(
            model,
            data,
            {"arm1": arm1, "arm2": arm2},
        )

    def test_trajectory_and_manual_targets(self, loop, two_arm_controller):
        """One arm runs a trajectory, other arm has targets set externally."""
        ctrl = two_arm_controller
        loop.set_controller(ctrl)

        positions = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
        traj = make_trajectory(positions, entity="arm1")
        ctrl.start_trajectory("arm1", traj)

        # Simulate teleop setting targets on arm2 each tick
        for i in range(3):
            target = np.array([0.5 + i * 0.1, 0.5 + i * 0.1])
            ctrl.set_arm_target("arm2", target)
            loop.tick()

        # arm2 should have the last target we set — exact in kinematic
        np.testing.assert_allclose(
            ctrl.data.qpos[ctrl._arms["arm2"].joint_qpos_indices],
            [0.7, 0.7],
            atol=1e-10,
        )

    def test_per_arm_abort_only_affects_target(self, loop, two_arm_controller):
        """Aborting one arm doesn't stop the other."""
        ctrl = two_arm_controller
        loop.set_controller(ctrl)
        registry = OwnershipRegistry(["arm1", "arm2"])

        positions = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]])
        traj1 = make_trajectory(positions, entity="arm1")
        traj2 = make_trajectory(positions, entity="arm2")

        future1 = ctrl.start_trajectory("arm1", traj1, abort_fn=lambda: registry.is_aborted("arm1"))
        future2 = ctrl.start_trajectory("arm2", traj2, abort_fn=lambda: registry.is_aborted("arm2"))

        loop.tick()
        loop.tick()

        # Abort only arm1
        registry.set_abort("arm1")
        loop.tick()

        assert future1.done()
        assert future1.result() is False  # aborted
        assert not future2.done()  # still running

    def test_preempt_aborts_runner_same_tick(self, loop, two_arm_controller):
        """Preempt sets abort, runner sees it on the same tick before command runs."""
        ctrl = two_arm_controller
        loop.set_controller(ctrl)
        registry = OwnershipRegistry(["arm1", "arm2"])

        positions = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]])
        traj = make_trajectory(positions, entity="arm1")

        registry.acquire("arm1", OwnerKind.TRAJECTORY, traj)
        future = ctrl.start_trajectory("arm1", traj, abort_fn=lambda: registry.is_aborted("arm1"))

        loop.tick()
        assert not future.done()

        teleop_owner = object()
        registry.preempt("arm1", OwnerKind.TELEOP, teleop_owner)
        cleared = []

        def do_activate():
            registry.clear_abort("arm1")
            cleared.append(True)

        loop.submit(do_activate)

        loop.tick()

        assert future.done()
        assert future.result() is False  # runner aborted
        assert cleared == [True]  # command also ran
        assert not registry.is_aborted("arm1")


class TestBackgroundThreadKinematic:
    """Test that background threads can submit work to kinematic mode."""

    def test_submit_from_background_thread(self, loop, controller):
        loop.set_controller(controller)

        result_holder = []

        def bg_work():
            future = loop.submit(lambda: 42)
            result_holder.append(future)

        t = threading.Thread(target=bg_work)
        t.start()
        t.join()

        loop.tick()
        assert result_holder[0].result() == 42
