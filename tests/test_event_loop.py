# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for tick-driven event loop execution.

Verifies that TrajectoryRunner + tick() correctly advances trajectories,
supports concurrent teleop target-setting, and handles per-arm abort.
All tests are headless (no viewer) so timing is deterministic.
"""

import threading

import numpy as np
import pytest
from conftest import MockArm, make_trajectory

from mj_manipulator.config import PhysicsExecutionConfig
from mj_manipulator.event_loop import PhysicsEventLoop
from mj_manipulator.ownership import OwnerKind, OwnershipRegistry
from mj_manipulator.physics_controller import PhysicsController


@pytest.fixture
def loop():
    return PhysicsEventLoop()


@pytest.fixture
def controller(model_and_data, mock_arm):
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


class TestTickDrivenExecution:
    """Test that tick() drives trajectory runners to completion."""

    def test_tick_advances_runner(self, loop, controller):
        loop.set_controller(controller)

        positions = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
        traj = make_trajectory(positions, entity="test_arm")
        future = controller.start_trajectory("test_arm", traj)

        # Each tick should advance one waypoint + step physics
        loop.tick()
        assert not future.done()
        np.testing.assert_allclose(
            controller._arms["test_arm"].target_position, [0.0, 0.0]
        )

        loop.tick()
        np.testing.assert_allclose(
            controller._arms["test_arm"].target_position, [0.1, 0.1]
        )

    def test_tick_completes_trajectory(self, loop, controller):
        loop.set_controller(controller)

        positions = np.array([[0.0, 0.0], [0.1, 0.1]])
        traj = make_trajectory(positions, entity="test_arm")
        future = controller.start_trajectory("test_arm", traj)

        # Advance through waypoints + convergence
        for _ in range(3000):
            loop.tick()
            if future.done():
                break

        assert future.done()
        assert future.result() is True

    def test_tick_steps_physics_every_cycle(self, loop, controller, model_and_data):
        """Physics always steps, even with no runners."""
        _, data = model_and_data
        loop.set_controller(controller)

        t0 = data.time
        for _ in range(10):
            loop.tick()
        assert data.time > t0

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


class TestTickDrivenBimanual:
    """Test concurrent trajectory + teleop target writing."""

    @pytest.fixture
    def two_arm_controller(self, model_and_data):
        model, data = model_and_data
        arm1 = MockArm("arm1", model, data)
        arm2 = MockArm("arm2", model, data)
        return PhysicsController(
            model,
            data,
            {"arm1": arm1, "arm2": arm2},
            config=PhysicsExecutionConfig(
                control_dt=0.002,
                position_tolerance=0.3,
                velocity_tolerance=1.0,
                convergence_timeout_steps=2000,
            ),
        )

    def test_trajectory_and_manual_targets(self, loop, two_arm_controller):
        """One arm runs a trajectory, other arm has targets set externally."""
        ctrl = two_arm_controller
        loop.set_controller(ctrl)

        # Start trajectory on arm1
        positions = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
        traj = make_trajectory(positions, entity="arm1")
        future = ctrl.start_trajectory("arm1", traj)

        # Simulate teleop setting targets on arm2 each tick
        for i in range(3):
            target = np.array([0.5 + i * 0.1, 0.5 + i * 0.1])
            ctrl.set_arm_target("arm2", target)
            loop.tick()

        # arm2 should have the last target we set
        np.testing.assert_allclose(
            ctrl._arms["arm2"].target_position, [0.7, 0.7]
        )
        # arm1 should be advancing through its trajectory
        assert ctrl._arms["arm1"].target_position[0] > 0.0 or future.done()

    def test_per_arm_abort_only_affects_target(self, loop, two_arm_controller):
        """Aborting one arm doesn't stop the other."""
        ctrl = two_arm_controller
        loop.set_controller(ctrl)
        registry = OwnershipRegistry(["arm1", "arm2"])

        positions = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2],
                              [0.3, 0.3], [0.4, 0.4]])
        traj1 = make_trajectory(positions, entity="arm1")
        traj2 = make_trajectory(positions, entity="arm2")

        future1 = ctrl.start_trajectory(
            "arm1", traj1, abort_fn=lambda: registry.is_aborted("arm1")
        )
        future2 = ctrl.start_trajectory(
            "arm2", traj2, abort_fn=lambda: registry.is_aborted("arm2")
        )

        # Advance a couple ticks
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

        positions = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2],
                              [0.3, 0.3], [0.4, 0.4]])
        traj = make_trajectory(positions, entity="arm1")

        # Acquire ownership for trajectory (as _execute_tick_driven does)
        registry.acquire("arm1", OwnerKind.TRAJECTORY, traj)
        future = ctrl.start_trajectory(
            "arm1", traj, abort_fn=lambda: registry.is_aborted("arm1")
        )

        loop.tick()  # advance waypoint 0
        assert not future.done()

        # Simulate what _activate_teleop does: preempt (sets abort),
        # then submit _do_activate (clears abort)
        teleop_owner = object()
        registry.preempt("arm1", OwnerKind.TELEOP, teleop_owner)
        cleared = []

        def do_activate():
            registry.clear_abort("arm1")
            cleared.append(True)

        loop.submit(do_activate)

        # Single tick: advance_all sees abort FIRST, then command clears it
        loop.tick()

        assert future.done()
        assert future.result() is False  # runner aborted
        assert cleared == [True]  # command also ran
        assert not registry.is_aborted("arm1")  # abort cleared for next use


class TestBackgroundThreadExecution:
    """Test that background threads can submit work and block on results."""

    def test_submit_from_background_thread(self, loop, controller):
        loop.set_controller(controller)

        result_holder = []

        def bg_work():
            # Submit from background thread
            future = loop.submit(lambda: 42)
            result_holder.append(future)

        t = threading.Thread(target=bg_work)
        t.start()
        t.join()

        # Tick to process the command
        loop.tick()
        assert result_holder[0].result() == 42
