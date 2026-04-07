# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Ownership state transition tests.

Verifies every realistic mode change on a single arm:
idle → teleop → trajectory → idle, and all combinations.
These map the state machine that the ownership registry,
event loop, and SimContext must support together.
"""

import numpy as np
import pytest
from conftest import MockArm, make_trajectory

from mj_manipulator.config import PhysicsExecutionConfig
from mj_manipulator.event_loop import PhysicsEventLoop
from mj_manipulator.ownership import OwnerKind, OwnershipRegistry
from mj_manipulator.physics_controller import PhysicsController


@pytest.fixture
def env(model_and_data):
    model, data = model_and_data
    arm = MockArm("arm", model, data)
    loop = PhysicsEventLoop()
    ctrl = PhysicsController(
        model, data, {"arm": arm},
        config=PhysicsExecutionConfig(
            control_dt=0.002,
            position_tolerance=0.3,
            velocity_tolerance=1.0,
            convergence_timeout_steps=2000,
        ),
    )
    loop.set_controller(ctrl)
    registry = OwnershipRegistry(["arm"])
    return loop, ctrl, registry


def _make_traj():
    positions = np.array([[i * 0.01, i * 0.01] for i in range(10)])
    return make_trajectory(positions, entity="arm")


class TestIdleTransitions:
    def test_idle_to_trajectory(self, env):
        loop, ctrl, reg = env
        traj = _make_traj()
        assert reg.acquire("arm", OwnerKind.TRAJECTORY, traj)
        assert reg.owner_of("arm")[0] == OwnerKind.TRAJECTORY

    def test_idle_to_teleop(self, env):
        _, _, reg = env
        owner = object()
        assert reg.acquire("arm", OwnerKind.TELEOP, owner)
        assert reg.owner_of("arm")[0] == OwnerKind.TELEOP

    def test_idle_acquire_clears_stale_abort(self, env):
        _, _, reg = env
        reg.set_abort("arm")
        reg.acquire("arm", OwnerKind.TRAJECTORY, object())
        assert not reg.is_aborted("arm")


class TestTeleopTransitions:
    def test_teleop_to_idle(self, env):
        _, _, reg = env
        owner = object()
        reg.acquire("arm", OwnerKind.TELEOP, owner)
        reg.release("arm", owner)
        assert reg.owner_of("arm")[0] == OwnerKind.IDLE

    def test_teleop_blocks_trajectory_acquire(self, env):
        """Trajectory cannot acquire an arm that teleop owns."""
        _, _, reg = env
        teleop = object()
        reg.acquire("arm", OwnerKind.TELEOP, teleop)
        assert not reg.acquire("arm", OwnerKind.TRAJECTORY, object())
        assert reg.owner_of("arm")[0] == OwnerKind.TELEOP

    def test_teleop_to_teleop_preempt_fails(self, env):
        """Cannot preempt teleop with teleop (same priority)."""
        _, _, reg = env
        teleop1 = object()
        reg.acquire("arm", OwnerKind.TELEOP, teleop1)
        with pytest.raises(ValueError, match="Cannot preempt"):
            reg.preempt("arm", OwnerKind.TELEOP, object())


class TestTrajectoryTransitions:
    def test_trajectory_to_idle(self, env):
        _, _, reg = env
        traj = object()
        reg.acquire("arm", OwnerKind.TRAJECTORY, traj)
        reg.release("arm", traj)
        assert reg.owner_of("arm")[0] == OwnerKind.IDLE

    def test_trajectory_preempted_by_teleop(self, env):
        """Teleop can preempt a running trajectory."""
        loop, ctrl, reg = env
        traj = _make_traj()
        reg.acquire("arm", OwnerKind.TRAJECTORY, traj)
        future = ctrl.start_trajectory(
            "arm", traj, abort_fn=lambda: reg.is_aborted("arm")
        )

        loop.tick()  # advance one waypoint
        assert not future.done()

        # Preempt with teleop
        teleop = object()
        reg.preempt("arm", OwnerKind.TELEOP, teleop)

        loop.tick()  # runner sees abort
        assert future.done()
        assert future.result() is False
        assert reg.owner_of("arm") == (OwnerKind.TELEOP, teleop)

    def test_trajectory_completes_releases_to_idle(self, env):
        """After trajectory finishes, ownership should be releasable."""
        loop, ctrl, reg = env
        traj = _make_traj()
        reg.acquire("arm", OwnerKind.TRAJECTORY, traj)
        future = ctrl.start_trajectory("arm", traj)

        for _ in range(3000):
            loop.tick()
            if future.done():
                break

        assert future.done()
        # Simulate what _execute_tick_driven's finally block does
        kind, owner = reg.owner_of("arm")
        if owner is traj:
            reg.release("arm", traj)
        assert reg.owner_of("arm")[0] == OwnerKind.IDLE


class TestFullCycles:
    """Test realistic multi-step mode cycling on a single arm."""

    def test_idle_teleop_trajectory_teleop_idle(self, env):
        """idle → teleop → (deactivate) → trajectory → idle → teleop → idle"""
        loop, ctrl, reg = env

        # 1. Activate teleop
        teleop = object()
        assert reg.acquire("arm", OwnerKind.TELEOP, teleop)
        assert reg.owner_of("arm")[0] == OwnerKind.TELEOP

        # 2. Deactivate teleop (user clicks deactivate)
        reg.release("arm", teleop)
        assert reg.owner_of("arm")[0] == OwnerKind.IDLE

        # 3. Run trajectory
        traj = _make_traj()
        assert reg.acquire("arm", OwnerKind.TRAJECTORY, traj)
        future = ctrl.start_trajectory("arm", traj)
        for _ in range(3000):
            loop.tick()
            if future.done():
                break
        reg.release("arm", traj)
        assert reg.owner_of("arm")[0] == OwnerKind.IDLE

        # 4. Activate teleop again
        teleop2 = object()
        assert reg.acquire("arm", OwnerKind.TELEOP, teleop2)
        assert reg.owner_of("arm")[0] == OwnerKind.TELEOP

        # 5. Deactivate
        reg.release("arm", teleop2)
        assert reg.owner_of("arm")[0] == OwnerKind.IDLE

    def test_execute_while_teleop_active(self, env):
        """Trajectory must deactivate teleop first (as SimContext does)."""
        loop, ctrl, reg = env

        # Teleop active
        teleop = object()
        reg.acquire("arm", OwnerKind.TELEOP, teleop)

        # Execute needs to deactivate teleop, then acquire
        kind, owner = reg.owner_of("arm")
        assert kind == OwnerKind.TELEOP
        reg.release("arm", owner)  # deactivate teleop
        assert reg.owner_of("arm")[0] == OwnerKind.IDLE

        traj = _make_traj()
        assert reg.acquire("arm", OwnerKind.TRAJECTORY, traj)

        future = ctrl.start_trajectory("arm", traj)
        for _ in range(3000):
            loop.tick()
            if future.done():
                break
        reg.release("arm", traj)
        assert reg.owner_of("arm")[0] == OwnerKind.IDLE

    def test_preempt_during_execute_then_reexecute(self, env):
        """Preempt trajectory with teleop, deactivate teleop, run new trajectory."""
        loop, ctrl, reg = env

        # 1. Start trajectory
        traj1 = _make_traj()
        reg.acquire("arm", OwnerKind.TRAJECTORY, traj1)
        future1 = ctrl.start_trajectory(
            "arm", traj1, abort_fn=lambda: reg.is_aborted("arm")
        )
        loop.tick()

        # 2. Preempt with teleop
        teleop = object()
        reg.preempt("arm", OwnerKind.TELEOP, teleop)
        loop.tick()  # runner aborts
        assert future1.result() is False

        # 3. Deactivate teleop
        reg.clear_abort("arm")
        reg.release("arm", teleop)

        # 4. Run new trajectory
        traj2 = _make_traj()
        assert reg.acquire("arm", OwnerKind.TRAJECTORY, traj2)
        future2 = ctrl.start_trajectory("arm", traj2)
        for _ in range(3000):
            loop.tick()
            if future2.done():
                break
        assert future2.result() is True
        reg.release("arm", traj2)
        assert reg.owner_of("arm")[0] == OwnerKind.IDLE
