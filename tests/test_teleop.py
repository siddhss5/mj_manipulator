"""Tests for TeleopController."""

import numpy as np
import pytest
import time

from mj_manipulator.teleop import TeleopConfig, TeleopController, TeleopFrame, TeleopState


# -- Mock objects for testing without MuJoCo ---------------------------------


class MockIKSolver:
    """Returns configurable IK solutions."""

    def __init__(self, solutions=None):
        self.solutions = solutions  # list of np.ndarray, or None for failure

    def solve(self, pose, q_init=None):
        if self.solutions is None:
            return []
        return [np.array(s) for s in self.solutions]


class MockGripper:
    def get_actual_position(self):
        return 0.5


class MockArm:
    """Minimal Arm-like object for testing."""

    def __init__(self, dof=6, ik_solutions=None):
        self._q = np.zeros(dof)
        self._ee_pose = np.eye(4)
        self.ik_solver = MockIKSolver(ik_solutions)
        self.gripper = MockGripper()

        class _Config:
            name = "test_arm"
        self.config = _Config()

    def get_joint_positions(self):
        return self._q.copy()

    def get_ee_pose(self):
        return self._ee_pose.copy()


class MockContext:
    """Minimal ExecutionContext for testing."""

    def __init__(self):
        self.last_arm = None
        self.last_q = None
        self.last_qd = None
        self.step_count = 0

    def step_cartesian(self, arm_name, position, velocity=None):
        self.last_arm = arm_name
        self.last_q = np.array(position)
        self.last_qd = np.array(velocity) if velocity is not None else None
        self.step_count += 1


# -- Tests -------------------------------------------------------------------


class TestTeleopLifecycle:

    def test_initial_state_is_idle(self):
        arm = MockArm()
        ctx = MockContext()
        ctrl = TeleopController(arm, ctx)
        assert ctrl.state == TeleopState.IDLE
        assert not ctrl.is_active

    def test_activate_returns_ee_pose(self):
        arm = MockArm()
        arm._ee_pose[0, 3] = 0.5
        ctx = MockContext()
        ctrl = TeleopController(arm, ctx)
        pose = ctrl.activate()
        assert ctrl.is_active
        assert pose[0, 3] == 0.5

    def test_deactivate(self):
        arm = MockArm()
        ctx = MockContext()
        ctrl = TeleopController(arm, ctx)
        ctrl.activate()
        ctrl.deactivate()
        assert not ctrl.is_active
        assert ctrl.state == TeleopState.IDLE

    def test_step_without_activate_returns_idle(self):
        arm = MockArm()
        ctx = MockContext()
        ctrl = TeleopController(arm, ctx)
        state = ctrl.step()
        assert state == TeleopState.IDLE
        assert ctx.step_count == 0


class TestPoseInput:

    def test_pose_tracking_with_valid_ik(self):
        q_solution = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        arm = MockArm(ik_solutions=[q_solution])
        ctx = MockContext()
        ctrl = TeleopController(arm, ctx)
        ctrl.activate()

        ctrl.set_target_pose(np.eye(4))
        state = ctrl.step()

        assert state == TeleopState.TRACKING
        assert ctx.step_count == 1
        np.testing.assert_array_almost_equal(ctx.last_q, q_solution)

    def test_pose_unreachable_when_no_ik(self):
        arm = MockArm(ik_solutions=None)
        ctx = MockContext()
        ctrl = TeleopController(arm, ctx)
        ctrl.activate()

        ctrl.set_target_pose(np.eye(4))
        state = ctrl.step()

        assert state == TeleopState.UNREACHABLE
        assert ctx.step_count == 0

    def test_picks_closest_ik_solution(self):
        q_far = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        q_close = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        arm = MockArm(ik_solutions=[q_far, q_close])
        ctx = MockContext()
        ctrl = TeleopController(arm, ctx)
        ctrl.activate()

        ctrl.set_target_pose(np.eye(4))
        ctrl.step()

        np.testing.assert_array_almost_equal(ctx.last_q, q_close)

    def test_rejects_large_joint_jump(self):
        q_far = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        arm = MockArm(ik_solutions=[q_far])
        ctx = MockContext()
        config = TeleopConfig(max_joint_delta=0.1)
        ctrl = TeleopController(arm, ctx, config=config)
        ctrl.activate()

        ctrl.set_target_pose(np.eye(4))
        state = ctrl.step()

        assert state == TeleopState.UNREACHABLE
        assert ctx.step_count == 0

    def test_velocity_feedforward(self):
        q_solution = np.array([0.01, 0.02, 0.03, 0.01, 0.02, 0.03])
        arm = MockArm(ik_solutions=[q_solution])
        ctx = MockContext()
        ctrl = TeleopController(arm, ctx)
        ctrl.activate()

        ctrl.set_target_pose(np.eye(4))
        ctrl.step()

        assert ctx.last_qd is not None
        # Velocity should be (q_target - q_current) / dt
        expected_qd = q_solution / ctrl._config.twist_dt
        np.testing.assert_array_almost_equal(ctx.last_qd, expected_qd, decimal=1)


class TestIdleTimeout:

    def test_idle_after_timeout(self):
        q_solution = np.array([0.01] * 6)
        arm = MockArm(ik_solutions=[q_solution])
        ctx = MockContext()
        config = TeleopConfig(idle_timeout=0.01)
        ctrl = TeleopController(arm, ctx, config=config)
        ctrl.activate()

        ctrl.set_target_pose(np.eye(4))
        ctrl.step()
        assert ctrl.state == TeleopState.TRACKING

        time.sleep(0.02)
        state = ctrl.step()
        assert state == TeleopState.IDLE

    def test_no_idle_with_continuous_input(self):
        q_solution = np.array([0.01] * 6)
        arm = MockArm(ik_solutions=[q_solution])
        ctx = MockContext()
        config = TeleopConfig(idle_timeout=0.5)
        ctrl = TeleopController(arm, ctx, config=config)
        ctrl.activate()

        for _ in range(5):
            ctrl.set_target_pose(np.eye(4))
            state = ctrl.step()
            assert state == TeleopState.TRACKING


class TestInputSwitching:

    def test_pose_clears_twist(self):
        arm = MockArm(ik_solutions=[np.array([0.01] * 6)])
        ctx = MockContext()
        ctrl = TeleopController(arm, ctx)
        ctrl.activate()

        ctrl.set_target_twist(np.array([0.1, 0, 0, 0, 0, 0]))
        ctrl.set_target_pose(np.eye(4))
        state = ctrl.step()

        # Should use pose path (tracking via IK)
        assert state == TeleopState.TRACKING


class TestRecording:

    def test_record_frames(self):
        q_solution = np.array([0.01] * 6)
        arm = MockArm(ik_solutions=[q_solution])
        ctx = MockContext()
        ctrl = TeleopController(arm, ctx)
        ctrl.activate()

        ctrl.start_recording()
        assert ctrl.is_recording

        for _ in range(5):
            ctrl.set_target_pose(np.eye(4))
            ctrl.step()

        frames = ctrl.stop_recording()
        assert not ctrl.is_recording
        assert len(frames) == 5
        assert all(isinstance(f, TeleopFrame) for f in frames)

    def test_recording_captures_joint_positions(self):
        q_solution = np.array([0.01, 0.02, 0.03, 0.01, 0.02, 0.03])
        arm = MockArm(ik_solutions=[q_solution])
        ctx = MockContext()
        ctrl = TeleopController(arm, ctx)
        ctrl.activate()

        ctrl.start_recording()
        ctrl.set_target_pose(np.eye(4))
        ctrl.step()
        frames = ctrl.stop_recording()

        assert len(frames) == 1
        assert frames[0].ee_pose.shape == (4, 4)
        assert frames[0].gripper_position == 0.5

    def test_no_recording_when_idle(self):
        arm = MockArm(ik_solutions=None)
        ctx = MockContext()
        ctrl = TeleopController(arm, ctx)
        ctrl.activate()

        ctrl.start_recording()
        ctrl.set_target_pose(np.eye(4))
        ctrl.step()  # UNREACHABLE — should not record
        frames = ctrl.stop_recording()

        assert len(frames) == 0

    def test_deactivate_stops_recording(self):
        q_solution = np.array([0.01] * 6)
        arm = MockArm(ik_solutions=[q_solution])
        ctx = MockContext()
        ctrl = TeleopController(arm, ctx)
        ctrl.activate()
        ctrl.start_recording()
        ctrl.deactivate()
        assert not ctrl.is_recording

    def test_timestamps_monotonic(self):
        q_solution = np.array([0.01] * 6)
        arm = MockArm(ik_solutions=[q_solution])
        ctx = MockContext()
        ctrl = TeleopController(arm, ctx)
        ctrl.activate()
        ctrl.start_recording()

        for _ in range(3):
            ctrl.set_target_pose(np.eye(4))
            ctrl.step()

        frames = ctrl.stop_recording()
        for i in range(1, len(frames)):
            assert frames[i].timestamp >= frames[i - 1].timestamp
