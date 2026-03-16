"""Tests for manipulation primitives (pickup, place).

Uses mocks to test orchestration logic without MuJoCo or real planning.
"""

from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest

from mj_manipulator.primitives import pickup, place
from mj_manipulator.trajectory import Trajectory


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


_UNSET = object()


@dataclass
class _MockConfig:
    name: str


def _make_trajectory() -> Trajectory:
    """Minimal 2-waypoint trajectory for mock returns."""
    return Trajectory(
        timestamps=np.array([0.0, 1.0]),
        positions=np.zeros((2, 2)),
        velocities=np.zeros((2, 2)),
        accelerations=np.zeros((2, 2)),
        joint_names=["j1", "j2"],
        entity="test_arm",
    )


def _make_arm(
    *,
    plan_tsrs_result=_UNSET,
    plan_pose_result=_UNSET,
) -> MagicMock:
    """Create a mock Arm with configurable planning results."""
    arm = MagicMock()
    arm.config = _MockConfig(name="test_arm")

    # Default: return a geometric path (list of arrays)
    if plan_tsrs_result is _UNSET:
        plan_tsrs_result = [np.zeros(2), np.ones(2)]
    arm.plan_to_tsrs.return_value = plan_tsrs_result

    if plan_pose_result is _UNSET:
        plan_pose_result = [np.zeros(2), np.ones(2)]
    arm.plan_to_pose.return_value = plan_pose_result

    arm.retime.return_value = _make_trajectory()
    arm.get_ee_pose.return_value = np.eye(4)

    return arm


def _make_context(*, execute_ok: bool = True, grasp_result="mug") -> MagicMock:
    """Create a mock ExecutionContext."""
    ctx = MagicMock()
    ctx.execute.return_value = execute_ok

    arm_ctrl = MagicMock()
    arm_ctrl.grasp.return_value = grasp_result
    ctx.arm.return_value = arm_ctrl

    return ctx


# ---------------------------------------------------------------------------
# pickup tests
# ---------------------------------------------------------------------------


class TestPickup:
    """Tests for the pickup primitive."""

    def test_success(self):
        arm = _make_arm()
        ctx = _make_context(grasp_result="mug")
        tsrs = [MagicMock()]

        result = pickup(ctx, arm, "mug", tsrs)

        assert result is True
        arm.plan_to_tsrs.assert_called_once_with(
            tsrs, constraint_tsrs=None, timeout=30.0,
        )
        ctx.arm("test_arm").grasp.assert_called_once_with("mug")

    def test_planning_fails(self):
        arm = _make_arm(plan_tsrs_result=None)
        ctx = _make_context()
        tsrs = [MagicMock()]

        result = pickup(ctx, arm, "mug", tsrs)

        assert result is False
        ctx.execute.assert_not_called()

    def test_execution_fails(self):
        arm = _make_arm()
        ctx = _make_context(execute_ok=False)
        tsrs = [MagicMock()]

        result = pickup(ctx, arm, "mug", tsrs)

        assert result is False
        # grasp should not be attempted after execution failure
        ctx.arm("test_arm").grasp.assert_not_called()

    def test_grasp_fails(self):
        arm = _make_arm()
        ctx = _make_context(grasp_result=None)
        tsrs = [MagicMock()]

        result = pickup(ctx, arm, "mug", tsrs)

        assert result is False

    def test_lift_planning_fails_still_succeeds(self):
        """Grasp succeeded but lift planning fails — should still return True."""
        arm = _make_arm(plan_pose_result=None)
        ctx = _make_context(grasp_result="mug")
        tsrs = [MagicMock()]

        result = pickup(ctx, arm, "mug", tsrs)

        assert result is True

    def test_constraint_tsrs_passed_through(self):
        arm = _make_arm()
        ctx = _make_context(grasp_result="mug")
        goal_tsrs = [MagicMock()]
        constraint = [MagicMock()]

        pickup(ctx, arm, "mug", goal_tsrs, constraint_tsrs=constraint)

        arm.plan_to_tsrs.assert_called_once_with(
            goal_tsrs, constraint_tsrs=constraint, timeout=30.0,
        )

    def test_custom_timeout(self):
        arm = _make_arm()
        ctx = _make_context(grasp_result="mug")
        tsrs = [MagicMock()]

        pickup(ctx, arm, "mug", tsrs, timeout=60.0)

        arm.plan_to_tsrs.assert_called_once_with(
            tsrs, constraint_tsrs=None, timeout=60.0,
        )

    def test_no_lift_when_zero_height(self):
        arm = _make_arm()
        ctx = _make_context(grasp_result="mug")
        tsrs = [MagicMock()]

        pickup(ctx, arm, "mug", tsrs, lift_height=0.0)

        # plan_to_pose should not be called (no lift)
        arm.plan_to_pose.assert_not_called()

    def test_lift_pose_is_above_current(self):
        """Lift target should be current EE pose + lift_height in Z."""
        ee_pose = np.eye(4)
        ee_pose[:3, 3] = [0.5, 0.1, 0.3]

        arm = _make_arm()
        arm.get_ee_pose.return_value = ee_pose
        ctx = _make_context(grasp_result="mug")
        tsrs = [MagicMock()]

        pickup(ctx, arm, "mug", tsrs, lift_height=0.10)

        call_args = arm.plan_to_pose.call_args
        target_pose = call_args[0][0]
        assert target_pose[2, 3] == pytest.approx(0.4)  # 0.3 + 0.10
        assert target_pose[0, 3] == pytest.approx(0.5)  # X unchanged
        assert target_pose[1, 3] == pytest.approx(0.1)  # Y unchanged


# ---------------------------------------------------------------------------
# place tests
# ---------------------------------------------------------------------------


class TestPlace:
    """Tests for the place primitive."""

    def test_success(self):
        arm = _make_arm()
        ctx = _make_context()
        tsrs = [MagicMock()]

        result = place(ctx, arm, tsrs, object_name="mug")

        assert result is True
        arm.plan_to_tsrs.assert_called_once_with(
            tsrs, constraint_tsrs=None, timeout=30.0,
        )
        ctx.arm("test_arm").release.assert_called_once_with("mug")

    def test_planning_fails(self):
        arm = _make_arm(plan_tsrs_result=None)
        ctx = _make_context()
        tsrs = [MagicMock()]

        result = place(ctx, arm, tsrs)

        assert result is False
        ctx.execute.assert_not_called()

    def test_execution_fails(self):
        arm = _make_arm()
        ctx = _make_context(execute_ok=False)
        tsrs = [MagicMock()]

        result = place(ctx, arm, tsrs)

        assert result is False
        # release should not be attempted after execution failure
        ctx.arm("test_arm").release.assert_not_called()

    def test_release_with_none_object(self):
        """object_name=None should pass None to release."""
        arm = _make_arm()
        ctx = _make_context()
        tsrs = [MagicMock()]

        place(ctx, arm, tsrs, object_name=None)

        ctx.arm("test_arm").release.assert_called_once_with(None)

    def test_retract_planning_fails_still_succeeds(self):
        """Release succeeded but retract planning fails — still True."""
        arm = _make_arm(plan_pose_result=None)
        ctx = _make_context()
        tsrs = [MagicMock()]

        result = place(ctx, arm, tsrs)

        assert result is True

    def test_no_retract_when_zero_height(self):
        arm = _make_arm()
        ctx = _make_context()
        tsrs = [MagicMock()]

        place(ctx, arm, tsrs, retract_height=0.0)

        arm.plan_to_pose.assert_not_called()

    def test_constraint_tsrs_passed_through(self):
        arm = _make_arm()
        ctx = _make_context()
        goal_tsrs = [MagicMock()]
        constraint = [MagicMock()]

        place(ctx, arm, goal_tsrs, constraint_tsrs=constraint)

        arm.plan_to_tsrs.assert_called_once_with(
            goal_tsrs, constraint_tsrs=constraint, timeout=30.0,
        )

    def test_retract_pose_is_above_current(self):
        ee_pose = np.eye(4)
        ee_pose[:3, 3] = [0.4, -0.2, 0.8]

        arm = _make_arm()
        arm.get_ee_pose.return_value = ee_pose
        ctx = _make_context()
        tsrs = [MagicMock()]

        place(ctx, arm, tsrs, retract_height=0.07)

        call_args = arm.plan_to_pose.call_args
        target_pose = call_args[0][0]
        assert target_pose[2, 3] == pytest.approx(0.87)  # 0.8 + 0.07
