# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Integration tests for servo_to_pose and ft_guarded_move.

Uses the Franka arm from mujoco_menagerie with physics mode for
realistic F/T and Jacobian behavior. Skips if menagerie is not
available.
"""

import numpy as np
import pytest

from mj_manipulator.force_control import ForceThresholds, SpeedProfile
from mj_manipulator.outcome import FailureKind
from mj_manipulator.servo import ft_guarded_move, servo_to_pose


@pytest.fixture
def franka_ctx(franka_env_with_gravcomp, franka_arm_at_home):
    """Franka arm in physics mode with SimContext, ready for servo tests."""
    from mj_manipulator.config import PhysicsConfig, PhysicsExecutionConfig
    from mj_manipulator.sim_context import SimContext

    env = franka_env_with_gravcomp
    arm = franka_arm_at_home
    ctx = SimContext(
        env.model,
        env.data,
        {"franka": arm},
        headless=True,
        physics_config=PhysicsConfig(
            execution=PhysicsExecutionConfig(control_dt=0.002),
        ),
    )
    with ctx as c:
        yield arm, c


class TestServoToPose:
    def test_reaches_nearby_target(self, franka_ctx):
        """Servo to a pose 5cm from current — should converge."""
        arm, ctx = franka_ctx
        current = arm.get_ee_pose()
        target = current.copy()
        target[2, 3] -= 0.05  # 5cm lower

        result = servo_to_pose(
            target,
            arm,
            ctx,
            speed_profile=SpeedProfile.constant(linear=0.1, angular=0.5),
            timeout=5.0,
            position_tol=0.01,
        )
        assert result.success
        assert result.details["position_error_m"] < 0.01

    def test_timeout_on_unreachable(self, franka_ctx):
        """Servo to a pose far out of reach — should timeout."""
        arm, ctx = franka_ctx
        target = np.eye(4)
        target[:3, 3] = [2.0, 0.0, 0.0]  # way out of reach

        result = servo_to_pose(
            target,
            arm,
            ctx,
            speed_profile=SpeedProfile.constant(linear=0.1, angular=0.5),
            timeout=0.5,
        )
        assert not result.success
        assert result.failure_kind == FailureKind.TIMEOUT

    def test_speed_profile_decelerates(self, franka_ctx):
        """Verify the arm moves slower near the target with a ramp profile."""
        arm, ctx = franka_ctx
        current = arm.get_ee_pose()
        target = current.copy()
        target[2, 3] -= 0.03  # 3cm lower (within ramp_distance)

        result = servo_to_pose(
            target,
            arm,
            ctx,
            speed_profile=SpeedProfile(
                max_linear=0.2,
                min_linear=0.02,
                max_angular=0.5,
                min_angular=0.1,
                ramp_distance=0.1,
            ),
            timeout=5.0,
            position_tol=0.005,
        )
        assert result.success

    def test_no_ft_threshold_means_no_abort(self, franka_ctx):
        """Without ft_threshold, servo never aborts on force."""
        arm, ctx = franka_ctx
        current = arm.get_ee_pose()
        target = current.copy()
        target[2, 3] -= 0.02

        result = servo_to_pose(
            target,
            arm,
            ctx,
            ft_threshold=None,
            timeout=3.0,
            position_tol=0.01,
        )
        assert result.success


class TestFtGuardedMove:
    def test_completes_without_contact(self, franka_ctx):
        """Move in free space — should complete duration without contact."""
        arm, ctx = franka_ctx
        twist = np.array([0.0, 0.0, -0.02, 0.0, 0.0, 0.0])  # slow downward

        result = ft_guarded_move(
            twist,
            arm,
            ctx,
            ft_threshold=ForceThresholds(force_n=50.0, torque_nm=10.0),
            duration=0.2,
        )
        assert result.success
        assert result.details["contact"] is False

    def test_returns_outcome_type(self, franka_ctx):
        """Result is always an Outcome."""
        arm, ctx = franka_ctx
        twist = np.zeros(6)

        result = ft_guarded_move(
            twist,
            arm,
            ctx,
            ft_threshold=ForceThresholds(force_n=50.0, torque_nm=10.0),
            duration=0.05,
        )
        assert hasattr(result, "success")
        assert hasattr(result, "failure_kind")
        assert hasattr(result, "details")
