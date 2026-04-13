# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for the velocity/acceleration safety layer in PhysicsController.

The safety layer enforces kinematic limits at the one chokepoint all
code paths pass through — the ``step()`` / ``step_reactive()`` methods
that write to ``data.ctrl``. Three response modes: WARN (log only),
CLAMP (limit the command), FAULT (halt the arm).

Uses a minimal MuJoCo model with a single revolute joint so the tests
are fast and don't depend on any robot-specific assets.
"""

from __future__ import annotations

import logging

import mujoco
import numpy as np
import pytest

from mj_manipulator.config import (
    KinematicLimits,
    PhysicsExecutionConfig,
    SafetyResponse,
)
from mj_manipulator.physics_controller import PhysicsController

# ---------------------------------------------------------------------------
# Minimal MuJoCo model: 2-DOF arm with position actuators
# ---------------------------------------------------------------------------

_MINIMAL_XML = """
<mujoco>
  <option timestep="0.001"/>
  <worldbody>
    <body>
      <joint name="j0" type="hinge" axis="0 0 1"/>
      <geom size="0.05 0.3" type="capsule"/>
      <body pos="0 0 0.6">
        <joint name="j1" type="hinge" axis="0 1 0"/>
        <geom size="0.04 0.2" type="capsule"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <general name="a0" joint="j0" gainprm="100" biastype="affine" biasprm="0 -100 -5"/>
    <general name="a1" joint="j1" gainprm="100" biastype="affine" biasprm="0 -100 -5"/>
  </actuator>
</mujoco>
"""


class _FakeArm:
    """Minimal arm stub with kinematic limits for safety tests."""

    class _Config:
        def __init__(self, name, limits):
            self.name = name
            self.kinematic_limits = limits
            self.joint_names = ["j0", "j1"]

    def __init__(self, model, data, *, vel_limit=2.0, acc_limit=20.0):
        self._model = model
        self._data = data
        self.config = self._Config(
            "test_arm",
            KinematicLimits(
                velocity=np.array([vel_limit, vel_limit]),
                acceleration=np.array([acc_limit, acc_limit]),
            ),
        )
        self.joint_qpos_indices = [0, 1]
        self.joint_qvel_indices = [0, 1]
        self.actuator_ids = [0, 1]
        self.dof = 2
        self.gripper = None

    def get_joint_positions(self):
        return self._data.qpos[:2].copy()


@pytest.fixture
def model_and_data():
    model = mujoco.MjModel.from_xml_string(_MINIMAL_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


@pytest.fixture
def arm(model_and_data):
    model, data = model_and_data
    return _FakeArm(model, data)


def _make_controller(model, data, arm, *, safety_response=SafetyResponse.CLAMP):
    config = PhysicsExecutionConfig(safety_response=safety_response)
    return PhysicsController(
        model,
        data,
        {"test_arm": arm},
        config=config,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNoViolation:
    def test_small_velocity_passes_through(self, model_and_data, arm):
        model, data = model_and_data
        ctrl = _make_controller(model, data, arm)
        ctrl.set_arm_target("test_arm", np.array([0.1, 0.1]), np.array([1.0, 1.0]))
        ctrl.step()
        # q_cmd = 0.1 + 0.1 * 1.0 = 0.2 per joint
        expected = np.array([0.1 + 0.1 * 1.0, 0.1 + 0.1 * 1.0])
        np.testing.assert_allclose(data.ctrl[:2], expected, atol=1e-6)


class TestVelocityClamp:
    def test_velocity_clamped(self, model_and_data, arm, caplog):
        model, data = model_and_data
        ctrl = _make_controller(model, data, arm)
        # Command 5.0 rad/s on joint 0, limit is 2.0
        ctrl.set_arm_target("test_arm", np.array([0.0, 0.0]), np.array([5.0, 1.0]))
        with caplog.at_level(logging.WARNING):
            ctrl.step()
        # Velocity should be clamped to 2.0, q_cmd = 0.0 + 0.1 * 2.0 = 0.2
        assert data.ctrl[0] == pytest.approx(0.0 + 0.1 * 2.0, abs=1e-6)
        # Joint 1 is within limits, q_cmd = 0.0 + 0.1 * 1.0 = 0.1
        assert data.ctrl[1] == pytest.approx(0.0 + 0.1 * 1.0, abs=1e-6)
        assert "velocity" in caplog.text.lower()


class TestAccelerationClamp:
    def test_acceleration_clamped(self, model_and_data, arm, caplog):
        model, data = model_and_data
        ctrl = _make_controller(model, data, arm)
        # First step: establish a baseline (first_step flag clears)
        ctrl.set_arm_target("test_arm", np.array([0.0, 0.0]), np.array([0.5, 0.0]))
        ctrl.step()
        # Second step: also non-zero, clears first_step
        ctrl.set_arm_target("test_arm", np.array([0.0, 0.0]), np.array([0.5, 0.0]))
        ctrl.step()
        # Third step: velocity jumps from 0.5 to 2.0 (within vel limit)
        # acceleration = (2.0 - 0.5) / 0.008 = 187.5 rad/s² (exceeds limit of 20)
        ctrl.set_arm_target("test_arm", np.array([0.0, 0.0]), np.array([2.0, 0.0]))
        with caplog.at_level(logging.WARNING):
            ctrl.step()
        # Acceleration clamped: max_delta = 20.0 * 0.008 = 0.16 rad/s
        # Clamped velocity = 0.5 + 0.16 = 0.66
        assert data.ctrl[0] == pytest.approx(0.0 + 0.1 * 0.66, abs=1e-3)
        assert "acceleration" in caplog.text.lower()


class TestWarnMode:
    def test_warn_does_not_modify_command(self, model_and_data, arm, caplog):
        model, data = model_and_data
        ctrl = _make_controller(model, data, arm, safety_response=SafetyResponse.WARN)
        ctrl.set_arm_target("test_arm", np.array([0.0, 0.0]), np.array([5.0, 5.0]))
        with caplog.at_level(logging.WARNING):
            ctrl.step()
        # Command passes through unclamped: q_cmd = 0.0 + 0.1 * 5.0 = 0.5
        assert data.ctrl[0] == pytest.approx(0.5, abs=1e-6)
        assert data.ctrl[1] == pytest.approx(0.5, abs=1e-6)
        assert "velocity" in caplog.text.lower()


class TestFaultMode:
    def test_fault_halts_arm(self, model_and_data, arm, caplog):
        model, data = model_and_data
        ctrl = _make_controller(model, data, arm, safety_response=SafetyResponse.FAULT)
        ctrl.set_arm_target("test_arm", np.array([0.5, 0.5]), np.array([5.0, 5.0]))
        with caplog.at_level(logging.ERROR):
            ctrl.step()
        # Arm should hold current qpos (≈0, since we haven't moved)
        assert abs(data.ctrl[0]) < 0.01
        assert abs(data.ctrl[1]) < 0.01
        assert "FAULT" in caplog.text
        # Verify faulted flag
        state = ctrl._arms["test_arm"]
        assert state.faulted is True

    def test_faulted_arm_ignores_commands(self, model_and_data, arm):
        model, data = model_and_data
        ctrl = _make_controller(model, data, arm, safety_response=SafetyResponse.FAULT)
        # Trigger fault
        ctrl.set_arm_target("test_arm", np.array([0.0, 0.0]), np.array([5.0, 5.0]))
        ctrl.step()
        # Now send a normal command — should still hold
        ctrl.set_arm_target("test_arm", np.array([1.0, 1.0]), np.array([0.5, 0.5]))
        ctrl.step()
        assert abs(data.ctrl[0]) < 0.01  # still holding ~0

    def test_clear_fault_resumes(self, model_and_data, arm):
        model, data = model_and_data
        ctrl = _make_controller(model, data, arm, safety_response=SafetyResponse.FAULT)
        # Trigger fault
        ctrl.set_arm_target("test_arm", np.array([0.0, 0.0]), np.array([5.0, 5.0]))
        ctrl.step()
        assert ctrl._arms["test_arm"].faulted is True
        # Clear fault
        ctrl.clear_fault("test_arm")
        assert ctrl._arms["test_arm"].faulted is False
        # Normal command should work
        ctrl.set_arm_target("test_arm", np.array([0.5, 0.5]), np.array([1.0, 1.0]))
        ctrl.step()
        expected = 0.5 + 0.1 * 1.0  # within limits
        assert data.ctrl[0] == pytest.approx(expected, abs=0.1)


class TestNoLimitsConfigured:
    def test_mock_arm_without_limits_skips_check(self, model_and_data):
        """Arms without kinematic_limits (test mocks) skip the safety check."""
        model, data = model_and_data

        class _BareArm:
            class _Config:
                name = "bare_arm"

            config = _Config()
            joint_qpos_indices = [0, 1]
            joint_qvel_indices = [0, 1]
            actuator_ids = [0, 1]
            dof = 2
            gripper = None

            def get_joint_positions(self):
                return data.qpos[:2].copy()

        ctrl = _make_controller(model, data, _BareArm())
        # Extreme velocity — should pass through unchecked
        ctrl.set_arm_target("test_arm", np.array([0.0, 0.0]), np.array([100.0, 100.0]))
        ctrl.step()
        assert data.ctrl[0] == pytest.approx(0.0 + 0.1 * 100.0, abs=1e-6)


class TestFirstCycleGraceful:
    def test_first_step_with_nonzero_velocity(self, model_and_data, arm):
        """First cycle has prev_velocity=0. A moderate velocity shouldn't
        trigger acceleration violation if within velocity limits."""
        model, data = model_and_data
        # Use tight acceleration limit
        arm_tight = _FakeArm(model, data, vel_limit=2.0, acc_limit=500.0)
        ctrl = _make_controller(model, data, arm_tight)
        ctrl.set_arm_target("test_arm", np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        ctrl.step()
        # vel=1.0, accel=1.0/0.008=125 < 500 → no violation
        assert data.ctrl[0] == pytest.approx(0.0 + 0.1 * 1.0, abs=1e-6)
