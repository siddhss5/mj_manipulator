# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for :mod:`mj_manipulator.load_signals`.

Each signal encapsulates a small amount of per-signal projection
logic (e.g. \"project to world Z\", \"take norm of joint torques\").
The tests pin that logic against fake Arm / Gripper stubs so the
axis conventions can't silently drift.

Integration on a real MuJoCo arm is covered by
``test_grasp_verifier.py`` (for the protocol) and end-to-end
recycling demo runs (for the numeric plausibility of the baseline
values in physics sim).
"""

from __future__ import annotations

import numpy as np

from mj_manipulator.load_signals import (
    GripperPositionSignal,
    JointTorqueSignal,
    LoadSignal,
    WristFTSignal,
)

# ---------------------------------------------------------------------------
# Fake Arm / Gripper for signal tests
# ---------------------------------------------------------------------------


class FakeArm:
    """Minimal Arm stub: exposes the attributes the signal classes read."""

    def __init__(
        self,
        *,
        has_ft_sensor: bool = True,
        wrench_world: np.ndarray | None = None,
        joint_torques: np.ndarray | None = None,
    ):
        self.has_ft_sensor = has_ft_sensor
        self._wrench_world = wrench_world
        self._joint_torques = joint_torques

    def get_ft_wrench_world(self) -> np.ndarray:
        if self._wrench_world is None:
            return np.full(6, np.nan)
        return self._wrench_world

    def get_joint_torques(self) -> np.ndarray:
        if self._joint_torques is None:
            return np.full(6, np.nan)
        return self._joint_torques


class FakeGripper:
    def __init__(self, position: float = 0.5):
        self.position = position

    def get_actual_position(self) -> float:
        return self.position


# ---------------------------------------------------------------------------
# GripperPositionSignal
# ---------------------------------------------------------------------------


class TestGripperPositionSignal:
    def test_reads_current_position(self):
        gripper = FakeGripper(position=0.42)
        signal = GripperPositionSignal(gripper)
        assert signal.read() == 0.42

    def test_returns_none_on_exception(self):
        """If the gripper raises for any reason (disconnected hardware,
        stale model handle), the signal returns None so the verifier
        skips it rather than propagating the exception."""

        class BrokenGripper:
            def get_actual_position(self):
                raise RuntimeError("nope")

        signal = GripperPositionSignal(BrokenGripper())
        assert signal.read() is None

    def test_satisfies_protocol(self):
        signal = GripperPositionSignal(FakeGripper())
        assert isinstance(signal, LoadSignal)
        assert signal.name == "gripper_position"


# ---------------------------------------------------------------------------
# WristFTSignal — the interesting one
# ---------------------------------------------------------------------------


class TestWristFTSignal:
    """Pin the world-Z projection convention.

    The F/T sensor's local frame rotates with the wrist flange, so
    the raw local reading projects gravity differently depending on
    arm pose. Reading world Z (via :meth:`Arm.get_ft_wrench_world`)
    gives a pose-independent \"how much vertical load is the arm
    carrying\" that's directly comparable across poses.

    The signal returns the Z component directly (not the norm, not
    the absolute value) so the verifier's ``abs(val) < abs(base)``
    comparison can work symmetrically regardless of the sign
    convention MuJoCo's :class:`mujoco.force` sensor uses for the
    child-to-parent wrench.
    """

    def test_returns_world_z_component(self):
        """Signal.read() should return exactly wrench_world[2], not
        the magnitude, not the L2 norm, not any other projection."""
        arm = FakeArm(wrench_world=np.array([1.5, -2.3, -0.7, 0.1, 0.2, 0.3]))
        signal = WristFTSignal(arm)
        assert signal.read() == -0.7  # the Z force, signed

    def test_ignores_x_and_y_forces(self):
        """A held object only shows up in Z (gravity axis). Lateral
        contact forces on X/Y should not contaminate the signal."""
        arm = FakeArm(wrench_world=np.array([100.0, 100.0, -0.5, 0, 0, 0]))
        signal = WristFTSignal(arm)
        assert signal.read() == -0.5

    def test_ignores_torques(self):
        """Linear forces only — torques live in wrench_world[3:] and
        must not contaminate the scalar."""
        arm = FakeArm(wrench_world=np.array([0.0, 0.0, -1.0, 50.0, 50.0, 50.0]))
        signal = WristFTSignal(arm)
        assert signal.read() == -1.0

    def test_returns_none_when_arm_lacks_sensor(self):
        """Franka / arms without wrist F/T return None so the
        verifier skips the signal entirely."""
        arm = FakeArm(has_ft_sensor=False)
        signal = WristFTSignal(arm)
        assert signal.read() is None

    def test_returns_none_in_kinematic_mode(self):
        """When ``ft_valid`` is False, get_ft_wrench_world() returns
        all-NaN, and the signal should translate that to None."""
        arm = FakeArm(has_ft_sensor=True, wrench_world=None)  # None → NaN fallback
        signal = WristFTSignal(arm)
        assert signal.read() is None

    def test_satisfies_protocol(self):
        arm = FakeArm(wrench_world=np.zeros(6))
        signal = WristFTSignal(arm)
        assert isinstance(signal, LoadSignal)
        assert signal.name == "wrist_ft_force_z"

    def test_signed_value_flows_through_verifier_decision(self):
        """Regression test for the sign convention: verify_grasp uses
        abs(val) < abs(base), so signed-Z should work symmetrically.
        If someone later 'fixes' the signal to return abs(Z), this
        test catches the silent information loss."""
        from mj_manipulator.grasp_verifier import (
            VerifierFacts,
            VerifierParams,
            verify_grasp,
        )

        # Baseline was -0.7 N (gravity pulling a held object down).
        # Live reading is -0.05 N (object dropped, only tare residuals).
        # abs(0.05) < abs(0.7) * 0.7 = 0.49 → object lost.
        facts = VerifierFacts(
            object_name="can_0",
            empty_at_fully_closed=False,
            gripper_position=0.5,
            signal_values={"wrist_ft_force_z": -0.05},
            signal_baselines={"wrist_ft_force_z": -0.7},
        )
        assert verify_grasp(facts, VerifierParams()) is False


# ---------------------------------------------------------------------------
# JointTorqueSignal
# ---------------------------------------------------------------------------


class TestJointTorqueSignal:
    def test_returns_norm_of_joint_torques(self):
        arm = FakeArm(joint_torques=np.array([3.0, 4.0, 0.0, 0.0, 0.0, 0.0]))
        signal = JointTorqueSignal(arm)
        assert signal.read() == 5.0

    def test_returns_none_when_nan(self):
        arm = FakeArm(joint_torques=np.full(6, np.nan))
        signal = JointTorqueSignal(arm)
        assert signal.read() is None

    def test_returns_none_on_empty_array(self):
        arm = FakeArm(joint_torques=np.array([]))
        signal = JointTorqueSignal(arm)
        assert signal.read() is None

    def test_satisfies_protocol(self):
        arm = FakeArm(joint_torques=np.zeros(6))
        signal = JointTorqueSignal(arm)
        assert isinstance(signal, LoadSignal)
        assert signal.name == "joint_torque_effort"
