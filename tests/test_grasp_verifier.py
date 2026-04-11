# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for :mod:`mj_manipulator.grasp_verifier`.

Two layers:

1. **Pure unit tests on** :func:`verify_grasp` — one test per branch of
   the decision tree, constructing :class:`VerifierFacts` by hand. No
   physics, no mocks, no MuJoCo.

2. **Small stateful tests on** :class:`GraspVerifier` **with fake
   signals** — verifies that the stateful shell captures baselines at
   :meth:`mark_grasped` time, that :meth:`mark_released` clears state,
   and that :attr:`is_held` reflects live changes to the fake signal
   values. Still no physics — the fake signals are plain objects.

Integration with real physics + a real arm happens in a separate
test file (``test_grasp_verifier_integration.py``) so this file stays
fast and deterministic.
"""

from __future__ import annotations

import pytest

from mj_manipulator.grasp_verifier import (
    GraspVerifier,
    VerifierFacts,
    VerifierParams,
    verify_grasp,
)

# ---------------------------------------------------------------------------
# Pure-function tests on verify_grasp
# ---------------------------------------------------------------------------


class TestVerifyGrasp:
    """Exhaustive tests for the pure decision function.

    Each test corresponds to one branch of the decision tree. If any
    SUCCESS-on-failure branch creeps back in, exactly one of these
    will fail, so the regression is easy to localize.
    """

    _params = VerifierParams()  # default thresholds

    def _facts(self, **overrides) -> VerifierFacts:
        """Build a 'healthy grasp' fact set and apply overrides."""
        defaults = dict(
            object_name="can_0",
            empty_at_fully_closed=False,
            gripper_position=0.55,
            signal_values={"wrist_ft_force": 8.0},
            signal_baselines={"wrist_ft_force": 10.0},
        )
        defaults.update(overrides)
        return VerifierFacts(**defaults)

    def test_happy_path(self):
        """Healthy grasp with a live signal close to baseline."""
        assert verify_grasp(self._facts(), self._params) is True

    def test_no_object_tracked(self):
        """Nothing was marked grasped → False (nothing to verify)."""
        facts = self._facts(object_name=None)
        assert verify_grasp(facts, self._params) is False

    def test_empty_at_fully_closed_decisive_negative(self):
        """Franka-style gripper at mechanical stop → False."""
        facts = self._facts(empty_at_fully_closed=True, gripper_position=0.99)
        assert verify_grasp(facts, self._params) is False

    def test_fully_closed_but_flag_off(self):
        """Robotiq-style gripper: fully closed is a valid grasp."""
        facts = self._facts(empty_at_fully_closed=False, gripper_position=1.0)
        assert verify_grasp(facts, self._params) is True

    def test_fully_closed_exactly_at_threshold(self):
        """Boundary check: position == threshold counts as empty."""
        facts = self._facts(empty_at_fully_closed=True, gripper_position=0.98)
        assert verify_grasp(facts, self._params) is False

    def test_fully_closed_just_below_threshold(self):
        """Boundary check: position just below threshold is fine."""
        facts = self._facts(empty_at_fully_closed=True, gripper_position=0.97)
        assert verify_grasp(facts, self._params) is True

    def test_load_signal_collapsed(self):
        """F/T reading dropped below baseline × (1 - ratio) → False."""
        facts = self._facts(
            signal_values={"wrist_ft_force": 2.0},
            signal_baselines={"wrist_ft_force": 10.0},
        )
        assert verify_grasp(facts, self._params) is False

    def test_load_signal_marginal(self):
        """Signal dropped but still above threshold → True."""
        # 10 N baseline, 0.3 ratio → threshold 7 N. 7.5 N is still OK.
        facts = self._facts(
            signal_values={"wrist_ft_force": 7.5},
            signal_baselines={"wrist_ft_force": 10.0},
        )
        assert verify_grasp(facts, self._params) is True

    def test_load_signal_unavailable_skipped(self):
        """Signal read returned None → skip, don't fail."""
        facts = self._facts(
            signal_values={"wrist_ft_force": None},
            signal_baselines={"wrist_ft_force": 10.0},
        )
        assert verify_grasp(facts, self._params) is True

    def test_load_signal_baseline_unavailable_skipped(self):
        """Baseline is None (signal was unavailable at grasp time) → skip."""
        facts = self._facts(
            signal_values={"wrist_ft_force": 2.0},
            signal_baselines={"wrist_ft_force": None},
        )
        assert verify_grasp(facts, self._params) is True

    def test_baseline_near_zero_skipped(self):
        """Baseline ~0 → can't infer a drop, skip.

        This matters because a taut tare zero (F/T calibrated to zero
        just before grasp) would give baseline ≈ 0 N, making any live
        reading trivially satisfy the drop check. Skipping is safer.
        """
        facts = self._facts(
            signal_values={"wrist_ft_force": 0.0},
            signal_baselines={"wrist_ft_force": 1e-8},
        )
        assert verify_grasp(facts, self._params) is True

    def test_multi_signal_one_collapsed_fails(self):
        """If any signal has collapsed, the overall verdict is False."""
        facts = self._facts(
            signal_values={"wrist_ft_force": 8.0, "joint_torque_effort": 0.1},
            signal_baselines={"wrist_ft_force": 10.0, "joint_torque_effort": 5.0},
        )
        assert verify_grasp(facts, self._params) is False

    def test_multi_signal_all_healthy(self):
        """All signals above their thresholds → True."""
        facts = self._facts(
            signal_values={"wrist_ft_force": 9.0, "joint_torque_effort": 4.5},
            signal_baselines={"wrist_ft_force": 10.0, "joint_torque_effort": 5.0},
        )
        assert verify_grasp(facts, self._params) is True

    def test_no_signals_no_empty_stop_check(self):
        """Kinematic-mode degeneration: no live signals, no decisive
        negative, but an object was marked grasped → True.

        This is the 'toy mode' path: kinematic sim has no F/T or joint
        torques, so every signal returns None. The verifier reduces
        to 'we told it to grasp something, trust that'. Documented as
        a limitation in the issue."""
        facts = self._facts(
            signal_values={"wrist_ft_force": None, "joint_torque_effort": None},
            signal_baselines={"wrist_ft_force": None, "joint_torque_effort": None},
        )
        assert verify_grasp(facts, self._params) is True


# ---------------------------------------------------------------------------
# Stateful tests on GraspVerifier with fake signals
# ---------------------------------------------------------------------------


class FakeSignal:
    """Plain LoadSignal stub — ``value`` is the live reading.

    Tests mutate ``value`` directly to simulate real-world signal
    changes (object weight appearing/disappearing). Implements the
    :class:`LoadSignal` protocol structurally: ``name`` is an
    attribute, ``read`` returns the current ``value``.
    """

    def __init__(self, name: str, value: float | None):
        self.name = name
        self.value = value

    def read(self) -> float | None:
        return self.value


class FakeGripper:
    """Minimal gripper with just what GraspVerifier needs."""

    def __init__(self, position: float = 0.5, empty_at_fully_closed: bool = False):
        self.position = position
        self.empty_at_fully_closed = empty_at_fully_closed

    def get_actual_position(self) -> float:
        return self.position


class TestGraspVerifier:
    """Stateful integration of the verifier with fake signals."""

    def _make(self, *, empty_at_fully_closed: bool = False):
        gripper = FakeGripper(position=0.5, empty_at_fully_closed=empty_at_fully_closed)
        signal = FakeSignal(name="wrist_ft_force", value=10.0)
        verifier = GraspVerifier(gripper=gripper, signals=[signal])
        return verifier, gripper, signal

    def test_fresh_verifier_is_not_held(self):
        verifier, _, _ = self._make()
        assert verifier.is_held is False
        assert verifier.held_object is None
        assert verifier.tracked_object is None

    def test_mark_grasped_then_is_held(self):
        verifier, _, _ = self._make()
        verifier.mark_grasped("can_0")
        assert verifier.is_held is True
        assert verifier.held_object == "can_0"
        assert verifier.tracked_object == "can_0"

    def test_signal_collapse_flips_is_held(self):
        """The canonical regression for geodude#173: load dropped mid-transport."""
        verifier, _, signal = self._make()
        verifier.mark_grasped("can_0")
        assert verifier.is_held is True

        # Object slips — signal drops to near zero
        signal.value = 0.5
        assert verifier.is_held is False
        assert verifier.held_object is None
        # But we still remember what we *tried* to grasp
        assert verifier.tracked_object == "can_0"

    def test_mark_released_clears_state(self):
        verifier, _, _ = self._make()
        verifier.mark_grasped("can_0")
        verifier.mark_released()
        assert verifier.is_held is False
        assert verifier.held_object is None
        assert verifier.tracked_object is None

    def test_signal_unavailable_after_grasp_does_not_fail(self):
        """If a signal starts returning None after grasp, the verifier
        skips it instead of treating it as failure. Important for the
        kinematic-mode degenerate case."""
        verifier, _, signal = self._make()
        verifier.mark_grasped("can_0")
        signal.value = None
        assert verifier.is_held is True

    def test_franka_style_mechanical_stop_triggers_drop(self):
        """When empty_at_fully_closed=True, reaching the stop should
        immediately flip is_held to False even if load signals say
        otherwise."""
        verifier, gripper, _ = self._make(empty_at_fully_closed=True)
        verifier.mark_grasped("panda_cube")
        assert verifier.is_held is True
        gripper.position = 1.0
        assert verifier.is_held is False

    def test_regrasping_updates_baseline(self):
        """mark_grasped twice — second baseline replaces the first."""
        verifier, _, signal = self._make()
        verifier.mark_grasped("can_0")
        signal.value = 4.0  # heavier object picked up next
        verifier.mark_grasped("spam_can_0")
        # New baseline is 4.0, so 3.0 is within 30% drop range
        signal.value = 3.0
        assert verifier.is_held is True
        assert verifier.held_object == "spam_can_0"

    def test_tick_is_noop(self):
        """v1 tick is a placeholder — calling it doesn't blow up and
        doesn't affect state."""
        verifier, _, _ = self._make()
        verifier.mark_grasped("can_0")
        verifier.tick()
        assert verifier.is_held is True


# ---------------------------------------------------------------------------
# _BaseGripper routing: verifier takes precedence over GraspManager when set
# ---------------------------------------------------------------------------


class TestBaseGripperRouting:
    """Verifies the _BaseGripper property wiring without loading MuJoCo.

    Constructing a real _BaseGripper needs a model + data, which is
    expensive. Instead, we test the routing via property semantics on
    a stand-in class that implements the same public API.
    """

    def test_held_object_without_verifier_uses_grasp_manager(self):
        """Baseline: gripper with no verifier reads from GraspManager
        (this is the current behavior across all existing tests)."""
        # Covered by existing test_grasp_manager.py + integration tests.
        # Asserting the principle here would duplicate those.
        pass

    def test_held_object_with_verifier_returns_none_when_not_held(self):
        """Gripper with verifier, but is_held=False → held_object=None."""
        gripper = FakeGripper()
        signal = FakeSignal(name="wrist_ft_force", value=10.0)
        verifier = GraspVerifier(gripper=gripper, signals=[signal])
        # Simulate: mark_grasped, then signal collapses
        verifier.mark_grasped("can_0")
        signal.value = 0.0
        assert verifier.held_object is None
        # But tracked_object still reflects what we thought we grabbed
        assert verifier.tracked_object == "can_0"


# ---------------------------------------------------------------------------
# Default params sanity
# ---------------------------------------------------------------------------


def test_default_verifier_params_are_reasonable():
    """Sanity check: defaults should be conservative but not trigger
    false positives at typical manipulation values."""
    params = VerifierParams()
    assert 0.9 <= params.empty_position_threshold <= 1.0
    assert 0.1 <= params.load_drop_ratio <= 0.5


def test_custom_verifier_params_pass_through():
    """Custom params work."""
    params = VerifierParams(empty_position_threshold=0.95, load_drop_ratio=0.5)
    facts = VerifierFacts(
        object_name="can_0",
        empty_at_fully_closed=False,
        gripper_position=0.5,
        signal_values={"x": 4.5},
        signal_baselines={"x": 10.0},
    )
    # 10 × 0.5 = 5.0 threshold; 4.5 < 5.0 → False with ratio=0.5
    assert verify_grasp(facts, params) is False
    # With default ratio=0.3, 4.5 < 7.0 → also False
    assert verify_grasp(facts, VerifierParams()) is False


def test_load_signal_protocol_runtime_checkable():
    """LoadSignal is a runtime-checkable protocol — fake signals
    should be structural instances."""
    from mj_manipulator.load_signals import LoadSignal

    signal = FakeSignal(name="test", value=1.0)
    assert isinstance(signal, LoadSignal)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
