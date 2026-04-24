# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for Outcome, FailureKind, and convenience constructors."""

import pytest

from mj_manipulator.outcome import FailureKind, Outcome, failure, success


class TestOutcomeInvariants:
    """Enforce the structural invariants on Outcome."""

    def test_success_with_no_failure_fields(self):
        o = Outcome(success=True)
        assert o.success
        assert o.failure_kind is None
        assert o.failure_code is None

    def test_success_with_details(self):
        o = Outcome(success=True, details={"distance_m": 0.003})
        assert o.success
        assert o.details["distance_m"] == 0.003

    def test_success_rejects_failure_kind(self):
        with pytest.raises(ValueError, match="success=True but failure_kind"):
            Outcome(success=True, failure_kind=FailureKind.TIMEOUT)

    def test_success_rejects_failure_code(self):
        with pytest.raises(ValueError, match="success=True but failure_code"):
            Outcome(success=True, failure_code="test:bad")

    def test_failure_requires_failure_kind(self):
        with pytest.raises(ValueError, match="success=False requires failure_kind"):
            Outcome(success=False)

    def test_failure_with_kind_only(self):
        o = Outcome(success=False, failure_kind=FailureKind.PLANNING_FAILED)
        assert not o.success
        assert o.failure_kind == FailureKind.PLANNING_FAILED
        assert o.failure_code is None

    def test_failure_with_kind_and_code(self):
        o = Outcome(
            success=False,
            failure_kind=FailureKind.SAFETY_ABORTED,
            failure_code="acquire:ft_exceeded",
            details={"force_n": 25.3},
        )
        assert o.failure_kind == FailureKind.SAFETY_ABORTED
        assert o.failure_code == "acquire:ft_exceeded"
        assert o.details["force_n"] == 25.3

    def test_frozen(self):
        o = Outcome(success=True)
        with pytest.raises(AttributeError):
            o.success = False


class TestOutcomeBool:
    """Outcome supports if/else via __bool__."""

    def test_success_is_truthy(self):
        assert Outcome(success=True)
        assert bool(Outcome(success=True)) is True

    def test_failure_is_falsy(self):
        o = Outcome(success=False, failure_kind=FailureKind.TIMEOUT)
        assert not o
        assert bool(o) is False

    def test_if_pattern(self):
        result = success()
        if result:
            passed = True
        else:
            passed = False
        assert passed

    def test_if_not_pattern(self):
        result = failure(FailureKind.PLANNING_FAILED, "test:no_plan")
        if not result:
            failed = True
        else:
            failed = False
        assert failed


class TestOutcomeRepr:
    def test_success_repr(self):
        assert repr(Outcome(success=True)) == "Outcome(success=True)"

    def test_failure_repr(self):
        o = failure(FailureKind.TIMEOUT, "servo:timeout")
        assert "TIMEOUT" not in repr(o)  # uses .value, not enum name
        assert "'timeout'" in repr(o)
        assert "'servo:timeout'" in repr(o)


class TestConvenienceConstructors:
    def test_success_no_details(self):
        o = success()
        assert o.success
        assert o.details == {}

    def test_success_with_kwargs(self):
        o = success(distance_m=0.003, contact=True)
        assert o.success
        assert o.details == {"distance_m": 0.003, "contact": True}

    def test_failure_minimal(self):
        o = failure(FailureKind.EXECUTION_FAILED)
        assert not o.success
        assert o.failure_kind == FailureKind.EXECUTION_FAILED
        assert o.failure_code is None

    def test_failure_with_code_and_details(self):
        o = failure(FailureKind.SAFETY_ABORTED, "test:force", force_n=25.0)
        assert o.failure_code == "test:force"
        assert o.details["force_n"] == 25.0


class TestFailureKind:
    """Every kind represents a distinct recovery strategy."""

    def test_all_kinds_have_unique_values(self):
        values = [k.value for k in FailureKind]
        assert len(values) == len(set(values))

    def test_branching_on_kind(self):
        """Tasks branch on FailureKind, not failure_code."""
        outcomes = [
            failure(FailureKind.PLANNING_FAILED, "a:1"),
            failure(FailureKind.SAFETY_ABORTED, "b:2"),
            failure(FailureKind.TIMEOUT, "c:3"),
        ]
        retryable = [o for o in outcomes if o.failure_kind != FailureKind.SAFETY_ABORTED]
        assert len(retryable) == 2
