# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Structured outcome types for manipulation behaviors.

Every behavior and primitive returns an ``Outcome`` instead of a raw bool.
Tasks branch on ``FailureKind`` (the category), never on ``failure_code``
(the specific condition). This gives callers a fixed set of recovery
strategies while preserving full diagnostic detail.

Usage::

    result = servo_to_pose(target, arm, ctx, ft_threshold=threshold)
    if not result.success:
        if result.failure_kind == FailureKind.SAFETY_ABORTED:
            return result  # don't retry safety
        if result.failure_kind == FailureKind.TIMEOUT:
            logger.warning("Servo timed out: %s", result.failure_code)

    # result.details has numeric context for logging/analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FailureKind(Enum):
    """Category of failure. Tasks branch on these, never on failure_code.

    Each kind maps to a recovery strategy:

    - PRECONDITION_FAILED: missing input, check prerequisites
    - PERCEPTION_FAILED: retry detection, use fallback
    - PLANNING_FAILED: try different goal, relax constraints
    - EXECUTION_FAILED: retry, recover to safe state
    - SAFETY_ABORTED: do NOT retry, escalate to operator
    - VERIFICATION_FAILED: action ran but outcome unconfirmed
    - TIMEOUT: retry with longer timeout or different approach
    """

    PRECONDITION_FAILED = "precondition_failed"
    PERCEPTION_FAILED = "perception_failed"
    PLANNING_FAILED = "planning_failed"
    EXECUTION_FAILED = "execution_failed"
    SAFETY_ABORTED = "safety_aborted"
    VERIFICATION_FAILED = "verification_failed"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class Outcome:
    """Structured result from a behavior or primitive.

    Invariants:
    - ``success=True`` implies ``failure_kind`` and ``failure_code`` are None
    - ``success=False`` requires ``failure_kind`` to be set
    - Tasks branch on ``failure_kind``, never on ``failure_code``
    - ``failure_code`` format: ``"behavior_name:specific_condition"``
    - ``details`` carries numeric context (forces, distances, durations)

    Examples::

        # Success
        Outcome(success=True, details={"distance_m": 0.003})

        # Planning failure
        Outcome(
            success=False,
            failure_kind=FailureKind.PLANNING_FAILED,
            failure_code="move_above:no_ik_solution",
            details={"target_pose": pose.tolist()},
        )

        # Safety abort
        Outcome(
            success=False,
            failure_kind=FailureKind.SAFETY_ABORTED,
            failure_code="acquire_food:ft_exceeded",
            details={"force_n": 25.3, "threshold_n": 20.0},
        )
    """

    success: bool
    failure_kind: FailureKind | None = None
    failure_code: str | None = None
    details: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.success:
            if self.failure_kind is not None:
                raise ValueError("success=True but failure_kind is set")
            if self.failure_code is not None:
                raise ValueError("success=True but failure_code is set")
        else:
            if self.failure_kind is None:
                raise ValueError(
                    "success=False requires failure_kind. Use FailureKind to indicate the failure category."
                )

    def __bool__(self) -> bool:
        """Allow ``if result:`` as shorthand for ``if result.success:``."""
        return self.success

    def __repr__(self) -> str:
        if self.success:
            return "Outcome(success=True)"
        return f"Outcome(success=False, failure_kind={self.failure_kind.value!r}, failure_code={self.failure_code!r})"


# Convenience constructors for common outcomes
def success(**details) -> Outcome:
    """Create a successful outcome."""
    return Outcome(success=True, details=details)


def failure(
    kind: FailureKind,
    code: str | None = None,
    **details,
) -> Outcome:
    """Create a failed outcome."""
    return Outcome(
        success=False,
        failure_kind=kind,
        failure_code=code,
        details=details,
    )
