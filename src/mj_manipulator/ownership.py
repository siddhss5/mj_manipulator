# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Arm ownership registry for concurrent multi-arm control.

Tracks which controller (teleop, trajectory, gripper) owns each arm,
with per-arm abort flags. Enables bimanual concurrent control: teleop
one arm while the other executes a trajectory.

Usage::

    registry = OwnershipRegistry(["left", "right"])

    # Trajectory acquires an arm
    registry.acquire("right", OwnerKind.TRAJECTORY, runner)

    # Teleop preempts (aborts trajectory, acquires arm)
    registry.preempt("right", OwnerKind.TELEOP, controller)

    # Per-arm abort (doesn't affect other arms)
    registry.set_abort("left")
"""

from __future__ import annotations

import logging
import threading
from enum import Enum

logger = logging.getLogger(__name__)


class OwnerKind(Enum):
    """Type of controller that can own an arm."""

    IDLE = "idle"
    TELEOP = "teleop"
    TRAJECTORY = "trajectory"
    GRIPPER = "gripper"


# Priority order: higher value can preempt lower.
_PRIORITY = {
    OwnerKind.IDLE: 0,
    OwnerKind.TRAJECTORY: 1,
    OwnerKind.TELEOP: 2,
    OwnerKind.GRIPPER: 3,
}


class OwnershipRegistry:
    """Per-arm ownership and abort tracking.

    Thread-safe. All public methods acquire the internal lock.

    Args:
        arm_names: Names of the arms to track.
    """

    def __init__(self, arm_names: list[str]) -> None:
        self._lock = threading.Lock()
        self._owners: dict[str, tuple[OwnerKind, object | None]] = {name: (OwnerKind.IDLE, None) for name in arm_names}
        self._abort_events: dict[str, threading.Event] = {name: threading.Event() for name in arm_names}

    @property
    def arm_names(self) -> list[str]:
        """Registered arm names."""
        return list(self._owners.keys())

    def acquire(self, arm_name: str, kind: OwnerKind, owner: object) -> bool:
        """Try to acquire an arm. Succeeds only if the arm is idle.

        Args:
            arm_name: Arm to acquire.
            kind: Type of controller acquiring.
            owner: Reference to the acquiring controller.

        Returns:
            True if acquired, False if arm is already owned.
        """
        with self._lock:
            self._check_arm(arm_name)
            current_kind, _ = self._owners[arm_name]
            if current_kind != OwnerKind.IDLE:
                logger.debug(
                    "Cannot acquire %s: owned by %s",
                    arm_name,
                    current_kind.value,
                )
                return False
            self._owners[arm_name] = (kind, owner)
            self._abort_events[arm_name].clear()
            logger.debug("Acquired %s for %s", arm_name, kind.value)
            return True

    def force_release_all(self) -> None:
        """Reset all arms to IDLE regardless of current owner.

        Used by the global e-stop — unconditionally releases
        everything so the system is in a clean state.
        """
        with self._lock:
            for name in self._owners:
                self._owners[name] = (OwnerKind.IDLE, None)
                self._abort_events[name].clear()

    def release(self, arm_name: str, owner: object) -> None:
        """Release an arm. Only the current owner can release.

        Args:
            arm_name: Arm to release.
            owner: Must match the current owner.
        """
        with self._lock:
            self._check_arm(arm_name)
            current_kind, current_owner = self._owners[arm_name]
            if current_owner is not owner:
                logger.warning(
                    "Cannot release %s: owned by %r, not %r",
                    arm_name,
                    current_owner,
                    owner,
                )
                return
            self._owners[arm_name] = (OwnerKind.IDLE, None)
            self._abort_events[arm_name].clear()
            logger.debug("Released %s (was %s)", arm_name, current_kind.value)

    def owner_of(self, arm_name: str) -> tuple[OwnerKind, object | None]:
        """Query current ownership of an arm.

        Returns:
            Tuple of (OwnerKind, owner_reference).
        """
        with self._lock:
            self._check_arm(arm_name)
            return self._owners[arm_name]

    def preempt(self, arm_name: str, kind: OwnerKind, owner: object) -> None:
        """Abort the current owner and acquire the arm.

        If the new kind has higher priority than the current owner, sets the
        abort flag (so the current owner stops on its next check) and takes
        ownership. If the arm is idle, just acquires it.

        Args:
            arm_name: Arm to preempt.
            kind: Type of controller preempting.
            owner: Reference to the preempting controller.

        Raises:
            ValueError: If the new kind has lower priority than current owner.
        """
        with self._lock:
            self._check_arm(arm_name)
            current_kind, current_owner = self._owners[arm_name]

            if current_kind == OwnerKind.IDLE:
                self._owners[arm_name] = (kind, owner)
                self._abort_events[arm_name].clear()
                logger.debug("Acquired idle %s for %s", arm_name, kind.value)
                return

            if _PRIORITY[kind] <= _PRIORITY[current_kind]:
                raise ValueError(
                    f"Cannot preempt {arm_name}: {kind.value} (priority "
                    f"{_PRIORITY[kind]}) cannot preempt {current_kind.value} "
                    f"(priority {_PRIORITY[current_kind]})"
                )

            # Signal the current owner to stop
            self._abort_events[arm_name].set()
            self._owners[arm_name] = (kind, owner)
            logger.info(
                "Preempted %s: %s → %s",
                arm_name,
                current_kind.value,
                kind.value,
            )

    def set_abort(self, arm_name: str) -> None:
        """Set the abort flag for one arm."""
        with self._lock:
            self._check_arm(arm_name)
        self._abort_events[arm_name].set()

    def clear_abort(self, arm_name: str) -> None:
        """Clear the abort flag for one arm."""
        with self._lock:
            self._check_arm(arm_name)
        self._abort_events[arm_name].clear()

    def is_aborted(self, arm_name: str) -> bool:
        """Check if the abort flag is set for one arm."""
        return self._abort_events[arm_name].is_set()

    def abort_all(self) -> None:
        """Set abort flags for all arms."""
        with self._lock:
            for event in self._abort_events.values():
                event.set()

    def clear_all(self) -> None:
        """Clear abort flags for all arms."""
        with self._lock:
            for event in self._abort_events.values():
                event.clear()

    def _check_arm(self, arm_name: str) -> None:
        if arm_name not in self._owners:
            raise ValueError(f"Unknown arm: {arm_name}")
