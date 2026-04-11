# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Shared base class for gripper implementations.

Not part of the public API. Subclass _BaseGripper to implement a concrete
gripper that satisfies the Gripper protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco

from mj_manipulator.contacts import iter_contacts

if TYPE_CHECKING:
    from mj_manipulator.grasp_manager import GraspManager
    from mj_manipulator.grasp_verifier import GraspVerifier


class _BaseGripper:
    """Common gripper logic shared by all implementations.

    Provides default implementations for Gripper protocol properties and
    shared helper methods (contact detection, geometric grasp detection).
    Subclasses must implement ``_apply_kinematic_position`` and
    ``get_actual_position``.
    """

    # Does fully-closed mean "no object held"? True for grippers whose
    # fingers physically touch (Franka). False for grippers with large
    # finger travel where fully-closed is still a valid grasp (Robotiq).
    empty_at_fully_closed: bool = False

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        arm_name: str,
        actuator_name: str,
        gripper_body_names: list[str],
        attachment_body: str,
        ctrl_open: float,
        ctrl_closed: float,
        grasp_manager: GraspManager | None = None,
    ):
        self._model = model
        self._data = data
        self._arm_name = arm_name
        self._gripper_body_names = gripper_body_names
        self._attachment_body = attachment_body
        self._ctrl_open = ctrl_open
        self._ctrl_closed = ctrl_closed
        self._grasp_manager = grasp_manager
        self._grasp_verifier: GraspVerifier | None = None
        self._candidate_objects: list[str] | None = None

        # Resolve actuator ID
        if actuator_name:
            self._actuator_id: int | None = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                actuator_name,
            )
            if self._actuator_id == -1:
                raise ValueError(f"Actuator '{actuator_name}' not found in model")
        else:
            self._actuator_id = None

        # Cache gripper body IDs for contact detection
        self._gripper_body_ids: set[int] = set()
        for name in gripper_body_names:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid != -1:
                self._gripper_body_ids.add(bid)

    # -- Gripper protocol properties ----------------------------------------

    @property
    def arm_name(self) -> str:
        return self._arm_name

    @property
    def gripper_body_names(self) -> list[str]:
        return self._gripper_body_names

    @property
    def attachment_body(self) -> str:
        return self._attachment_body

    @property
    def actuator_id(self) -> int | None:
        return self._actuator_id

    @property
    def ctrl_open(self) -> float:
        return self._ctrl_open

    @property
    def ctrl_closed(self) -> float:
        return self._ctrl_closed

    @property
    def is_holding(self) -> bool:
        if self._grasp_verifier is not None:
            return self._grasp_verifier.is_held
        if self._grasp_manager is None:
            return False
        return len(self._grasp_manager.get_grasped_by(self._arm_name)) > 0

    @property
    def held_object(self) -> str | None:
        if self._grasp_verifier is not None:
            return self._grasp_verifier.held_object
        if self._grasp_manager is None:
            return None
        held = self._grasp_manager.get_grasped_by(self._arm_name)
        return held[0] if held else None

    @property
    def grasp_verifier(self) -> GraspVerifier | None:
        """Sensor-based grasp health check, if configured.

        When set, :attr:`is_holding` and :attr:`held_object` route
        through the verifier instead of reading from
        :class:`GraspManager` bookkeeping. Set this after
        constructing the gripper and the signal sources it needs —
        see :class:`GraspVerifier` for wiring examples.

        Leaving it unset preserves the legacy path (``is_holding``
        reads ``GraspManager``), which is the current default
        everywhere and requires no per-robot changes.
        """
        return self._grasp_verifier

    @grasp_verifier.setter
    def grasp_verifier(self, verifier: GraspVerifier | None) -> None:
        self._grasp_verifier = verifier

    def set_candidate_objects(self, objects: list[str] | None) -> None:
        self._candidate_objects = objects

    # -- Kinematic control (template methods) ----------------------------------

    def _apply_kinematic_position(self, t: float) -> None:
        """Set gripper joints to position ``t`` and call ``mj_forward``.

        Must be implemented by subclasses. ``t`` ranges from 0.0 (open) to
        1.0 (closed).
        """
        raise NotImplementedError

    def kinematic_close(self, steps: int = 50) -> str | None:
        """Close gripper kinematically until contact.

        Incrementally closes the gripper, checking for contact with candidate
        objects at each step.

        Args:
            steps: Number of incremental steps to check for contact.

        Returns:
            Name of grasped object, or None if nothing grasped.
        """
        grasped = None

        for i in range(steps + 1):
            t = i / steps
            self._apply_kinematic_position(t)

            self_contact, external_body = self._scan_contacts()

            # Stop before gripper finger-to-finger self-contact.  If fingers
            # are about to touch each other with no object between them, back
            # off one step so the collision checker doesn't see an invalid
            # gripper self-collision at the start of subsequent planning.
            if self_contact:
                if i > 0:
                    self._apply_kinematic_position((i - 1) / steps)
                break

            if external_body:
                # Only stop for candidate objects (ignore incidental contacts)
                if self._candidate_objects is None or external_body in self._candidate_objects:
                    grasped = external_body
                    break

        return grasped

    def kinematic_open(self) -> None:
        """Open gripper kinematically."""
        self._apply_kinematic_position(0.0)

    # -- Shared helpers -----------------------------------------------------

    def _scan_contacts(self) -> tuple[bool, str | None]:
        """Single-pass contact scan over all active contacts.

        Returns:
            (self_contact, external_body) where self_contact is True if two
            distinct gripper bodies touch each other, and external_body is the
            name of any external body in contact with the gripper (candidate
            objects are preferred over incidental contacts).
        """
        self_contact = False
        candidate_body: str | None = None
        first_external: str | None = None

        for b1, b2, _ in iter_contacts(self._model, self._data):
            b1_gripper = b1 in self._gripper_body_ids
            b2_gripper = b2 in self._gripper_body_ids

            if b1_gripper and b2_gripper and b1 != b2:
                self_contact = True
            elif b1_gripper and not b2_gripper:
                name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, b2)
                if self._candidate_objects and name in self._candidate_objects:
                    candidate_body = name
                elif first_external is None:
                    first_external = name
            elif b2_gripper and not b1_gripper:
                name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, b1)
                if self._candidate_objects and name in self._candidate_objects:
                    candidate_body = name
                elif first_external is None:
                    first_external = name

        return self_contact, candidate_body or first_external
