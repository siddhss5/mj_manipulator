# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Runtime check for \"is the held object still held\".

The :class:`GraspVerifier` answers a single question: given what we
told the arm to grasp and what the sensors are reading right now, do
we still believe the object is in the gripper? It replaces
contact-inspection-based checks (e.g. the ``iter_contacts`` post-check
in geodude's original :class:`LiftBase`) that only work in simulation.

The decision logic is split out as a pure function
:func:`verify_grasp` taking a :class:`VerifierFacts` dataclass, so the
logic is unit-testable without any simulation state, mocks, or
hardware. The :class:`GraspVerifier` class is the stateful shell:
it owns the list of :class:`~mj_manipulator.load_signals.LoadSignal`
instances, records a baseline at :meth:`mark_grasped` time, and
assembles facts from live reads when :attr:`is_held` is queried.

Design notes:

- Signals that return ``None`` from their :meth:`read` method are
  skipped, not interpreted as lost load. This is how kinematic sim
  degrades gracefully — every load signal returns ``None`` there and
  :attr:`is_held` reduces to \"did anyone call ``mark_grasped``?\".
- The decision function is robot-agnostic. Geodude composes a verifier
  with ``[GripperPositionSignal(robotiq), WristFTSignal(ur5e)]``;
  Franka composes one with ``[GripperPositionSignal(panda),
  JointTorqueSignal(franka)]``. Same class, different signal list.
- The :attr:`held_object` property is the bookkeeping name (what we
  told the arm to grasp). It goes to ``None`` when :attr:`is_held`
  is False, so downstream consumers that currently read
  ``gripper.held_object`` as \"what am I holding right now?\" get a
  real answer instead of stale bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mj_manipulator.load_signals import LoadSignal

if TYPE_CHECKING:
    from mj_manipulator.protocols import Gripper


# ---------------------------------------------------------------------------
# Pure decision function — unit-testable without any physics or mocks
# ---------------------------------------------------------------------------


@dataclass
class VerifierParams:
    """Tunable parameters for :func:`verify_grasp`.

    Defaults are deliberately conservative — we'd rather declare a
    healthy grasp \"probably dropped\" and let recovery run a sanity
    check than silently transport an object that isn't there.
    """

    empty_position_threshold: float = 0.98
    """Normalized gripper position [0, 1] at or above which we treat
    the gripper as \"at the mechanical stop\". Only consulted when
    :attr:`VerifierFacts.empty_at_fully_closed` is True."""

    load_drop_ratio: float = 0.3
    """Fraction of baseline below which a signal is considered
    collapsed. A signal whose magnitude drops below
    ``abs(baseline) * (1 - load_drop_ratio)`` triggers a FAILURE
    verdict. 0.3 = signal must drop by more than 30% from baseline."""


@dataclass
class VerifierFacts:
    """Inputs to :func:`verify_grasp`.

    Holds the minimal snapshot the decision function needs: what we
    were told we grasped, whether fully-closed means \"empty\" for
    this gripper, the current position reading, and each load
    signal's current value and baseline. Everything is already a
    plain Python type — no MuJoCo, no numpy, no mocks.
    """

    object_name: str | None
    """The object the grasp sequence believes it picked up, or None
    if no grasp is currently in progress."""

    empty_at_fully_closed: bool
    """Does this gripper's ``ctrl_closed`` position mean \"fingers
    touching, no object between them\"? True for Franka/ABH, False
    for Robotiq 2F-140."""

    gripper_position: float | None
    """Current normalized gripper position, or None if unreadable."""

    signal_values: dict[str, float | None] = field(default_factory=dict)
    """Current scalar reading per signal name. ``None`` means the
    signal is not available on this arm right now."""

    signal_baselines: dict[str, float | None] = field(default_factory=dict)
    """Per-signal baseline recorded at :meth:`GraspVerifier.mark_grasped`
    time. ``None`` means the baseline wasn't captured for that
    signal (signal was unavailable at grasp time)."""


def verify_grasp(facts: VerifierFacts, params: VerifierParams) -> bool:
    """Pure decision: given a snapshot, is the object still held?

    Walks a small decision tree:

    1. **No object tracked** → False (nothing to verify; we're not
       currently trying to hold anything).
    2. **Decisive empty-stop negative**: if the gripper is a
       fully-closed-means-empty type and the position is at or above
       the threshold → False. The fingers are touching each other
       with nothing in between.
    3. **Load-drop check**: for every signal that has a usable
       baseline and a live reading, check whether the live value has
       collapsed below ``|baseline| * (1 - load_drop_ratio)``. If any
       signal has → False.
    4. **Otherwise** → True.

    Signals with ``None`` readings (either baseline or live) are
    skipped rather than treated as failures. A very small baseline
    (``|baseline| < 1e-6``) is also skipped — we can't infer a drop
    from nothing.

    Args:
        facts: Current snapshot of grasp state.
        params: Thresholds controlling the decision.

    Returns:
        True if the held object is still held, False otherwise.
    """
    if facts.object_name is None:
        return False

    if (
        facts.empty_at_fully_closed
        and facts.gripper_position is not None
        and facts.gripper_position >= params.empty_position_threshold
    ):
        return False

    for name, val in facts.signal_values.items():
        base = facts.signal_baselines.get(name)
        if val is None or base is None:
            continue
        if abs(base) < 1e-6:
            continue
        if abs(val) < abs(base) * (1.0 - params.load_drop_ratio):
            return False

    return True


# ---------------------------------------------------------------------------
# Stateful shell — owns signals, captures baselines, answers queries
# ---------------------------------------------------------------------------


class GraspVerifier:
    """Per-arm \"is the held object still held\" check.

    Composes a list of :class:`LoadSignal` instances that the arm
    actually has (gripper position, wrist F/T, joint torques, ...)
    and a reference to the gripper (for the ``empty_at_fully_closed``
    decisive-negative branch). Records a baseline from every signal
    when :meth:`mark_grasped` is called; the :attr:`is_held` property
    compares live reads against the baseline via :func:`verify_grasp`.

    The verifier is instantaneous — :attr:`is_held` re-reads every
    signal each time it's queried. There's no rolling average or
    debounce in v1; adding one means extending :meth:`tick` and
    filtering in :meth:`_collect_facts`, but the v1 signals are all
    physical quantities that already change smoothly, so
    instantaneous reads are good enough to start with.

    Example — Geodude (UR5e + Robotiq 2F-140)::

        verifier = GraspVerifier(
            gripper=robot.left.gripper,
            signals=[
                GripperPositionSignal(robot.left.gripper),
                WristFTSignal(robot.left.arm),
            ],
        )
        robot.left.gripper.grasp_verifier = verifier

    Example — Franka (empty_at_fully_closed=True, joint torques)::

        verifier = GraspVerifier(
            gripper=franka.gripper,
            signals=[
                GripperPositionSignal(franka.gripper),
                JointTorqueSignal(franka.arm),
            ],
        )
    """

    def __init__(
        self,
        gripper: Gripper,
        signals: list[LoadSignal],
        *,
        params: VerifierParams | None = None,
    ):
        self._gripper = gripper
        self._signals = list(signals)
        self._params = params if params is not None else VerifierParams()
        self._object_name: str | None = None
        self._baselines: dict[str, float | None] = {}

    # -- Public API ----------------------------------------------------------

    @property
    def held_object(self) -> str | None:
        """The object we believe is currently held, or None.

        This is *not* raw bookkeeping — it only returns the grasped
        name when :attr:`is_held` agrees. A stale baseline + dropped
        object yields ``None``. Consumers that want \"what did the
        grasp sequence think it grabbed\" without the health check
        should read :attr:`tracked_object` instead.
        """
        if self._object_name is None:
            return None
        return self._object_name if self.is_held else None

    @property
    def tracked_object(self) -> str | None:
        """The object name recorded at :meth:`mark_grasped` time, or None.

        This is the raw bookkeeping — it does *not* re-check live
        signals. Useful for diagnostics (\"we thought we grabbed X,
        but the verifier says we dropped it\") and for
        :class:`~mj_manipulator.bt.nodes` that need to know what the
        grasp sequence was *attempting* even if it failed.
        """
        return self._object_name

    @property
    def is_held(self) -> bool:
        """Live check: are the signals consistent with still holding the object?

        Re-reads every signal each call. Returns False if no object
        is currently tracked (nothing to check) or if the decision
        function says the held object has dropped.
        """
        return verify_grasp(self._collect_facts(), self._params)

    def mark_grasped(self, object_name: str) -> None:
        """Begin tracking a grasp and capture a baseline from every signal.

        Called by the grasp sequence (:meth:`SimArmController.grasp`
        or its hardware equivalent) right after the close motion
        succeeds. Records the current value of every signal so future
        :attr:`is_held` calls can detect drops.
        """
        self._object_name = object_name
        self._baselines = {s.name: s.read() for s in self._signals}

    def mark_released(self) -> None:
        """Stop tracking. :attr:`is_held` returns False until the next
        :meth:`mark_grasped`.
        """
        self._object_name = None
        self._baselines = {}

    def tick(self) -> None:
        """Per-control-cycle hook.

        No-op in v1 — signals are read instantaneously on every
        :attr:`is_held` call. Reserved for future debouncing /
        rolling-average logic; keeping the hook point means execute
        loops can wire it up now without needing a protocol change
        later.
        """
        return None

    # -- Internals -----------------------------------------------------------

    def _collect_facts(self) -> VerifierFacts:
        """Gather live readings into a :class:`VerifierFacts` snapshot."""
        try:
            gripper_position: float | None = float(self._gripper.get_actual_position())
        except Exception:
            gripper_position = None
        return VerifierFacts(
            object_name=self._object_name,
            empty_at_fully_closed=getattr(self._gripper, "empty_at_fully_closed", False),
            gripper_position=gripper_position,
            signal_values={s.name: s.read() for s in self._signals},
            signal_baselines=dict(self._baselines),
        )
