# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Runtime check for \"is the held object still held\".

:class:`GraspVerifier` is a small sticky state machine driven by
per-tick signal verification. It answers a single question — *are
the live sensor signals consistent with still holding the object we
grasped?* — from whatever signals the arm actually has (gripper
position, wrist F/T, joint torques). It replaces contact-inspection
post-checks like geodude's original `LiftBase._compute_source_contacts`
that only work in simulation.

Three states:

- ``IDLE`` — not tracking a grasp. `is_held` is False.
- ``HOLDING`` — `mark_grasped` was called, the baseline was captured,
  and subsequent ticks have agreed with it. `is_held` is True.
- ``LOST`` — the state machine was in HOLDING and a tick observed a
  signal collapse. **Sticky**: only the next `mark_grasped` can
  leave this state. `is_held` is False.

Stickiness matches reality: a dropped object doesn't self-heal, so
momentarily-flickering signals shouldn't push us back to HOLDING.
If noise produces false positives, the right fix is a settling
window (already built in) or an N-consecutive-tick debounce (future).

The decision logic itself lives as a pure function
:func:`verify_grasp` taking a :class:`VerifierFacts` dataclass, so
the branch logic is unit-testable without any simulation state,
mocks, or hardware. :class:`GraspVerifier` is the stateful shell
that records baselines, ticks on a schedule, and logs transitions.

Design points:

- **Tick-driven, not live-query.** `is_held` is a plain state read,
  cheap and consistent across multiple reads in the same cycle. The
  work of re-reading signals happens in :meth:`tick`, which the
  execution context calls exactly once per control cycle. Consumers
  never tick manually.
- **Signals with ``None`` readings are skipped, not treated as
  failure.** Kinematic sim has no F/T or joint-torque data, so every
  load signal returns ``None``; the verifier falls back to \"trust
  that `mark_grasped` was called\" in that mode.
- **Baseline settling.** Right after `close_gripper` finishes, the
  constraint solver is still settling and the F/T reading is
  transient. :class:`VerifierParams.settling_ticks` defaults to 5 —
  the first 5 ticks after `mark_grasped` don't run drop-detection,
  they just let physics settle. After that the baseline is live and
  drops trigger LOST transitions.
- **Release is explicit, drop detection is observational.** Commanded
  releases (`SimArmController.release()`) call :meth:`mark_released`
  directly — the caller's intent is authoritative, and we want
  `is_held` to flip immediately on the same tick (not after the
  settling window observes the load drop). The tick-driven
  observation path is reserved for *unintended* drops: an object
  slipping from the gripper mid-transport, a grasp that looked
  successful but wasn't holding, etc. Both paths end in the same
  ``is_held → False`` signal to downstream consumers.
- **Robot packages compose their own verifier.** Geodude uses
  `[GripperPositionSignal(robotiq), WristFTSignal(ur5e)]`; Franka
  uses `[GripperPositionSignal(panda), JointTorqueSignal(franka)]`.
  Same class, different signal list.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from mj_manipulator.load_signals import LoadSignal

if TYPE_CHECKING:
    from mj_manipulator.protocols import Gripper

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State machine primitives
# ---------------------------------------------------------------------------


class GraspState(Enum):
    """Sticky state for the grasp verifier.

    Transitions:

    - ``mark_grasped(name)``: any state → HOLDING (baseline captured,
      settling window begins)
    - ``tick()`` during HOLDING, post-settling, signals collapse:
      HOLDING → LOST
    - ``mark_released()``: any state → IDLE (manual override; not
      called from the normal release path)
    """

    IDLE = "idle"
    HOLDING = "holding"
    LOST = "lost"


# ---------------------------------------------------------------------------
# Pure decision function — unit-testable without any physics or mocks
# ---------------------------------------------------------------------------


@dataclass
class VerifierParams:
    """Tunable parameters for :func:`verify_grasp` and :class:`GraspVerifier`.

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
    ``abs(baseline) * (1 - load_drop_ratio)`` triggers a LOST
    transition. 0.3 = signal must drop by more than 30% from
    baseline."""

    settling_ticks: int = 5
    """Number of ticks after :meth:`GraspVerifier.mark_grasped`
    during which drop-detection is suppressed. The physics state is
    still settling right after the gripper finishes closing, and the
    F/T reading is transient; forcing a few ticks of warmup before
    the verifier goes live prevents spurious LOST transitions on the
    first tick after grasp completion. 5 × 8ms = 40ms at 125Hz."""


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
    if no grasp is currently tracked."""

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

    1. **No object tracked** → False.
    2. **Decisive empty-stop negative**: if the gripper is a
       fully-closed-means-empty type and the position is at or above
       the threshold → False.
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
# Stateful shell — owns signals, drives the state machine, ticks per cycle
# ---------------------------------------------------------------------------


class GraspVerifier:
    """Per-arm sticky state machine for \"is the held object still held\".

    Composes a list of :class:`LoadSignal` instances the arm actually
    has (gripper position, wrist F/T, joint torques, ...) and a
    reference to the gripper (for the ``empty_at_fully_closed``
    decisive-negative branch). Drives a three-state machine via
    :meth:`mark_grasped` and per-cycle :meth:`tick` calls.

    Example — Geodude (UR5e + Robotiq 2F-140)::

        verifier = GraspVerifier(
            gripper=robot.left.gripper,
            signals=[
                GripperPositionSignal(robot.left.gripper),
                WristFTSignal(robot.left.arm),
            ],
        )
        robot.left.gripper.grasp_verifier = verifier

    The verifier never ticks itself. The execution context
    (:class:`SimContext`) ticks every configured verifier exactly
    once per control cycle inside :meth:`SimContext.step` and inside
    every trajectory execution. Consumers only call :meth:`is_held`
    (cheap state read) and :meth:`mark_grasped`.
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
        self._state: GraspState = GraspState.IDLE
        self._object_name: str | None = None
        self._baselines: dict[str, float | None] = {}
        self._ticks_since_grasp: int = 0

    # -- Public API ----------------------------------------------------------

    @property
    def state(self) -> GraspState:
        """Current state of the grasp verifier.

        Useful for diagnostics (\"we were HOLDING, now we're LOST,
        something slipped\") and for tests that want to assert on
        exact state rather than the bool shorthand.
        """
        return self._state

    @property
    def is_held(self) -> bool:
        """True iff the state machine is in HOLDING.

        Plain state read — does not re-invoke :func:`verify_grasp`.
        Consistent across multiple reads in the same cycle. The work
        of deciding whether we still hold the object happens in
        :meth:`tick`, which the execution context calls once per
        control cycle.
        """
        return self._state is GraspState.HOLDING

    @property
    def held_object(self) -> str | None:
        """The object we're currently holding, or None.

        Returns the tracked name when :attr:`is_held` is True,
        otherwise None. A stale baseline + dropped object yields
        ``None``. Consumers that want *\"what did the grasp sequence
        try to grab?\"* regardless of current health should read
        :attr:`tracked_object` instead.
        """
        return self._object_name if self._state is GraspState.HOLDING else None

    @property
    def tracked_object(self) -> str | None:
        """The object name recorded at :meth:`mark_grasped` time, or None.

        This is the raw bookkeeping — it does *not* check the state
        machine. Useful for diagnostics (\"we thought we grabbed X,
        but the verifier says we dropped it\") and for recovery
        subtrees that need to know what the grasp sequence was
        *attempting* even if it failed. Goes to None on
        :meth:`mark_released` and stays at the last-grasp name
        while the state is LOST.
        """
        return self._object_name

    def mark_grasped(self, object_name: str) -> None:
        """Begin tracking a grasp, capture baseline, enter HOLDING.

        Called by the grasp sequence (:meth:`SimArmController.grasp`
        or its hardware equivalent) right after the close motion
        finishes. Records the current value of every signal so
        subsequent ticks can detect drops, and starts the settling
        window — drop-detection is suppressed for the first
        ``settling_ticks`` ticks to let physics settle.

        Transitions the state machine to HOLDING unconditionally,
        even if the previous state was LOST. A fresh grasp overrides
        any previous state.
        """
        self._object_name = object_name
        self._baselines = {s.name: s.read() for s in self._signals}
        self._state = GraspState.HOLDING
        self._ticks_since_grasp = 0

        if self._baselines:
            baseline_str = ", ".join(
                f"{name}={'unavailable' if value is None else f'{value:.3f}'}"
                for name, value in self._baselines.items()
            )
            logger.info(
                "GraspVerifier: HOLDING %s (baseline: %s, settling for %d ticks)",
                object_name,
                baseline_str,
                self._params.settling_ticks,
            )
        else:
            logger.info(
                "GraspVerifier: HOLDING %s (no load signals configured)",
                object_name,
            )

    def mark_released(self) -> None:
        """Force state to IDLE and clear baseline.

        Called by `SimArmController.release()` (and the hardware
        equivalent) when the caller explicitly commands a release.
        Intent is authoritative: the state flips immediately, with no
        wait for :meth:`tick` to observe the load drop.

        The tick-driven observation path (HOLDING → LOST on load
        collapse) is reserved for *unintended* drops — an object
        slipping mid-transport, a grasp that looked successful but
        wasn't holding, etc. Both paths end in ``is_held → False``.
        """
        if self._state is not GraspState.IDLE:
            logger.info("GraspVerifier: released %s manually (→ IDLE)", self._object_name)
        self._state = GraspState.IDLE
        self._object_name = None
        self._baselines = {}
        self._ticks_since_grasp = 0

    def tick(self) -> None:
        """Advance the state machine by one control cycle.

        No-op when IDLE or LOST. When HOLDING:

        - Increments the settling counter.
        - If still inside the settling window, does nothing else.
        - Otherwise re-reads every signal, runs :func:`verify_grasp`,
          and transitions to LOST if the verdict is False.

        Called by the execution context once per control cycle. Not
        meant to be called by user code — consumers read
        :attr:`is_held` between ticks.
        """
        if self._state is not GraspState.HOLDING:
            return

        self._ticks_since_grasp += 1
        if self._ticks_since_grasp <= self._params.settling_ticks:
            return

        facts = self._collect_facts()
        if not verify_grasp(facts, self._params):
            self._transition_to_lost(facts)

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

    def _transition_to_lost(self, facts: VerifierFacts) -> None:
        """Move HOLDING → LOST, clean up the sim weld, and log.

        Detaching the weld is important: otherwise the grasp_manager
        still thinks the arm is holding the object, the collision
        checker treats the (now-detached-in-physics) object as part
        of the arm, and subsequent plans report "start in collision"
        referencing a stale attachment. On hardware this cleanup is
        trivial (no sim weld to detach); here it mirrors what
        :meth:`SimArmController.release` does, minus the gripper open
        (the gripper is already closed — nothing to open).
        """
        reasons = []
        if (
            facts.empty_at_fully_closed
            and facts.gripper_position is not None
            and facts.gripper_position >= self._params.empty_position_threshold
        ):
            reasons.append(f"gripper at mechanical stop (pos={facts.gripper_position:.3f})")
        for name, val in facts.signal_values.items():
            base = facts.signal_baselines.get(name)
            if val is None or base is None or abs(base) < 1e-6:
                continue
            threshold = abs(base) * (1.0 - self._params.load_drop_ratio)
            if abs(val) < threshold:
                reasons.append(f"{name} dropped from {base:.3f} → {val:.3f} (threshold {threshold:.3f})")
        reason_str = "; ".join(reasons) if reasons else "unknown"
        logger.warning("GraspVerifier: LOST %s (reason: %s)", self._object_name, reason_str)

        # Clean up sim bookkeeping so subsequent plans don't see a
        # stale weld. Harmless when grasp_manager is None (hardware).
        gm = getattr(self._gripper, "_grasp_manager", None)
        if gm is not None and self._object_name is not None:
            try:
                gm.mark_released(self._object_name)
                gm.detach_object(self._object_name)
            except Exception as e:
                # Bookkeeping cleanup is best-effort — log but don't crash.
                logger.debug("GraspVerifier: cleanup failed for %s: %s", self._object_name, e)

        self._state = GraspState.LOST
