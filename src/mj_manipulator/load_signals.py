# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Live load signals for grasp verification.

A ``LoadSignal`` is a named, live, scalar-valued readout that says
something about whether the gripper is bearing a load. The
:class:`~mj_manipulator.grasp_verifier.GraspVerifier` composes a list
of these signals to decide whether a held object is still held — on
sim, on UR5e + wrist F/T, on Franka + joint-torque sensing, and on
arms with nothing but a gripper position sensor. The verifier doesn't
know or care which kind of signal it's reading; it just compares each
live value against a baseline recorded at grasp time.

The protocol is deliberately tiny:

1. ``name`` — a unique string identifier the verifier uses to key
   baselines by signal.
2. ``read()`` — return the current scalar value, or ``None`` when the
   signal is not available on this arm right now (e.g. kinematic sim
   has no F/T, or the arm has no wrist sensor). ``None`` is different
   from ``0.0``: ``0.0`` means \"no load was measured\", ``None``
   means \"no measurement is possible\" — the verifier skips ``None``
   signals rather than interpreting them as lost load.

This is the only place the \"does this robot have this sensor?\"
branch lives. Consumers (the verifier, runtime monitors) stay
sensor-agnostic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from mj_manipulator.arm import Arm
    from mj_manipulator.protocols import Gripper


@runtime_checkable
class LoadSignal(Protocol):
    """A live scalar that reflects whether the gripper is bearing a load.

    Implementations encapsulate any per-signal projection (force
    magnitude, torque norm, position) so the verifier can treat every
    signal as a plain ``float``. See the three built-ins below for
    representative shapes.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this signal (used as a baseline key)."""
        ...

    def read(self) -> float | None:
        """Return the current scalar value, or None if unavailable.

        ``None`` means \"this signal cannot be read on this arm right
        now\" — the verifier will skip it rather than treating the
        missing reading as lost load. This is how kinematic sim
        degrades gracefully (all load signals return ``None``, so the
        verifier falls back to whatever signals are still live).
        """
        ...


# ---------------------------------------------------------------------------
# Built-in signal sources
# ---------------------------------------------------------------------------


class GripperPositionSignal:
    """Normalized gripper position in ``[0.0, 1.0]`` where ``1.0`` is fully closed.

    Universally available: every gripper has a position readout. This
    signal is most useful as a *decisive negative*: if the gripper hit
    the mechanical stop (``position ≈ 1.0``) AND the gripper's
    ``empty_at_fully_closed`` flag is True (Franka, ABH), then the
    fingers physically touched each other with nothing in between and
    the grasp is empty. For grippers where fully-closed is a valid
    grasp (Robotiq 2F-140), the verifier's decisive-negative branch is
    disabled and this signal is informational only.
    """

    name: str = "gripper_position"

    def __init__(self, gripper: Gripper):
        self._gripper = gripper

    def read(self) -> float | None:
        try:
            return float(self._gripper.get_actual_position())
        except Exception:
            return None


class WristFTSignal:
    """Signed vertical (world Z) force at the wrist in newtons.

    Reads :meth:`Arm.get_ft_wrench_world` and returns the Z component
    of the linear force — the projection of the wrench onto the
    gravity axis. This is the quantity that *most directly*
    corresponds to \"is the gripper still holding the object's
    weight\", because a held object of mass ``m`` contributes a
    constant ``±m*g`` to the world-Z reading regardless of how the
    arm is posed. When the object drops, the contribution vanishes
    and the reading returns to whatever the taread baseline
    contained (nominally zero, if :meth:`Arm.tare_ft` was called
    immediately before grasping).

    Why world Z instead of the sensor-local frame:

    - The F/T sensor's local frame rotates with the wrist flange.
      The same held object projects differently onto each local axis
      depending on the arm's pose. A signed-Z check in the local
      frame would need pose-dependent thresholds — not general.
    - Projecting into the world frame makes the \"how much vertical
      load am I carrying\" signal pose-independent. Gravity is
      gravity regardless of how the arm is oriented.

    Why signed Z instead of the total-force magnitude:

    - ``np.linalg.norm(wrench[:3])`` mixes gravity load with lateral
      contact forces and angular-inertial bleed from each axis. A
      moving arm with a held object sees the magnitude swing around
      even when the grip is perfectly healthy, producing false LOST
      transitions.
    - The Z component is dominated by gravity (the thing we care
      about) and only contaminated by vertical acceleration. X/Y
      inertial forces from lateral motion don't touch it.

    **Caveat — inertial forces during motion.** When the arm is
    accelerating vertically, the Z reading includes the inertial
    force of the held object as well as its weight. A hard upward
    lift makes the reading grow; a hard deceleration at the top of
    the lift makes it shrink transiently, sometimes below the
    drop-detection threshold. The verifier's settling window
    (:attr:`VerifierParams.settling_ticks`) handles the immediate
    post-grasp transient, but does not handle mid-trajectory motion.
    Consumers that need reliable mid-motion drop detection should
    additionally gate verification on arm quiescence (e.g.
    ``|qvel| < threshold``) — not done in this signal.

    Returns ``None`` when the arm has no F/T sensor configured or
    when ``ft_valid`` is False (kinematic sim), which matches the
    verifier's \"skip this signal\" semantics.
    """

    name: str = "wrist_ft_force_z"

    def __init__(self, arm: Arm):
        self._arm = arm

    def read(self) -> float | None:
        if not self._arm.has_ft_sensor:
            return None
        wrench_world = self._arm.get_ft_wrench_world()
        if np.isnan(wrench_world[0]):
            return None
        return float(wrench_world[2])


class JointTorqueSignal:
    """Aggregate joint-torque magnitude in N·m, or None in kinematic mode.

    Reads :meth:`Arm.get_joint_torques` and returns the Euclidean norm
    across all joints. With gravity compensation active, this reduces
    (at rest) largely to whatever external load the arm is working
    against — the weight of a held object being the dominant term.
    The signal drops when the load disappears, which is exactly what
    the verifier looks for.

    Useful for arms with no wrist F/T (Franka, which exposes joint
    torques via ``tau_ext`` on hardware). Returns ``None`` in
    kinematic sim where ``ft_valid`` is False.
    """

    name: str = "joint_torque_effort"

    def __init__(self, arm: Arm):
        self._arm = arm

    def read(self) -> float | None:
        torques = self._arm.get_joint_torques()
        if torques.size == 0 or np.isnan(torques[0]):
            return None
        return float(np.linalg.norm(torques))
