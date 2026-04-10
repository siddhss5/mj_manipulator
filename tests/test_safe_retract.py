# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for :func:`mj_manipulator.safe_retract.safe_retract`.

Covers:
- Kinematic and physics-mode reachability (home + low reach-down pose)
- Input validation (angular twist raises, zero twist no-ops)
- Baseline-contact awareness (starting with a contact is tolerated)
- New-contact abort (collision mid-trajectory stops the motion)

All physics-mode tests require the Franka menagerie. They skip
automatically if the menagerie isn't available (via
``franka_env_with_gravcomp`` in conftest.py).
"""

from __future__ import annotations

import numpy as np
import pytest
from mj_environment import Environment

from mj_manipulator.arms.franka import FRANKA_HOME
from mj_manipulator.safe_retract import safe_retract

# ---------------------------------------------------------------------------
# Input validation — no physics needed
# ---------------------------------------------------------------------------


class TestSafeRetractValidation:
    """Tests that exercise safe_retract's input validation."""

    def test_angular_twist_raises(self, franka_arm_at_home):
        """A twist with non-zero angular components is not supported."""
        from mj_manipulator.sim_context import SimContext

        arm = franka_arm_at_home
        with SimContext(
            arm.env.model,
            arm.env.data,
            {arm.config.name: arm},
            physics=False,
            headless=True,
        ) as ctx:
            twist = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.1])
            with pytest.raises(NotImplementedError, match="translational"):
                safe_retract(arm, ctx, twist=twist, max_distance=0.1)

    def test_zero_twist_returns_zero(self, franka_arm_at_home):
        """A zero twist is a no-op, not an error."""
        from mj_manipulator.sim_context import SimContext

        arm = franka_arm_at_home
        with SimContext(
            arm.env.model,
            arm.env.data,
            {arm.config.name: arm},
            physics=False,
            headless=True,
        ) as ctx:
            twist = np.zeros(6)
            distance = safe_retract(arm, ctx, twist=twist, max_distance=0.1)
            assert distance == 0.0


# ---------------------------------------------------------------------------
# Reachability — kinematic and physics
# ---------------------------------------------------------------------------


# A "reach down" configuration — same one used by verify_cartesian_lift.py
FRANKA_REACH_DOWN = np.array([0.0, 0.6, 0.0, -2.0, 0.0, 2.6, -0.7853], dtype=float)


class TestSafeRetractReachability:
    """safe_retract moves the EE along the commanded twist in clean scenes."""

    def _run(self, franka_arm_at_home, home_q, physics: bool, distance: float = 0.15):
        """Shared setup: move arm to home_q, run safe_retract, return deltas."""
        import mujoco

        from mj_manipulator.sim_context import SimContext

        arm = franka_arm_at_home
        env = arm.env

        # Override the fixture's home with the requested config
        for i, idx in enumerate(arm.joint_qpos_indices):
            env.data.qpos[idx] = home_q[i]
        for idx in arm.joint_qvel_indices:
            env.data.qvel[idx] = 0.0
        mujoco.mj_forward(env.model, env.data)

        with SimContext(
            env.model,
            env.data,
            {arm.config.name: arm},
            physics=physics,
            headless=True,
        ) as ctx:
            ee_id = arm.ee_site_id
            start_pos = env.data.site_xpos[ee_id].copy()
            start_rot = env.data.site_xmat[ee_id].reshape(3, 3).copy()

            twist = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
            reported = safe_retract(arm, ctx, twist=twist, max_distance=distance)

            end_pos = env.data.site_xpos[ee_id].copy()
            end_rot = env.data.site_xmat[ee_id].reshape(3, 3).copy()

        delta = end_pos - start_pos
        rot_err = float(np.linalg.norm(end_rot @ start_rot.T - np.eye(3), ord="fro"))
        return {
            "reported": reported,
            "x_drift": float(delta[0]),
            "y_drift": float(delta[1]),
            "z_travel": float(delta[2]),
            "rot_err": rot_err,
        }

    def test_kinematic_home_reaches_target(self, franka_arm_at_home):
        """Kinematic mode from home reaches the commanded 15 cm lift."""
        m = self._run(franka_arm_at_home, FRANKA_HOME, physics=False)
        assert abs(m["z_travel"] - 0.15) < 0.005  # within 5 mm
        assert abs(m["x_drift"]) < 0.002
        assert abs(m["y_drift"]) < 0.002
        assert m["rot_err"] < 0.02
        assert abs(m["reported"] - m["z_travel"]) < 0.005

    def test_physics_home_reaches_target(self, franka_arm_at_home):
        """Physics mode from home reaches the commanded 15 cm lift within tolerance."""
        m = self._run(franka_arm_at_home, FRANKA_HOME, physics=True)
        assert abs(m["z_travel"] - 0.15) < 0.005
        assert abs(m["x_drift"]) < 0.002
        assert abs(m["y_drift"]) < 0.002
        assert m["rot_err"] < 0.02

    def test_physics_low_pose_reaches_target(self, franka_arm_at_home):
        """Physics mode from a reach-down pose reaches the commanded lift.

        This is the demo's actual failure case before the fix: the arm was
        drifting sideways ~35 mm instead of lifting straight up.
        """
        m = self._run(franka_arm_at_home, FRANKA_REACH_DOWN, physics=True)
        assert abs(m["z_travel"] - 0.15) < 0.005
        assert abs(m["x_drift"]) < 0.002
        assert abs(m["y_drift"]) < 0.002
        assert m["rot_err"] < 0.02


# ---------------------------------------------------------------------------
# Collision-aware abort
# ---------------------------------------------------------------------------


# Franka scene with a ceiling box 8 cm above the EE at home. The lift
# should reach the ceiling after ~5 cm (ceiling at ~62 cm, EE starts at
# ~57 cm, ~3 cm of clearance before fingertip/hand contacts the box).
_CEILING_SCENE = """
<mujoco>
  <worldbody>
    <body name="ceiling" pos="0.555 0 0.64">
      <geom type="box" size="0.3 0.3 0.01" rgba="0.4 0.4 0.4 1"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def franka_with_ceiling():
    """Franka + a ceiling box 8 cm above the home EE pose.

    The ceiling is thin and positioned so the Franka hand will contact it
    after a few cm of upward lift. Used to test that safe_retract stops on
    new contact rather than pushing through.
    """
    try:
        import mujoco

        from mj_manipulator.arms.franka import (
            add_franka_ee_site,
            add_franka_gravcomp,
            create_franka_arm,
        )
        from mj_manipulator.menagerie import menagerie_scene
    except (ImportError, FileNotFoundError):
        pytest.skip("mujoco_menagerie or mj_environment not available")

    try:
        scene = menagerie_scene("franka_emika_panda")
    except FileNotFoundError:
        pytest.skip("mujoco_menagerie not found")

    spec = mujoco.MjSpec.from_file(str(scene))
    add_franka_ee_site(spec)
    add_franka_gravcomp(spec)

    # Add a ceiling box directly to the spec so its mesh paths resolve
    ceiling = spec.worldbody.add_body()
    ceiling.name = "ceiling"
    ceiling.pos = [0.555, 0.0, 0.64]  # ~8 cm above FRANKA_HOME EE z=0.566
    g = ceiling.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [0.3, 0.3, 0.01]
    g.rgba = [0.4, 0.4, 0.4, 1.0]

    env = Environment.from_model(spec.compile())
    arm = create_franka_arm(env)
    for i, idx in enumerate(arm.joint_qpos_indices):
        env.data.qpos[idx] = FRANKA_HOME[i]
    for idx in arm.joint_qvel_indices:
        env.data.qvel[idx] = 0.0
    mujoco.mj_forward(env.model, env.data)
    return arm


class TestSafeRetractCollision:
    """Tests for the baseline-contact / new-contact abort logic."""

    def test_stops_on_new_contact(self, franka_with_ceiling):
        """Lift command of 15 cm stops early when hitting a ceiling ~8 cm up.

        The ceiling is a thin box at z=0.64, ~8 cm above the home EE pose
        at z=0.566. The Franka hand is longer than just the EE site, so
        contact happens somewhere before the EE itself reaches the ceiling
        — expect motion to halt between 1 cm and 10 cm of z travel.
        """
        import mujoco

        from mj_manipulator.sim_context import SimContext

        arm = franka_with_ceiling
        env = arm.env

        mujoco.mj_forward(env.model, env.data)
        start_pos = env.data.site_xpos[arm.ee_site_id].copy()

        with SimContext(
            env.model,
            env.data,
            {arm.config.name: arm},
            physics=True,
            headless=True,
        ) as ctx:
            twist = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
            safe_retract(arm, ctx, twist=twist, max_distance=0.15)

        end_pos = env.data.site_xpos[arm.ee_site_id].copy()
        z_travel = end_pos[2] - start_pos[2]

        # Expect to stop well before 15 cm — somewhere in (1 cm, 10 cm).
        assert 0.01 < z_travel < 0.10, (
            f"Expected early stop, got z_travel={z_travel * 1000:.1f}mm"
        )

    def test_tolerates_baseline_contact(self, franka_arm_at_home):
        """An object already in contact at t=0 does not abort the lift.

        Approximates the post-grasp case: the held object is in contact
        with a surface when the lift starts. safe_retract records this as
        a baseline and only aborts on *new* pairs.
        """
        from mj_manipulator.sim_context import SimContext

        arm = franka_arm_at_home
        env = arm.env

        # We simulate this by monkey-patching the baseline detection to
        # report an extra pair. This is less invasive than adding a whole
        # new spec with a touching body, and verifies the same logic: a
        # contact present at the start is ignored, new ones are not.
        from mj_manipulator import safe_retract as safe_retract_mod

        real_get_pairs = safe_retract_mod._get_contact_pairs
        fake_pair = (999, 1000)  # synthetic "already in contact at t=0"

        def mock_get_pairs(model, data):
            real = real_get_pairs(model, data)
            return real | {fake_pair}

        safe_retract_mod._get_contact_pairs = mock_get_pairs
        try:
            with SimContext(
                env.model,
                env.data,
                {arm.config.name: arm},
                physics=True,
                headless=True,
            ) as ctx:
                twist = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
                distance = safe_retract(arm, ctx, twist=twist, max_distance=0.1)

            # Baseline includes the fake pair, but it's also present the
            # whole time, so it never appears as "new" and execution runs
            # to completion.
            assert abs(distance - 0.1) < 0.005
        finally:
            safe_retract_mod._get_contact_pairs = real_get_pairs
