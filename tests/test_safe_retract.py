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
        assert 0.01 < z_travel < 0.10, f"Expected early stop, got z_travel={z_travel * 1000:.1f}mm"


# ---------------------------------------------------------------------------
# Joint-space smoothness — catches IK branch-flips
# ---------------------------------------------------------------------------
#
# The failure mode: IK solver returns solutions that don't include the arm's
# current config even when the current pose is the target. plan_cartesian_path
# picks the "nearest" of the wrong branches, so the arm jumps to a different
# branch on the very first waypoint. The motion is Cartesian-straight for
# the end-effector but internally involves a huge joint-space sweep.
#
# We test this by calling plan_cartesian_path directly and inspecting the
# joint path for unreasonable per-segment jumps.


class TestCartesianPathSmoothness:
    """plan_cartesian_path should not jump IK branches on the first waypoint."""

    def test_first_waypoint_is_close_to_start(self, franka_arm_at_home):
        """A 5mm step up shouldn't require a huge joint motion."""
        import mujoco

        from mj_manipulator.cartesian_path import plan_cartesian_path, translational_waypoints

        arm = franka_arm_at_home

        # Set to home explicitly
        for i, idx in enumerate(arm.joint_qpos_indices):
            arm.env.data.qpos[idx] = FRANKA_HOME[i]
        mujoco.mj_forward(arm.env.model, arm.env.data)

        start_pose = arm.get_ee_pose()
        q_start = arm.get_joint_positions()

        # Single 5mm step up
        waypoints = translational_waypoints(start_pose, np.array([0.0, 0.0, 1.0]), 0.005, segment_length=0.005)
        trajectory = plan_cartesian_path(arm, waypoints, q_start=q_start)

        # The first commanded joint position should be within a reasonable
        # distance of q_start. A 5mm Cartesian step on a 7-DOF arm away
        # from singularity should be ~0.05-0.15 rad per joint at most,
        # vector norm well under 0.5 rad.
        q_first = trajectory.positions[0]
        jump = float(np.linalg.norm(q_first - q_start))
        assert jump < 0.5, (
            f"First commanded config is {jump:.3f} rad from q_start; "
            f"expected << 0.5 rad for a 5mm Cartesian step. This is the IK "
            f"branch-flip bug (#124)."
        )

    def test_joint_path_has_no_discontinuities(self, franka_arm_at_home):
        """Consecutive joint configs should be within one segment's IK jump."""
        import mujoco

        from mj_manipulator.cartesian_path import plan_cartesian_path, translational_waypoints

        arm = franka_arm_at_home

        for i, idx in enumerate(arm.joint_qpos_indices):
            arm.env.data.qpos[idx] = FRANKA_HOME[i]
        mujoco.mj_forward(arm.env.model, arm.env.data)

        start_pose = arm.get_ee_pose()
        q_start = arm.get_joint_positions()

        # 10cm lift at 5mm segments → 20 waypoints
        waypoints = translational_waypoints(start_pose, np.array([0.0, 0.0, 1.0]), 0.10, segment_length=0.005)
        trajectory = plan_cartesian_path(arm, waypoints, q_start=q_start)

        # Walk through the sampled joint path and verify no step exceeds a
        # physically sensible bound. A 5mm EE step should yield ~0.1 rad
        # norm max; we allow 0.3 to leave margin for retiming resampling.
        positions = np.asarray(trajectory.positions)
        for i in range(1, len(positions)):
            step = float(np.linalg.norm(positions[i] - positions[i - 1]))
            assert step < 0.3, (
                f"Joint path step {i}: {step:.3f} rad between samples. "
                f"Expected << 0.3 rad per control step for a smooth Cartesian retract."
            )

    def test_safe_retract_does_not_cause_large_initial_jump(self, franka_arm_at_home):
        """safe_retract should not produce a commanded jump on its first control cycle.

        This is the end-to-end version of test_first_waypoint_is_close_to_start —
        it catches the IK branch-flip as manifest in the retract trajectory.
        """
        import mujoco

        from mj_manipulator.safe_retract import safe_retract
        from mj_manipulator.sim_context import SimContext

        arm = franka_arm_at_home
        env = arm.env

        for i, idx in enumerate(arm.joint_qpos_indices):
            env.data.qpos[idx] = FRANKA_HOME[i]
        mujoco.mj_forward(env.model, env.data)

        q_start = arm.get_joint_positions()

        # Run safe_retract in kinematic mode (no PD, direct joint sets)
        # so we can measure the commanded-vs-actual gap precisely.
        with SimContext(
            env.model,
            env.data,
            {arm.config.name: arm},
            physics=False,
            headless=True,
        ) as ctx:
            twist = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
            safe_retract(arm, ctx, twist=twist, max_distance=0.05)

        q_end = arm.get_joint_positions()
        total_joint_motion = float(np.linalg.norm(q_end - q_start))

        # A 5cm Cartesian lift at Franka home should correspond to modest
        # joint motion — roughly 0.3-0.6 rad total norm. If we see > 2 rad
        # norm it means the arm swept through a branch flip mid-trajectory.
        assert total_joint_motion < 1.5, (
            f"safe_retract produced total joint motion of {total_joint_motion:.2f} rad "
            f"for a 5cm lift — symptom of IK branch-flip (#124)."
        )
