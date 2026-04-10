# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Headless regression test for safe_retract.

Runs safe_retract in three scenarios against a clean Franka in an empty
scene and asserts strict bounds on the resulting EE motion:

  1. Home pose, kinematic mode — planning + perfect-tracking execution.
  2. Home pose, physics mode — planning + PD control + gravcomp.
  3. Pre-grasp-like low pose, physics mode — the demo's actual failure
     case, where the arm reaches down toward the raised plate.

For each scenario we command a 15 cm upward lift (twist = [0,0,0.10,...])
and report:

    z travel   — actual EE +Z delta  (expected +150.00 mm)
    x drift    — actual EE X delta   (expected    +0.00)
    y drift    — actual EE Y delta   (expected    +0.00)
    rot error  — ||R_end R_start^T - I||_F  (expected ~0)
    Δq (deg)   — per-joint joint-space delta

This exercises the full scripted-Cartesian execution path
(``plan_cartesian_path`` → TOPP-RA retime → ``ctx.execute``) in both
kinematic and physics modes, and is the regression test for the fix to
personalrobotics/mj_manipulator#68.

Usage:
    cd mj_manipulator
    uv run python demos/verify_cartesian_lift.py
"""

from __future__ import annotations

import sys

import mujoco
import numpy as np
from mj_environment import Environment

from mj_manipulator.arms.franka import (
    FRANKA_HOME,
    add_franka_ee_site,
    add_franka_gravcomp,
    create_franka_arm,
)
from mj_manipulator.menagerie import menagerie_scene
from mj_manipulator.safe_retract import safe_retract
from mj_manipulator.sim_context import SimContext

# Lift parameters — match the pickup subtree default
LIFT_TWIST = np.array([0.0, 0.0, 0.10, 0.0, 0.0, 0.0])
LIFT_DISTANCE = 0.15  # meters

# A "reach down" configuration where the Franka EE hovers above the raised
# plate (z≈0.05). Hardcoded instead of IK-solved so the test is
# deterministic and independent of the IK branch selection.
FRANKA_PREGRASP = np.array(
    [0.0, 0.6, 0.0, -2.0, 0.0, 2.6, -0.7853],
    dtype=float,
)


def _make_env() -> Environment:
    scene_path = menagerie_scene("franka_emika_panda")
    if not scene_path.exists():
        print(f"ERROR: Franka scene not found at {scene_path}")
        sys.exit(1)
    spec = mujoco.MjSpec.from_file(str(scene_path))
    add_franka_ee_site(spec)
    add_franka_gravcomp(spec)
    return Environment.from_model(spec.compile())


def _set_arm(arm, env, q: np.ndarray) -> None:
    for i, idx in enumerate(arm.joint_qpos_indices):
        env.data.qpos[idx] = q[i]
    for idx in arm.joint_qvel_indices:
        env.data.qvel[idx] = 0.0
    mujoco.mj_forward(env.model, env.data)


def _measure(arm, env, ctx):
    """Run safe_retract via ``ctx`` and return a dict of measurements."""
    mujoco.mj_forward(env.model, env.data)

    ee_id = arm.ee_site_id
    start_pos = env.data.site_xpos[ee_id].copy()
    start_rot = env.data.site_xmat[ee_id].reshape(3, 3).copy()
    start_q = arm.get_joint_positions().copy()

    reported = safe_retract(
        arm,
        ctx,
        twist=LIFT_TWIST,
        max_distance=LIFT_DISTANCE,
    )

    end_pos = env.data.site_xpos[ee_id].copy()
    end_rot = env.data.site_xmat[ee_id].reshape(3, 3).copy()
    end_q = arm.get_joint_positions().copy()

    delta = end_pos - start_pos
    rot_err = float(np.linalg.norm(end_rot @ start_rot.T - np.eye(3), ord="fro"))
    joint_delta = np.rad2deg(end_q - start_q)

    return {
        "reported": reported,
        "x_drift": float(delta[0]),
        "y_drift": float(delta[1]),
        "z_travel": float(delta[2]),
        "rot_err": rot_err,
        "joint_delta_deg": joint_delta,
        "start_ee": start_pos,
        "end_ee": end_pos,
    }


def _print_row(label: str, m: dict) -> None:
    print(f"  {label}")
    print(f"    start EE   = [{m['start_ee'][0]:+.3f}, {m['start_ee'][1]:+.3f}, {m['start_ee'][2]:+.3f}] m")
    print(f"    end EE     = [{m['end_ee'][0]:+.3f}, {m['end_ee'][1]:+.3f}, {m['end_ee'][2]:+.3f}] m")
    print(f"    reported   = {m['reported'] * 1000:+7.2f} mm")
    print(f"    z travel   = {m['z_travel'] * 1000:+7.2f} mm   (expected  +150.00)")
    print(f"    x drift    = {m['x_drift'] * 1000:+7.2f} mm   (expected    +0.00)")
    print(f"    y drift    = {m['y_drift'] * 1000:+7.2f} mm   (expected    +0.00)")
    print(f"    rot err    = {m['rot_err']:.5f}       (expected     0.00000)")
    jdelta = m["joint_delta_deg"]
    jstr = " ".join(f"{x:+6.2f}" for x in jdelta)
    print(f"    Δq (deg)   = [{jstr}]")


def _run_scenario(label: str, home_q: np.ndarray, physics: bool) -> dict:
    env = _make_env()
    arm = create_franka_arm(env)
    _set_arm(arm, env, home_q)
    with SimContext(
        env.model,
        env.data,
        {arm.config.name: arm},
        physics=physics,
        headless=True,
    ) as ctx:
        return _measure(arm, env, ctx)


def main() -> int:
    print("=" * 72)
    print("  safe_retract regression test")
    print("=" * 72)
    print(f"  twist       = {LIFT_TWIST.tolist()}")
    print(f"  distance    = {LIFT_DISTANCE * 1000:.0f} mm")
    print()

    print("Scenario 1: FRANKA_HOME, kinematic mode")
    m1 = _run_scenario("kinematic/home", FRANKA_HOME, physics=False)
    _print_row("kinematic/home", m1)
    print()

    print("Scenario 2: FRANKA_HOME, physics mode (PD + gravcomp)")
    m2 = _run_scenario("physics/home", FRANKA_HOME, physics=True)
    _print_row("physics/home", m2)
    print()

    print("Scenario 3: FRANKA_PREGRASP (low reach-down), physics mode")
    m3 = _run_scenario("physics/low", FRANKA_PREGRASP, physics=True)
    _print_row("physics/low", m3)
    print()

    print("=" * 72)
    print("  Verdict")
    print("=" * 72)
    all_pass = True
    for name, m in [("kinematic/home", m1), ("physics/home", m2), ("physics/low", m3)]:
        z_ok = abs(m["z_travel"] - LIFT_DISTANCE) < 0.005  # within 5 mm
        xy_ok = abs(m["x_drift"]) < 0.002 and abs(m["y_drift"]) < 0.002  # < 2 mm drift
        rot_ok = m["rot_err"] < 0.02
        passed = z_ok and xy_ok and rot_ok
        all_pass = all_pass and passed
        print(
            f"  {name:20s}  {'PASS' if passed else 'FAIL'}  "
            f"(z={'ok' if z_ok else 'bad'}, "
            f"xy={'ok' if xy_ok else 'bad'}, "
            f"rot={'ok' if rot_ok else 'bad'})"
        )
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
