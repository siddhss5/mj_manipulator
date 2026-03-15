"""Analytical IK demo with EAIK — UR5e and Franka Panda.

Showcases the MuJoCoEAIKSolver which extracts joint axes (H) and position
offsets (P) directly from MuJoCo models to build EAIK HPRobots. No DH
parameters or frame calibration needed — EAIK FK matches MuJoCo FK exactly.

Demonstrates:
  1. Automatic kinematic extraction from MuJoCo
  2. 6-DOF analytical IK (UR5e) — direct solve
  3. 7-DOF IK via joint discretization (Franka) — lock joint 5
  4. FK ↔ IK round-trip verification
  5. Multiple configurations with solution counts

Usage:
    cd mj_manipulator
    uv run python demos/ik_solver.py
"""

import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from mj_environment import Environment
from mj_manipulator.arms.franka import (
    FRANKA_HOME,
    add_franka_ee_site,
    create_franka_arm,
)
from mj_manipulator.arms.ur5e import UR5E_HOME, create_ur5e_arm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parent.parent.parent  # robot-code/
MENAGERIE = WORKSPACE / "mujoco_menagerie"
UR5E_SCENE = MENAGERIE / "universal_robots_ur5e" / "scene.xml"
FRANKA_SCENE = MENAGERIE / "franka_emika_panda" / "scene.xml"


def print_header(title):
    print(f"\n{'=' * 64}")
    print(f"  {title}")
    print(f"{'=' * 64}")


# ---------------------------------------------------------------------------
# Demo 1: Kinematic extraction
# ---------------------------------------------------------------------------
def demo_kinematics(arm, label):
    """Show the H/P vectors extracted from MuJoCo for EAIK."""
    solver = arm.ik_solver
    print_header(f"{label} - Kinematic Extraction ({arm.dof}-DOF)")

    print(f"\n  Joint axes H ({solver.H.shape}):")
    for i, h in enumerate(solver.H):
        print(f"    Joint {i+1}: [{h[0]:7.4f}, {h[1]:7.4f}, {h[2]:7.4f}]")

    print(f"\n  Position offsets P ({solver.P.shape}):")
    for i, p in enumerate(solver.P):
        label_p = f"Joint {i+1}" if i < len(solver.H) else "EE"
        print(f"    {label_p:>7s}: [{p[0]:8.5f}, {p[1]:8.5f}, {p[2]:8.5f}]")

    if solver.robot is not None:
        family = solver.robot.getKinematicFamily()
        known = solver.robot.hasKnownDecomposition()
        print(f"\n  Kinematic family: {family}")
        print(f"  Known decomposition: {known}")
    else:
        print(f"\n  7-DOF with joint {solver.fixed_joint_index + 1} discretized")
        print(f"  Discretization steps: {len(solver.discretize_values)}")


# ---------------------------------------------------------------------------
# Demo 2: IK solve with full analysis
# ---------------------------------------------------------------------------
def demo_ik_analysis(arm, q_test, config_name, label):
    """Solve IK for a configuration and analyze all solutions."""
    # Set arm to test config and compute FK
    for i, idx in enumerate(arm.joint_qpos_indices):
        arm.env.data.qpos[idx] = q_test[i]
    mujoco.mj_forward(arm.env.model, arm.env.data)

    pose = arm.get_ee_pose()
    pos = pose[:3, 3]

    print(f"\n  Config '{config_name}':")
    print(f"    q = {np.array2string(q_test, precision=3)}")
    print(f"    EE pos = {np.array2string(pos, precision=4)}")

    t0 = time.perf_counter()
    all_solutions = arm.ik_solver.solve(pose)
    t_solve = (time.perf_counter() - t0) * 1000

    # Filter to within-limits solutions (avoids redundant solve call)
    lower, upper = arm.get_joint_limits()
    valid_solutions = [
        q for q in all_solutions
        if np.all(q >= lower) and np.all(q <= upper)
    ]

    print(f"    Solutions: {len(all_solutions)} total, "
          f"{len(valid_solutions)} within limits  ({t_solve:.1f} ms)")

    if not all_solutions:
        print("    WARNING: No IK solutions found!")
        return

    # FK verification for all solutions
    errors = []
    for q in all_solutions:
        fk_pose = arm.forward_kinematics(q)
        err = np.linalg.norm(fk_pose[:3, 3] - pos)
        errors.append(err)

    errors = np.array(errors)
    print(f"    FK errors: min={errors.min()*1000:.4f} mm, "
          f"max={errors.max()*1000:.4f} mm, "
          f"mean={errors.mean()*1000:.4f} mm")

    # Show a few solutions
    n_show = min(3, len(valid_solutions))
    for i in range(n_show):
        q = valid_solutions[i]
        fk = arm.forward_kinematics(q)
        err = np.linalg.norm(fk[:3, 3] - pos)
        print(f"    Sol {i+1}: q={np.array2string(q, precision=3)}"
              f"  err={err*1000:.4f}mm")


def demo_ik_configs(arm, configs, label):
    """Run IK analysis across multiple configurations."""
    print_header(f"{label} - IK Analysis ({arm.dof}-DOF)")
    for name, q in configs.items():
        demo_ik_analysis(arm, q, name, label)


# ---------------------------------------------------------------------------
# Demo 3: FK ↔ IK round-trip
# ---------------------------------------------------------------------------
def demo_roundtrip(arm, label):
    """Show that IK(FK(q)) recovers configurations with high precision."""
    print_header(f"{label} - FK ↔ IK Round-Trip")

    q_original = arm.get_joint_positions()
    pose = arm.get_ee_pose()
    pos = pose[:3, 3]

    solutions = arm.ik_solver.solve_valid(pose)
    print(f"\n  Original q: {np.array2string(q_original, precision=4)}")
    print(f"  EE pos:     {np.array2string(pos, precision=4)}")
    print(f"  IK found:   {len(solutions)} valid solutions")

    if not solutions:
        print("  No solutions — skipping round-trip check")
        return

    # Find closest solution to original q
    best_q_err = float("inf")
    best_pos_err = float("inf")
    for q in solutions:
        q_err = np.linalg.norm(q - q_original)
        fk = arm.forward_kinematics(q)
        pos_err = np.linalg.norm(fk[:3, 3] - pos)
        if q_err < best_q_err:
            best_q_err = q_err
            best_pos_err = pos_err

    print(f"\n  Closest solution:")
    print(f"    Joint-space error: {best_q_err:.6f} rad")
    print(f"    Position error:    {best_pos_err*1000:.4f} mm")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not MENAGERIE.exists():
        print(f"ERROR: mujoco_menagerie not found at {MENAGERIE}")
        sys.exit(1)

    # === UR5e (6-DOF, direct analytical IK) ===
    ur5e_env = Environment(str(UR5E_SCENE))
    ur5e = create_ur5e_arm(ur5e_env)
    for i, idx in enumerate(ur5e.joint_qpos_indices):
        ur5e_env.data.qpos[idx] = UR5E_HOME[i]
    mujoco.mj_forward(ur5e_env.model, ur5e_env.data)

    demo_kinematics(ur5e, "UR5e")

    ur5e_configs = {
        "home": UR5E_HOME.copy(),
        "offset_1": UR5E_HOME + np.array([0.2, -0.1, 0.15, -0.1, 0.2, 0.1]),
        "offset_2": UR5E_HOME + np.array([-0.3, 0.2, -0.1, 0.3, -0.1, 0.0]),
        "stretched": np.array([0.0, -1.5708, 0.0, 0.0, 0.0, 0.0]),
    }
    demo_ik_configs(ur5e, ur5e_configs, "UR5e")
    demo_roundtrip(ur5e, "UR5e")

    # === Franka Panda (7-DOF, joint-5 discretization) ===
    spec = mujoco.MjSpec.from_file(str(FRANKA_SCENE))
    add_franka_ee_site(spec)
    franka_dir = FRANKA_SCENE.parent
    tmp_path = franka_dir / "_demo_franka_ee.xml"
    try:
        tmp_path.write_text(spec.to_xml())
        franka_env = Environment(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)

    franka = create_franka_arm(franka_env)
    for i, idx in enumerate(franka.joint_qpos_indices):
        franka_env.data.qpos[idx] = FRANKA_HOME[i]
    mujoco.mj_forward(franka_env.model, franka_env.data)

    demo_kinematics(franka, "Franka Panda")

    franka_configs = {
        "home": FRANKA_HOME.copy(),
        "offset_1": FRANKA_HOME + np.array([0.2, -0.1, 0.0, 0.3, 0.0, -0.2, 0.1]),
        "offset_2": FRANKA_HOME + np.array([-0.1, 0.2, 0.1, 0.0, -0.1, 0.1, 0.0]),
    }
    demo_ik_configs(franka, franka_configs, "Franka Panda")
    demo_roundtrip(franka, "Franka Panda")

    print_header(
        "DONE - EAIK Analytical IK\n"
        "  UR5e: 6-DOF direct solve\n"
        "  Franka: 7-DOF via joint-5 discretization\n"
        "  Kinematics extracted from MuJoCo — no DH parameters needed"
    )
    print()


if __name__ == "__main__":
    main()
