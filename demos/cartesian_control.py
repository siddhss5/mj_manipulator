"""Cartesian velocity control demo with UR5e and Franka Panda.

Demonstrates that the same QP-based twist controller works with both
6-DOF and 7-DOF arms using real robot models from mujoco_menagerie.

Shows three capabilities:
  1. Jacobian computation and rank analysis
  2. Cartesian twist to joint velocities (QP solver)
  3. Multi-step trajectory following via step_twist

Usage:
    cd mj_manipulator
    uv run python demos/cartesian_control.py
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

from mj_environment import Environment
from mj_manipulator.arms.franka import (
    FRANKA_HOME,
    FRANKA_VELOCITY_LIMITS,
    add_franka_ee_site,
    create_franka_arm,
)
from mj_manipulator.arms.ur5e import (
    UR5E_HOME,
    UR5E_VELOCITY_LIMITS,
    create_ur5e_arm,
)
from mj_manipulator.cartesian import (
    get_ee_jacobian,
    step_twist,
    twist_to_joint_velocity,
)

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
# Demo 1: Jacobian analysis
# ---------------------------------------------------------------------------
def demo_jacobian(arm, label):
    """Analyze the Jacobian at current configuration."""
    dof = arm.dof
    print_header(f"{label} - Jacobian Analysis ({dof}-DOF)")

    J = get_ee_jacobian(
        arm.env.model, arm.env.data, arm.ee_site_id, arm.joint_qvel_indices,
    )
    print(f"\n  Jacobian shape: {J.shape}")
    print(f"  Jacobian rank:  {np.linalg.matrix_rank(J, tol=1e-6)}")
    print(f"  Frobenius norm: {np.linalg.norm(J):.4f}")

    sv = np.linalg.svd(J, compute_uv=False)
    print(f"  Singular values: {np.array2string(sv, precision=4)}")
    manipulability = float(np.prod(sv[:min(6, dof)]))
    print(f"  Manipulability:  {manipulability:.6f}")

    if dof == 7:
        print(f"  Null space dim:  1 (redundant arm)")
    elif dof == 6:
        print(f"  Null space dim:  0 (fully actuated at this config)")


# ---------------------------------------------------------------------------
# Demo 2: QP solver comparison
# ---------------------------------------------------------------------------
def demo_qp_solver(arm, vel_limits, label):
    """Compare QP solver results for different twists."""
    dof = arm.dof
    print_header(f"{label} - QP Solver ({dof}-DOF)")

    J = get_ee_jacobian(
        arm.env.model, arm.env.data, arm.ee_site_id, arm.joint_qvel_indices,
    )
    q_current = arm.get_joint_positions()
    q_min = -np.ones(dof) * 6.28
    q_max = np.ones(dof) * 6.28

    twists = {
        "X linear (5cm/s)": np.array([0.05, 0, 0, 0, 0, 0]),
        "Z linear (5cm/s)": np.array([0, 0, -0.05, 0, 0, 0]),
        "Y rotation (0.2r/s)": np.array([0, 0, 0, 0, 0.2, 0]),
        "Combined": np.array([0.03, 0, -0.02, 0, 0.1, 0]),
    }

    print(f"\n  {'Twist':<22s}  {'Achieved%':>9s}  {'Error':>8s}  {'Limiter':>12s}  "
          f"{'||qd||':>8s}")
    print(f"  {'-'*22}  {'-'*9}  {'-'*8}  {'-'*12}  {'-'*8}")

    for name, twist in twists.items():
        result = twist_to_joint_velocity(
            J=J, twist=twist, q_current=q_current,
            q_min=q_min, q_max=q_max, qd_max=vel_limits,
            dt=0.004,
        )
        limiter = result.limiting_factor or "none"
        qd_norm = np.linalg.norm(result.joint_velocities)
        print(
            f"  {name:<22s}  {result.achieved_fraction:>8.1%}  "
            f"{result.twist_error:>8.5f}  {limiter:>12s}  "
            f"{qd_norm:>8.4f}"
        )


# ---------------------------------------------------------------------------
# Demo 3: Multi-step trajectory via step_twist
# ---------------------------------------------------------------------------
def demo_step_twist(arm, vel_limits, label):
    """Execute multiple twist steps and show EE motion."""
    dof = arm.dof
    model, data = arm.env.model, arm.env.data
    ee_id = arm.ee_site_id
    qpos_idx = arm.joint_qpos_indices
    qvel_idx = arm.joint_qvel_indices
    print_header(f"{label} - Step Twist Trajectory ({dof}-DOF)")

    q_min = -np.ones(dof) * 6.28
    q_max = np.ones(dof) * 6.28
    dt = 0.004

    # Straight-line motion: 5cm/s in -Z for 20 steps (= 4mm total)
    twist = np.array([0, 0, -0.05, 0, 0, 0])
    n_steps = 20

    ee_start = data.site_xpos[ee_id].copy()
    q_dot_prev = None

    fractions = []
    for _ in range(n_steps):
        q_new, result = step_twist(
            model, data, ee_id, qpos_idx, qvel_idx,
            q_min=q_min, q_max=q_max, qd_max=vel_limits,
            twist=twist, dt=dt, q_dot_prev=q_dot_prev,
        )
        q_dot_prev = result.joint_velocities

        for j, idx in enumerate(qpos_idx):
            data.qpos[idx] = q_new[j]
        mujoco.mj_forward(model, data)
        fractions.append(result.achieved_fraction)

    ee_end = data.site_xpos[ee_id].copy()
    displacement = ee_end - ee_start
    distance = np.linalg.norm(displacement)

    print(f"\n  Direction: -Z (downward) at 5 cm/s")
    print(f"  Steps:     {n_steps} x {dt*1000:.0f}ms = {n_steps*dt*1000:.0f}ms")
    print(f"\n  EE start:  {np.array2string(ee_start, precision=4)}")
    print(f"  EE end:    {np.array2string(ee_end, precision=4)}")
    print(f"  Delta:     {np.array2string(displacement, precision=5)}")
    print(f"  Distance:  {distance*1000:.2f} mm")
    expected = abs(twist[2]) * n_steps * dt
    print(f"  Expected:  {expected*1000:.2f} mm")
    print(f"  Tracking:  {distance/expected*100:.1f}%")
    print(f"  Avg frac:  {np.mean(fractions):.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not MENAGERIE.exists():
        print(f"ERROR: mujoco_menagerie not found at {MENAGERIE}")
        sys.exit(1)

    # --- UR5e (6-DOF) ---
    ur5e_env = Environment(str(UR5E_SCENE))
    ur5e_arm = create_ur5e_arm(ur5e_env, with_ik=False)
    for i, idx in enumerate(ur5e_arm.joint_qpos_indices):
        ur5e_env.data.qpos[idx] = UR5E_HOME[i]
    mujoco.mj_forward(ur5e_env.model, ur5e_env.data)

    demo_jacobian(ur5e_arm, "UR5e")
    demo_qp_solver(ur5e_arm, UR5E_VELOCITY_LIMITS, "UR5e")
    demo_step_twist(ur5e_arm, UR5E_VELOCITY_LIMITS, "UR5e")

    # --- Franka Panda (7-DOF) ---
    spec = mujoco.MjSpec.from_file(str(FRANKA_SCENE))
    add_franka_ee_site(spec)
    franka_dir = FRANKA_SCENE.parent
    tmp_path = franka_dir / "_demo_franka_ee.xml"
    try:
        tmp_path.write_text(spec.to_xml())
        franka_env = Environment(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)

    franka_arm = create_franka_arm(franka_env, with_ik=False)
    for i, idx in enumerate(franka_arm.joint_qpos_indices):
        franka_env.data.qpos[idx] = FRANKA_HOME[i]
    mujoco.mj_forward(franka_env.model, franka_env.data)

    demo_jacobian(franka_arm, "Franka Panda")
    demo_qp_solver(franka_arm, FRANKA_VELOCITY_LIMITS, "Franka Panda")
    demo_step_twist(franka_arm, FRANKA_VELOCITY_LIMITS, "Franka Panda")

    print_header(
        "DONE - Same Cartesian controller code\n"
        "  works with UR5e (6-DOF) and Franka Panda (7-DOF)"
    )
    print()


if __name__ == "__main__":
    main()
