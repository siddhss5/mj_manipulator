"""Cartesian velocity control demo with UR5e and Franka Panda.

Shows CartesianController in three modes:
  1. Teleop  — step() in a simulated 125 Hz control loop
  2. Move    — move() for a constant-twist plan with a distance limit
  3. Move-to — move_to() tracking a nearby target pose

Also shows the internal QP solver table and Jacobian for reference.

Usage:
    cd mj_manipulator
    uv run python demos/cartesian_control.py
"""

import mujoco
import numpy as np

from mj_environment import Environment
from mj_manipulator import CartesianController
from mj_manipulator.arms.franka import FRANKA_HOME, add_franka_ee_site, create_franka_arm
from mj_manipulator.arms.ur5e import UR5E_HOME, create_ur5e_arm

# Internal functions — useful for introspection but not part of the public API
from mj_manipulator.cartesian import get_ee_jacobian, twist_to_joint_velocity
from mj_manipulator.menagerie import menagerie_scene

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
UR5E_SCENE = menagerie_scene("universal_robots_ur5e")
FRANKA_SCENE = menagerie_scene("franka_emika_panda")


def print_header(title):
    print(f"\n{'=' * 64}")
    print(f"  {title}")
    print(f"{'=' * 64}")


# ---------------------------------------------------------------------------
# Demo 1: Teleop — simulated 125 Hz control loop
# ---------------------------------------------------------------------------
def demo_teleop(arm, label):
    """Simulate a 125 Hz teleop loop: operator holds -Z at 5 cm/s for 2 s."""
    print_header(f"{label} — Teleop (step, 125 Hz × 2 s)")

    controller = CartesianController.from_arm(arm)
    model, data = arm.env.model, arm.env.data
    ee_id = arm.ee_site_id

    dt = 0.008        # 125 Hz
    n_steps = 250     # 2 seconds
    twist = np.array([0, 0, -0.05, 0, 0, 0])  # 5 cm/s downward

    ee_start = data.site_xpos[ee_id].copy()
    fractions = []

    for _ in range(n_steps):
        result = controller.step(twist, dt)
        mujoco.mj_forward(model, data)
        fractions.append(result.achieved_fraction)

    ee_end = data.site_xpos[ee_id].copy()
    displacement = np.linalg.norm(ee_end - ee_start)
    expected = abs(twist[2]) * n_steps * dt

    print(f"\n  Twist:      -Z at 5 cm/s for {n_steps * dt:.1f} s ({n_steps} steps)")
    print(f"  EE moved:   {displacement * 1000:.2f} mm  (expected {expected * 1000:.2f} mm)")
    print(f"  Tracking:   {displacement / expected * 100:.1f}%")
    print(f"  Avg frac:   {np.mean(fractions):.3f}")
    print(f"  Last limit: {result.limiting_factor or 'none'}")


# ---------------------------------------------------------------------------
# Demo 2: Move — constant twist with distance limit
# ---------------------------------------------------------------------------
def demo_move(arm, label):
    """Use move() to execute a 3 cm downward approach."""
    print_header(f"{label} — Scripted Move (move, 3 cm -Z)")

    controller = CartesianController.from_arm(arm)
    model, data = arm.env.model, arm.env.data

    result = controller.move(
        twist=np.array([0, 0, -0.05, 0, 0, 0]),
        dt=0.008,
        max_distance=0.03,
        step_fn=lambda: mujoco.mj_forward(model, data),
    )

    print(f"\n  Terminated: {result.terminated_by}")
    print(f"  Moved:      {result.distance_moved * 1000:.2f} mm")
    print(f"  Duration:   {result.duration * 1000:.1f} ms  ({result.duration / 0.008:.0f} steps)")


# ---------------------------------------------------------------------------
# Demo 3: Move-to — track a nearby target pose
# ---------------------------------------------------------------------------
def demo_move_to(arm, label):
    """Use move_to() to reach a pose offset 3 cm in Y from current EE."""
    print_header(f"{label} — Move-to (move_to, +3 cm Y)")

    controller = CartesianController.from_arm(arm)
    model, data = arm.env.model, arm.env.data
    ee_id = arm.ee_site_id

    mujoco.mj_forward(model, data)
    target = np.eye(4)
    target[:3, :3] = data.site_xmat[ee_id].reshape(3, 3)
    target[:3, 3] = data.site_xpos[ee_id] + np.array([0, 0.03, 0])

    result = controller.move_to(
        target,
        dt=0.008,
        max_duration=10.0,
        speed=0.05,
        position_tol=0.002,
        rotation_tol=0.05,
        step_fn=lambda: mujoco.mj_forward(model, data),
    )

    pos_err = np.linalg.norm(target[:3, 3] - data.site_xpos[ee_id])
    print(f"\n  Terminated: {result.terminated_by}")
    print(f"  Pos error:  {pos_err * 1000:.2f} mm")
    print(f"  Duration:   {result.duration * 1000:.1f} ms  ({result.duration / 0.008:.0f} steps)")


# ---------------------------------------------------------------------------
# Demo 4: Jacobian analysis (internal)
# ---------------------------------------------------------------------------
def demo_jacobian(arm, label):
    """Analyze the Jacobian at current configuration."""
    dof = arm.dof
    print_header(f"{label} — Jacobian Analysis ({dof}-DOF, internal)")

    J = get_ee_jacobian(arm.env.model, arm.env.data, arm.ee_site_id, arm.joint_qvel_indices)
    sv = np.linalg.svd(J, compute_uv=False)
    manipulability = float(np.prod(sv[:min(6, dof)]))

    print(f"\n  Jacobian shape:  {J.shape}")
    print(f"  Rank:            {np.linalg.matrix_rank(J, tol=1e-6)}")
    print(f"  Singular values: {np.array2string(sv, precision=4)}")
    print(f"  Manipulability:  {manipulability:.6f}")


# ---------------------------------------------------------------------------
# Demo 5: QP solver table (internal)
# ---------------------------------------------------------------------------
def demo_qp_solver(arm, label):
    """Show QP solver results for various twists without mutating state."""
    dof = arm.dof
    print_header(f"{label} — QP Solver Table ({dof}-DOF, internal)")

    model, data = arm.env.model, arm.env.data
    J = get_ee_jacobian(model, data, arm.ee_site_id, arm.joint_qvel_indices)
    q_current = arm.get_joint_positions()
    q_min, q_max = arm.get_joint_limits()
    qd_max = arm.config.kinematic_limits.velocity

    twists = {
        "X linear (5cm/s)":   np.array([0.05,  0,     0,    0,   0,   0]),
        "Z linear (5cm/s)":   np.array([0,      0,    -0.05, 0,   0,   0]),
        "Y rotation (0.2r/s)": np.array([0,     0,     0,    0,   0.2, 0]),
        "Combined":            np.array([0.03,  0,    -0.02, 0,   0.1, 0]),
    }

    print(f"\n  {'Twist':<22s}  {'Achieved%':>9s}  {'Limiter':>12s}  {'||qd||':>8s}")
    print(f"  {'-'*22}  {'-'*9}  {'-'*12}  {'-'*8}")

    for name, twist in twists.items():
        result = twist_to_joint_velocity(
            J=J, twist=twist, q_current=q_current,
            q_min=q_min, q_max=q_max, qd_max=qd_max, dt=0.008,
        )
        limiter = result.limiting_factor or "none"
        print(
            f"  {name:<22s}  {result.achieved_fraction:>8.1%}  "
            f"{limiter:>12s}  {np.linalg.norm(result.joint_velocities):>8.4f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _load_arm(robot):
    if robot == "ur5e":
        env = Environment(str(UR5E_SCENE))
        arm = create_ur5e_arm(env, with_ik=False)
        for i, idx in enumerate(arm.joint_qpos_indices):
            env.data.qpos[idx] = UR5E_HOME[i]
        mujoco.mj_forward(env.model, env.data)
        return arm, "UR5e"

    spec = mujoco.MjSpec.from_file(str(FRANKA_SCENE))
    add_franka_ee_site(spec)
    tmp = FRANKA_SCENE.parent / "_demo_franka_ee.xml"
    try:
        tmp.write_text(spec.to_xml())
        env = Environment(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)
    arm = create_franka_arm(env, with_ik=False)
    for i, idx in enumerate(arm.joint_qpos_indices):
        env.data.qpos[idx] = FRANKA_HOME[i]
    mujoco.mj_forward(env.model, env.data)
    return arm, "Franka Panda"


def run_demos(arm, label):
    demo_teleop(arm, label)
    # Reset to home before each subsequent demo so joint limits aren't hit
    home = UR5E_HOME if "UR5e" in label else FRANKA_HOME
    for i, idx in enumerate(arm.joint_qpos_indices):
        arm.env.data.qpos[idx] = home[i]
    mujoco.mj_forward(arm.env.model, arm.env.data)

    demo_move(arm, label)

    for i, idx in enumerate(arm.joint_qpos_indices):
        arm.env.data.qpos[idx] = home[i]
    mujoco.mj_forward(arm.env.model, arm.env.data)

    demo_move_to(arm, label)

    for i, idx in enumerate(arm.joint_qpos_indices):
        arm.env.data.qpos[idx] = home[i]
    mujoco.mj_forward(arm.env.model, arm.env.data)

    demo_jacobian(arm, label)
    demo_qp_solver(arm, label)


def main():
    ur5e_arm, ur5e_label = _load_arm("ur5e")
    run_demos(ur5e_arm, ur5e_label)

    franka_arm, franka_label = _load_arm("franka")
    run_demos(franka_arm, franka_label)

    print_header(
        "DONE — CartesianController works with\n"
        "  UR5e (6-DOF) and Franka Panda (7-DOF)"
    )
    print()


if __name__ == "__main__":
    main()
