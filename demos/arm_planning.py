"""Motion planning demo with UR5e and Franka Panda.

Demonstrates the Arm class's motion planning capabilities using arm factories:
  1. plan_to_configuration: collision-free joint-space paths
  2. plan_to_pose: pose goal via point TSR (planner handles IK)
  3. Trajectory retiming: time-parameterized trajectories

Usage:
    cd mj_manipulator
    uv run python demos/arm_planning.py
"""

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
# Demo 1: Plan to configuration
# ---------------------------------------------------------------------------
def demo_plan_to_config(arm, q_goal, label):
    """Plan a collision-free path from current config to a goal."""
    print_header(f"{label} - Plan to Configuration ({arm.dof}-DOF)")

    q_start = arm.get_joint_positions()
    print(f"\n  Start: {np.array2string(q_start, precision=4)}")
    print(f"  Goal:  {np.array2string(q_goal, precision=4)}")
    print(f"  Joint delta: {np.linalg.norm(q_goal - q_start):.3f} rad")

    path = arm.plan_to_configuration(q_goal, timeout=10.0, seed=42)

    if path is None:
        print("  Planning FAILED (no collision-free path found)")
        return

    print(f"  Path:  {len(path)} waypoints")

    # Show start/end FK positions
    fk_start = arm.forward_kinematics(path[0])
    fk_end = arm.forward_kinematics(path[-1])
    ee_dist = np.linalg.norm(fk_end[:3, 3] - fk_start[:3, 3])
    print(f"  EE travel: {ee_dist*1000:.1f} mm")


# ---------------------------------------------------------------------------
# Demo 2: Trajectory retiming
# ---------------------------------------------------------------------------
def demo_trajectory(arm, q_goal, label):
    """Plan then retime (time-parameterized output)."""
    print_header(f"{label} - Trajectory Retiming ({arm.dof}-DOF)")

    path = arm.plan_to_configuration(q_goal, timeout=10.0, seed=42)

    if path is None:
        print("  Planning failed")
        return

    traj = arm.retime(path)

    print(f"\n  Duration: {traj.duration:.3f} s")
    print(f"  Samples:  {len(traj.positions)}")

    # Show first and last position
    print(f"  q[0]:     {np.array2string(traj.positions[0], precision=4)}")
    print(f"  q[-1]:    {np.array2string(traj.positions[-1], precision=4)}")

    # Goal match
    goal_err = np.linalg.norm(traj.positions[-1] - q_goal)
    print(f"  Goal err: {goal_err:.6f} rad")


# ---------------------------------------------------------------------------
# Demo 3: Plan to pose (IK + planning)
# ---------------------------------------------------------------------------
def demo_plan_to_pose(arm, label):
    """Plan to a target EE pose (planner handles IK via point TSR)."""
    print_header(f"{label} - Plan to Pose ({arm.dof}-DOF)")

    current_pose = arm.get_ee_pose()
    target_pose = current_pose.copy()
    target_pose[0, 3] += 0.05  # Shift 5cm in +X

    print(f"\n  Current EE: {np.array2string(current_pose[:3, 3], precision=4)}")
    print(f"  Target EE:  {np.array2string(target_pose[:3, 3], precision=4)}")

    path = arm.plan_to_pose(target_pose, timeout=10.0, seed=42)

    if path is None:
        print("  Plan to pose FAILED (IK or planning failed)")
        return

    print(f"  Path: {len(path)} waypoints")

    # Verify final waypoint FK
    fk_final = arm.forward_kinematics(path[-1])
    pos_err = np.linalg.norm(fk_final[:3, 3] - target_pose[:3, 3])
    print(f"  Final FK error: {pos_err*1000:.3f} mm")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # --- UR5e ---
    ur5e_env = Environment(str(UR5E_SCENE))
    ur5e = create_ur5e_arm(ur5e_env)
    for i, idx in enumerate(ur5e.joint_qpos_indices):
        ur5e_env.data.qpos[idx] = UR5E_HOME[i]
    mujoco.mj_forward(ur5e_env.model, ur5e_env.data)

    ur5e_goal = UR5E_HOME + np.array([0.3, -0.2, 0.1, -0.1, 0.2, 0.0])
    demo_plan_to_config(ur5e, ur5e_goal, "UR5e")
    demo_trajectory(ur5e, ur5e_goal, "UR5e")
    demo_plan_to_pose(ur5e, "UR5e")

    # --- Franka ---
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

    franka_goal = FRANKA_HOME + np.array([0.2, -0.1, 0.0, 0.3, 0.0, -0.2, 0.1])
    demo_plan_to_config(franka, franka_goal, "Franka Panda")
    demo_trajectory(franka, franka_goal, "Franka Panda")
    demo_plan_to_pose(franka, "Franka Panda")

    print_header(
        "DONE - Same planning API\n"
        "  works with UR5e (6-DOF) and Franka Panda (7-DOF)"
    )
    print()


if __name__ == "__main__":
    main()
