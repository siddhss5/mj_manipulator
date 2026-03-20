"""SimContext execution demo with UR5e and Franka Panda.

Demonstrates the SimContext execution layer — the same ExecutionContext protocol
that will be used on real hardware. Shows all three execution patterns:

  1. Batch trajectory execution:  ctx.execute(trajectory)
  2. Streaming joint control:     ctx.step({"arm_name": q_target})
  3. Streaming cartesian control:  ctx.step_cartesian("arm_name", q_target)

Each pattern runs in both physics mode (PD control with settling) and
kinematic mode (perfect tracking, no dynamics).

Usage:
    cd mj_manipulator
    uv run python demos/sim_context.py
"""

import mujoco
import numpy as np

from mj_environment import Environment
from mj_manipulator.arms.franka import (
    FRANKA_HOME,
    add_franka_ee_site,
    create_franka_arm,
)
from mj_manipulator.arms.ur5e import UR5E_HOME, create_ur5e_arm
from mj_manipulator.config import PhysicsConfig, PhysicsExecutionConfig
from mj_manipulator.menagerie import menagerie_scene
from mj_manipulator.sim_context import SimContext

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
UR5E_SCENE = menagerie_scene("universal_robots_ur5e")
FRANKA_SCENE = menagerie_scene("franka_emika_panda")


def print_header(title):
    print(f"\n{'=' * 64}")
    print(f"  {title}")
    print(f"{'=' * 64}")


def _set_arm_positions(arm, env, positions):
    """Set arm joint positions directly in MuJoCo data."""
    for i, idx in enumerate(arm.joint_qpos_indices):
        env.data.qpos[idx] = positions[i]
    mujoco.mj_forward(env.model, env.data)


# ---------------------------------------------------------------------------
# Demo 1: Batch trajectory execution (ctx.execute)
# ---------------------------------------------------------------------------
def demo_execute(arm, env, home, goal, label, physics_config):
    """Plan a trajectory and execute it through SimContext."""
    print_header(f"{label} - Batch Trajectory Execution ({arm.dof}-DOF)")

    # Plan a path
    path = arm.plan_to_configuration(goal, timeout=10.0, seed=42)
    if path is None:
        print("  Planning failed — skipping")
        return

    traj = arm.retime(path)
    print(f"  Trajectory: {traj.num_waypoints} waypoints, {traj.duration:.3f}s")

    for mode, physics in [("Physics", True), ("Kinematic", False)]:
        # Reset to home
        _set_arm_positions(arm, env, home)

        kwargs = {"headless": True}
        if physics:
            kwargs["physics_config"] = physics_config
        else:
            kwargs["physics"] = False

        with SimContext(
            env.model, env.data, {arm.config.name: arm}, **kwargs,
        ) as ctx:
            result = ctx.execute(traj)

        final_q = arm.get_joint_positions()
        err = np.linalg.norm(final_q - goal)
        print(f"  {mode:10s}  success={result}  goal_err={np.degrees(err):.2f}°")


# ---------------------------------------------------------------------------
# Demo 2: Streaming joint control (ctx.step)
# ---------------------------------------------------------------------------
def demo_step(arm, env, home, goal, label, physics_config, n_steps=50):
    """Interpolate to a goal using streaming step() calls."""
    print_header(f"{label} - Streaming Joint Control ({arm.dof}-DOF)")

    for mode, physics in [("Physics", True), ("Kinematic", False)]:
        _set_arm_positions(arm, env, home)

        kwargs = {"headless": True}
        if physics:
            kwargs["physics_config"] = physics_config
        else:
            kwargs["physics"] = False

        with SimContext(
            env.model, env.data, {arm.config.name: arm}, **kwargs,
        ) as ctx:
            # Linear interpolation from home → goal
            for i in range(n_steps + 1):
                alpha = i / n_steps
                q_target = home * (1 - alpha) + goal * alpha
                ctx.step({arm.config.name: q_target})

        final_q = arm.get_joint_positions()
        err = np.linalg.norm(final_q - goal)
        print(f"  {mode:10s}  {n_steps} steps  goal_err={np.degrees(err):.2f}°")


# ---------------------------------------------------------------------------
# Demo 3: Streaming cartesian control (ctx.step_cartesian)
# ---------------------------------------------------------------------------
def demo_step_cartesian(arm, env, home, goal, label, physics_config, n_steps=50):
    """Interpolate to a goal using streaming step_cartesian() calls."""
    print_header(f"{label} - Streaming Cartesian Control ({arm.dof}-DOF)")

    for mode, physics in [("Physics", True), ("Kinematic", False)]:
        _set_arm_positions(arm, env, home)

        kwargs = {"headless": True}
        if physics:
            kwargs["physics_config"] = physics_config
        else:
            kwargs["physics"] = False

        with SimContext(
            env.model, env.data, {arm.config.name: arm}, **kwargs,
        ) as ctx:
            # Linear interpolation with velocity estimate
            dt = ctx.control_dt
            for i in range(n_steps + 1):
                alpha = i / n_steps
                q_target = home * (1 - alpha) + goal * alpha
                velocity = (goal - home) / (n_steps * dt) if i < n_steps else None
                ctx.step_cartesian(arm.config.name, q_target, velocity)

        final_q = arm.get_joint_positions()
        err = np.linalg.norm(final_q - goal)
        print(f"  {mode:10s}  {n_steps} steps  goal_err={np.degrees(err):.2f}°")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Physics config with generous tolerances for menagerie models
    physics_config = PhysicsConfig(
        execution=PhysicsExecutionConfig(
            control_dt=0.008,
            position_tolerance=0.15,
            velocity_tolerance=0.5,
            convergence_timeout_steps=5000,
        ),
    )

    # --- UR5e (6-DOF) ---
    ur5e_env = Environment(str(UR5E_SCENE))
    ur5e = create_ur5e_arm(ur5e_env, with_ik=False)
    _set_arm_positions(ur5e, ur5e_env, UR5E_HOME)

    ur5e_goal = UR5E_HOME + np.array([0.3, -0.2, 0.1, -0.1, 0.2, 0.0])

    demo_execute(ur5e, ur5e_env, UR5E_HOME, ur5e_goal, "UR5e", physics_config)
    demo_step(ur5e, ur5e_env, UR5E_HOME, ur5e_goal, "UR5e", physics_config)
    demo_step_cartesian(ur5e, ur5e_env, UR5E_HOME, ur5e_goal, "UR5e", physics_config)

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

    franka = create_franka_arm(franka_env, with_ik=False)
    _set_arm_positions(franka, franka_env, FRANKA_HOME)

    franka_goal = FRANKA_HOME + np.array([0.2, -0.1, 0.0, 0.3, 0.0, -0.2, 0.1])

    demo_execute(franka, franka_env, FRANKA_HOME, franka_goal, "Franka", physics_config)
    demo_step(franka, franka_env, FRANKA_HOME, franka_goal, "Franka", physics_config)
    demo_step_cartesian(
        franka, franka_env, FRANKA_HOME, franka_goal, "Franka", physics_config,
    )

    print_header(
        "DONE - Same SimContext API\n"
        "  works with UR5e (6-DOF) and Franka Panda (7-DOF)\n"
        "  in both physics and kinematic modes"
    )
    print()


if __name__ == "__main__":
    main()
