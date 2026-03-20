#!/usr/bin/env python3
"""Recycling demo — pick up soda cans and drop them in a recycling bin.

End-to-end integration demo showing the full workspace stack:

  Repos used
  ----------
  prl_assets      — real soda can + recycle bin models (with meshes)
  asset_manager   — can geometry metadata for TSR construction
  mj_environment  — Environment.from_model() wrapping MjSpec-built scenes
  tsr.hands       — Robotiq2F140 / FrankaHand grasp templates
  mujoco_menagerie — UR5e + Franka base scenes
  geodude_assets  — Robotiq 2F-140 gripper XML
  mj_manipulator  — arm/gripper factories, GraspManager, SimContext

  Robots
  ------
  UR5e + Robotiq 2F-140
  Franka Panda   + Franka Hand

  Each robot picks up three soda cans from the table and drops them into
  a floor-standing recycling bin. TSR grasp templates are generated from
  the can's physical dimensions (read from asset metadata) using the
  appropriate hand model.

Usage:
    cd mj_manipulator
    uv run mjpython demos/recycling.py --robot ur5e
    uv run mjpython demos/recycling.py --robot franka
    uv run mjpython demos/recycling.py --robot both
    uv run mjpython demos/recycling.py --robot ur5e --physics
    uv run mjpython demos/recycling.py --robot ur5e --headless
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from asset_manager import AssetManager
from mj_environment import Environment
from mj_manipulator.cartesian import step_twist
from mj_manipulator.arms.franka import (
    FRANKA_HOME,
    add_franka_ee_site,
    create_franka_arm,
)
from mj_manipulator.arms.ur5e import UR5E_HOME, UR5E_ROBOTIQ_EE_SITE, create_ur5e_arm
from mj_manipulator.config import PhysicsConfig, PhysicsExecutionConfig
from mj_manipulator.grasp_manager import GraspManager
from mj_manipulator.grippers.franka import FrankaGripper
from mj_manipulator.grippers.robotiq import RobotiqGripper
from mj_manipulator.menagerie import menagerie_scene
from mj_manipulator.sim_context import SimContext
from prl_assets import OBJECTS_DIR
from tsr.hands import FrankaHand, Robotiq2F140

if TYPE_CHECKING:
    from mj_manipulator.arm import Arm

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logging.getLogger("toppra").setLevel(logging.WARNING)  # suppress planning noise
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
import geodude_assets

UR5E_SCENE   = menagerie_scene("universal_robots_ur5e")
FRANKA_SCENE = menagerie_scene("franka_emika_panda")
ROBOTIQ_MODEL = geodude_assets.MODELS_DIR / "robotiq_2f140" / "2f140.xml"

# ---------------------------------------------------------------------------
# prl_assets — real soda can + recycling bin
# ---------------------------------------------------------------------------
_ASSETS  = AssetManager(str(OBJECTS_DIR))
_CAN_GP  = _ASSETS.get("can")["geometric_properties"]   # radius=0.033, height=0.123
_CAN_XML = _ASSETS.get_path("can",         "mujoco")
_BIN_XML = _ASSETS.get_path("yellow_tote", "mujoco")

# Body names produced by MjSpec.attach_body() — prefix + original body name in XML
CAN_BODY_NAMES = [f"can_{i}/can" for i in range(3)]

# Bin on the floor at y = −0.70 (same side as cans but farther away).
# At UR5E_HOME the arm extends toward +y, so placing the bin at y = −0.70
# avoids any collision at the home configuration.
# EE is placed 15 cm above the tote opening; the grasped can drops in on release.
BIN_POS        = np.array([0.25, -0.70, 0.0])
_TOTE_HEIGHT   = _ASSETS.get("yellow_tote")["geometric_properties"]["outer_size"][2]
BIN_OPENING_Z  = BIN_POS[2] + _TOTE_HEIGHT   # 0.29 m
BIN_PLACE_Z    = BIN_OPENING_Z + 0.15


# ---------------------------------------------------------------------------
# TSR grasp generation
# ---------------------------------------------------------------------------
_ROBOTIQ = Robotiq2F140()
_FRANKA  = FrankaHand()

# Place pose: top-down palm-down above the bin opening
_TOP_DOWN = np.array([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1],
], dtype=float)


def make_grasp_tsrs(T_center: np.ndarray, robot_type: str) -> list:
    """Generate grasp TSRs for a soda can from asset metadata.

    TSR convention: reference at the can's bottom centre, z up.
    T_center is the MuJoCo body pose (geometric centre); we shift
    to the bottom face before calling instantiate().
    """
    hand = _FRANKA if robot_type == "franka" else _ROBOTIQ
    T_bottom = T_center.copy()
    T_bottom[2, 3] -= _CAN_GP["height"] / 2   # centre → bottom
    templates = hand.grasp_cylinder(_CAN_GP["radius"], _CAN_GP["height"])
    return [t.instantiate(T_bottom) for t in templates]


def compute_place_pose() -> np.ndarray:
    """Top-down pose with EE just inside the yellow tote opening."""
    pose = _TOP_DOWN.copy()
    pose[:3, 3] = [BIN_POS[0], BIN_POS[1], BIN_PLACE_Z]
    return pose


# ---------------------------------------------------------------------------
# Scene composition
# ---------------------------------------------------------------------------


def _attach_objects(spec: mujoco.MjSpec, can_positions: list[list[float]]) -> None:
    """Attach prl_assets cans + recycle bin to the spec via MjSpec.

    Each can spec is loaded from prl_assets and attached at its table position.
    The recycle bin is placed on the floor. Both retain their original mesh
    references — MuJoCo resolves them from each spec's modelfiledir.
    """
    for i, pos in enumerate(can_positions):
        can_spec = mujoco.MjSpec.from_file(_CAN_XML)
        f = spec.worldbody.add_frame()
        f.pos = pos
        f.attach_body(can_spec.worldbody.first_body(), prefix=f"can_{i}/")

    bin_spec = mujoco.MjSpec.from_file(_BIN_XML)
    f = spec.worldbody.add_frame()
    f.pos = list(BIN_POS)
    f.attach_body(bin_spec.worldbody.first_body(), prefix="bin/")


# ---------------------------------------------------------------------------
# Setup functions per robot
# ---------------------------------------------------------------------------


def setup_ur5e():
    """Compose UR5e + Robotiq + cans + recycle bin scene."""
    for path, label in [(UR5E_SCENE, "UR5e scene"), (ROBOTIQ_MODEL, "Robotiq model")]:
        if not path.exists():
            print(f"ERROR: {label} not found at {path}")
            sys.exit(1)

    spec = mujoco.MjSpec.from_file(str(UR5E_SCENE))
    robotiq_spec = mujoco.MjSpec.from_file(str(ROBOTIQ_MODEL))

    # Attach Robotiq 2F-140 at the UR5e flange
    wrist = spec.worldbody.find_child("wrist_3_link")
    frame = wrist.add_frame()
    frame.pos  = [0, 0.1, 0]
    frame.quat = [-1, 1, 0, 0]
    frame.attach_body(robotiq_spec.worldbody.first_body(), prefix="gripper/")

    # Three cans on the table (surface at z ≈ 0.46 in menagerie scene), all at y < 0.
    table_z = 0.46
    can_positions = [
        [0.40, -0.15, table_z],
        [0.50, -0.15, table_z],
        [0.45, -0.28, table_z],
    ]
    _attach_objects(spec, can_positions)

    model = spec.compile()
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    env = Environment.from_model(model, data)

    gm      = GraspManager(env.model, env.data)
    gripper = RobotiqGripper(env.model, env.data, "ur5e", prefix="gripper/", grasp_manager=gm)
    arm     = create_ur5e_arm(env, ee_site=UR5E_ROBOTIQ_EE_SITE, gripper=gripper, grasp_manager=gm)

    for i, idx in enumerate(arm.joint_qpos_indices):
        env.data.qpos[idx] = UR5E_HOME[i]
    mujoco.mj_forward(env.model, env.data)

    return env, arm, UR5E_HOME


def setup_franka():
    """Compose Franka + cans + recycle bin scene."""
    if not FRANKA_SCENE.exists():
        print(f"ERROR: Franka scene not found at {FRANKA_SCENE}")
        sys.exit(1)

    spec = mujoco.MjSpec.from_file(str(FRANKA_SCENE))
    add_franka_ee_site(spec)

    # Three cans on the table (surface at z ≈ 0.46 in menagerie scene), all at y < 0.
    table_z = 0.46
    can_positions = [
        [0.40, -0.15, table_z],
        [0.48, -0.15, table_z],
        [0.44, -0.28, table_z],
    ]
    _attach_objects(spec, can_positions)

    model = spec.compile()
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    env = Environment.from_model(model, data)

    gm      = GraspManager(env.model, env.data)
    gripper = FrankaGripper(env.model, env.data, "franka", grasp_manager=gm)
    arm     = create_franka_arm(env, gripper=gripper, grasp_manager=gm)

    for i, idx in enumerate(arm.joint_qpos_indices):
        env.data.qpos[idx] = FRANKA_HOME[i]
    gripper.kinematic_open()  # default qpos=0 is fully closed → self-collision
    mujoco.mj_forward(env.model, env.data)

    return env, arm, FRANKA_HOME


# ---------------------------------------------------------------------------
# Pickup / place helpers
# ---------------------------------------------------------------------------


def cartesian_lift(ctx: SimContext, arm: Arm, height: float = 0.05) -> None:
    """Lift the EE straight up using Jacobian-based cartesian control.

    Bypasses CBiRRT — used immediately after grasping to clear the object
    off the table before planning the place motion. (The collision checker
    flags grasped-object-table contacts as invalid at the start config.)
    """
    model = arm.env.model
    data  = arm.env.data
    q_lower, q_upper = arm.get_joint_limits()
    qd_max = arm.config.kinematic_limits.velocity

    lift_speed = 0.10  # m/s
    dt         = 0.004
    n_steps    = max(1, int(height / (lift_speed * dt)))
    twist      = np.array([0.0, 0.0, lift_speed, 0.0, 0.0, 0.0])

    q_dot_prev = None
    for _ in range(n_steps):
        q_new, result = step_twist(
            model, data,
            arm.ee_site_id,
            arm.joint_qpos_indices,
            arm.joint_qvel_indices,
            q_min=q_lower, q_max=q_upper, qd_max=qd_max,
            twist=twist, dt=dt, q_dot_prev=q_dot_prev,
        )
        q_dot_prev = result.joint_velocities
        for j, idx in enumerate(arm.joint_qpos_indices):
            data.qpos[idx] = q_new[j]
        if arm.grasp_manager is not None:
            arm.grasp_manager.update_attached_poses(data)
        mujoco.mj_forward(model, data)
    ctx.sync()


def pickup(ctx: SimContext, arm: Arm, body_name: str, robot_type: str) -> bool:
    """Plan to a TSR-sampled grasp pose, execute, close gripper, lift."""
    logger.info("Planning grasp for %s...", body_name)

    T_center   = arm.env.get_body_pose(body_name)
    grasp_tsrs = make_grasp_tsrs(T_center, robot_type)

    path = arm.plan_to_tsrs(grasp_tsrs, timeout=10.0)
    if path is None:
        logger.warning("Grasp planning failed for %s", body_name)
        return False

    traj = arm.retime(path)
    logger.info("Executing grasp (%d waypoints, %.2fs)...", traj.num_waypoints, traj.duration)
    if not ctx.execute(traj):
        logger.warning("Grasp execution failed")
        return False

    logger.info("Closing gripper on %s...", body_name)
    grasped = ctx.arm(arm.config.name).grasp(body_name)
    if not grasped:
        logger.warning("Grasp failed for %s", body_name)
        return False

    cartesian_lift(ctx, arm, height=0.10)
    return True


def place(ctx: SimContext, arm: Arm, body_name: str) -> bool:
    """Plan to the place pose above the bin, execute, release."""
    logger.info("Planning placement above recycling bin...")
    place_pose = compute_place_pose()
    path = arm.plan_to_pose(place_pose, timeout=10.0)
    if path is None:
        logger.warning("Place planning failed")
        return False

    traj = arm.retime(path)
    logger.info(
        "Executing placement (%d waypoints, %.2fs)...",
        traj.num_waypoints, traj.duration,
    )
    if not ctx.execute(traj):
        logger.warning("Place execution failed")
        return False

    logger.info("Opening gripper...")
    ctx.arm(arm.config.name).release(body_name)
    return True


# ---------------------------------------------------------------------------
# Main recycling loop
# ---------------------------------------------------------------------------


def run_recycling(
    robot_type: str,
    *,
    physics: bool = False,
    headless: bool = False,
    cycles: int = 3,
):
    """Run the recycling demo for a single robot type."""
    print(f"\n{'=' * 60}")
    print(f"  Recycling Demo — {robot_type.upper()}")
    print(f"  Mode: {'Physics' if physics else 'Kinematic'}")
    print(f"{'=' * 60}")

    if robot_type == "ur5e":
        env, arm, home = setup_ur5e()
    elif robot_type == "franka":
        env, arm, home = setup_franka()
    else:
        print(f"ERROR: Unknown robot type: {robot_type}")
        return

    physics_config = PhysicsConfig(
        execution=PhysicsExecutionConfig(
            control_dt=0.008,
            position_tolerance=0.15,
            velocity_tolerance=0.5,
            convergence_timeout_steps=5000,
        ),
    ) if physics else None

    with SimContext(
        env.model, env.data, {arm.config.name: arm},
        physics=physics,
        headless=headless,
        physics_config=physics_config,
    ) as ctx:
        for cycle, body_name in enumerate(CAN_BODY_NAMES[:cycles], 1):
            print(f"\n--- Cycle {cycle}: {body_name} ---")
            print(f"  Can at: {env.get_body_pose(body_name)[:3, 3].round(3)}")

            if not pickup(ctx, arm, body_name, robot_type):
                print(f"  FAILED to pick up {body_name}")
                continue

            print(f"  Picked up {body_name}")

            if not place(ctx, arm, body_name):
                print(f"  FAILED to place {body_name}")
                continue

            print(f"  Dropped {body_name} into bin")

            # In kinematic mode, hide the object so it doesn't float in the air.
            # In physics mode the can physically falls into the bin on release.
            if not physics:
                env.hide_freebody(body_name)

            home_path = arm.plan_to_configuration(home, timeout=10.0)
            if home_path is not None:
                ctx.execute(arm.retime(home_path))
                print("  Returned to home")
            else:
                print("  Could not plan home path — skipping")

            ctx.sync()

        print(f"\nCompleted {min(cycles, len(CAN_BODY_NAMES))} cycles.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Recycling demo — mj_manipulator integration")
    parser.add_argument(
        "--robot", choices=["ur5e", "franka", "both"], default="ur5e",
        help="Which robot to run (default: ur5e)",
    )
    parser.add_argument("--physics",  action="store_true", help="Enable physics simulation")
    parser.add_argument("--headless", action="store_true", help="Run without viewer")
    parser.add_argument("--cycles",   type=int, default=3,  help="Number of cans to recycle")
    parser.add_argument("--debug",    action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    robots = ["ur5e", "franka"] if args.robot == "both" else [args.robot]
    for robot in robots:
        run_recycling(
            robot,
            physics=args.physics,
            headless=args.headless,
            cycles=args.cycles,
        )

    print(f"\n{'=' * 60}")
    print("  DONE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
