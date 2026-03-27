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
from mj_manipulator.cartesian import CartesianController
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
from tsr.placement import StablePlacer

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
    # Side grasps only — more stable for cylindrical objects
    templates = hand.grasp_cylinder_side(_CAN_GP["radius"], _CAN_GP["height"])
    return [t.instantiate(T_bottom) for t in templates]


def compute_place_pose() -> np.ndarray:
    """Top-down pose with EE just inside the yellow tote opening."""
    pose = _TOP_DOWN.copy()
    pose[:3, 3] = [BIN_POS[0], BIN_POS[1], BIN_PLACE_Z]
    return pose


# ---------------------------------------------------------------------------
# Scene composition
# ---------------------------------------------------------------------------


def _add_table(spec: mujoco.MjSpec, pos: list[float], size: list[float]) -> None:
    """Add a simple table to the scene."""
    table = spec.worldbody.add_body()
    table.name = "table"
    table.pos = pos
    g = table.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = size
    g.rgba = [0.4, 0.3, 0.2, 1.0]


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


def _add_table_and_cans(spec: mujoco.MjSpec, n_cans: int = 3) -> None:
    """Add a table with cans placed via TSR sampling."""
    table_half = [0.15, 0.15, 0.23]  # 30x30cm surface, 46cm tall
    table_center = [0.45, -0.20, table_half[2]]
    _add_table(spec, pos=table_center, size=table_half)

    table_surface = np.eye(4)
    table_surface[:3, 3] = [table_center[0], table_center[1], table_half[2] * 2]
    placer = StablePlacer(table_half[0], table_half[1])
    place_templates = placer.place_cylinder(_CAN_GP["radius"], _CAN_GP["height"])

    # The can XML has body pos="0 0 0.0615" (half-height offset baked in),
    # so the attach frame goes at the table surface, not the can center.
    can_body_offset_z = _CAN_GP["height"] / 2
    min_separation = _CAN_GP["radius"] * 3  # cans must be 3 radii apart
    can_positions = []
    for _ in range(n_cans):
        tsr = place_templates[0].instantiate(table_surface)
        for _attempt in range(50):
            pose = tsr.sample()
            pos = pose[:3, 3]
            # Check separation from already-placed cans
            if all(np.linalg.norm(pos[:2] - np.array(p[:2])) > min_separation
                   for p in can_positions):
                break
        result = list(pos)
        result[2] -= can_body_offset_z
        can_positions.append(result)
    _attach_objects(spec, can_positions)


def _fix_franka_grip_force(model: mujoco.MjModel, target_force: float = 140.0) -> None:
    """Scale Franka gripper actuator to match real hardware grip force.

    The menagerie model's actuator8 under-produces grip force. The real
    Franka hand grips at 140N (70N per finger). We scale both gain and
    bias proportionally so the actuator reaches target_force at full close
    while ctrl=255 can still hold the fingers open.

    Args:
        model: Compiled MjModel (modified in place).
        target_force: Desired grip force at full close [N]. Default 140N
            from the Franka Emika datasheet.
    """
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
    if aid < 0:
        return

    # Actuator force model: force = gain * ctrl + bias[1] * length + bias[2] * velocity
    # To grip at target_force when ctrl=0 and grasping a typical object:
    #   target_force = |bias[1]| * length_grasp
    # To hold open when ctrl=255:
    #   gain * 255 = |bias[1]| * length_open  (zero net force)
    length_open = 0.08  # 2 × finger_joint open position (0.04m each)
    length_grasp = 0.066  # typical grasp (e.g., soda can r=33mm)

    new_bias1 = -target_force / length_grasp
    new_gain = abs(new_bias1) * length_open / 255.0

    # Scale damping proportionally to bias spring
    old_bias1 = model.actuator_biasprm[aid, 1]
    scale = new_bias1 / old_bias1 if old_bias1 != 0 else 1.0

    model.actuator_biasprm[aid, 1] = new_bias1
    model.actuator_biasprm[aid, 2] *= scale
    model.actuator_gainprm[aid, 0] = new_gain


def _compile_and_create_arm(spec, robot_type):
    """Compile spec, create Environment + Arm + Gripper."""
    model = spec.compile()
    if robot_type == "franka":
        _fix_franka_grip_force(model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    env = Environment.from_model(model, data)
    gm = GraspManager(env.model, env.data)

    if robot_type == "ur5e":
        gripper = RobotiqGripper(env.model, env.data, "ur5e", prefix="gripper/", grasp_manager=gm)
        arm = create_ur5e_arm(env, ee_site=UR5E_ROBOTIQ_EE_SITE, gripper=gripper, grasp_manager=gm)
        home = UR5E_HOME
    else:
        gripper = FrankaGripper(env.model, env.data, "franka", grasp_manager=gm)
        arm = create_franka_arm(env, gripper=gripper, grasp_manager=gm)
        gripper.kinematic_open()  # default qpos=0 is fully closed
        # Set actuator ctrl to hold fingers open in physics mode
        if gripper.actuator_id is not None:
            env.data.ctrl[gripper.actuator_id] = gripper.ctrl_open
        home = FRANKA_HOME

    for i, idx in enumerate(arm.joint_qpos_indices):
        env.data.qpos[idx] = home[i]
    mujoco.mj_forward(env.model, env.data)

    return env, arm, home


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

    _add_table_and_cans(spec)
    return _compile_and_create_arm(spec, "ur5e")


def setup_franka():
    """Compose Franka + cans + recycle bin scene."""
    if not FRANKA_SCENE.exists():
        print(f"ERROR: Franka scene not found at {FRANKA_SCENE}")
        sys.exit(1)

    spec = mujoco.MjSpec.from_file(str(FRANKA_SCENE))
    add_franka_ee_site(spec)

    _add_table_and_cans(spec)
    return _compile_and_create_arm(spec, "franka")


# ---------------------------------------------------------------------------
# Pickup / place helpers
# ---------------------------------------------------------------------------


def cartesian_lift(ctx: SimContext, arm: Arm, height: float = 0.05) -> None:
    """Lift the EE straight up using Jacobian-based cartesian control."""
    arm_name = arm.config.name
    ctrl = CartesianController.from_arm(
        arm, step_fn=lambda q, qd: ctx.step_cartesian(arm_name, q, qd),
    )
    ctrl.move(
        np.array([0.0, 0.0, 0.10, 0.0, 0.0, 0.0]),  # 10 cm/s upward
        dt=0.004,
        max_distance=height,
    )
    ctx.sync()


def pickup(
    ctx: SimContext, arm: Arm, grasp_tsrs: list,
    tsr_to_object: list[str],
) -> str | None:
    """Plan to any grasp TSR, execute, close gripper, lift.

    Returns the name of the grasped object, or None on failure.
    """
    logger.info("Planning grasp (%d TSRs)...", len(grasp_tsrs))

    try:
        result = arm.plan_to_tsrs(grasp_tsrs, timeout=10.0, return_details=True)
    except Exception:
        result = None
    if result is None or not result.success:
        logger.warning("Grasp planning failed")
        return None

    # Determine which object the planner chose
    body_name = tsr_to_object[result.goal_index]
    logger.info("Planner chose %s (TSR %d)", body_name, result.goal_index)

    traj = arm.retime(result.path)
    logger.info("Executing grasp (%d waypoints, %.2fs)...", traj.num_waypoints, traj.duration)
    if not ctx.execute(traj):
        logger.warning("Grasp execution failed")
        return None

    logger.info("Closing gripper on %s...", body_name)
    grasped = ctx.arm(arm.config.name).grasp(body_name)
    if not grasped:
        logger.warning("Grasp failed for %s", body_name)
        return None

    cartesian_lift(ctx, arm, height=0.10)
    return grasped


def place(ctx: SimContext, arm: Arm, body_name: str) -> bool:
    """Plan to the place pose above the bin, execute, release."""
    logger.info("Planning placement above recycling bin...")
    place_pose = compute_place_pose()
    try:
        path = arm.plan_to_pose(place_pose, timeout=10.0)
    except Exception:
        path = None
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
        # Re-open Franka gripper — viewer launch resets qpos to model defaults
        if robot_type == "franka" and arm.gripper is not None:
            arm.gripper.kinematic_open()
            if arm.gripper.actuator_id is not None:
                env.data.ctrl[arm.gripper.actuator_id] = arm.gripper.ctrl_open
            ctx.sync()

        arm_name = arm.config.name
        _step_fn = lambda q, qd: ctx.step_cartesian(arm_name, q, qd)

        def recover():
            """Release any held object, retract up, return home."""
            # Release anything held
            ctx.arm(arm_name).release()
            # Cartesian retract upward to clear collisions
            ctrl = CartesianController.from_arm(arm, step_fn=_step_fn)
            ctrl.move(np.array([0., 0., 0.10, 0., 0., 0.]), dt=0.004, max_distance=0.10)
            # Plan home
            try:
                home_path = arm.plan_to_configuration(home, timeout=10.0)
            except Exception:
                home_path = None
            if home_path is not None:
                ctx.execute(arm.retime(home_path))
            ctx.sync()

        remaining = list(CAN_BODY_NAMES[:cycles])

        for cycle in range(1, cycles + 1):
            if not ctx.is_running() or not remaining:
                break

            # Combine TSRs from all remaining cans
            all_tsrs, tsr_to_object = [], []
            for body_name in remaining:
                T_center = env.get_body_pose(body_name)
                tsrs = make_grasp_tsrs(T_center, robot_type)
                for _ in tsrs:
                    tsr_to_object.append(body_name)
                all_tsrs.extend(tsrs)

            print(f"\n--- Cycle {cycle}: {len(remaining)} cans, {len(all_tsrs)} TSRs ---")

            grasped = pickup(ctx, arm, all_tsrs, tsr_to_object)
            if grasped is None:
                print(f"  FAILED to pick up any can")
                recover()
                continue

            print(f"  Picked up {grasped}")

            if not place(ctx, arm, grasped):
                print(f"  FAILED to place {grasped}")
                recover()
                continue

            print(f"  Dropped {grasped} into bin")

            if not physics:
                env.hide_freebody(grasped)
            remaining.remove(grasped)

            try:
                home_path = arm.plan_to_configuration(home, timeout=10.0)
            except Exception:
                home_path = None
            if home_path is not None:
                ctx.execute(arm.retime(home_path))
                print("  Returned to home")
            else:
                print("  Could not plan home — recovering")
                recover()

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
