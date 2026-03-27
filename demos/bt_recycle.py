#!/usr/bin/env python3
"""Recycling demo using behavior trees.

Same task as recycling.py (pick up cans, drop in bin), but orchestrated
via py_trees behavior trees instead of sequential function calls.

Usage:
    uv run mjpython demos/bt_recycle.py --robot ur5e
    uv run mjpython demos/bt_recycle.py --robot ur5e --physics
    uv run mjpython demos/bt_recycle.py --robot ur5e --headless --cycles 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import mujoco
import numpy as np
import py_trees
from asset_manager import AssetManager
from mj_environment import Environment
from py_trees.common import Access, Status
from prl_assets import OBJECTS_DIR
from tsr import TSR
from tsr.hands import FrankaHand, Robotiq2F140
from tsr.placement import StablePlacer

from mj_manipulator.arms.franka import (
    FRANKA_HOME,
    add_franka_ee_site,
    create_franka_arm,
)
from mj_manipulator.arms.ur5e import UR5E_HOME, UR5E_ROBOTIQ_EE_SITE, create_ur5e_arm
from mj_manipulator.bt import pickup_with_recovery, place_with_recovery
from mj_manipulator.config import PhysicsConfig, PhysicsExecutionConfig
from mj_manipulator.grasp_manager import GraspManager
from mj_manipulator.grippers.franka import FrankaGripper
from mj_manipulator.grippers.robotiq import RobotiqGripper
from mj_manipulator.menagerie import menagerie_scene
from mj_manipulator.sim_context import SimContext

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logging.getLogger("toppra").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scene + geometry (same as recycling.py)
# ---------------------------------------------------------------------------

import geodude_assets

UR5E_SCENE = menagerie_scene("universal_robots_ur5e")
FRANKA_SCENE = menagerie_scene("franka_emika_panda")
ROBOTIQ_MODEL = geodude_assets.MODELS_DIR / "robotiq_2f140" / "2f140.xml"

_ASSETS = AssetManager(str(OBJECTS_DIR))
_CAN_GP = _ASSETS.get("can")["geometric_properties"]
_CAN_XML = _ASSETS.get_path("can", "mujoco")
_BIN_XML = _ASSETS.get_path("yellow_tote", "mujoco")
_BIN_HEIGHT = _ASSETS.get("yellow_tote")["geometric_properties"]["outer_size"][2]

_ROBOTIQ = Robotiq2F140()
_FRANKA = FrankaHand()

CAN_BODY_NAMES = [f"can_{i}/can" for i in range(3)]
BIN_POS = np.array([0.25, -0.70, 0.0])


def _add_table(spec, pos, size):
    table = spec.worldbody.add_body()
    table.name = "table"
    table.pos = pos
    g = table.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = size
    g.rgba = [0.4, 0.3, 0.2, 1.0]


def setup_scene(robot_type):
    """Build scene with robot + table + cans + bin."""
    if robot_type == "ur5e":
        spec = mujoco.MjSpec.from_file(str(UR5E_SCENE))
        robotiq_spec = mujoco.MjSpec.from_file(str(ROBOTIQ_MODEL))
        wrist = spec.worldbody.find_child("wrist_3_link")
        frame = wrist.add_frame()
        frame.pos = [0, 0.1, 0]
        frame.quat = [-1, 1, 0, 0]
        frame.attach_body(robotiq_spec.worldbody.first_body(), prefix="gripper/")
    else:
        spec = mujoco.MjSpec.from_file(str(FRANKA_SCENE))
        add_franka_ee_site(spec)

    # Table + cans
    table_half = [0.15, 0.15, 0.23]
    table_center = [0.45, -0.20, table_half[2]]
    _add_table(spec, pos=table_center, size=table_half)

    table_surface = np.eye(4)
    table_surface[:3, 3] = [table_center[0], table_center[1], table_half[2] * 2]
    placer = StablePlacer(table_half[0], table_half[1])
    place_templates = placer.place_cylinder(_CAN_GP["radius"], _CAN_GP["height"])
    can_body_offset_z = _CAN_GP["height"] / 2
    min_sep = _CAN_GP["radius"] * 3

    can_positions = []
    for _ in range(3):
        tsr = place_templates[0].instantiate(table_surface)
        for _ in range(50):
            pose = tsr.sample()
            pos = pose[:3, 3]
            if all(np.linalg.norm(pos[:2] - np.array(p[:2])) > min_sep for p in can_positions):
                break
        result = list(pos)
        result[2] -= can_body_offset_z
        can_positions.append(result)

    for i, pos in enumerate(can_positions):
        can_spec = mujoco.MjSpec.from_file(_CAN_XML)
        f = spec.worldbody.add_frame()
        f.pos = pos
        f.attach_body(can_spec.worldbody.first_body(), prefix=f"can_{i}/")

    bin_spec = mujoco.MjSpec.from_file(_BIN_XML)
    f = spec.worldbody.add_frame()
    f.pos = list(BIN_POS)
    f.attach_body(bin_spec.worldbody.first_body(), prefix="bin/")

    # Compile + create arm
    model = spec.compile()
    if robot_type == "franka":
        # Fix grip force
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
        if aid >= 0:
            length_grasp = 0.066
            length_open = 0.08
            new_bias1 = -140.0 / length_grasp
            old_bias1 = model.actuator_biasprm[aid, 1]
            scale = new_bias1 / old_bias1 if old_bias1 != 0 else 1.0
            model.actuator_biasprm[aid, 1] = new_bias1
            model.actuator_biasprm[aid, 2] *= scale
            model.actuator_gainprm[aid, 0] = abs(new_bias1) * length_open / 255.0

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
        gripper.kinematic_open()
        home = FRANKA_HOME

    for i, idx in enumerate(arm.joint_qpos_indices):
        env.data.qpos[idx] = home[i]
    mujoco.mj_forward(env.model, env.data)

    return env, arm, home


def make_grasp_tsrs(T_center, robot_type):
    hand = _FRANKA if robot_type == "franka" else _ROBOTIQ
    T_bottom = T_center.copy()
    T_bottom[2, 3] -= _CAN_GP["height"] / 2
    templates = hand.grasp_cylinder_side(_CAN_GP["radius"], _CAN_GP["height"])
    return [t.instantiate(T_bottom) for t in templates]


def make_place_tsrs():
    place_pose = np.array([
        [1, 0, 0, BIN_POS[0]],
        [0, -1, 0, BIN_POS[1]],
        [0, 0, -1, BIN_POS[2] + _BIN_HEIGHT + 0.15],
        [0, 0, 0, 1],
    ], dtype=float)
    return [TSR(T0_w=place_pose)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(robot_type, *, physics=False, headless=False, cycles=3):
    print(f"\n{'='*60}")
    print(f"  BT Recycling Demo — {robot_type.upper()}")
    print(f"  Mode: {'Physics' if physics else 'Kinematic'}")
    print(f"{'='*60}")

    env, arm, home = setup_scene(robot_type)
    ns = f"/{arm.config.name}"

    # Build tree
    root = py_trees.composites.Sequence(
        name="recycle_cycle", memory=True,
        children=[pickup_with_recovery(ns), place_with_recovery(ns)],
    )
    tree = py_trees.trees.BehaviourTree(root=root)

    physics_config = PhysicsConfig(
        execution=PhysicsExecutionConfig(
            control_dt=0.008, position_tolerance=0.15,
            velocity_tolerance=0.5, convergence_timeout_steps=5000,
        ),
    ) if physics else None

    with SimContext(
        env.model, env.data, {arm.config.name: arm},
        physics=physics, headless=headless, physics_config=physics_config,
    ) as ctx:
        if robot_type == "franka" and arm.gripper is not None:
            arm.gripper.kinematic_open()
            if arm.gripper.actuator_id is not None:
                env.data.ctrl[arm.gripper.actuator_id] = arm.gripper.ctrl_open
            ctx.sync()

        # Set up blackboard
        bb = py_trees.blackboard.Client(name="demo")
        for key in ["/context", f"{ns}/arm", f"{ns}/arm_name",
                     f"{ns}/grasp_tsrs", f"{ns}/place_tsrs", f"{ns}/grasped",
                     f"{ns}/goal_tsr_index", f"{ns}/tsr_to_object",
                     f"{ns}/timeout", f"{ns}/object_name", f"{ns}/goal_config"]:
            bb.register_key(key=key, access=Access.WRITE)

        bb.set("/context", ctx)
        bb.set(f"{ns}/arm", arm)
        bb.set(f"{ns}/arm_name", arm.config.name)
        bb.set(f"{ns}/timeout", 10.0)
        bb.set(f"{ns}/goal_config", home)

        # Set place TSRs once (same for all cycles)
        bb.set(f"{ns}/place_tsrs", make_place_tsrs())

        # Track which TSRs belong to which can (for goal_tsr_index lookup)
        remaining_cans = list(CAN_BODY_NAMES[:cycles])

        for cycle in range(1, cycles + 1):
            if not ctx.is_running() or not remaining_cans:
                break

            print(f"\n--- Cycle {cycle}: {len(remaining_cans)} cans remaining ---")

            # Combine grasp TSRs from ALL remaining cans
            all_grasp_tsrs = []
            tsr_to_can = []  # maps TSR index → can body name
            for body_name in remaining_cans:
                T_center = env.get_body_pose(body_name)
                tsrs = make_grasp_tsrs(T_center, robot_type)
                for _ in tsrs:
                    tsr_to_can.append(body_name)
                all_grasp_tsrs.extend(tsrs)

            print(f"  {len(all_grasp_tsrs)} TSRs from {len(remaining_cans)} cans")

            bb.set(f"{ns}/grasp_tsrs", all_grasp_tsrs)
            bb.set(f"{ns}/tsr_to_object", tsr_to_can)
            bb.set(f"{ns}/object_name", remaining_cans[0])  # fallback

            # Reset tree and tick
            for node in root.iterate():
                node.status = Status.INVALID
            tree.tick()

            if root.status == Status.SUCCESS:
                # The Grasp node resolved the object from tsr_to_object
                # and wrote it to object_name and grasped
                grasped_can = bb.get(f"{ns}/grasped")
                if grasped_can is None:
                    grasped_can = remaining_cans[0]
                print(f"  Picked up and placed {grasped_can}")

                if not physics:
                    env.hide_freebody(grasped_can)
                remaining_cans.remove(grasped_can)

                # Return home
                try:
                    home_path = arm.plan_to_configuration(home, timeout=10.0)
                except Exception:
                    home_path = None
                if home_path is not None:
                    ctx.execute(arm.retime(home_path))
            else:
                print(f"  FAILED (recovery attempted)")

            ctx.sync()

        print(f"\nCompleted {min(cycles, len(CAN_BODY_NAMES))} cycles.")


def main():
    parser = argparse.ArgumentParser(description="BT Recycling demo")
    parser.add_argument("--robot", choices=["ur5e", "franka", "both"], default="ur5e")
    parser.add_argument("--physics", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--cycles", type=int, default=3)
    args = parser.parse_args()

    robots = ["ur5e", "franka"] if args.robot == "both" else [args.robot]
    for robot in robots:
        run(robot, physics=args.physics, headless=args.headless, cycles=args.cycles)

    print(f"\n{'='*60}")
    print("  DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
