#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Debug: place Franka at a TSR grasp pose and hold for inspection.

Usage: uv run mjpython demos/_debug_franka_grasp.py
"""

import mujoco
import numpy as np
from asset_manager import AssetManager
from mj_environment import Environment
from prl_assets import OBJECTS_DIR
from tsr.hands import FrankaHand

from mj_manipulator.arms.franka import (
    FRANKA_HOME,
    add_franka_ee_site,
    add_franka_gravcomp,
    create_franka_arm,
)
from mj_manipulator.grasp_manager import GraspManager
from mj_manipulator.grippers.franka import FrankaGripper
from mj_manipulator.sim_context import SimContext

assets = AssetManager(str(OBJECTS_DIR))
can_gp = assets.get("can")["geometric_properties"]
hand = FrankaHand()

# Build scene
spec = mujoco.MjSpec.from_file(
    str(__import__("mj_manipulator.menagerie", fromlist=["menagerie_scene"]).menagerie_scene("franka_emika_panda"))
)
add_franka_ee_site(spec)
add_franka_gravcomp(spec)

table = spec.worldbody.add_body()
table.name = "table"
table.pos = [0.45, -0.2, 0.23]
g = table.add_geom()
g.type = mujoco.mjtGeom.mjGEOM_BOX
g.size = [0.15, 0.15, 0.23]

can_spec = mujoco.MjSpec.from_file(str(assets.get_path("can", "mujoco")))
f = spec.worldbody.add_frame()
f.pos = [0.45, -0.2, 0.46]
f.attach_body(can_spec.worldbody.first_body(), prefix="can_0/")

model = spec.compile()
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
env = Environment.from_model(model, data)

gm = GraspManager(env.model, env.data)
gripper = FrankaGripper(env.model, env.data, "franka", grasp_manager=gm)
arm = create_franka_arm(env, gripper=gripper, grasp_manager=gm)

for i, idx in enumerate(arm.joint_qpos_indices):
    env.data.qpos[idx] = FRANKA_HOME[i]
gripper.kinematic_open()
mujoco.mj_forward(env.model, env.data)

# Generate a deep side grasp TSR and plan to it
T_center = env.get_body_pose("can_0/can")
T_bottom = T_center.copy()
T_bottom[2, 3] -= can_gp["height"] / 2
templates = hand.grasp_cylinder_side(can_gp["radius"], can_gp["height"])
grasp_tsrs = [t.instantiate(T_bottom) for t in templates]

print(f"Can at: {T_center[:3, 3].round(3)}")
print(f"{len(grasp_tsrs)} grasp TSRs")
print(f"finger_length: {hand.finger_length}")

path = arm.plan_to_tsrs(grasp_tsrs, timeout=15.0)
if path is None:
    print("Planning failed")
    exit(1)

traj = arm.retime(path)
print(f"Plan found: {traj.num_waypoints} waypoints")


def finger_pos():
    positions = []
    for jname in ["finger_joint1", "finger_joint2"]:
        jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            positions.append(env.data.qpos[env.model.jnt_qposadr[jid]])
    return positions


print(f"\nBefore SimContext: fingers={finger_pos()}")

with SimContext(env.model, env.data, {"franka": arm}, physics=False, headless=False) as ctx:
    print(f"After SimContext enter: fingers={finger_pos()}")
    # Re-open fingers — viewer launch may have reset qpos to model defaults
    gripper.kinematic_open()
    print(f"After re-open in ctx: fingers={finger_pos()}")
    print(f"Arm joint_qpos_indices: {list(arm.joint_qpos_indices)}")
    print(f"Trajectory DOF: {traj.dof}")
    print(f"Trajectory entity: {traj.entity}")

    # Check finger joint qpos addresses
    for jname in ["finger_joint1", "finger_joint2"]:
        jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        print(f"  {jname}: qpos_adr={env.model.jnt_qposadr[jid]}")

    # Execute first waypoint manually to see if fingers change
    wp = traj.positions[0]
    print(f"First waypoint ({len(wp)} values): {wp.round(3)}")
    executor = ctx._executors.get("franka")
    print(f"Executor joint_qpos_indices: {list(executor.joint_qpos_indices)}")

    ctx.execute(traj)
    print(f"After execute: fingers={finger_pos()}")

    ctx.sync()

    ee = arm.get_ee_pose()
    print(f"\nEE pos: {ee[:3, 3].round(4)}")
    print(f"EE z (approach): {ee[:3, 2].round(3)}")
    print(
        f"Distance EE to can axis: {np.sqrt((ee[0, 3] - T_center[0, 3]) ** 2 + (ee[1, 3] - T_center[1, 3]) ** 2):.4f}"
    )
    print(f"Fingers: {finger_pos()}")

    # Re-open fingers to see where they should be
    gripper.kinematic_open()
    ctx.sync()
    print(f"After re-open: fingers={finger_pos()}")

    print("\nViewer open — inspect. Close viewer to exit.")
    while ctx.is_running():
        ctx.sync()
