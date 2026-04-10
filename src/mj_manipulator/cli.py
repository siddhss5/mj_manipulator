# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""CLI entry point for mj_manipulator interactive console.

Launches an IPython console with a Franka Panda, physics simulation,
and viser web viewer with teleop. Objects from prl_assets can be
spawned into the scene.

Usage::

    python -m mj_manipulator                          # Franka + viser
    python -m mj_manipulator --physics                # with physics sim
    python -m mj_manipulator --objects '{"can": 3}'   # spawn objects
    python -m mj_manipulator --no-viser               # headless
"""

from __future__ import annotations

import argparse
import sys

from mj_manipulator.robot import RobotBase


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mj_manipulator",
        description="Interactive manipulation console",
    )
    parser.add_argument("--physics", action="store_true", help="Enable physics simulation")
    parser.add_argument("--no-viser", action="store_true", help="Disable viser web viewer")
    parser.add_argument("--objects", type=str, default=None, help="Objects JSON, e.g. '{\"can\": 3}'")
    args = parser.parse_args()

    objects = {}
    if args.objects:
        import json

        objects = json.loads(args.objects)

    print("\nLoading Franka...", flush=True)

    robot = _setup_franka(objects)

    def commands():
        """Print available commands."""
        print(
            """
Quick Reference
===============

Scene:
  robot.find_objects()              — list all graspable objects
  robot.get_object_pose("can_0")    — 4x4 pose matrix
  robot.holding()                   — (arm, name) or None

Actions:
  robot.pickup()                    — pick up nearest reachable object
  robot.pickup("can_0")             — pick up specific object
  robot.place("yellow_tote")        — place in recycling bin
  robot.place("worktop")            — place on table surface
  robot.go_home()                   — return arm to ready

Arm:
  robot.arms["franka"].get_ee_pose()
  robot.arms["franka"].get_joint_positions()

Scene:
  reset()                           — re-scatter objects, arm to ready

Teleop:
  Click "Activate" in the viser viewer

IPython:
  robot.<tab>             — tab completion
  ?robot.pickup           — docstring
  commands()              — this help
"""
        )

    from mj_manipulator.console import start_console

    start_console(
        robot,
        physics=args.physics,
        viser=not args.no_viser,
        robot_name="Franka",
        extra_ns={"commands": commands, "reset": robot.reset},
    )


class _SimpleRobot(RobotBase):
    """Minimal robot for the CLI demo.

    Inherits all convenience methods from RobotBase:
    pickup, place, go_home, find_objects, holding, get_object_pose.
    """

    def __init__(self, env, arm, home_config, has_objects=False):
        from mj_manipulator.grasp_manager import GraspManager as _GM

        gm = arm.grasp_manager or _GM(env.model, env.data)
        super().__init__(
            model=env.model,
            data=env.data,
            arms={arm.config.name: arm},
            grasp_manager=gm,
            named_poses={"ready": {arm.config.name: list(home_config)}},
        )
        self._env = env
        self._has_objects = has_objects
        self._objects_config = None  # set externally after creation

    def reset(self):
        """Reset scene to initial state, then re-scatter objects.

        Same pattern as geodude: mj_resetData → hold controller → release
        grasps → hide objects → re-scatter.
        """
        import mujoco

        self.request_abort()
        self.clear_abort()

        # Reset MuJoCo state (qpos, qvel, ctrl all to defaults)
        mujoco.mj_resetData(self.model, self.data)

        # Restore arm to home (model defaults may not be the right config)
        ready = self.named_poses.get("ready", {})
        for arm_name, arm in self.arms.items():
            if arm_name in ready:
                for i, idx in enumerate(arm.joint_qpos_indices):
                    self.data.qpos[idx] = ready[arm_name][i]
            if arm.gripper and arm.gripper.actuator_id is not None:
                self.data.ctrl[arm.gripper.actuator_id] = arm.gripper.ctrl_open

        # Sync controller targets to new positions
        if self._context is not None:
            self._context.hold()

        # Release grasps
        for arm_name in list(self.arms.keys()):
            for obj in list(self.grasp_manager.get_grasped_by(arm_name)):
                self.grasp_manager.mark_released(obj)
                self.grasp_manager.detach_object(obj)

        # Hide all objects, then re-scatter
        if self._env.registry is not None:
            for obj_type in list(self._env.registry.objects.keys()):
                for name in list(self._env.registry.objects[obj_type]["instances"]):
                    if self._env.registry.is_active(name):
                        self._env.registry.hide(name)

        mujoco.mj_forward(self.model, self.data)

        if self._objects_config and self._env.registry is not None:
            all_objects = dict(self._objects_config)
            all_objects["yellow_tote"] = 1
            _scatter_objects(self._env, all_objects)

        if self._context is not None:
            self._context.sync()

        print("Scene reset.")

    @property
    def grasp_source(self):
        if self._grasp_source is None:
            if self._has_objects:
                from mj_manipulator.grasp_sources.prl_assets import PrlAssetsGraspSource

                registry = getattr(self._env, "registry", None)
                self._grasp_source = PrlAssetsGraspSource(
                    self.model,
                    self.data,
                    self.grasp_manager,
                    self.arms,
                    registry=registry,
                )
        return super().grasp_source


def _setup_franka(objects):
    """Set up Franka Panda with optional prl_assets objects."""
    import mujoco
    from mj_environment import Environment

    from mj_manipulator.arms.franka import (
        FRANKA_HOME,
        add_franka_ee_site,
        add_franka_gravcomp,
        create_franka_arm,
        fix_franka_grip_force,
    )
    from mj_manipulator.grasp_manager import GraspManager
    from mj_manipulator.grippers.franka import FrankaGripper
    from mj_manipulator.menagerie import menagerie_scene

    scene_path = menagerie_scene("franka_emika_panda")
    if not scene_path.exists():
        print(f"ERROR: Franka scene not found at {scene_path}")
        print("Run ./setup.sh from the robot-code workspace to clone mujoco_menagerie.")
        sys.exit(1)

    # Assemble robot via MjSpec, then use Environment.from_spec()
    # which handles objects + registry with native path resolution.
    spec = mujoco.MjSpec.from_file(str(scene_path))
    add_franka_ee_site(spec)
    add_franka_gravcomp(spec)

    if objects:
        from prl_assets import OBJECTS_DIR

        # Flat plate on the ground in front of the robot for objects
        plate = spec.worldbody.add_body()
        plate.name = "plate"
        plate.pos = [0.5, 0.0, 0.005]
        g = plate.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [0.30, 0.30, 0.005]
        g.rgba = [0.6, 0.6, 0.6, 1.0]
        # Worktop site on plate surface — enables robot.place("worktop")
        s = plate.add_site()
        s.name = "worktop"
        s.pos = [0, 0, 0.005]
        s.size = [0.25, 0.25, 0.001]
        s.type = mujoco.mjtGeom.mjGEOM_BOX
        s.rgba = [0, 0, 0, 0]

        # Add yellow_tote to the scene config for recycling
        scene_config = dict(objects)
        scene_config["yellow_tote"] = 1

        env = Environment.from_spec(spec, objects_dir=str(OBJECTS_DIR), scene_config=scene_config)
    else:
        model = spec.compile()
        env = Environment.from_model(model)

    # Fix menagerie Franka's under-powered grip (140N target from datasheet)
    fix_franka_grip_force(env.model)

    gm = GraspManager(env.model, env.data)
    gripper = FrankaGripper(env.model, env.data, "franka", grasp_manager=gm)
    arm = create_franka_arm(env, gripper=gripper, grasp_manager=gm)

    gripper.kinematic_open()
    if gripper.actuator_id is not None:
        env.data.ctrl[gripper.actuator_id] = gripper.ctrl_open

    for i, idx in enumerate(arm.joint_qpos_indices):
        env.data.qpos[idx] = FRANKA_HOME[i]
    mujoco.mj_forward(env.model, env.data)

    # Activate and scatter objects (include tote from scene_config)
    if objects and env.registry is not None:
        all_objects = dict(objects)
        all_objects["yellow_tote"] = 1
        _scatter_objects(env, all_objects)

    robot = _SimpleRobot(env, arm, FRANKA_HOME, has_objects=bool(objects))
    robot._objects_config = objects if objects else None
    return robot


def _scatter_objects(env, objects: dict):
    """Activate and scatter objects on the plate, position tote to the side."""
    import mujoco
    import numpy as np
    from asset_manager import AssetManager
    from prl_assets import OBJECTS_DIR
    from tsr.placement import StablePlacer

    assets = AssetManager(str(OBJECTS_DIR))

    # Plate surface: ground level in front of robot
    plate_surface = np.eye(4)
    plate_surface[:3, 3] = [0.5, 0.0, 0.01]
    placer = StablePlacer(0.25, 0.25)

    placed_positions = []
    for obj_type, count in objects.items():
        if isinstance(count, int):
            pass
        elif isinstance(count, dict):
            count = count.get("count", 1)
        else:
            continue

        # Position tote to the side, not on the plate
        try:
            gp = assets.get(obj_type)["geometric_properties"]
        except (KeyError, TypeError):
            continue

        geo = gp.get("type")
        if geo in ("open_box", "tote"):
            # Place container to the side
            env.registry.activate(obj_type, pos=[-0.5, 0.0, 0.0])
            mujoco.mj_forward(env.model, env.data)
            continue

        if geo == "cylinder":
            templates = placer.place_cylinder(gp["radius"], gp["height"])
        elif geo == "box":
            templates = placer.place_box(gp["size"][0], gp["size"][1], gp["size"][2])
        else:
            continue

        if not templates:
            continue

        for _ in range(count):
            tsr = templates[0].instantiate(plate_surface)
            for _attempt in range(50):
                T = tsr.sample()
                pos = T[:3, 3]
                if all(np.linalg.norm(pos[:2] - np.array(p[:2])) > 0.06 for p in placed_positions):
                    break

            name = env.registry.activate(obj_type, pos=list(pos))
            placed_positions.append(list(pos))

            body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
            jnt_id = env.model.body_jntadr[body_id]
            qpos_adr = env.model.jnt_qposadr[jnt_id]
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, T[:3, :3].flatten())
            env.data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat

    mujoco.mj_forward(env.model, env.data)
