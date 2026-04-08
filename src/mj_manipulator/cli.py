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
        extra_ns={"commands": commands},
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

    from mj_manipulator.arms.franka import FRANKA_HOME, add_franka_ee_site, create_franka_arm
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

    if objects:
        from prl_assets import OBJECTS_DIR

        env = Environment.from_spec(spec, objects_dir=str(OBJECTS_DIR), scene_config=objects)
    else:
        model = spec.compile()
        env = Environment.from_model(model)

    gm = GraspManager(env.model, env.data)
    gripper = FrankaGripper(env.model, env.data, "franka", grasp_manager=gm)
    arm = create_franka_arm(env, gripper=gripper, grasp_manager=gm)

    gripper.kinematic_open()
    if gripper.actuator_id is not None:
        env.data.ctrl[gripper.actuator_id] = gripper.ctrl_open

    for i, idx in enumerate(arm.joint_qpos_indices):
        env.data.qpos[idx] = FRANKA_HOME[i]
    mujoco.mj_forward(env.model, env.data)

    return _SimpleRobot(env, arm, FRANKA_HOME, has_objects=bool(objects))
