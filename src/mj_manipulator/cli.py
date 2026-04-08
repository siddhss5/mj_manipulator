# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""CLI entry point for mj_manipulator interactive console.

Launches an IPython console with a single-arm robot (UR5e or Franka),
physics simulation, and viser web viewer with teleop.

Usage::

    python -m mj_manipulator                        # UR5e + viser (default)
    python -m mj_manipulator --robot franka          # Franka + viser
    python -m mj_manipulator --physics               # with physics simulation
    python -m mj_manipulator --objects '{"can": 3}'  # spawn objects
    python -m mj_manipulator --no-viser              # headless, no viewer
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mj_manipulator",
        description="Interactive manipulation console",
    )
    parser.add_argument(
        "--robot",
        choices=["ur5e", "franka"],
        default="ur5e",
        help="Robot type (default: ur5e)",
    )
    parser.add_argument("--physics", action="store_true", help="Enable physics simulation")
    parser.add_argument("--no-viser", action="store_true", help="Disable viser web viewer")
    parser.add_argument("--objects", type=str, default=None, help="Objects JSON, e.g. '{\"can\": 3}'")
    args = parser.parse_args()

    objects = {}
    if args.objects:
        import json

        objects = json.loads(args.objects)

    print(f"\nLoading {args.robot}...", flush=True)

    robot = _setup_robot(args.robot, objects)

    from mj_manipulator.console import start_console

    start_console(
        robot,
        physics=args.physics,
        viser=not args.no_viser,
        robot_name=args.robot.upper(),
    )


def _setup_robot(robot_type: str, objects: dict):
    """Create a single-arm robot with optional objects."""

    from mj_manipulator.menagerie import menagerie_scene

    if robot_type == "ur5e":
        scene_path = menagerie_scene("universal_robots_ur5e")
        if not scene_path.exists():
            print(f"ERROR: UR5e scene not found at {scene_path}")
            print("Run ./setup.sh from the robot-code workspace to clone mujoco_menagerie.")
            sys.exit(1)
        robot = _setup_ur5e(scene_path, objects)
    elif robot_type == "franka":
        scene_path = menagerie_scene("franka_emika_panda")
        if not scene_path.exists():
            print(f"ERROR: Franka scene not found at {scene_path}")
            print("Run ./setup.sh from the robot-code workspace to clone mujoco_menagerie.")
            sys.exit(1)
        robot = _setup_franka(scene_path, objects)
    else:
        print(f"Unknown robot: {robot_type}")
        sys.exit(1)

    return robot


class _SimpleRobot:
    """Minimal ManipulationRobot for single-arm use with the generic console.

    This is the reference implementation showing how little code is needed
    to bring up a new robot with the full console experience.
    """

    def __init__(self, env, arm, home_config, grasp_source=None):
        import threading

        self.model = env.model
        self.data = env.data
        self._env = env
        self._arm = arm
        self.arms = {arm.config.name: arm}
        from mj_manipulator.grasp_manager import GraspManager as _GM

        self.grasp_manager = arm.grasp_manager or _GM(self.model, self.data)
        self.named_poses = {"ready": {arm.config.name: list(home_config)}}
        self._grasp_source = grasp_source
        self._context = None
        self._abort_event = threading.Event()

    @property
    def grasp_source(self):
        if self._grasp_source is None:
            self._grasp_source = _NullGraspSource()
        return self._grasp_source

    @property
    def _active_context(self):
        return self._context

    @_active_context.setter
    def _active_context(self, ctx):
        self._context = ctx

    def sim(self, *, physics=True, headless=False, viewer=None, event_loop=None):
        from mj_manipulator.sim_context import SimContext

        inner = SimContext(
            self.model,
            self.data,
            self.arms,
            physics=physics,
            headless=headless,
            viewer=viewer,
            event_loop=event_loop,
            abort_fn=self.is_abort_requested,
        )
        return _SimContextWrapper(inner, self)

    def request_abort(self):
        if self._context is not None and hasattr(self._context, "ownership") and self._context.ownership is not None:
            self._context.ownership.abort_all()
        self._abort_event.set()

    def clear_abort(self):
        if self._context is not None and hasattr(self._context, "ownership") and self._context.ownership is not None:
            self._context.ownership.clear_all()
        self._abort_event.clear()

    def is_abort_requested(self):
        return self._abort_event.is_set()


class _SimContextWrapper:
    """Sets robot._active_context on enter/exit."""

    def __init__(self, inner, robot):
        self._inner = inner
        self._robot = robot

    def __enter__(self):
        ctx = self._inner.__enter__()
        self._robot._active_context = ctx
        return ctx

    def __exit__(self, *args):
        self._robot._active_context = None
        return self._inner.__exit__(*args)


class _NullGraspSource:
    """GraspSource that returns empty results (no objects to grasp)."""

    def get_grasps(self, object_name, hand_type):
        return []

    def get_placements(self, destination, object_name):
        return []

    def get_graspable_objects(self):
        return []

    def get_place_destinations(self, object_name):
        return []


def _setup_ur5e(scene_path, objects):
    """Set up UR5e + Robotiq."""
    import geodude_assets
    import mujoco
    from mj_environment import Environment

    from mj_manipulator.arms.ur5e import UR5E_HOME, UR5E_ROBOTIQ_EE_SITE, create_ur5e_arm
    from mj_manipulator.grasp_manager import GraspManager
    from mj_manipulator.grippers.robotiq import RobotiqGripper

    spec = mujoco.MjSpec.from_file(str(scene_path))

    # Attach Robotiq gripper
    robotiq_path = geodude_assets.MODELS_DIR / "robotiq_2f140" / "2f140.xml"
    robotiq_spec = mujoco.MjSpec.from_file(str(robotiq_path))
    wrist = spec.worldbody.find_child("wrist_3_link")
    frame = wrist.add_frame()
    frame.pos = [0, 0.1, 0]
    frame.quat = [-1, 1, 0, 0]
    frame.attach_body(robotiq_spec.worldbody.first_body(), prefix="gripper/")

    # Add objects if specified
    if objects:
        _attach_prl_objects(spec, objects)

    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    env = Environment.from_model(model, data)
    gm = GraspManager(env.model, env.data)

    gripper = RobotiqGripper(env.model, env.data, "ur5e", prefix="gripper/", grasp_manager=gm)
    arm = create_ur5e_arm(env, ee_site=UR5E_ROBOTIQ_EE_SITE, gripper=gripper, grasp_manager=gm)

    for i, idx in enumerate(arm.joint_qpos_indices):
        env.data.qpos[idx] = UR5E_HOME[i]
    mujoco.mj_forward(env.model, env.data)

    return _SimpleRobot(env, arm, UR5E_HOME)


def _setup_franka(scene_path, objects):
    """Set up Franka Panda."""
    import mujoco
    from mj_environment import Environment

    from mj_manipulator.arms.franka import FRANKA_HOME, add_franka_ee_site, create_franka_arm
    from mj_manipulator.grasp_manager import GraspManager
    from mj_manipulator.grippers.franka import FrankaGripper

    spec = mujoco.MjSpec.from_file(str(scene_path))
    add_franka_ee_site(spec)

    if objects:
        _attach_prl_objects(spec, objects)

    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    env = Environment.from_model(model, data)
    gm = GraspManager(env.model, env.data)

    gripper = FrankaGripper(env.model, env.data, "franka", grasp_manager=gm)
    arm = create_franka_arm(env, gripper=gripper, grasp_manager=gm)

    gripper.kinematic_open()
    if gripper.actuator_id is not None:
        env.data.ctrl[gripper.actuator_id] = gripper.ctrl_open

    for i, idx in enumerate(arm.joint_qpos_indices):
        env.data.qpos[idx] = FRANKA_HOME[i]
    mujoco.mj_forward(env.model, env.data)

    return _SimpleRobot(env, arm, FRANKA_HOME)


def _attach_prl_objects(spec, objects: dict):
    """Attach prl_assets objects to the scene."""
    import mujoco
    from prl_assets import OBJECTS_DIR

    # Simple table
    table = spec.worldbody.add_body()
    table.name = "table"
    table.pos = [0.45, -0.20, 0.23]
    g = table.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [0.15, 0.15, 0.23]
    g.rgba = [0.4, 0.3, 0.2, 1.0]

    idx = 0
    for obj_type, count in objects.items():
        from asset_manager import AssetManager

        assets = AssetManager(str(OBJECTS_DIR))
        try:
            xml_path = assets.get_path(obj_type, "mujoco")
        except (KeyError, TypeError):
            continue
        for i in range(count):
            obj_spec = mujoco.MjSpec.from_file(xml_path)
            f = spec.worldbody.add_frame()
            # Spread objects on the table
            x = 0.40 + (idx % 3) * 0.05
            y = -0.25 + (idx // 3) * 0.05
            f.pos = [x, y, 0.46 + 0.05]
            f.attach_body(obj_spec.worldbody.first_body(), prefix=f"{obj_type}_{i}/")
            idx += 1
