# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""KUKA iiwa 14 + Robotiq 2F-85 assembly for mj_manipulator scenarios.

The iiwa ships as a bare arm in mujoco_menagerie (no gripper). This
module shows the "attach your own gripper" workflow via MjSpec:

    1. Load the iiwa 14 scene
    2. Add an EE site + gravcomp
    3. Load the Robotiq 2F-85 scene (from menagerie)
    4. Add a worktop plate (if the scenario has objects)
    5. MjSpec.attach the 2F-85 at the iiwa's grasp_site
    6. Build the Environment
    7. Wire RobotiqGripper + arm + RobotBase

Compare with :mod:`franka_setup` — Franka ships with a hand, so that
demo skips the gripper-attach step. The rest (RobotBase subclass with
``get_worktop_pose`` and ``reset``) is identical.

This file is skipped by scenario discovery — no ``scene = {...}``
at module level.
"""

from __future__ import annotations

import sys

import mujoco
import numpy as np

from mj_manipulator.robot import RobotBase
from mj_manipulator.scenarios import WorktopPose


def build_iiwa14_robot(scene: dict) -> "IIWA14DemoRobot":
    """Assemble a KUKA iiwa 14 with a Robotiq 2F-85 and apply a scene.

    Args:
        scene: Scenario scene dict with ``objects``, ``fixtures``,
            optional ``spawn_count``.

    Returns:
        A ready-to-use robot with the scene applied.
    """
    from mj_environment import Environment

    from mj_manipulator.arms.iiwa14 import (
        IIWA14_HOME,
        add_iiwa14_ee_site,
        add_iiwa14_gravcomp,
        create_iiwa14_arm,
    )
    from mj_manipulator.grasp_manager import GraspManager
    from mj_manipulator.grippers.robotiq import RobotiqGripper
    from mj_manipulator.menagerie import find_menagerie, menagerie_scene

    # 1. Load the iiwa scene from menagerie and add an EE site on link7.
    scene_path = menagerie_scene("kuka_iiwa_14")
    if not scene_path.exists():
        print(f"ERROR: iiwa 14 scene not found at {scene_path}")
        print("Run ./setup.sh from the workspace root to clone mujoco_menagerie.")
        sys.exit(1)

    spec = mujoco.MjSpec.from_file(str(scene_path))
    add_iiwa14_ee_site(spec)  # adds 'grasp_site' on link7
    add_iiwa14_gravcomp(spec)

    # 2. Add a worktop plate if the scenario needs objects.
    has_objects = bool(scene.get("objects") or scene.get("fixtures"))
    if has_objects:
        _add_iiwa14_worktop(spec)

    # 3. Attach the Robotiq 2F-85 at the grasp_site.
    gripper_xml = find_menagerie() / "robotiq_2f85" / "2f85.xml"
    if not gripper_xml.exists():
        print(f"ERROR: Robotiq 2F-85 not found at {gripper_xml}")
        sys.exit(1)
    gripper_spec = mujoco.MjSpec.from_file(str(gripper_xml))

    grasp_site = spec.site("grasp_site")
    spec.attach(gripper_spec, prefix="gripper/", site=grasp_site)

    # Skip finger↔finger self-collision — real Robotiq has a mechanical
    # hard stop before the fingertips touch; the menagerie model lets
    # them interpenetrate on empty close, producing a persistent
    # "start in collision" for the planner.
    exclude = spec.add_exclude()
    exclude.bodyname1 = "gripper/left_pad"
    exclude.bodyname2 = "gripper/right_pad"

    # 4. Build the Environment.
    if has_objects:
        from prl_assets import OBJECTS_DIR

        scene_config = dict(scene.get("objects") or {})
        for fixture_type in scene.get("fixtures") or {}:
            scene_config.setdefault(fixture_type, 1)
        env = Environment.from_spec(spec, objects_dir=str(OBJECTS_DIR), scene_config=scene_config)
    else:
        env = Environment.from_model(spec.compile())

    # 5. Arm + gripper + grasp manager.
    # The gripper's actuator is prefixed 'gripper/'. RobotiqGripper
    # appends 'fingers_actuator' and the body suffixes.
    gm = GraspManager(env.model, env.data)
    gripper = RobotiqGripper(env.model, env.data, "iiwa14", prefix="gripper/", grasp_manager=gm)

    # The arm's ee_site is 'grasp_site' — the site we added on link7.
    # The gripper's fingertip is the 2F-85's 'pinch' site, accessible
    # as 'gripper/pinch' after the attach. For grasp planning, we want
    # the TSRs to target the fingertip frame; use gripper/pinch.
    arm = create_iiwa14_arm(
        env,
        ee_site="gripper/pinch",
        gripper=gripper,
        grasp_manager=gm,
    )

    # 6. Open gripper, move to home.
    gripper.kinematic_open()
    if gripper.actuator_id is not None:
        env.data.ctrl[gripper.actuator_id] = gripper.ctrl_open
    for i, idx in enumerate(arm.joint_qpos_indices):
        env.data.qpos[idx] = IIWA14_HOME[i]
    mujoco.mj_forward(env.model, env.data)

    # 7. Wrap in a RobotBase subclass and apply the scene.
    robot = IIWA14DemoRobot(env, arm, list(IIWA14_HOME))
    robot.setup_scenario_scene(scene)
    return robot


def _add_iiwa14_worktop(spec: mujoco.MjSpec) -> None:
    """Add a 60×60×5 cm plate with a named ``worktop`` site on top.

    Positioned slightly further from the iiwa base than the Franka
    worktop, since the iiwa 14 has a longer reach (~820 mm vs ~855 mm
    for Franka, similar but slightly offset base height). The exact
    position is a bit of a tuning knob — the scenario system will
    error politely if objects don't fit within arm reach.
    """
    plate = spec.worldbody.add_body()
    plate.name = "plate"
    plate.pos = [0.6, 0.0, 0.025]

    g = plate.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [0.30, 0.30, 0.025]
    g.rgba = [0.6, 0.6, 0.6, 1.0]

    s = plate.add_site()
    s.name = "worktop"
    s.pos = [0, 0, 0.025]
    s.size = [0.25, 0.25, 0.001]
    s.type = mujoco.mjtGeom.mjGEOM_BOX
    s.rgba = [0, 0, 0, 0]


class IIWA14DemoRobot(RobotBase):
    """Minimal :class:`RobotBase` subclass for the iiwa 14 demo.

    Mirrors :class:`FrankaDemoRobot` — the only robot-specific bit is
    the arm name ("iiwa14"). Pretty clear sign this shell should be
    factored into a shared base class in a follow-up.
    """

    def __init__(self, env, arm, home: list[float]) -> None:
        from mj_manipulator.grasp_manager import GraspManager

        gm = arm.grasp_manager or GraspManager(env.model, env.data)
        super().__init__(
            model=env.model,
            data=env.data,
            arms={arm.config.name: arm},
            grasp_manager=gm,
            named_poses={"ready": {arm.config.name: list(home)}},
        )
        self._env = env
        self._home = list(home)

    def get_worktop_pose(self) -> WorktopPose:
        """Read the ``worktop`` site from the model and return its pose."""
        wt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "worktop")
        if wt_id < 0:
            raise RuntimeError(
                "IIWA14DemoRobot.get_worktop_pose: no 'worktop' site in the model. "
                "Did you run build_iiwa14_robot with a scenario that has objects?"
            )
        pose = np.eye(4)
        pose[:3, 3] = self.data.site_xpos[wt_id]
        pose[:3, :3] = self.data.site_xmat[wt_id].reshape(3, 3)
        size = self.model.site_size[wt_id]
        return WorktopPose(pose=pose, size=(float(size[0]), float(size[1])))

    def reset(self, scene: dict) -> None:
        """Reset MuJoCo state and re-apply the scenario scene."""
        self.request_abort()
        self.clear_abort()

        mujoco.mj_resetData(self.model, self.data)

        for i, idx in enumerate(self.arms["iiwa14"].joint_qpos_indices):
            self.data.qpos[idx] = self._home[i]
        gripper = self.arms["iiwa14"].gripper
        if gripper and gripper.actuator_id is not None:
            self.data.ctrl[gripper.actuator_id] = gripper.ctrl_open

        if self._context is not None:
            self._context.hold()

        for arm_name in list(self.arms.keys()):
            for obj in list(self.grasp_manager.get_grasped_by(arm_name)):
                self.grasp_manager.mark_released(obj)
                self.grasp_manager.detach_object(obj)

        if self._env.registry is not None:
            for obj_type in list(self._env.registry.objects.keys()):
                for name in list(self._env.registry.objects[obj_type]["instances"]):
                    if self._env.registry.is_active(name):
                        self._env.registry.hide(name)

        mujoco.mj_forward(self.model, self.data)
        self.setup_scenario_scene(scene)

        if self._context is not None:
            self._context.sync()

        print("Scene reset.")

    @property
    def grasp_source(self):
        """Lazy PrlAssetsGraspSource — only instantiated if we have objects."""
        if self._grasp_source is None and self._env.registry is not None:
            from mj_manipulator.grasp_sources.prl_assets import PrlAssetsGraspSource

            self._grasp_source = PrlAssetsGraspSource(
                self.model,
                self.data,
                self.grasp_manager,
                self.arms,
                registry=self._env.registry,
            )
        return super().grasp_source
