# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Franka Panda assembly for mj_manipulator scenarios.

Reference implementation showing what it takes to bring an arm into
the mj_manipulator framework. Six steps, each commented:

    1. Load the model (MjSpec + gravcomp injection)
    2. Add a worktop surface (so ``scatter_on_surface`` has somewhere
       to place objects)
    3. Build the :class:`Environment`
    4. Construct arm, gripper, grasp manager
    5. Home the robot (qpos → home, gripper open)
    6. Wrap in a :class:`RobotBase` subclass and apply the scene

To bring your own arm: copy this file, swap the Franka-specific bits
(``menagerie_scene``, ``create_franka_arm``, ``FrankaGripper``,
``FRANKA_HOME``) for your own factory, adjust the worktop, done.

This file is skipped by scenario discovery — it has no ``scene = {...}``
assignment at module level.
"""

from __future__ import annotations

import sys

import mujoco
import numpy as np

from mj_manipulator.robot import RobotBase
from mj_manipulator.scenarios import WorktopPose


def build_franka_robot(scene: dict) -> "FrankaDemoRobot":
    """Assemble a Franka Panda robot and apply a scenario scene.

    Args:
        scene: Scenario scene dict with ``objects``, ``fixtures``,
            optional ``spawn_count``. Drives worktop presence and the
            :class:`mj_environment.Environment` object catalog.

    Returns:
        A ready-to-use robot with the scene applied.
    """
    from mj_environment import Environment

    from mj_manipulator.arms.franka import (
        FRANKA_HOME,
        add_franka_ee_site,
        add_franka_finger_exclude,
        add_franka_gravcomp,
        add_franka_pad_friction,
        create_franka_arm,
        fix_franka_grip_force,
    )
    from mj_manipulator.grasp_manager import GraspManager
    from mj_manipulator.grippers.franka import FrankaGripper
    from mj_manipulator.menagerie import menagerie_scene

    # 1. Load the Franka scene from mujoco_menagerie and post-process it.
    scene_path = menagerie_scene("franka_emika_panda")
    if not scene_path.exists():
        print(f"ERROR: Franka scene not found at {scene_path}")
        print("Run ./setup.sh from the workspace root to clone mujoco_menagerie.")
        sys.exit(1)

    spec = mujoco.MjSpec.from_file(str(scene_path))
    add_franka_ee_site(spec)  # adds a named 'ee' site the IK solver targets
    add_franka_gravcomp(spec)  # body-level gravcomp
    add_franka_pad_friction(spec)  # higher friction on finger pads
    add_franka_finger_exclude(spec)  # skip finger↔finger self-collision (models hard stop)

    # 2. Add a worktop plate if the scenario has objects to scatter.
    #    (A scenario without objects doesn't need a worktop.)
    has_objects = bool(scene.get("objects") or scene.get("fixtures"))
    if has_objects:
        _add_franka_worktop(spec)

    # 3. Build the Environment.
    if has_objects:
        from prl_assets import OBJECTS_DIR

        # Environment.from_spec needs a scene_config that lists every
        # object type that might appear (including fixtures), so the
        # registry pre-allocates enough body slots.
        scene_config = dict(scene.get("objects") or {})
        for fixture_type in scene.get("fixtures") or {}:
            scene_config.setdefault(fixture_type, 1)
        env = Environment.from_spec(spec, objects_dir=str(OBJECTS_DIR), scene_config=scene_config)
    else:
        env = Environment.from_model(spec.compile())

    fix_franka_grip_force(env.model)

    # 4. Arm + gripper + grasp manager. RobotBase.__init__ will
    # auto-wire a GraspVerifier on the gripper below.
    gm = GraspManager(env.model, env.data)
    gripper = FrankaGripper(env.model, env.data, "franka", grasp_manager=gm)
    arm = create_franka_arm(env, gripper=gripper, grasp_manager=gm)

    # 5. Open gripper, move to home.
    gripper.kinematic_open()
    if gripper.actuator_id is not None:
        env.data.ctrl[gripper.actuator_id] = gripper.ctrl_open
    for i, idx in enumerate(arm.joint_qpos_indices):
        env.data.qpos[idx] = FRANKA_HOME[i]
    mujoco.mj_forward(env.model, env.data)

    # 6. Wrap in a RobotBase subclass and apply the scene.
    robot = FrankaDemoRobot(env, arm, list(FRANKA_HOME))
    robot.setup_scenario_scene(scene)
    return robot


def _add_franka_worktop(spec: mujoco.MjSpec) -> None:
    """Add a 60×60×5 cm plate with a named ``worktop`` site on top.

    The ``worktop`` site is what :meth:`FrankaDemoRobot.get_worktop_pose`
    reads to tell the scenario system where to scatter objects.
    """
    plate = spec.worldbody.add_body()
    plate.name = "plate"
    plate.pos = [0.5, 0.0, 0.025]

    g = plate.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [0.30, 0.30, 0.025]
    g.rgba = [0.6, 0.6, 0.6, 1.0]

    # Invisible target site — scenario system reads position + size,
    # nothing is rendered.
    s = plate.add_site()
    s.name = "worktop"
    s.pos = [0, 0, 0.025]
    s.size = [0.25, 0.25, 0.001]
    s.type = mujoco.mjtGeom.mjGEOM_BOX
    s.rgba = [0, 0, 0, 0]


class FrankaDemoRobot(RobotBase):
    """Minimal :class:`RobotBase` subclass for the Franka demo.

    Implements the two scenario hooks ``get_worktop_pose`` and ``reset``.
    All manipulation primitives (pickup, place, go_home, find_objects,
    holding, get_object_pose) come from ``RobotBase``.
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
                "FrankaDemoRobot.get_worktop_pose: no 'worktop' site in the model. "
                "Did you run build_franka_robot with a scenario that has objects?"
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

        # Restore arm to home
        for i, idx in enumerate(self.arms["franka"].joint_qpos_indices):
            self.data.qpos[idx] = self._home[i]
        gripper = self.arms["franka"].gripper
        if gripper and gripper.actuator_id is not None:
            self.data.ctrl[gripper.actuator_id] = gripper.ctrl_open

        if self._context is not None:
            self._context.hold()

        # Release any active grasps
        for arm_name in list(self.arms.keys()):
            for obj in list(self.grasp_manager.get_grasped_by(arm_name)):
                self.grasp_manager.mark_released(obj)
                self.grasp_manager.detach_object(obj)

        # Hide everything currently active
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
