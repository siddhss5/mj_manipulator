# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Ready-to-use arm definitions for supported robots.

Provides factory functions that create fully configured Arm instances
with IK solvers, joint limits, and kinematic constants.

Supported arms:
    - UR5e (6-DOF): ``from mj_manipulator.arms.ur5e import create_ur5e_arm``
    - Franka Panda (7-DOF): ``from mj_manipulator.arms.franka import create_franka_arm``

Usage:
    from mj_environment import Environment
    from mj_manipulator.arms.ur5e import create_ur5e_arm

    env = Environment("path/to/ur5e/scene.xml")
    arm = create_ur5e_arm(env)
    path = arm.plan_to_pose(target_pose)

Adding a new arm:
    1. Create ``arms/<robot>.py`` with:
       - Joint names, home config, velocity/acceleration limits as constants
         (limits from datasheet, halved for conservative planning)
       - ``create_<robot>_arm(env, ...) -> Arm`` factory function
    2. Build ``ArmConfig`` with your constants and ``MuJoCoEAIKSolver`` for IK.
       The solver extracts kinematics (H/P vectors) from the MuJoCo model —
       no DH parameters needed.
       - 6-DOF arms: pass joint_ids and ee_site_id directly (see ``ur5e.py``)
       - 7-DOF arms: call ``find_locked_joint_index(H, P)`` once to discover
         which joint to lock for EAIK. Hardcode the result as a module constant
         (see ``franka.py`` and ``_FRANKA_LOCKED_JOINT_INDEX``).
    3. If the model lacks an EE site, provide ``add_<robot>_ee_site(spec)``
       (see ``franka.py``). Place the site at the palm/flange; z-axis = approach.
    4. Add tests in ``tests/test_arms.py``: factory creates valid Arm, FK-IK
       round-trip at home config, solutions within joint limits.
    5. Re-export the factory from this ``__init__.py``.

    See ``ur5e.py`` (6-DOF) and ``franka.py`` (7-DOF) as complete references.
    See ``README.md`` for a code skeleton.
"""

from mj_manipulator.arms.eaik_solver import find_locked_joint_index
from mj_manipulator.arms.franka import add_franka_ee_site, create_franka_arm
from mj_manipulator.arms.ur5e import create_ur5e_arm

__all__ = [
    "create_ur5e_arm",
    "create_franka_arm",
    "add_franka_ee_site",
    "find_locked_joint_index",
]
