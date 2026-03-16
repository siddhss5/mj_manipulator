"""Concrete gripper implementations for supported robots.

Provides gripper classes that satisfy the Gripper protocol, with both
physics-mode and kinematic-mode support.

Supported grippers:
    - Robotiq 2F-140: ``from mj_manipulator.grippers.robotiq import RobotiqGripper``
    - Franka Hand: ``from mj_manipulator.grippers.franka import FrankaGripper``
"""

from mj_manipulator.grippers.franka import FrankaGripper
from mj_manipulator.grippers.robotiq import RobotiqGripper

__all__ = [
    "RobotiqGripper",
    "FrankaGripper",
]
