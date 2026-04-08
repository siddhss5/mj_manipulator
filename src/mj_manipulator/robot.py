# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Robot protocol for generic manipulation.

Defines the minimal interface a robot must expose for the generic
console, primitives (pickup/place/go_home), and teleop to work.

A new robot implements this by composing mj_manipulator Arms::

    class MyRobot:
        def __init__(self, model_path):
            self.env = Environment(model_path)
            self.model = self.env.model
            self.data = self.env.data
            arm = create_ur5e_arm(self.env, "ur5e", with_gripper=True)
            self.arms = {"arm": arm}
            self.grasp_manager = GraspManager(self.model, self.data)
            self.grasp_source = MyGraspSource(...)
            self.named_poses = {"ready": {"arm": [0, -1.57, 1.57, ...]}}
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import mujoco

    from mj_manipulator.arm import Arm
    from mj_manipulator.grasp_manager import GraspManager
    from mj_manipulator.protocols import GraspSource


@runtime_checkable
class ManipulationRobot(Protocol):
    """Minimal interface for a robot that supports generic manipulation.

    The generic console, primitives, and teleop panels call only
    these methods and properties. Everything else is robot-specific.
    """

    @property
    def model(self) -> mujoco.MjModel: ...

    @property
    def data(self) -> mujoco.MjData: ...

    @property
    def arms(self) -> dict[str, Arm]:
        """All arms, keyed by name (e.g. {"left": arm, "right": arm})."""
        ...

    @property
    def grasp_source(self) -> GraspSource:
        """Provides grasp and placement TSRs for objects."""
        ...

    @property
    def grasp_manager(self) -> GraspManager:
        """Tracks grasped objects and kinematic attachments."""
        ...

    @property
    def named_poses(self) -> dict[str, dict[str, list[float]]]:
        """Named joint configurations (e.g. {"ready": {"left": [...], "right": [...]}})."""
        ...

    def sim(
        self,
        *,
        physics: bool = True,
        headless: bool = False,
        viewer: object = None,
        event_loop: object = None,
    ) -> object:
        """Create simulation execution context (context manager).

        Returns an object whose __enter__ yields a SimContext.
        """
        ...

    def request_abort(self) -> None:
        """Signal all running operations to stop."""
        ...

    def clear_abort(self) -> None:
        """Clear the abort flag."""
        ...

    def is_abort_requested(self) -> bool:
        """Check if an abort has been requested."""
        ...
