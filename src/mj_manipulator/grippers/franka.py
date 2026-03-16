"""Franka Emika Panda hand gripper implementation.

Supports both physics-mode (actuator-driven) and kinematic-mode (direct
joint position) operation. The Franka hand has two prismatic finger joints
with simple linear opening/closing.

Usage:
    from mj_manipulator.grippers.franka import FrankaGripper

    gripper = FrankaGripper(model, data, "franka")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

from mj_manipulator.grippers._base import _BaseGripper

if TYPE_CHECKING:
    from mj_manipulator.grasp_manager import GraspManager

# ---------------------------------------------------------------------------
# Franka hand constants
# ---------------------------------------------------------------------------

# Finger joint name suffixes.
_FINGER_JOINT_SUFFIXES = ["finger_joint1", "finger_joint2"]

# Body name suffixes for contact detection.
_BODY_SUFFIXES = ["hand", "left_finger", "right_finger"]

# Attachment body (finger that objects weld to during kinematic grasping).
_ATTACHMENT_BODY_SUFFIX = "left_finger"

# Finger joint range (slide joints, meters).
_FINGER_OPEN = 0.04   # Fully open position
_FINGER_CLOSED = 0.0   # Fully closed position


# ---------------------------------------------------------------------------
# FrankaGripper
# ---------------------------------------------------------------------------


class FrankaGripper(_BaseGripper):
    """Franka Emika Panda hand gripper.

    The Franka hand is a simple parallel jaw gripper with two prismatic
    finger joints driven by a tendon actuator. In kinematic mode, the
    fingers are linearly interpolated between open and closed positions.

    The ``prefix`` parameter handles namespacing in multi-robot scenes.
    For the standard menagerie model, use ``prefix=""`` (default).

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        arm_name: Which arm this gripper belongs to.
        prefix: MuJoCo name prefix for all gripper elements.
        grasp_manager: Optional grasp state tracker.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        arm_name: str,
        prefix: str = "",
        grasp_manager: GraspManager | None = None,
    ):
        # Resolve actuator
        actuator_name = f"{prefix}actuator8"

        # Resolve body names
        body_names = [f"{prefix}{s}" for s in _BODY_SUFFIXES]

        # Attachment body
        attachment_body = f"{prefix}{_ATTACHMENT_BODY_SUFFIX}"

        super().__init__(
            model=model,
            data=data,
            arm_name=arm_name,
            actuator_name=actuator_name,
            gripper_body_names=body_names,
            attachment_body=attachment_body,
            ctrl_open=0.0,
            ctrl_closed=255.0,
            grasp_manager=grasp_manager,
        )

        # Resolve finger joint qpos indices
        self._finger_qpos_indices: list[int] = []
        for suffix in _FINGER_JOINT_SUFFIXES:
            full_name = f"{prefix}{suffix}"
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, full_name,
            )
            if joint_id != -1:
                self._finger_qpos_indices.append(model.jnt_qposadr[joint_id])

    def _apply_kinematic_position(self, t: float) -> None:
        if not self._finger_qpos_indices:
            return
        pos = _FINGER_OPEN + t * (_FINGER_CLOSED - _FINGER_OPEN)
        for idx in self._finger_qpos_indices:
            self._data.qpos[idx] = pos
        mujoco.mj_forward(self._model, self._data)

    def get_actual_position(self) -> float:
        """Get actual gripper position (0=open, 1=closed).

        Reads finger_joint1 position and maps from [0.04, 0.0] to [0, 1].
        """
        if not self._finger_qpos_indices:
            return 0.0

        finger_pos = self._data.qpos[self._finger_qpos_indices[0]]
        if abs(_FINGER_OPEN - _FINGER_CLOSED) < 1e-8:
            return 0.0

        t = (_FINGER_OPEN - finger_pos) / (_FINGER_OPEN - _FINGER_CLOSED)
        return float(np.clip(t, 0.0, 1.0))
