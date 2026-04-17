# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Grasp state management for manipulation.

Tracks which objects are grasped and handles kinematic attachments so grasped
objects move with the gripper. Collision detection is handled entirely in
software by the collision checker — no MuJoCo collision group changes needed.
"""

import logging

import mujoco
import numpy as np

from mj_manipulator.contacts import iter_contacts

logger = logging.getLogger(__name__)


class GraspManager:
    """Manages grasp state and kinematic attachments.

    Tracks which objects are grasped by which arm, and maintains kinematic
    attachments so objects move with the gripper during manipulation.

    The collision checker uses ``is_grasped()`` to determine how to filter
    contacts — gripper-to-object contacts are allowed, while arm-to-object
    and object-to-environment contacts are flagged as collisions.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.grasped: dict[str, str] = {}  # object_name -> arm_name
        # Kinematic attachments: object_name -> (gripper_body_name, T_gripper_object)
        self._attachments: dict[str, tuple[str, np.ndarray]] = {}

    def mark_grasped(self, object_name: str, arm: str) -> None:
        """Mark an object as grasped by the specified arm."""
        if object_name in self.grasped:
            return
        self.grasped[object_name] = arm

    def mark_released(self, object_name: str) -> None:
        """Mark an object as released."""
        if object_name not in self.grasped:
            return
        del self.grasped[object_name]

    def get_grasped_by(self, arm: str) -> list[str]:
        """Get list of objects currently grasped by the specified arm."""
        return [obj for obj, holder in self.grasped.items() if holder == arm]

    def is_grasped(self, object_name: str) -> bool:
        """Check if an object is currently grasped."""
        return object_name in self.grasped

    def get_holder(self, object_name: str) -> str | None:
        """Get the arm holding an object, or None if not grasped."""
        return self.grasped.get(object_name)

    def attach_object(self, object_name: str, gripper_body_name: str) -> None:
        """Attach an object to a gripper for kinematic manipulation.

        Computes and stores the relative transform between gripper and object
        so the object can move with the gripper.

        Args:
            object_name: Name of the object body in MuJoCo.
            gripper_body_name: Name of the gripper body to attach to.
        """
        T_world_gripper = self._get_body_pose(gripper_body_name)
        T_world_object = self._get_body_pose(object_name)
        T_gripper_object = np.linalg.inv(T_world_gripper) @ T_world_object
        self._attachments[object_name] = (gripper_body_name, T_gripper_object)

    def detach_object(self, object_name: str) -> None:
        """Detach an object from kinematic attachment."""
        self._attachments.pop(object_name, None)

    def is_attached(self, object_name: str) -> bool:
        """Check if an object is kinematically attached."""
        return object_name in self._attachments

    def get_attached_objects(self) -> list[str]:
        """Get list of all kinematically attached objects."""
        return list(self._attachments.keys())

    def get_attachment_body(self, object_name: str) -> str | None:
        """Get the gripper body name that an object is attached to."""
        if object_name not in self._attachments:
            return None
        gripper_body_name, _ = self._attachments[object_name]
        return gripper_body_name

    def get_grasp_transform(self, object_name: str) -> np.ndarray | None:
        """Get the grasp transform T_gripper_object for an attached object.

        Returns the 4x4 transform from gripper frame to object frame,
        recorded at grasp time.  Returns None if the object is not attached.
        """
        if object_name not in self._attachments:
            return None
        _, T_gripper_object = self._attachments[object_name]
        return T_gripper_object.copy()

    def update_attached_poses(self, data: mujoco.MjData | None = None) -> None:
        """Update poses of all kinematically attached objects.

        Call this after moving the gripper to update attached object positions.

        Args:
            data: MjData to update (defaults to self.data). Pass a temporary
                  MjData during collision checking to avoid viewer flickering.
        """
        if data is None:
            data = self.data

        for object_name, (gripper_body_name, T_gripper_object) in self._attachments.items():
            T_world_gripper = self._get_body_pose_from_data(gripper_body_name, data)
            T_world_object = T_world_gripper @ T_gripper_object
            self._set_body_pose_in_data(object_name, T_world_object, data)

    # -- Internal pose helpers --

    def _get_body_pose(self, body_name: str) -> np.ndarray:
        """Get the 4x4 pose matrix of a body from self.data."""
        return self._get_body_pose_from_data(body_name, self.data)

    def _get_body_pose_from_data(self, body_name: str, data: mujoco.MjData) -> np.ndarray:
        """Get the 4x4 pose matrix of a body from specified MjData."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"Body '{body_name}' not found in model")

        pos = data.xpos[body_id].copy()
        mat = data.xmat[body_id].reshape(3, 3).copy()

        T = np.eye(4)
        T[:3, :3] = mat
        T[:3, 3] = pos
        return T

    def _set_body_pose_in_data(self, body_name: str, T: np.ndarray, data: mujoco.MjData) -> None:
        """Set the pose of a freejoint body in specified MjData."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"Body '{body_name}' not found in model")

        joint_id = self.model.body_jntadr[body_id]
        if joint_id == -1:
            raise ValueError(f"Body '{body_name}' has no joint — cannot set pose")

        if self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"Body '{body_name}' joint is not a freejoint")

        qpos_adr = self.model.jnt_qposadr[joint_id]

        pos = T[:3, 3]
        mat = T[:3, :3]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat.flatten())

        data.qpos[qpos_adr : qpos_adr + 3] = pos
        data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat


def find_contacted_object(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    gripper_body_names: list[str],
    candidate_objects: list[str] | None = None,
) -> str | None:
    """Find the object with the most gripper contacts (sim only).

    Simple contact-count heuristic for the nameless grasp path
    (REPL / teleop "close the gripper on whatever is there"). The
    BT / primitives path always passes an explicit object name and
    uses ``GraspVerifier`` for post-grasp validation. On hardware,
    the nameless path is identified by ``PerceptionService`` after
    gripper close.

    Args:
        model: MuJoCo model.
        data: MuJoCo data (after ``mj_forward``).
        gripper_body_names: All gripper body names (pads, fingers, etc.).
        candidate_objects: Optional filter — only consider these objects.

    Returns:
        Name of the most-contacted object body, or None.
    """
    gripper_ids: set[int] = set()
    for name in gripper_body_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid != -1:
            gripper_ids.add(bid)
    if not gripper_ids:
        return None

    candidate_ids: set[int] | None = None
    if candidate_objects is not None:
        candidate_ids = set()
        for name in candidate_objects:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid != -1:
                candidate_ids.add(bid)

    contact_counts: dict[int, int] = {}
    for body1, body2, _ in iter_contacts(model, data):
        if body1 in gripper_ids and body2 not in gripper_ids:
            other = body2
        elif body2 in gripper_ids and body1 not in gripper_ids:
            other = body1
        else:
            continue
        if candidate_ids is not None and other not in candidate_ids:
            continue
        contact_counts[other] = contact_counts.get(other, 0) + 1

    if not contact_counts:
        return None

    best = max(contact_counts, key=contact_counts.get)
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, best)
