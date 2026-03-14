"""Grasp state management for manipulation.

Tracks which objects are grasped and handles kinematic attachments so grasped
objects move with the gripper. Collision detection is handled entirely in
software by the collision checker — no MuJoCo collision group changes needed.
"""

import logging

import mujoco
import numpy as np

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

    def _set_body_pose_in_data(
        self, body_name: str, T: np.ndarray, data: mujoco.MjData
    ) -> None:
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


def detect_grasped_object(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    gripper_body_names: list[str],
    candidate_objects: list[str] | None = None,
    require_bilateral: bool = True,
    finger_groups: dict[str, list[str]] | None = None,
    debug: bool = False,
) -> str | None:
    """Detect which object (if any) is being grasped by the gripper.

    Checks MuJoCo contacts to find objects in contact with gripper bodies.
    For realistic grasp detection, requires bilateral contact (both finger
    groups touching the object).

    Args:
        model: MuJoCo model.
        data: MuJoCo data (after mj_forward).
        gripper_body_names: All gripper body names (pads, fingers, etc.).
        candidate_objects: Optional filter — only consider these objects.
        require_bilateral: If True, requires contact with both finger groups.
        finger_groups: Mapping of group name to body names for bilateral
            detection. Example: ``{"left": ["gripper/left_pad"],
            "right": ["gripper/right_pad"]}``. If None, infers from body
            names containing "/left_" or "/right_".
        debug: If True, log detailed contact info.

    Returns:
        Name of grasped object, or None if nothing is grasped.
    """
    # Build body ID sets
    all_gripper_body_ids: set[int] = set()
    group_body_ids: dict[str, set[int]] = {}

    if finger_groups is not None:
        # Explicit finger groups
        for group_name, bodies in finger_groups.items():
            group_body_ids[group_name] = set()
            for name in bodies:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                if body_id != -1:
                    all_gripper_body_ids.add(body_id)
                    group_body_ids[group_name].add(body_id)
        # Also add any gripper bodies not in explicit groups
        for name in gripper_body_names:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id != -1:
                all_gripper_body_ids.add(body_id)
    else:
        # Infer from naming convention (left/right)
        group_body_ids = {"left": set(), "right": set()}
        for name in gripper_body_names:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id != -1:
                all_gripper_body_ids.add(body_id)
                if "/left_" in name or name.endswith("/left_pad") or name.endswith("/left_follower"):
                    group_body_ids["left"].add(body_id)
                elif "/right_" in name or name.endswith("/right_pad") or name.endswith("/right_follower"):
                    group_body_ids["right"].add(body_id)
            elif debug:
                logger.warning(f"Gripper body not found: {name}")

    if not all_gripper_body_ids:
        return None

    # Get candidate object body IDs
    candidate_body_ids: set[int] | None = None
    if candidate_objects is not None:
        candidate_body_ids = set()
        for name in candidate_objects:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id != -1:
                candidate_body_ids.add(body_id)

    # Track contacts per object: body_id -> {group_name: bool, ..., "count": int}
    object_contacts: dict[int, dict] = {}

    for i in range(data.ncon):
        contact = data.contact[i]
        body1 = model.geom_bodyid[contact.geom1]
        body2 = model.geom_bodyid[contact.geom2]

        gripper_body = None
        other_body = None
        if body1 in all_gripper_body_ids:
            gripper_body = body1
            other_body = body2
        elif body2 in all_gripper_body_ids:
            gripper_body = body2
            other_body = body1

        if gripper_body is None or other_body in all_gripper_body_ids:
            continue

        if candidate_body_ids is not None and other_body not in candidate_body_ids:
            continue

        if other_body not in object_contacts:
            entry: dict = {"count": 0}
            for gname in group_body_ids:
                entry[gname] = False
            object_contacts[other_body] = entry

        for gname, gids in group_body_ids.items():
            if gripper_body in gids:
                object_contacts[other_body][gname] = True
        object_contacts[other_body]["count"] += 1

    if not object_contacts:
        return None

    # Filter by bilateral contact requirement
    non_empty_groups = [g for g, ids in group_body_ids.items() if ids]
    if require_bilateral and len(non_empty_groups) >= 2:
        bilateral = {
            bid: info
            for bid, info in object_contacts.items()
            if all(info.get(g, False) for g in non_empty_groups)
        }
        if bilateral:
            object_contacts = bilateral
        else:
            if debug:
                logger.info("No bilateral contacts found")
            return None

    best_body_id = max(object_contacts, key=lambda x: object_contacts[x]["count"])
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, best_body_id)
