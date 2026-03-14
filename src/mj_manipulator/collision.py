"""Grasp-aware collision checking for motion planning.

Collision detection is handled entirely in software by filtering MuJoCo
contacts. No collision group changes needed — we check grasp state to
determine which contacts are expected (gripper-object) vs invalid
(arm-object, object-environment).

Unifies three previous classes (GraspAwareCollisionChecker,
CollisionChecker, SimpleCollisionChecker) into one.
"""

from __future__ import annotations

import logging

import mujoco
import numpy as np

from mj_manipulator.grasp_manager import GraspManager

logger = logging.getLogger(__name__)


class CollisionChecker:
    """Grasp-aware collision checker for motion planning.

    Checks for collisions between the robot arm and environment, AND
    non-adjacent self-collisions (e.g., forearm hitting gripper). Adjacent
    link collisions are filtered by MuJoCo's ``<exclude>`` tags.

    Supports two modes:

    **Live mode** (``grasp_manager`` provided): Reads grasp state from a
    live GraspManager. Uses temporary MjData to avoid viewer flickering.
    Use for single-threaded planning.

    **Snapshot mode** (``grasped_objects`` + ``attachments`` provided):
    Uses frozen grasp state passed at construction. Each instance owns
    its MjData. Use for parallel/multi-threaded planning.

    **Simple mode** (neither provided): No grasp awareness. Any contact
    involving the arm is a collision.

    Implements pycbirrt's ``CollisionChecker`` protocol (``is_valid(q)``).
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        joint_names: list[str],
        *,
        grasp_manager: GraspManager | None = None,
        grasped_objects: frozenset[tuple[str, str]] | None = None,
        attachments: dict[str, tuple[str, np.ndarray]] | None = None,
    ):
        """Initialize collision checker.

        Args:
            model: MuJoCo model (shared, read-only in snapshot mode).
            data: MuJoCo data. In live mode, a temporary copy is created
                internally. In snapshot mode, this should be a private copy.
            joint_names: Names of joints to control.
            grasp_manager: Live GraspManager for single-threaded use.
            grasped_objects: Frozen grasp state for thread-safe use.
                Frozenset of ``(object_name, arm_name)`` tuples.
            attachments: Frozen attachment state for thread-safe use.
                Dict of ``{object_name: (gripper_body_name, T_gripper_object)}``.
        """
        self.model = model
        self._grasp_manager = grasp_manager

        # Snapshot state (for thread-safe mode)
        self._grasped_objects = grasped_objects or frozenset()
        self._attachments = attachments or {}

        # In live mode, create temp data to avoid viewer flickering.
        # In snapshot mode, use the provided data directly (caller owns it).
        if grasp_manager is not None:
            self._live_data = data  # keep reference for qpos copy
            self.data = mujoco.MjData(model)
        else:
            self._live_data = None
            self.data = data

        # Get joint qpos indices
        self.joint_indices: list[int] = []
        for name in joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id == -1:
                raise ValueError(f"Joint '{name}' not found in model")
            self.joint_indices.append(model.jnt_qposadr[joint_id])

        # Build set of body IDs belonging to this arm (including gripper children)
        self._arm_body_ids: set[int] = set()
        for name in joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            body_id = model.jnt_bodyid[joint_id]
            self._arm_body_ids.add(body_id)
            self._add_child_bodies(body_id)

    # -- Public API (pycbirrt CollisionChecker protocol) --

    def is_valid(self, q: np.ndarray) -> bool:
        """Check if a configuration is collision-free.

        Args:
            q: Joint configuration (only the controlled joints).

        Returns:
            True if collision-free.
        """
        data = self._prepare_data(q)

        # Update attached object poses, then regenerate contacts
        self._update_attached_poses(data)
        mujoco.mj_forward(self.model, data)

        return self._count_invalid_contacts(data) == 0

    def is_valid_batch(self, qs: np.ndarray) -> np.ndarray:
        """Check multiple configurations for collisions."""
        results = np.zeros(len(qs), dtype=bool)
        for i, q in enumerate(qs):
            results[i] = self.is_valid(q)
        return results

    def is_arm_in_collision(
        self, q: np.ndarray | None = None, min_penetration: float = 0.005
    ) -> bool:
        """Check if arm links are colliding with environment.

        For reactive cartesian control: allows gripper-object and
        grasped-object-environment contacts, but flags arm-link-environment
        contacts (forearm hitting base, etc.).

        Args:
            q: Joint configuration. If None, uses current data state.
            min_penetration: Minimum penetration depth (meters) to report.

        Returns:
            True if arm is in collision with environment.
        """
        if q is not None:
            data = self._prepare_data(q)
            self._update_attached_poses(data)
            mujoco.mj_forward(self.model, data)
        else:
            data = self._live_data if self._live_data is not None else self.data

        for i in range(data.ncon):
            contact = data.contact[i]
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]

            body1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1)
            body2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2)

            body1_is_arm = body1 in self._arm_body_ids
            body2_is_arm = body2 in self._arm_body_ids
            body1_is_grasped = body1_name is not None and self._is_grasped(body1_name)
            body2_is_grasped = body2_name is not None and self._is_grasped(body2_name)

            body1_is_gripper = body1_is_arm and body1_name is not None and "gripper" in body1_name
            body2_is_gripper = body2_is_arm and body2_name is not None and "gripper" in body2_name

            body1_is_arm_link = body1_is_arm and not body1_is_gripper
            body2_is_arm_link = body2_is_arm and not body2_is_gripper

            body1_is_env = not body1_is_arm and not body1_is_grasped
            body2_is_env = not body2_is_arm and not body2_is_grasped

            if (body1_is_arm_link and body2_is_env) or (body2_is_arm_link and body1_is_env):
                penetration = -contact.dist
                if penetration >= min_penetration:
                    return True

        return False

    def debug_contacts(self, q: np.ndarray) -> None:
        """Print all contacts for a configuration."""
        data = self._prepare_data(q)
        self._update_attached_poses(data)
        mujoco.mj_forward(self.model, data)

        print(f"Total contacts: {data.ncon}")
        self._count_invalid_contacts(data, debug=True)

    # -- Internal helpers --

    def _prepare_data(self, q: np.ndarray) -> mujoco.MjData:
        """Set joint positions and run forward kinematics."""
        if self._live_data is not None:
            # Live mode: copy from live data to temp
            self.data.qpos[:] = self._live_data.qpos
            self.data.qvel[:] = self._live_data.qvel

        for i, qpos_idx in enumerate(self.joint_indices):
            self.data.qpos[qpos_idx] = q[i]

        mujoco.mj_forward(self.model, self.data)
        return self.data

    def _is_grasped(self, body_name: str) -> bool:
        """Check if a body is grasped (works in both modes)."""
        if self._grasp_manager is not None:
            return self._grasp_manager.is_grasped(body_name)
        return any(obj == body_name for obj, _ in self._grasped_objects)

    def _get_attachment_body(self, object_name: str) -> str | None:
        """Get the gripper body an object is attached to."""
        if self._grasp_manager is not None:
            return self._grasp_manager.get_attachment_body(object_name)
        if object_name in self._attachments:
            return self._attachments[object_name][0]
        return None

    def _update_attached_poses(self, data: mujoco.MjData) -> None:
        """Update poses of attached objects in the given data."""
        if self._grasp_manager is not None:
            self._grasp_manager.update_attached_poses(data)
            return

        # Snapshot mode: inline pose update
        for obj_name, (gripper_body_name, T_gripper_object) in self._attachments.items():
            gripper_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, gripper_body_name
            )
            if gripper_id == -1:
                continue

            pos = data.xpos[gripper_id].copy()
            mat = data.xmat[gripper_id].reshape(3, 3).copy()
            T_world_gripper = np.eye(4)
            T_world_gripper[:3, :3] = mat
            T_world_gripper[:3, 3] = pos

            T_world_object = T_world_gripper @ T_gripper_object
            self._set_body_pose(obj_name, T_world_object, data)

    def _set_body_pose(
        self, body_name: str, T: np.ndarray, data: mujoco.MjData
    ) -> None:
        """Set the pose of a freejoint body."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            return

        joint_id = self.model.body_jntadr[body_id]
        if joint_id == -1:
            return

        if self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            return

        qpos_adr = self.model.jnt_qposadr[joint_id]
        pos = T[:3, 3]
        mat = T[:3, :3]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat.flatten())

        data.qpos[qpos_adr : qpos_adr + 3] = pos
        data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat

    def _count_invalid_contacts(
        self, data: mujoco.MjData, debug: bool = False
    ) -> int:
        """Count contacts that indicate invalid collisions.

        Treats grasped objects as part of the robot:
        - Gripper-to-grasped-object contacts are ALLOWED
        - Grasped-object-to-arm contacts are INVALID
        - Grasped-object-to-environment contacts are INVALID
        - Non-adjacent arm self-collisions are INVALID
        """
        invalid_count = 0

        for i in range(data.ncon):
            contact = data.contact[i]
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]

            body1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1)
            body2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2)

            body1_is_arm = body1 in self._arm_body_ids
            body2_is_arm = body2 in self._arm_body_ids
            body1_is_grasped = body1_name is not None and self._is_grasped(body1_name)
            body2_is_grasped = body2_name is not None and self._is_grasped(body2_name)

            body1_is_robot = body1_is_arm or body1_is_grasped
            body2_is_robot = body2_is_arm or body2_is_grasped

            if not body1_is_robot and not body2_is_robot:
                continue

            if body1_is_robot and body2_is_robot:
                if self._is_gripper_object_contact(
                    body1, body1_name, body1_is_arm, body1_is_grasped,
                    body2, body2_name, body2_is_arm, body2_is_grasped,
                ):
                    if debug:
                        print(f"  [OK] Gripper-object: {body1_name} <-> {body2_name}")
                    continue
                if debug:
                    print(f"  [INVALID] Self-collision: {body1_name} <-> {body2_name}")
                invalid_count += 1
                continue

            if debug:
                robot = body1_name if body1_is_robot else body2_name
                env = body2_name if body1_is_robot else body1_name
                print(f"  [INVALID] Robot-environment: {robot} <-> {env}")
            invalid_count += 1

        return invalid_count

    def _is_gripper_object_contact(
        self,
        body1: int, body1_name: str | None, body1_is_arm: bool, body1_is_grasped: bool,
        body2: int, body2_name: str | None, body2_is_arm: bool, body2_is_grasped: bool,
    ) -> bool:
        """Check if contact is between a grasped object and its holding gripper."""
        if body1_is_grasped and body2_is_arm:
            grasped_name = body1_name
            arm_body_id = body2
        elif body2_is_grasped and body1_is_arm:
            grasped_name = body2_name
            arm_body_id = body1
        else:
            return False

        gripper_body_name = self._get_attachment_body(grasped_name)
        if gripper_body_name is None:
            return False

        gripper_base_name = self._get_gripper_base_name(gripper_body_name)
        if gripper_base_name is None:
            gripper_base_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, gripper_body_name
            )
        else:
            gripper_base_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, gripper_base_name
            )

        if gripper_base_id == -1:
            return False

        return arm_body_id in self._get_body_and_descendants(gripper_base_id)

    def _get_gripper_base_name(self, attachment_body_name: str) -> str | None:
        """Find gripper base body name from attachment body name.

        Given "robot/gripper/right_follower", returns "robot/gripper/base"
        if it exists. This allows contacts with all gripper parts.
        """
        parts = attachment_body_name.rsplit("/", 1)
        if len(parts) < 2:
            return None

        gripper_base_name = f"{parts[0]}/base"
        body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, gripper_base_name
        )
        return gripper_base_name if body_id != -1 else None

    def _get_body_and_descendants(self, body_id: int) -> set[int]:
        """Get a body ID and all its descendant body IDs."""
        result = {body_id}
        for i in range(self.model.nbody):
            if self.model.body_parentid[i] == body_id:
                result.update(self._get_body_and_descendants(i))
        return result

    def _add_child_bodies(self, parent_id: int) -> None:
        """Recursively add child bodies to arm body set."""
        for i in range(self.model.nbody):
            if self.model.body_parentid[i] == parent_id and i not in self._arm_body_ids:
                self._arm_body_ids.add(i)
                self._add_child_bodies(i)
