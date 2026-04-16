# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""GraspSource backed by prl_assets object geometry.

Generates grasp and placement TSRs by reading object dimensions from
prl_assets meta.yaml files. Works with any robot — only needs MuJoCo
model/data and a GraspManager.

Usage::

    from mj_manipulator.grasp_sources.prl_assets import PrlAssetsGraspSource

    source = PrlAssetsGraspSource(model, data, grasp_manager, arms)
    tsrs = source.get_grasps("can_0", "parallel_jaw")
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from mj_manipulator.arm import Arm
    from mj_manipulator.grasp_manager import GraspManager

logger = logging.getLogger(__name__)

_CONTAINER_TYPES = frozenset(("open_box", "tote"))


class PrlAssetsGraspSource:
    """GraspSource backed by prl_assets geometry.

    Implements the mj_manipulator GraspSource protocol by loading object
    dimensions from prl_assets' AssetManager and generating TSRs using
    the tsr library's hand models.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        grasp_manager: For tracking grasped objects.
        arms: Dict of arm name → Arm (for grasp transform computation).
        registry: Optional object registry with is_active() method.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        grasp_manager: GraspManager,
        arms: dict[str, Arm],
        registry: object | None = None,
        perception: object | None = None,
    ) -> None:
        self._model = model
        self._data = data
        self._gm = grasp_manager
        self._arms = arms
        self._registry = registry
        self._perception = perception

    def get_grasps(self, object_name: str, hand_type: str) -> list:
        """Get grasp TSRs for an object from its prl_assets geometry."""
        obj_type = _instance_to_type(object_name)
        if obj_type is None:
            return []
        return self._generate_tsrs_for_object(object_name, obj_type, hand_type)

    def get_placements(self, destination: str, object_name: str) -> list:
        """Get placement TSRs for an object at a destination."""
        dest_type = _instance_to_type(destination)
        if dest_type is None:
            return self._get_site_placements(destination, object_name)

        held_height = self._get_held_object_height()
        T_gripper_object = self._get_grasp_transform()
        return self._generate_place_tsrs(
            destination,
            dest_type,
            held_height=held_height,
            T_gripper_object=T_gripper_object,
        )

    def get_graspable_objects(self) -> list[str]:
        """Get all graspable objects currently in the scene."""
        objects = self._find_scene_objects(None)
        return [body_name for body_name, _ in objects]

    def get_place_destinations(self, object_name: str) -> list[str]:
        """Get valid placement destinations for an object."""
        from asset_manager import AssetManager
        from prl_assets import OBJECTS_DIR

        assets = AssetManager(str(OBJECTS_DIR))

        destinations = []
        for i in range(self._model.nbody):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, i)
            if not name:
                continue
            if not self._is_active(name):
                continue
            m = re.match(r"^(.+?)_(\d+)$", name)
            if not m:
                continue
            obj_type = m.group(1)
            try:
                gp = assets.get(obj_type)["geometric_properties"]
                geo_type = gp.get("type")
                if geo_type in _CONTAINER_TYPES or geo_type in ("box", "cylinder"):
                    destinations.append(name)
            except (KeyError, TypeError):
                continue

        # Also check for worktop site
        try:
            wt_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "worktop")
            if wt_id >= 0:
                destinations.append("worktop")
        except Exception:
            pass

        return destinations

    # -- Internal helpers -------------------------------------------------------

    def _is_active(self, body_name: str) -> bool:
        """Check if a body is active (visible) in the scene."""
        if self._registry is None:
            return True
        try:
            return self._registry.is_active(body_name)
        except Exception:
            return True

    def _get_body_pose(self, body_name: str) -> np.ndarray:
        """Get 4x4 world-frame pose of a MuJoCo body.

        Uses the PerceptionService when available, falling back to
        direct ``data.xpos`` reads for backward compatibility.
        """
        if self._perception is not None:
            pose = self._perception.get_pose(body_name)
            if pose is not None:
                return pose
            raise ValueError(f"Object not found or not active: {body_name}")
        bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            raise ValueError(f"Body not found: {body_name}")
        pose = np.eye(4)
        pose[:3, :3] = self._data.xmat[bid].reshape(3, 3)
        pose[:3, 3] = self._data.xpos[bid]
        return pose

    def _holding(self) -> tuple[str, str] | None:
        """Get (arm_name, object_name) if any arm is holding an object."""
        for arm_name, arm in self._arms.items():
            if arm.gripper and arm.gripper.is_holding and arm.gripper.held_object:
                return (arm_name, arm.gripper.held_object)
        return None

    def _find_scene_objects(self, target: str | None) -> list[tuple[str, str]]:
        """Find objects in the scene matching a target specification."""
        from asset_manager import AssetManager
        from prl_assets import OBJECTS_DIR

        assets = AssetManager(str(OBJECTS_DIR))

        all_bodies = []
        for i in range(self._model.nbody):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, i)
            if not name:
                continue
            if not self._is_active(name):
                continue
            all_bodies.append(name)

        if target is not None:
            instance_match = re.match(r"^(.+)_(\d+)$", target)
            if instance_match:
                obj_type = instance_match.group(1)
                if target in all_bodies:
                    return [(target, obj_type)]
                return []

            matches = []
            for body in all_bodies:
                m = re.match(r"^(.+?)_(\d+)$", body)
                if m and m.group(1) == target:
                    if not self._gm.is_grasped(body):
                        matches.append((body, target))
            return matches

        matches = []
        known_types = set()
        for body in all_bodies:
            m = re.match(r"^(.+?)_(\d+)$", body)
            if not m:
                continue
            obj_type = m.group(1)
            if self._gm.is_grasped(body):
                continue
            if obj_type not in known_types:
                try:
                    assets.get(obj_type)["geometric_properties"]
                    known_types.add(obj_type)
                except (KeyError, TypeError):
                    continue
            if obj_type in known_types:
                matches.append((body, obj_type))
        return matches

    def _generate_tsrs_for_object(self, body_name: str, obj_type: str, hand_type: str = "parallel_jaw") -> list:
        """Generate grasp TSRs for a single object from its prl_assets geometry."""
        from asset_manager import AssetManager
        from prl_assets import OBJECTS_DIR

        assets = AssetManager(str(OBJECTS_DIR))
        try:
            gp = assets.get(obj_type)["geometric_properties"]
        except (KeyError, TypeError):
            return []

        try:
            obj_pose = self._get_body_pose(body_name)
        except ValueError:
            return []

        hand = _get_hand(hand_type)
        if gp.get("type") == "cylinder":
            T_bottom = obj_pose.copy()
            local_z = obj_pose[:3, 2]
            T_bottom[:3, 3] -= local_z * (gp["height"] / 2)
            templates = hand.grasp_cylinder_side(gp["radius"], gp["height"])
            return [t.instantiate(T_bottom) for t in templates]
        elif gp.get("type") == "box":
            size = gp["size"]
            T_bottom = obj_pose.copy()
            local_z = obj_pose[:3, 2]
            T_bottom[:3, 3] -= local_z * (size[2] / 2)
            templates = []
            for grasp_fn in [
                hand.grasp_box_face_x,
                hand.grasp_box_face_y,
                hand.grasp_box_top,
                hand.grasp_box_bottom,
            ]:
                try:
                    templates.extend(grasp_fn(size[0], size[1], size[2]))
                except ValueError as e:
                    logger.info(
                        "%s: skipping %s — %s",
                        body_name,
                        grasp_fn.__name__,
                        e,
                    )
            return [t.instantiate(T_bottom) for t in templates]

        return []

    def _generate_place_tsrs(
        self,
        body_name: str,
        dest_type: str,
        held_height: float = 0.0,
        T_gripper_object: np.ndarray | None = None,
    ) -> list:
        """Generate placement TSRs — dispatches between container and surface."""
        from asset_manager import AssetManager
        from prl_assets import OBJECTS_DIR

        assets = AssetManager(str(OBJECTS_DIR))
        try:
            gp = assets.get(dest_type)["geometric_properties"]
        except (KeyError, TypeError):
            return []

        geo_type = gp.get("type")

        if geo_type in _CONTAINER_TYPES:
            return self._generate_container_drop_tsrs(body_name, dest_type, held_height)

        try:
            dest_pose = self._get_body_pose(body_name)
        except (ValueError, AttributeError):
            return []

        faces = _get_upward_faces(dest_pose, gp)
        if not faces:
            return []

        held_type = self._get_held_object_type()
        all_tsrs = []
        for surface_pose, hx, hy in faces:
            tsrs = _generate_surface_place_tsrs(
                surface_pose,
                hx,
                hy,
                held_type,
                T_gripper_object=T_gripper_object,
            )
            all_tsrs.extend(tsrs)
        return all_tsrs

    def _generate_container_drop_tsrs(
        self,
        body_name: str,
        dest_type: str,
        held_height: float = 0.0,
    ) -> list:
        """Generate drop-zone TSRs for a container."""
        from asset_manager import AssetManager
        from prl_assets import OBJECTS_DIR
        from tsr import TSR

        assets = AssetManager(str(OBJECTS_DIR))
        try:
            meta = assets.get(dest_type)
            gp = meta["geometric_properties"]
            policy = meta.get("policy", {}).get("placement", {})
        except (KeyError, TypeError):
            return []

        if gp.get("type") not in _CONTAINER_TYPES:
            return []

        try:
            dest_pose = self._get_body_pose(body_name)
        except ValueError:
            return []

        outer = gp["outer_size"]
        wall = gp.get("wall_thickness", 0.003)
        margin = policy.get("drop_zone_margin", 0.05)

        hx = (outer[0] / 2) - wall - margin
        hy = (outer[1] / 2) - wall - margin

        local_z = dest_pose[:3, 2]
        clearance = max(0.25, held_height + 0.10)
        drop_pos = dest_pose[:3, 3] + local_z * (outer[2] + clearance)

        approach = -local_z
        gripper_x = dest_pose[:3, 0]
        gripper_y = np.cross(approach, gripper_x)

        T0_w = np.eye(4)
        T0_w[:3, 0] = gripper_x
        T0_w[:3, 1] = gripper_y
        T0_w[:3, 2] = approach
        T0_w[:3, 3] = drop_pos

        Bw = np.zeros((6, 2))
        Bw[0, :] = [-hx, hx]
        Bw[1, :] = [-hy, hy]
        Bw[2, :] = [-0.02, 0.05]
        Bw[5, :] = [-np.pi, np.pi]

        return [TSR(T0_w=T0_w, Bw=Bw)]

    def _get_held_object_height(self) -> float:
        """Get the height of the currently held object, or 0 if unknown."""
        from asset_manager import AssetManager
        from prl_assets import OBJECTS_DIR

        held = self._holding()
        if not held:
            return 0.0

        _, obj_name = held
        assets = AssetManager(str(OBJECTS_DIR))
        obj_type = _instance_to_type(obj_name)
        if not obj_type:
            return 0.0

        try:
            gp = assets.get(obj_type)["geometric_properties"]
            if gp["type"] == "cylinder":
                return max(gp["height"], gp["radius"] * 2)
            elif gp["type"] == "box":
                return max(gp["size"])
        except (KeyError, TypeError):
            pass
        return 0.0

    def _get_held_object_type(self) -> str | None:
        """Get the prl_assets type of the currently held object."""
        held = self._holding()
        if not held:
            return None
        _, obj_name = held
        return _instance_to_type(obj_name)

    def _get_grasp_transform(self) -> np.ndarray | None:
        """Get T_site_object for the currently held object."""
        held = self._holding()
        if not held:
            return None
        side, obj_name = held

        T_body_object = self._gm.get_grasp_transform(obj_name)
        if T_body_object is None:
            return None

        arm = self._arms[side]
        body_name = arm.gripper.attachment_body
        body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        site_id = arm.ee_site_id

        T_world_body = np.eye(4)
        T_world_body[:3, :3] = self._data.xmat[body_id].reshape(3, 3)
        T_world_body[:3, 3] = self._data.xpos[body_id]

        T_world_site = np.eye(4)
        T_world_site[:3, :3] = self._data.site_xmat[site_id].reshape(3, 3)
        T_world_site[:3, 3] = self._data.site_xpos[site_id]

        T_body_site = np.linalg.inv(T_world_body) @ T_world_site
        T_site_object = np.linalg.inv(T_body_site) @ T_body_object
        return T_site_object

    def _get_site_placements(self, site_name: str, object_name: str) -> list:
        """Get placement TSRs for a named site (e.g. 'worktop')."""
        try:
            wt_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        except Exception:
            return []
        if wt_id < 0:
            return []

        wt_size = self._model.site_size[wt_id]
        surface_pose = np.eye(4)
        surface_pose[:3, 3] = self._data.site_xpos[wt_id].copy()

        held_type = self._get_held_object_type()
        T_gripper_object = self._get_grasp_transform()

        return _generate_surface_place_tsrs(
            surface_pose,
            float(wt_size[0]),
            float(wt_size[1]),
            held_type,
            T_gripper_object=T_gripper_object,
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _get_hand(hand_type: str):
    """Get the TSR hand model for a gripper type.

    ``hand_type`` is typically ``gripper.hand_type`` (set by the gripper
    class). Matches are substring-based, case-insensitive:

    - contains ``"franka"`` → :class:`tsr.hands.FrankaHand`
    - contains ``"2f85"``   → :class:`tsr.hands.Robotiq2F85`
    - anything else (including ``"robotiq_2f140"``, ``"robotiq"``)
      → :class:`tsr.hands.Robotiq2F140`

    The 2F-140 default preserves backward compatibility with Geodude
    and any other caller that still reports ``hand_type="robotiq"``.
    """
    from tsr.hands import FrankaHand, Robotiq2F85, Robotiq2F140

    ht = hand_type.lower()
    if "franka" in ht:
        return FrankaHand()
    if "2f85" in ht:
        return Robotiq2F85()
    return Robotiq2F140()


def _instance_to_type(name: str) -> str | None:
    """Extract object type from instance name (e.g. 'can_0' → 'can')."""
    m = re.match(r"^(.+?)_(\d+)$", name)
    return m.group(1) if m else None


_UPWARD_THRESHOLD = 0.95


def _get_upward_faces(
    dest_pose: np.ndarray,
    gp: dict,
) -> list[tuple[np.ndarray, float, float]]:
    """Enumerate flat faces pointing upward."""
    R = dest_pose[:3, :3]
    origin = dest_pose[:3, 3]
    geo_type = gp.get("type")

    candidates: list[tuple[np.ndarray, float, float, float]] = []

    if geo_type == "box":
        lx, ly, lz = gp["size"]
        candidates.append((np.array([0, 0, +1.0]), lz / 2, lx / 2, ly / 2))
        candidates.append((np.array([0, 0, -1.0]), lz / 2, lx / 2, ly / 2))
        candidates.append((np.array([0, +1.0, 0]), ly / 2, lx / 2, lz / 2))
        candidates.append((np.array([0, -1.0, 0]), ly / 2, lx / 2, lz / 2))
        candidates.append((np.array([+1.0, 0, 0]), lx / 2, ly / 2, lz / 2))
        candidates.append((np.array([-1.0, 0, 0]), lx / 2, ly / 2, lz / 2))
    elif geo_type == "cylinder":
        r, h = gp["radius"], gp["height"]
        candidates.append((np.array([0, 0, +1.0]), h / 2, r, r))
        candidates.append((np.array([0, 0, -1.0]), h / 2, r, r))

    results = []
    up = np.array([0.0, 0.0, 1.0])
    for local_normal, offset, hx, hy in candidates:
        normal_world = R @ local_normal
        if normal_world @ up < _UPWARD_THRESHOLD:
            continue

        face_center = origin + R @ (local_normal * offset)
        surface_pose = np.eye(4)
        surface_pose[:3, 3] = face_center
        surface_pose[:3, 2] = normal_world

        abs_normal = np.abs(local_normal)
        if abs_normal[0] < 0.5:
            local_x = np.array([1.0, 0, 0])
        elif abs_normal[1] < 0.5:
            local_x = np.array([0, 1.0, 0])
        else:
            local_x = np.array([0, 0, 1.0])
        surface_x = R @ local_x
        surface_x -= normal_world * (surface_x @ normal_world)
        surface_x /= np.linalg.norm(surface_x)
        surface_pose[:3, 0] = surface_x
        surface_pose[:3, 1] = np.cross(normal_world, surface_x)

        results.append((surface_pose, hx, hy))

    return results


def _generate_surface_place_tsrs(
    surface_pose: np.ndarray,
    surface_hx: float,
    surface_hy: float,
    held_obj_type: str | None,
    T_gripper_object: np.ndarray | None = None,
) -> list:
    """Generate stable placement TSRs for a flat surface."""
    from asset_manager import AssetManager
    from prl_assets import OBJECTS_DIR
    from tsr.placement import StablePlacer

    if held_obj_type is None:
        return []

    assets = AssetManager(str(OBJECTS_DIR))
    try:
        gp = assets.get(held_obj_type)["geometric_properties"]
    except (KeyError, TypeError):
        return []

    margin = 0.05
    placer = StablePlacer(
        table_x=max(0.01, surface_hx - margin),
        table_y=max(0.01, surface_hy - margin),
    )

    geo_type = gp.get("type")
    if geo_type == "cylinder":
        templates = placer.place_cylinder(gp["radius"], gp["height"], subject=held_obj_type)
    elif geo_type == "box":
        templates = placer.place_box(gp["size"][0], gp["size"][1], gp["size"][2], subject=held_obj_type)
    elif geo_type == "sphere":
        templates = placer.place_sphere(gp["radius"], subject=held_obj_type)
    else:
        return []

    if not templates:
        return []
    template = templates[0]

    if T_gripper_object is not None:
        import dataclasses

        T_object_gripper = np.linalg.inv(T_gripper_object)
        template = dataclasses.replace(template, Tw_e=template.Tw_e @ T_object_gripper)

    tsr = template.instantiate(surface_pose)
    clearance = 0.005
    tsr.Bw[2, :] += clearance

    return [tsr]
