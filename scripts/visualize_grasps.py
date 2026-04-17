#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Disembodied-hand grasp visualizer: teleport a bare gripper to sampled TSRs.

The fastest way to diagnose a grasp problem is to remove the arm and IK
from the picture and look at just the gripper and the object. This script
does exactly that — builds a minimal scene with a free-floating gripper
and a target object, samples TSRs from the same templates the grasp
source uses in production, and teleports the gripper to each sample so
you can eyeball the approach geometry.

When to run it:

- Adding a new gripper and unsure whether ``FINGER_LENGTH`` is right.
- Arm stops "1 cm short" during grasp and you suspect the tool frame.
- Adding a new object shape and want to see where side/top grasps go.
- Debugging "why does IK fail for every grasp TSR?" — if the TSR
  geometry is wrong in this script, IK will fail for good reason.

Usage::

    uv run python scripts/visualize_grasps.py --gripper robotiq_2f85 --object can
    uv run python scripts/visualize_grasps.py --gripper franka        --object spam_can
    uv run python scripts/visualize_grasps.py --gripper robotiq_2f140 --object can

Controls (in the viser browser window):

- **Next sample** — resample the current template (new random yaw).
- **Next / Prev template** — cycle through grasp families
  (shallow → mid → deep × roll 0° / roll 180° for cylinders, four
  faces + top + bottom for boxes, etc.).
- **Sweep (25×N)** — batch-sample 25 grasps per template and report
  per-template collision rate in the stats panel.

Per-tick readouts:

- **Collision** — ``✅ CLEAR`` or ``❌ IN COLLISION`` for the current pose.
  This uses MuJoCo's contact solver, restricted to contacts between
  gripper geoms and the target object (pad↔pad self-contacts are
  ignored).
- **Contacts** — names of the offending gripper geom(s), if any.
- **Per-template hit rate** — running tally of in-collision samples
  per template, updated on every click.

The gripper is teleported via a freejoint — no arm, no IK, no
trajectory. If the gripper lands "inside" the object the TSR/hand
parameters are wrong:

- Palm inside housing → ``grasp_site`` placed at the mounting body
  instead of at the housing's forward edge. Fix per "Adding a New
  Gripper" step 3 in the repo README.
- Fingers too short → ``FINGER_LENGTH`` too small, TSR stops the
  palm short of the target.
- Fingers miss the object in the y-axis → grasp_site rotation wrong
  (finger-opening axis isn't TSR +y).

For automated pass/fail plus specific fix suggestions, see the
companion script :mod:`validate_gripper`.

For the math behind the palm-vs-housing rule and the empirical
grip-force findings that drove this tooling, see the deep-dive at
``mj_manipulator/docs/grippers.md``.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# Gripper registry — one entry per supported gripper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GripperSpec:
    """How to load a given gripper into a standalone visualizer scene.

    Every supported gripper needs an XML file containing a single
    kinematic tree rooted at a body we can attach under a freejoint, a
    ``hand_type`` string that :func:`mj_manipulator.grasp_sources.prl_assets._get_hand`
    can resolve to a TSR ``ParallelJawGripper`` subclass, and enough
    info to either find or synthesize a ``grasp_site`` at the canonical
    TSR EE frame (z=approach, y=finger-opening, x=palm-normal).
    """

    xml_path_resolver: "object"
    """Callable returning a ``Path`` to the gripper XML. Lazy so we don't
    import menagerie/geodude_assets until the user actually asks for
    that gripper."""

    hand_type: str
    """Label forwarded to ``_get_hand`` and TSR grasp generation."""

    add_grasp_site: bool = False
    """If True, this script adds a grasp_site at ``grasp_site_base_body``
    with ``grasp_site_pos`` / ``grasp_site_quat``. If False, the XML
    already contains a site named ``grasp_site_name``."""

    grasp_site_name: str = "grasp_site"
    grasp_site_base_body: str = ""
    grasp_site_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    grasp_site_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    finger_open_qpos: dict[str, float] = field(default_factory=dict)
    """Optional map of joint name → qpos value for opening the gripper
    kinematically (no actuator, no physics). If empty, the default qpos
    from the XML keyframe (or zero) is used."""


def _2f85_xml() -> Path:
    from mj_manipulator.menagerie import find_menagerie

    return find_menagerie() / "robotiq_2f85" / "2f85.xml"


def _2f140_xml() -> Path:
    # geodude_assets 2F-140 is the only one we have; no menagerie equivalent.
    import importlib

    ga = importlib.import_module("geodude_assets")
    return Path(ga.__file__).parent / "models" / "robotiq_2f140" / "2f140.xml"


def _franka_hand_xml() -> Path:
    from mj_manipulator.menagerie import find_menagerie

    return find_menagerie() / "franka_emika_panda" / "hand.xml"


GRIPPERS: dict[str, GripperSpec] = {
    "robotiq_2f85": GripperSpec(
        xml_path_resolver=_2f85_xml,
        hand_type="robotiq_2f85",
        add_grasp_site=True,
        grasp_site_name="grasp_site",
        grasp_site_base_body="base_mount",
        # Palm at the forward edge of the base housing; -90° about z
        # aligns the gripper's opening-along-x to the TSR y-convention.
        # The z-offset is pulled from Robotiq2F85 so the single source
        # of truth stays in the TSR hand class — must match
        # iiwa14_setup.py's grasp_site definition exactly.
        grasp_site_pos=(0.0, 0.0, 0.094),  # = Robotiq2F85.PALM_OFFSET_FROM_BASE_MOUNT
        grasp_site_quat=(0.7071, 0.0, 0.0, -0.7071),
    ),
    "robotiq_2f140": GripperSpec(
        # grasp_site is already baked into the geodude_assets 2f140.xml
        # at pos=[0,0,0.1] with the same -90° z rotation.
        xml_path_resolver=_2f140_xml,
        hand_type="robotiq_2f140",
        add_grasp_site=False,
        grasp_site_name="grasp_site",
    ),
    "franka": GripperSpec(
        # Franka hand.xml has no site. Palm = forward edge of the hand
        # body (the metal collar) = hand + [0, 0, 0.0753] = finger-joint
        # origin + 17 mm forward. Hand frame already has z=approach,
        # y=opening, so identity rotation. Must match add_franka_ee_site
        # in arms/franka.py exactly.
        xml_path_resolver=_franka_hand_xml,
        hand_type="franka",
        add_grasp_site=True,
        grasp_site_name="grasp_site",
        grasp_site_base_body="hand",
        grasp_site_pos=(0.0, 0.0, 0.0753),  # = FrankaHand.PALM_OFFSET_FROM_HAND
        grasp_site_quat=(1.0, 0.0, 0.0, 0.0),
        # Open the fingers for visualization (default qpos leaves them
        # closed, which is unhelpful for eyeballing aperture).
        finger_open_qpos={"finger_joint1": 0.04, "finger_joint2": 0.04},
    ),
}


# ---------------------------------------------------------------------------
# Object registry — pulls from prl_assets at runtime
# ---------------------------------------------------------------------------


@dataclass
class ObjectShape:
    """Loaded geometry for a prl_assets object type."""

    name: str
    type: str  # "cylinder" | "box" | "sphere"
    params: dict  # {"radius":..., "height":...} or {"size":[hx,hy,hz]} or {"radius":...}


def load_object(obj_type: str) -> ObjectShape:
    """Load a prl_assets object type, returning geometry for visualization."""
    from asset_manager import AssetManager
    from prl_assets import OBJECTS_DIR

    assets = AssetManager(str(OBJECTS_DIR))
    gp = assets.get(obj_type)["geometric_properties"]
    return ObjectShape(name=obj_type, type=gp["type"], params=dict(gp))


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------


def build_scene(gripper: GripperSpec, obj: ObjectShape) -> tuple[mujoco.MjModel, mujoco.MjData, dict]:
    """Assemble the visualization scene as an MjSpec → compile.

    Returns (model, data, ids) where ``ids`` has indices into the model
    for the freejoint, gripper_root body, grasp site, and object body.
    """
    gripper_xml = gripper.xml_path_resolver()
    if not gripper_xml.is_file():
        raise FileNotFoundError(f"Gripper XML not found: {gripper_xml}")
    gripper_spec = mujoco.MjSpec.from_file(str(gripper_xml))

    # Optionally stamp a canonical grasp_site if the gripper XML doesn't
    # have one.
    if gripper.add_grasp_site:
        base = gripper_spec.body(gripper.grasp_site_base_body)
        if base is None:
            raise RuntimeError(
                f"Gripper {gripper.hand_type}: no body named {gripper.grasp_site_base_body!r} "
                f"to attach grasp_site under."
            )
        s = base.add_site()
        s.name = gripper.grasp_site_name
        s.pos = list(gripper.grasp_site_pos)
        s.quat = list(gripper.grasp_site_quat)

    # Host spec: world body with floor, target object, and a floating
    # gripper root (freejoint) at the origin.
    host = mujoco.MjSpec()
    # Disable gravity — the gripper has no actuator and we don't want
    # it falling between teleports.
    host.option.gravity = [0.0, 0.0, 0.0]

    # Floor — visual only, no collision (we never step physics).
    floor = host.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [2.0, 2.0, 0.1]
    floor.rgba = [0.7, 0.75, 0.8, 1.0]
    floor.contype = 0
    floor.conaffinity = 0

    # Target object body at (0.5, 0, 0.1) — in front of the world origin,
    # waist-height. Static.
    target = host.worldbody.add_body()
    target.name = "target"
    target.pos = [0.5, 0.0, 0.1]

    tg = target.add_geom()
    tg.name = "target_geom"
    if obj.type == "cylinder":
        r = float(obj.params["radius"])
        h = float(obj.params["height"])
        tg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        tg.size = [r, h / 2, 0.0]
        tg.rgba = [0.85, 0.3, 0.3, 0.6]
    elif obj.type == "box":
        sx, sy, sz = [float(v) for v in obj.params["size"]]
        tg.type = mujoco.mjtGeom.mjGEOM_BOX
        tg.size = [sx / 2, sy / 2, sz / 2]
        tg.rgba = [0.3, 0.6, 0.85, 0.6]
    elif obj.type == "sphere":
        r = float(obj.params["radius"])
        tg.type = mujoco.mjtGeom.mjGEOM_SPHERE
        tg.size = [r, 0.0, 0.0]
        tg.rgba = [0.3, 0.85, 0.4, 0.6]
    else:
        raise ValueError(f"Unsupported object geometry type: {obj.type}")
    # Keep the target colliding (contype/conaffinity defaults) so the
    # solver generates contacts we can report on. The gripper XMLs
    # already ship with collision geoms in groups that collide with
    # anything in the world.

    # A small visible cross at the object origin to help orient the eye.
    origin_marker = target.add_site()
    origin_marker.name = "target_origin"
    origin_marker.type = mujoco.mjtGeom.mjGEOM_SPHERE
    origin_marker.size = [0.005, 0, 0]
    origin_marker.rgba = [1, 1, 0, 1]

    # Floating gripper: body with a freejoint, and a site to attach
    # the gripper spec at. Start the freejoint "parked" off to the side
    # so the first frame isn't misleading.
    root = host.worldbody.add_body()
    root.name = "gripper_root"
    root.pos = [0.0, 0.0, 0.5]
    root.add_freejoint()
    attach = root.add_site()
    attach.name = "attach"

    host.attach(gripper_spec, prefix="g/", site=attach)

    model = host.compile()
    data = mujoco.MjData(model)

    # Kinematically open fingers if we have a recipe for it.
    for jname, qval in gripper.finger_open_qpos.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"g/{jname}")
        if jid < 0:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            qadr = model.jnt_qposadr[jid]
            data.qpos[qadr] = qval

    # Compute T_root_site ONCE with the freejoint at identity — this is
    # the constant offset we need to invert later to place the gripper.
    # Free joints have no name via ``body.add_freejoint()`` — look them
    # up by body instead.
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_root")
    fj_joint_id = int(model.body_jntadr[root_id])
    if fj_joint_id < 0 or int(model.jnt_type[fj_joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise RuntimeError("Expected a free joint on gripper_root; got none.")
    fj_addr = int(model.jnt_qposadr[fj_joint_id])
    data.qpos[fj_addr : fj_addr + 3] = [0.0, 0.0, 0.5]
    data.qpos[fj_addr + 3 : fj_addr + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"g/{gripper.grasp_site_name}")
    if site_id < 0:
        raise RuntimeError(
            f"grasp_site not found in compiled model (looked for 'g/{gripper.grasp_site_name}'). "
            f"Available sites: "
            f"{[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(model.nsite)]}"
        )

    T_world_root = np.eye(4)
    T_world_root[:3, :3] = data.xmat[root_id].reshape(3, 3)
    T_world_root[:3, 3] = data.xpos[root_id]
    T_world_site = np.eye(4)
    T_world_site[:3, :3] = data.site_xmat[site_id].reshape(3, 3)
    T_world_site[:3, 3] = data.site_xpos[site_id]
    T_root_site = np.linalg.inv(T_world_root) @ T_world_site

    target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
    target_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom")

    # Collect every geom under the gripper subtree for collision
    # attribution. Walk the body tree from gripper_root down — any geom
    # attached to a descendant body is a "gripper geom".
    gripper_body_ids: set[int] = set()
    stack = [root_id]
    while stack:
        bid = stack.pop()
        gripper_body_ids.add(bid)
        for cid in range(model.nbody):
            if int(model.body_parentid[cid]) == bid and cid != bid:
                stack.append(cid)
    gripper_geom_ids: set[int] = {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in gripper_body_ids}

    ids = {
        "freejoint_qpos": int(fj_addr),
        "root_body": int(root_id),
        "site": int(site_id),
        "target_body": int(target_id),
        "target_geom": int(target_geom_id),
        "gripper_geoms": gripper_geom_ids,
        "T_root_site": T_root_site,
    }
    return model, data, ids


# ---------------------------------------------------------------------------
# TSR generation
# ---------------------------------------------------------------------------


def generate_templates(gripper: GripperSpec, obj: ObjectShape) -> list:
    """Return a flat list of ``TSRTemplate`` objects for the (gripper, object) pair.

    Reuses the same hand class / grasp primitives the production grasp
    source uses, so what you see here is what the planner will try.
    """
    from mj_manipulator.grasp_sources.prl_assets import _get_hand

    hand = _get_hand(gripper.hand_type)
    templates = []
    if obj.type == "cylinder":
        r = float(obj.params["radius"])
        h = float(obj.params["height"])
        templates.extend(hand.grasp_cylinder_side(r, h))
    elif obj.type == "box":
        sx, sy, sz = [float(v) for v in obj.params["size"]]
        for fn in (hand.grasp_box_face_x, hand.grasp_box_face_y, hand.grasp_box_top, hand.grasp_box_bottom):
            try:
                templates.extend(fn(sx, sy, sz))
            except ValueError:
                continue
    elif obj.type == "sphere":
        r = float(obj.params["radius"])
        templates.extend(hand.grasp_sphere(r))
    else:
        raise ValueError(f"Unsupported object geometry: {obj.type}")

    if not templates:
        raise RuntimeError(f"No grasp templates generated for {gripper.hand_type} × {obj.name}.")
    return templates


# ---------------------------------------------------------------------------
# Teleport utilities
# ---------------------------------------------------------------------------


def _mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to wxyz quaternion."""
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q


def teleport_gripper(model, data, ids, T_world_site: np.ndarray) -> None:
    """Place the gripper so its grasp_site lands at ``T_world_site``."""
    T_root_site = ids["T_root_site"]
    T_world_root = T_world_site @ np.linalg.inv(T_root_site)

    fj_adr = ids["freejoint_qpos"]
    data.qpos[fj_adr : fj_adr + 3] = T_world_root[:3, 3]
    data.qpos[fj_adr + 3 : fj_adr + 7] = _mat_to_quat_wxyz(T_world_root[:3, :3])
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def check_collision(model, data, ids) -> tuple[bool, list[str]]:
    """Report whether the gripper is in collision with the target.

    ``mj_forward`` was already called by :func:`teleport_gripper` — the
    contact buffer is fresh. We inspect ``data.contact[:data.ncon]`` and
    return (any_hit, list-of-colliding-gripper-geom-names).

    Self-contacts inside the gripper (pads touching each other etc.) are
    ignored — only contacts between a gripper geom and the target geom
    count as a "grasp is bad" signal.
    """
    target_geom = ids["target_geom"]
    gripper_geoms = ids["gripper_geoms"]

    hits: list[str] = []
    for i in range(int(data.ncon)):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 == target_geom and g2 in gripper_geoms) or (g2 == target_geom and g1 in gripper_geoms):
            other = g2 if g1 == target_geom else g1
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or f"geom{other}"
            hits.append(name)
    return (len(hits) > 0, hits)


def get_object_pose(model, data, ids, obj: ObjectShape) -> np.ndarray:
    """4×4 world pose of the *template-reference* frame.

    For cylinders and boxes, the grasp templates expect the reference
    frame at the *bottom* face (as per ``PrlAssetsGraspSource``).
    Sphere templates expect the center.
    """
    bid = ids["target_body"]
    T = np.eye(4)
    T[:3, :3] = data.xmat[bid].reshape(3, 3)
    T[:3, 3] = data.xpos[bid]

    if obj.type == "cylinder":
        h = float(obj.params["height"])
        T[:3, 3] -= T[:3, 2] * (h / 2)
    elif obj.type == "box":
        sz = float(obj.params["size"][2])
        T[:3, 3] -= T[:3, 2] * (sz / 2)
    # sphere: reference = center — no shift.
    return T


# ---------------------------------------------------------------------------
# Viser panel
# ---------------------------------------------------------------------------


def _run_viewer(model, data, ids, gripper: GripperSpec, obj: ObjectShape, templates: list) -> None:
    """Launch mj_viser with a panel to step through grasp samples."""
    from mj_viser import MujocoViewer
    from mj_viser.panels import PanelBase

    state = {
        "template_idx": 0,
        "sample_count": 0,
        "current_pose": None,  # 4x4 ndarray or None
        "dirty": True,  # trigger a first teleport on initial sync
        "lock": threading.Lock(),
    }

    T_ref = get_object_pose(model, data, ids, obj)

    def resample() -> None:
        with state["lock"]:
            tmpl = templates[state["template_idx"]]
            tsr = tmpl.instantiate(T_ref)
            state["current_pose"] = tsr.sample()
            state["sample_count"] += 1
            state["dirty"] = True

    # Per-template collision statistics, populated incrementally as the
    # user clicks around.
    stats: dict[int, dict] = {i: {"samples": 0, "collisions": 0} for i in range(len(templates))}

    class GraspViz(PanelBase):
        def __init__(self_) -> None:
            self_._template_label = None
            self_._info_label = None
            self_._collision_label = None
            self_._contact_label = None
            self_._stats_label = None

        def name(self_) -> str:
            return "Grasp Viz"

        def setup(self_, gui, viewer) -> None:
            with gui.add_folder("Grasp Viz"):
                gui.add_markdown(f"**Gripper:** {gripper.hand_type}\n\n**Object:** {obj.name} ({obj.type})")
                gui.add_markdown(f"**Templates:** {len(templates)}")

                def on_next_sample(_event) -> None:
                    resample()

                def on_next_template(_event) -> None:
                    with state["lock"]:
                        state["template_idx"] = (state["template_idx"] + 1) % len(templates)
                    resample()

                def on_prev_template(_event) -> None:
                    with state["lock"]:
                        state["template_idx"] = (state["template_idx"] - 1) % len(templates)
                    resample()

                def on_sweep(_event) -> None:
                    # Sample each template 25 times, tally collision rate.
                    N = 25
                    with state["lock"]:
                        for i in range(len(templates)):
                            for _ in range(N):
                                pose = templates[i].instantiate(T_ref).sample()
                                teleport_gripper(viewer.model, viewer.data, ids, pose)
                                hit, _ = check_collision(viewer.model, viewer.data, ids)
                                stats[i]["samples"] += 1
                                if hit:
                                    stats[i]["collisions"] += 1
                        # Restore the currently-viewed pose.
                        state["dirty"] = True

                gui.add_button("Next sample").on_click(on_next_sample)
                gui.add_button("Next template").on_click(on_next_template)
                gui.add_button("Prev template").on_click(on_prev_template)
                gui.add_button("Sweep (25×N)").on_click(on_sweep)

                self_._template_label = gui.add_text("Template", initial_value="", disabled=True)
                self_._info_label = gui.add_text("Sample #", initial_value="", disabled=True)
                self_._collision_label = gui.add_text("Collision", initial_value="—", disabled=True)
                self_._contact_label = gui.add_text("Contacts", initial_value="", disabled=True)
                self_._stats_label = gui.add_text("Per-template hit rate", initial_value="", disabled=True)

        def on_sync(self_, viewer) -> None:
            with state["lock"]:
                if state["dirty"] and state["current_pose"] is not None:
                    teleport_gripper(viewer.model, viewer.data, ids, state["current_pose"])
                    state["dirty"] = False
                tmpl = templates[state["template_idx"]]

                in_collision, hit_names = check_collision(viewer.model, viewer.data, ids)

                # Tally running stats for the current template, but only
                # when the user clicked "Next sample" (sample_count > 0)
                # rather than on every redraw.
                if state.get("stat_tick") != state["sample_count"]:
                    stats[state["template_idx"]]["samples"] += 1
                    if in_collision:
                        stats[state["template_idx"]]["collisions"] += 1
                    state["stat_tick"] = state["sample_count"]

                if self_._template_label is not None:
                    label = tmpl.name or tmpl.variant or f"template {state['template_idx']}"
                    self_._template_label.value = f"[{state['template_idx'] + 1}/{len(templates)}] {label}"
                if self_._info_label is not None:
                    self_._info_label.value = str(state["sample_count"])
                if self_._collision_label is not None:
                    self_._collision_label.value = "❌ IN COLLISION" if in_collision else "✅ CLEAR"
                if self_._contact_label is not None:
                    self_._contact_label.value = ", ".join(hit_names) if hit_names else "—"
                if self_._stats_label is not None:
                    rows = []
                    for i, s in stats.items():
                        if s["samples"] == 0:
                            rows.append(f"[{i}] —")
                            continue
                        rate = 100.0 * s["collisions"] / s["samples"]
                        rows.append(f"[{i}] {s['collisions']}/{s['samples']} ({rate:.0f}%)")
                    self_._stats_label.value = "  ".join(rows)

    resample()

    viewer = MujocoViewer(model, data, label=f"grasps: {gripper.hand_type} × {obj.name}")
    viewer.add_panel(GraspViz())
    viewer.launch_passive(open_browser=True)
    try:
        while viewer.is_running():
            viewer.sync()
            time.sleep(1.0 / 60.0)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gripper", required=True, choices=sorted(GRIPPERS.keys()))
    parser.add_argument(
        "--object",
        required=True,
        help="prl_assets object type name (e.g., 'can', 'spam_can', 'pop_tarts_case').",
    )
    args = parser.parse_args()

    gripper_spec = GRIPPERS[args.gripper]
    try:
        obj = load_object(args.object)
    except (KeyError, TypeError) as e:
        print(f"ERROR: could not load prl_assets object {args.object!r}: {e}", file=sys.stderr)
        return 1

    try:
        model, data, ids = build_scene(gripper_spec, obj)
    except Exception as e:
        print(f"ERROR: scene build failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1

    templates = generate_templates(gripper_spec, obj)
    print(f"Generated {len(templates)} grasp templates for {args.gripper} × {args.object}.", flush=True)
    for i, t in enumerate(templates):
        label = t.name or t.variant or f"template_{i}"
        print(f"  [{i}] {label}", flush=True)

    _run_viewer(model, data, ids, gripper_spec, obj, templates)
    return 0


if __name__ == "__main__":
    sys.exit(main())
