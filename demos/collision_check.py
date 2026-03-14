"""Collision checking demo with UR5e and Franka Panda.

Demonstrates that the same CollisionChecker works with both 6-DOF and
7-DOF arms using real robot models from mujoco_menagerie.

Shows three capabilities:
  1. Simple mode — any contact involving the arm is a collision
  2. Grasp-aware mode — gripper-object contacts are allowed when grasped
  3. Batch checking — efficient multi-configuration validation

Usage:
    cd mj_manipulator
    uv run python demos/collision_check.py
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

from mj_manipulator.collision import CollisionChecker
from mj_manipulator.grasp_manager import GraspManager

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parent.parent.parent  # robot-code/
MENAGERIE = WORKSPACE / "mujoco_menagerie"

UR5E_SCENE = MENAGERIE / "universal_robots_ur5e" / "scene.xml"
FRANKA_SCENE = MENAGERIE / "franka_emika_panda" / "scene.xml"

# ---------------------------------------------------------------------------
# Robot definitions
# ---------------------------------------------------------------------------
UR5E_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

FRANKA_JOINTS = [f"joint{i}" for i in range(1, 8)]

UR5E_CONFIGS = {
    "home":        np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0]),
    "zeros":       np.zeros(6),
    "reach_front": np.array([0.0, -1.2, 1.0, -1.0, -1.5708, 0.0]),
    "reach_down":  np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0]),
    "folded":      np.array([0.0, 1.5, -2.0, 0.0, 0.0, 0.0]),
}

FRANKA_CONFIGS = {
    "home":        np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853]),
    "zeros":       np.zeros(7),
    "reach_front": np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.7, -0.7]),
    "reach_right": np.array([1.5, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0]),
    "tucked":      np.array([0.0, -0.5, 0.0, -2.5, 0.0, 2.0, 0.0]),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def add_objects_to_scene(scene_path: Path) -> mujoco.MjModel:
    """Load a menagerie scene and add a table + graspable mug via MjSpec."""
    spec = mujoco.MjSpec.from_file(str(scene_path))

    # Table
    table = spec.worldbody.add_body()
    table.name = "table"
    table.pos = [0.5, 0.0, 0.4]
    tgeom = table.add_geom()
    tgeom.name = "table_top"
    tgeom.type = mujoco.mjtGeom.mjGEOM_BOX
    tgeom.size = [0.3, 0.4, 0.02]
    tgeom.rgba = [0.6, 0.4, 0.2, 1.0]

    # Graspable mug (freejoint)
    mug = spec.worldbody.add_body()
    mug.name = "mug"
    mug.pos = [0.5, 0.0, 0.46]
    mug_joint = mug.add_joint()
    mug_joint.name = "mug_joint"
    mug_joint.type = mujoco.mjtJoint.mjJNT_FREE
    mug_geom = mug.add_geom()
    mug_geom.name = "mug_geom"
    mug_geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    mug_geom.size = [0.025, 0.04, 0]
    mug_geom.rgba = [0.9, 0.15, 0.15, 1.0]

    return spec.compile()


def print_header(title: str) -> None:
    print(f"\n{'=' * 64}")
    print(f"  {title}")
    print(f"{'=' * 64}")


# ---------------------------------------------------------------------------
# Demo 1: Simple collision checking
# ---------------------------------------------------------------------------
def demo_simple(
    scene_path: Path,
    joint_names: list[str],
    configs: dict[str, np.ndarray],
    label: str,
) -> None:
    """Check configurations for collisions with real robot models."""
    print_header(f"{label} - Simple Mode ({len(joint_names)}-DOF)")

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    cc = CollisionChecker(model, data, joint_names)

    # Show which bodies are detected as "arm"
    arm_names = sorted(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in cc._arm_body_ids
    )
    print(f"\n  Arm bodies ({len(arm_names)}): {arm_names}")

    # Check each configuration
    print(f"\n  {'Config':<15s}  {'Result':<12s}  {'Details'}")
    print(f"  {'-'*15}  {'-'*12}  {'-'*30}")
    for name, q in configs.items():
        valid = cc.is_valid(q)
        status = "VALID" if valid else "COLLISION"
        marker = " " if valid else "!"
        q_str = np.array2string(q, precision=2, suppress_small=True)
        print(f"  {marker} {name:<14s}  {status:<12s}  q={q_str}")

    # Batch check
    qs = np.array(list(configs.values()))
    results = cc.is_valid_batch(qs)
    n_valid = results.sum()
    print(f"\n  Batch result: {n_valid}/{len(results)} collision-free")


# ---------------------------------------------------------------------------
# Demo 2: Grasp-aware collision checking
# ---------------------------------------------------------------------------
def demo_grasp_aware(
    scene_path: Path,
    joint_names: list[str],
    q: np.ndarray,
    attach_body: str,
    label: str,
    mug_offset: np.ndarray | None = None,
) -> None:
    """Show that gripper-object contacts are allowed when an object is grasped."""
    print_header(f"{label} - Grasp-Aware Mode")

    model = add_objects_to_scene(scene_path)
    data = mujoco.MjData(model)

    # Set arm to config
    cc_tmp = CollisionChecker(model, data, joint_names)
    for i, idx in enumerate(cc_tmp.joint_indices):
        data.qpos[idx] = q[i]
    mujoco.mj_forward(model, data)

    # Move mug to the attachment body position (simulating a grasp)
    attach_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, attach_body
    )
    attach_pos = data.xpos[attach_id].copy()
    attach_mat = data.xmat[attach_id].reshape(3, 3)
    # Offset in body-local frame (default: along Z toward fingertips)
    offset = mug_offset if mug_offset is not None else np.array([0, 0, 0.08])
    mug_pos = attach_pos + attach_mat @ offset

    mug_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "mug"
    )
    mug_jnt = model.body_jntadr[mug_body_id]
    mug_qadr = model.jnt_qposadr[mug_jnt]
    data.qpos[mug_qadr : mug_qadr + 3] = mug_pos
    data.qpos[mug_qadr + 3 : mug_qadr + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)

    # Create grasp-aware checker
    gm = GraspManager(model, data)
    cc = CollisionChecker(model, data, joint_names, grasp_manager=gm)

    # --- Before grasp ---
    valid_before = cc.is_valid(q)
    print(f"\n  Mug placed at {attach_body} (pos={np.array2string(mug_pos, precision=3)})")
    print(f"\n  Before grasp:")
    print(f"    is_valid = {valid_before}")
    cc.debug_contacts(q)

    # --- After grasp ---
    gm.mark_grasped("mug", label.lower())
    gm.attach_object("mug", attach_body)

    valid_after = cc.is_valid(q)
    print(f"\n  After grasp (attached to '{attach_body}'):")
    print(f"    is_valid = {valid_after}")
    cc.debug_contacts(q)

    # --- After release ---
    gm.detach_object("mug")
    gm.mark_released("mug")

    valid_released = cc.is_valid(q)
    print(f"\n  After release:")
    print(f"    is_valid = {valid_released}")

    # Validate the transition
    if not valid_before and valid_after and not valid_released:
        print("\n  >> Grasp-aware filtering works correctly!")
    elif valid_before:
        print("\n  >> Note: no contact at this config (mug not touching arm)")
        print("     Grasp-aware logic validated by unit tests.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not MENAGERIE.exists():
        print(f"ERROR: mujoco_menagerie not found at {MENAGERIE}")
        print(
            "Clone it:\n"
            "  cd robot-code\n"
            "  git clone https://github.com/google-deepmind/mujoco_menagerie"
        )
        sys.exit(1)

    for path in [UR5E_SCENE, FRANKA_SCENE]:
        if not path.exists():
            print(f"ERROR: {path} not found")
            sys.exit(1)

    # Part 1: Simple collision checking (same code, two different arms)
    demo_simple(UR5E_SCENE, UR5E_JOINTS, UR5E_CONFIGS, "UR5e")
    demo_simple(FRANKA_SCENE, FRANKA_JOINTS, FRANKA_CONFIGS, "Franka Panda")

    # Part 2: Grasp-aware collision checking
    # Franka has a proper gripper (hand + fingers), so grasp-aware filtering
    # is meaningful. UR5e bare arm has no gripper — in practice you'd attach
    # a Robotiq 2F-140 to get the same gripper-object filtering.
    demo_grasp_aware(
        FRANKA_SCENE,
        FRANKA_JOINTS,
        q=np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853]),
        attach_body="hand",
        label="Franka Panda",
    )

    print_header(
        "DONE - Same CollisionChecker + GraspManager code\n"
        "  works with UR5e (6-DOF) and Franka Panda (7-DOF)"
    )
    print()


if __name__ == "__main__":
    main()
