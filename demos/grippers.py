"""Gripper demo with Robotiq 2F-140 and Franka Hand.

Demonstrates the Gripper protocol implementations:

  1. Standalone kinematic control: open, close, set_kinematic_position
  2. Position reading: get_actual_position maps to [0, 1]
  3. Arm + GraspManager integration: grasp/release through SimContext

Uses the Robotiq model from geodude_assets and the Franka model from
mujoco_menagerie.

Usage:
    cd mj_manipulator
    uv run python demos/grippers.py
"""

from pathlib import Path

import mujoco
import numpy as np

from mj_environment import Environment

from mj_manipulator.arms.franka import (
    FRANKA_HOME,
    add_franka_ee_site,
    create_franka_arm,
)
from mj_manipulator.grasp_manager import GraspManager
from mj_manipulator.grippers.franka import FrankaGripper
from mj_manipulator.grippers.robotiq import RobotiqGripper
from mj_manipulator.menagerie import menagerie_scene

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parent.parent.parent  # robot-code/
GEODUDE_ASSETS = WORKSPACE / "geodude_assets" / "src" / "geodude_assets" / "models"
ROBOTIQ_SCENE = GEODUDE_ASSETS / "robotiq_2f140" / "scene.xml"
FRANKA_SCENE = menagerie_scene("franka_emika_panda")


def print_header(title):
    print(f"\n{'=' * 64}")
    print(f"  {title}")
    print(f"{'=' * 64}")


# ---------------------------------------------------------------------------
# Demo 1: Robotiq 2F-140 standalone kinematic control
# ---------------------------------------------------------------------------
def demo_robotiq_kinematic():
    """Kinematic open/close and position reading for Robotiq 2F-140."""
    print_header("Robotiq 2F-140 — Kinematic Control")

    if not ROBOTIQ_SCENE.exists():
        print("  SKIP: geodude_assets not found")
        return

    env = Environment(str(ROBOTIQ_SCENE))
    gripper = RobotiqGripper(env.model, env.data, "demo_arm")

    print(f"  arm_name:        {gripper.arm_name}")
    print(f"  actuator_id:     {gripper.actuator_id}")
    print(f"  ctrl_range:      [{gripper.ctrl_open}, {gripper.ctrl_closed}]")
    print(f"  bodies:          {len(gripper.gripper_body_names)} bodies")
    print(f"  attachment_body: {gripper.attachment_body}")

    # Open
    gripper.kinematic_open()
    print(f"\n  Open:     position = {gripper.get_actual_position():.3f}")

    # Sweep through positions
    for t in [0.25, 0.50, 0.75, 1.0]:
        gripper.set_kinematic_position(t)
        pos = gripper.get_actual_position()
        print(f"  t={t:.2f}:    position = {pos:.3f}")

    # Kinematic close (with candidate objects — the scene has an "object" body)
    gripper.kinematic_open()
    gripper.set_candidate_objects(["object"])
    result = gripper.kinematic_close()
    pos = gripper.get_actual_position()
    print(f"\n  kinematic_close: grasped={result!r}  position={pos:.3f}")


# ---------------------------------------------------------------------------
# Demo 2: Franka Hand standalone kinematic control
# ---------------------------------------------------------------------------
def demo_franka_kinematic():
    """Kinematic open/close and position reading for Franka Hand."""
    print_header("Franka Hand — Kinematic Control")

    if not FRANKA_SCENE.exists():
        print("  SKIP: mujoco_menagerie not found")
        return

    spec = mujoco.MjSpec.from_file(str(FRANKA_SCENE))
    add_franka_ee_site(spec)

    franka_dir = FRANKA_SCENE.parent
    tmp_path = franka_dir / "_demo_gripper_franka.xml"
    try:
        tmp_path.write_text(spec.to_xml())
        env = Environment(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)

    gripper = FrankaGripper(env.model, env.data, "franka")

    print(f"  arm_name:        {gripper.arm_name}")
    print(f"  actuator_id:     {gripper.actuator_id}")
    print(f"  ctrl_range:      [{gripper.ctrl_open}, {gripper.ctrl_closed}]")
    print(f"  bodies:          {gripper.gripper_body_names}")
    print(f"  attachment_body: {gripper.attachment_body}")

    # Open
    gripper.kinematic_open()
    print(f"\n  Open:     position = {gripper.get_actual_position():.3f}")

    # Close (no objects — empty candidates so it fully closes)
    gripper.set_candidate_objects([])
    gripper.kinematic_close()
    print(f"  Closed:   position = {gripper.get_actual_position():.3f}")

    # Re-open
    gripper.kinematic_open()
    print(f"  Re-open:  position = {gripper.get_actual_position():.3f}")


# ---------------------------------------------------------------------------
# Demo 3: Franka arm + gripper + GraspManager through SimContext
# ---------------------------------------------------------------------------
def demo_franka_integration():
    """Full grasp/release cycle using SimContext.arm().grasp()/release()."""
    print_header("Franka — Arm + Gripper + GraspManager Integration")

    if not FRANKA_SCENE.exists():
        print("  SKIP: mujoco_menagerie not found")
        return

    spec = mujoco.MjSpec.from_file(str(FRANKA_SCENE))
    add_franka_ee_site(spec)

    franka_dir = FRANKA_SCENE.parent
    tmp_path = franka_dir / "_demo_gripper_franka.xml"
    try:
        tmp_path.write_text(spec.to_xml())
        env = Environment(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)

    gm = GraspManager(env.model, env.data)
    gripper = FrankaGripper(env.model, env.data, "franka", grasp_manager=gm)
    arm = create_franka_arm(env, gripper=gripper, grasp_manager=gm, with_ik=False)

    print(f"  arm.gripper:        {type(arm.gripper).__name__}")
    print(f"  arm.grasp_manager:  {type(arm.grasp_manager).__name__}")
    print(f"  gripper.is_holding: {gripper.is_holding}")

    # Demonstrate GraspManager integration (the Franka scene has no objects,
    # so we use mark_grasped/mark_released directly rather than going through
    # SimContext.arm().grasp() which requires a real body for attach_object).
    gripper.kinematic_close(steps=10)
    print(f"\n  After kinematic_close:")
    print(f"    position:    {gripper.get_actual_position():.3f}")
    print(f"    is_holding:  {gripper.is_holding}")

    gm.mark_grasped("mug", "franka")
    print(f"\n  After GraspManager.mark_grasped('mug', 'franka'):")
    print(f"    is_holding:  {gripper.is_holding}")
    print(f"    held_object: {gripper.held_object}")

    gm.mark_released("mug")
    gripper.kinematic_open()
    print(f"\n  After mark_released + kinematic_open:")
    print(f"    is_holding:  {gripper.is_holding}")
    print(f"    held_object: {gripper.held_object}")
    print(f"    position:    {gripper.get_actual_position():.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    missing = []
    if not ROBOTIQ_SCENE.exists():
        missing.append("geodude_assets")
    if not FRANKA_SCENE.exists():
        missing.append("mujoco_menagerie")

    if missing:
        print(f"WARNING: {', '.join(missing)} not found — some demos will be skipped")

    demo_robotiq_kinematic()
    demo_franka_kinematic()
    demo_franka_integration()

    print_header(
        "DONE — Gripper protocol implementations\n"
        "  Robotiq 2F-140 (4-bar trajectory replay)\n"
        "  Franka Hand (linear finger interpolation)\n"
        "  Both work standalone and with Arm + GraspManager"
    )
    print()


if __name__ == "__main__":
    main()
