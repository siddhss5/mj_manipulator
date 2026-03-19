# Grasp-Aware Collision Detection

Real manipulation requires treating a grasped object as part of the robot: the gripper contacting the can is expected, but the forearm colliding with it is not. This document explains how `collision.py` handles this with software-based contact filtering.

## The Problem

When a robot grasps an object, collision detection faces a dilemma:

```
Before Grasp:                    After Grasp:

    Gripper                          Gripper
      |                                |
      v                                v
    [   ]                            [CAN]  ← object now "part of" robot
                                       |
                                       v
    [CAN]                           arm links below
      |
      v
    Table
```

Without special handling, the grasped object is in permanent contact with the gripper fingers, causing every configuration to appear invalid during motion planning.

**The naive fix** — ignoring all collisions with the grasped object — is dangerous: we still need to detect collisions between the object and the environment (tables, bins, other obstacles).

## The Solution: Software Contact Filtering

Rather than manipulating MuJoCo collision groups (fragile, hard to debug), `CollisionChecker` lets MuJoCo generate all contacts, then filters them in software using grasp state:

```
1. Set robot configuration
                ↓
2. Move attached objects with gripper (kinematic attachment)
                ↓
3. mj_forward() → MuJoCo generates ALL contacts
                ↓
4. _count_invalid_contacts(): filter in software
   ┌─────────────────────────────────────────────────┐
   │  Gripper ↔ Grasped Object    →  ALLOWED         │
   │  Arm Link ↔ Grasped Object   →  INVALID         │
   │  Grasped Object ↔ Env        →  INVALID         │
   │  Arm ↔ Environment           →  INVALID         │
   │  Pure env-env contacts       →  ignored          │
   └─────────────────────────────────────────────────┘
                ↓
5. Return: invalid_contacts == 0
```

This keeps collision group configuration constant — no state to synchronize.

## Two Modes

`CollisionChecker` supports two operating modes to handle single-threaded and parallel planning:

### Live Mode (single-threaded planning)

Pass a `GraspManager` at construction. The checker reads live grasp state and creates an internal `MjData` copy to avoid viewer flickering:

```python
checker = CollisionChecker(
    model, data, joint_names,
    grasp_manager=grasp_manager,   # reads live state
)
```

### Snapshot Mode (parallel/multi-threaded planning)

Pass frozen grasp state instead. Each `CollisionChecker` instance owns its `MjData`, enabling safe parallel use:

```python
checker = CollisionChecker(
    model, private_data, joint_names,
    grasped_objects=frozenset({("can_0", "right")}),
    attachments={"can_0": ("gripper/right_pad", T_gripper_object)},
)
```

### Simple Mode (no grasp awareness)

Pass neither — any contact involving the arm is a collision:

```python
checker = CollisionChecker(model, data, joint_names)
```

## The GraspManager

`GraspManager` tracks which objects are grasped and maintains kinematic attachments so objects move with the gripper during planning.

### State Tracking

```python
grasp_manager.mark_grasped("can_0", arm="right")   # record grasp
grasp_manager.is_grasped("can_0")                  # → True
grasp_manager.get_holder("can_0")                  # → "right"
grasp_manager.mark_released("can_0")               # release
```

### Kinematic Attachments

For kinematic (non-physics) execution, grasped objects don't move automatically with the gripper. `attach_object` computes the relative transform and stores it:

```python
# After closing gripper:
grasp_manager.attach_object("can_0", "gripper/right_pad")

# Before each collision check:
grasp_manager.update_attached_poses(temp_data)
```

`update_attached_poses` propagates `T_world_gripper @ T_gripper_object` to update the object's freejoint qpos.

### Gripper Contact Filtering

A contact between a grasped object and an arm body is only allowed if the arm body belongs to the holding gripper. `CollisionChecker._is_gripper_object_contact` checks this by:

1. Looking up the attachment body via `get_attachment_body(object_name)`
2. Finding the gripper base body (`*/base`) to cover all finger bodies
3. Checking if the contacting arm body is a descendant of the gripper base

This allows contacts with fingers, pads, and other gripper parts while still flagging arm-link ↔ object collisions.

## Complete Grasp Lifecycle

```
1. APPROACH
   Object not grasped — planner avoids it as an obstacle.

2. GRASP DETECTED (gripper closes, contacts detected via detect_grasped_object)
   grasp_manager.mark_grasped("can_0", "right")
   grasp_manager.attach_object("can_0", "gripper/right_pad")

3. MANIPULATION (lift, move, place planning)
   Collision checker:
     ✓ allows gripper ↔ can contacts
     ✗ rejects arm ↔ can contacts
     ✗ rejects can ↔ environment contacts
   update_attached_poses() keeps can with gripper

4. RELEASE (gripper opens at target pose)
   grasp_manager.detach_object("can_0")
   grasp_manager.mark_released("can_0")
   Object returns to normal collision rules.
```

## Detecting a Grasp

`detect_grasped_object` checks MuJoCo contacts to find which object the gripper is holding. With `require_bilateral=True` (default), both finger groups must be in contact:

```python
from mj_manipulator.grasp_manager import detect_grasped_object

grasped = detect_grasped_object(
    model, data,
    gripper_body_names=arm.gripper.body_names,
    candidate_objects=["can_0", "can_1", "can_2"],
    require_bilateral=True,
)
# Returns "can_0" or None
```

Bilateral detection uses body name conventions (`/left_*`, `/right_*`) by default, or an explicit `finger_groups` dict for grippers with other naming.

## Reactive Cartesian Control

`is_arm_in_collision` (distinct from `is_valid`) checks for unexpected arm-link contacts during live execution. It allows gripper-object and grasped-object-environment contacts (expected during manipulation) but flags arm-link-environment contacts (forearm hitting the table):

```python
if checker.is_arm_in_collision(min_penetration=0.005):
    # Stop cartesian motion
```

## Why Not Collision Groups?

| Approach | Pros | Cons |
|----------|------|------|
| **MuJoCo `contype`/`conaffinity`** | Native filtering, no overhead | Complex state management; groups can desync; hard to debug |
| **Software filtering** (this approach) | Simple, explicit, debuggable | Negligible extra overhead processing all contacts |

Software filtering wins because:
1. **No state to manage** — collision groups stay constant; filtering logic is in one function
2. **Debuggable** — `checker.debug_contacts(q)` prints exactly why each contact was allowed or rejected
3. **Thread-safe** — snapshot mode freezes state at construction; no shared mutable state during parallel planning

## Further Reading

- [cartesian-control.md](cartesian-control.md) — QP-based Cartesian velocity control using the collision checker
- [collision.py](../src/mj_manipulator/collision.py) — `CollisionChecker` with live and snapshot modes
- [grasp_manager.py](../src/mj_manipulator/grasp_manager.py) — grasp state and kinematic attachments
- [contacts.py](../src/mj_manipulator/contacts.py) — `iter_contacts` iterator used throughout
