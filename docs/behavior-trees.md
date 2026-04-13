# How the Robot Picks Up and Places Objects

This guide explains how mj_manipulator's behavior tree (BT) system works. You don't need to know anything about py_trees to understand it — the concepts are simple.

## The big picture

A **behavior tree** is a sequence of steps. Each step either **succeeds** or **fails**. If a step fails, the whole sequence stops and reports failure. If every step succeeds, the sequence reports success.

Here's what a pickup looks like:

```
Pick up object
 ├── Find grasps for object
 ├── Plan arm path
 ├── Smooth trajectory
 ├── Move arm
 ├── Close gripper
 └── Lift arm off table
```

And a place:

```
Place object
 ├── Find placement poses
 ├── Plan arm path
 ├── Smooth trajectory
 ├── Move arm
 └── Open gripper
```

That's it. Find poses, plan, move, grasp (or release). Every step is visible, every step can fail, and the sequence stops on the first failure.

## What happens when a step fails?

| Step | Fails when | What happens next |
|------|-----------|-------------------|
| **Find grasps for object** | No graspable objects in the scene, or no grasp templates for this object type | Sequence stops. No arm motion attempted. |
| **Find placement poses** | No valid destinations, or no placement templates | Sequence stops. Object is still held. |
| **Plan arm path** | No collision-free path exists (object unreachable, arm blocked) | Sequence stops. Arm stays where it is. |
| **Smooth trajectory** | Never fails (TOPP-RA always succeeds on a valid path) | — |
| **Move arm** | Trajectory aborted (user interrupt, teleop preemption, or object dropped mid-motion) | Arm stops wherever it reached. |
| **Close gripper** | Gripper closed on nothing (no contact with the target object) | Sequence stops. Gripper is closed but empty. |
| **Open gripper** | Never fails | — |
| **Lift arm off table** | New collision detected during retraction | Arm stops mid-lift. Object may still be held. |

When the sequence fails, `robot.pickup()` / `robot.place()` runs recovery automatically: release the gripper, go home. The workspace is clean for the next command.

## Geodude's version

Geodude has a Vention linear base that lifts the whole arm vertically. Instead of lifting the arm's end-effector (which is what fixed-base arms like Franka do), Geodude lifts the base:

```
Geodude pickup
 ├── Pick up object
 │    ├── Find grasps for object
 │    ├── Plan arm path
 │    ├── Smooth trajectory
 │    ├── Move arm
 │    └── Close gripper
 └── Lift base to clear table
```

For bimanual operation, Geodude tries the right arm first, then falls back to the left:

```
Try both arms
 ├── Geodude pickup (right arm)
 └── Geodude pickup (left arm)
```

## Three ways to use this

### Level 1: Just call `robot.pickup()`

The simplest way. Handles everything — arm selection, grasp generation, planning, grasping, lifting, recovery on failure.

```python
robot.pickup("can")          # pick up any can
robot.place("recycle_bin")   # place it in the bin
```

Here's the full recycling demo — sort every object into a bin:

```python
def sort_all():
    while robot.pickup():
        robot.place("recycle_bin")
        robot.go_home()        # reset arm + base between cycles
    robot.go_home()
```

This is four lines because `robot.pickup()` and `robot.place()` handle
two layers of complexity internally:

**Inside the BT** (auto-generated, see tree diagrams above): find
grasps, plan, move, grasp/release. This is the action sequence — it
either succeeds or fails, and the BT stops on the first failure.

**Outside the BT** (Python control flow in the primitives layer):
- Try the right arm. If the BT fails, try the left arm.
- If both arms fail, release grippers, go home, return False.
- If pickup succeeds but place fails, release, go home, return False.
- `go_home()` between cycles resets the arm and base height.

The BT diagrams show what happens inside each `pickup()` / `place()`
call. The `sort_all()` code shows the loop, the arm fallback, and the
recovery. Both are sources of truth for their respective layers.

### Level 2: Use the pre-built subtrees

Two functions: `pickup(ns)` and `place(ns)`. Each builds a complete action sequence.

```python
from mj_manipulator.bt import pickup, place

tree = pickup("/right")   # the full sequence shown above
tree = place("/left")     # find placements → plan → move → release
```

### Level 3: Compose your own tree from individual nodes

Full control. You pick which nodes to use and in what order:

```python
from mj_manipulator.bt import PlanToTSRs, Retime, Execute, Grasp, GenerateGrasps
import py_trees

ns = "/right"

my_pickup = py_trees.composites.Sequence("My custom pickup", memory=True, children=[
    GenerateGrasps(ns=ns, name="Find grasps"),
    PlanToTSRs(ns=ns, tsrs_key="grasp_tsrs", name="Plan"),
    Retime(ns=ns, name="Smooth"),
    Execute(ns=ns, name="Move"),
    Grasp(ns=ns, name="Grasp"),
    # Add your custom step here:
    # WaveSadly(ns=ns, name="Celebrate"),
])
```

At this level you need to populate the blackboard yourself — see the **Blackboard reference** below.

## Recovery

The subtrees don't include recovery. If any step fails, the sequence stops and returns FAILURE. Recovery is handled by `robot.pickup()` / `robot.place()`:

1. Release the gripper (open it)
2. Go home (retract up if needed, plan to ready pose, move there)

When composing your own tree at Level 3, you handle failure however you like:

```python
if not tick_tree(my_tree):
    wave_sadly()
    robot.go_home()
```

## Available nodes

### Action nodes (used by the default subtrees)

| Node | What it does | Fails when |
|------|-------------|-----------|
| `GenerateGrasps` | Find candidate grasp poses for the target object | No objects or no grasp templates |
| `GeneratePlaceTSRs` | Find candidate placement poses for the destination | No destinations or no placement templates |
| `PlanToTSRs` | Find a collision-free arm path to a grasp or place pose | No reachable pose exists |
| `Retime` | Convert the path into a smooth, velocity-limited trajectory | Never |
| `Execute` | Move the arm along the trajectory | Motion aborted |
| `Grasp` | Close the gripper on the target object | Gripper closed on nothing |
| `Release` | Open the gripper | Never |
| `SafeRetract` | Lift the arm upward, stopping on new collisions | Never (aborts gracefully) |

### Building-block nodes (for custom trees)

| Node | What it does | Fails when |
|------|-------------|-----------|
| `PlanToConfig` | Plan a path to a specific joint configuration | No collision-free path |
| `CartesianMove` | Move end-effector along a direction using velocity control | Never |
| `CheckNotNearConfig` | Check if arm is far from a target configuration | Arm is already near target |
| `Sync` | Flush physics state and sync the viewer | Never |

## Blackboard reference

The blackboard is a shared key-value store that nodes read from and write to. Keys are prefixed with a namespace (`/left`, `/right`) for multi-arm support.

### Keys you set up before running the tree

| Key | Type | Description |
|-----|------|-------------|
| `/context` | ExecutionContext | The sim or hardware context |
| `{ns}/arm` | Arm | Arm instance |
| `{ns}/arm_name` | str | Arm name (e.g. "left") |
| `{ns}/timeout` | float | Planning timeout in seconds |
| `{ns}/object_name` | str or None | What to pick up (None = anything) |
| `{ns}/destination` | str or None | Where to place (None = anywhere valid) |
| `{ns}/grasp_source` | GraspSource | Provides grasp/place poses for objects |
| `{ns}/hand_type` | str | Gripper type (e.g. "robotiq") |

### Keys produced by the tree

| Key | Type | Written by | Description |
|-----|------|-----------|-------------|
| `{ns}/grasp_tsrs` | list[TSR] | GenerateGrasps | Candidate grasp poses |
| `{ns}/place_tsrs` | list[TSR] | GeneratePlaceTSRs | Candidate placement poses |
| `{ns}/path` | list[ndarray] | PlanToTSRs | Planned joint-space path |
| `{ns}/trajectory` | Trajectory | Retime | Time-parameterized trajectory |
| `{ns}/grasped` | str or None | Grasp | Name of grasped object |
| `{ns}/goal_tsr_index` | int | PlanToTSRs | Which pose the planner reached |
| `{ns}/tsr_to_object` | list[str] | GenerateGrasps | Maps pose index → object name |

## Generating tree diagrams

```python
import py_trees
from mj_manipulator.bt import pickup

tree = pickup("/arm")

# ASCII (terminals and docs):
print(py_trees.display.ascii_tree(tree))

# SVG (visual documentation):
py_trees.display.dot_tree(tree).write("pickup.svg", format="svg")
```

Pre-generated SVGs are in `docs/*.svg`.
