# mj_manipulator

Generic MuJoCo manipulator control: planning, execution, grasping, and cartesian control for any robot arm.

## Supported Robots

Pre-built arm factories in `mj_manipulator.arms`:

- **UR5e** (6-DOF) — `create_ur5e_arm(env)`
- **Franka Emika Panda** (7-DOF) — `create_franka_arm(env)`
- **KUKA LBR iiwa 14** (7-DOF) — `create_iiwa14_arm(env)`

Bundled demos for Franka (with its built-in hand) and iiwa 14 (with a menagerie Robotiq 2F-85 attached). See [Scenarios and Demos](#scenarios-and-demos) to run them, [Adding a New Arm](#adding-a-new-arm) to bring your own.

## Installation

```bash
uv add mj-manipulator
```

For development:
```bash
git clone <repo-url> mj_manipulator
cd mj_manipulator
uv sync --extra dev
uv run pytest tests/ -v
```

## Quick Start

```python
from mj_environment import Environment
from mj_manipulator.arms.ur5e import create_ur5e_arm

env = Environment("path/to/ur5e/scene.xml")
arm = create_ur5e_arm(env)

# Plan to a joint configuration
path = arm.plan_to_configuration(q_goal)

# Plan to an end-effector pose (planner handles IK via TSRs)
path = arm.plan_to_pose(target_pose)

# Time-parameterize any path with TOPP-RA
traj = arm.retime(path)
print(f"Duration: {traj.duration:.2f}s, {len(traj.positions)} samples")
```

For 7-DOF arms like Franka (menagerie model needs an EE site added):
```python
import mujoco
from mj_manipulator.arms.franka import create_franka_arm, add_franka_ee_site

spec = mujoco.MjSpec.from_file("path/to/franka/scene.xml")
add_franka_ee_site(spec)
# Save XML, create Environment, then:
arm = create_franka_arm(env)
```

## Planning API

All planning methods are thin pass-throughs to the underlying planner ([pycbirrt](https://github.com/personalrobotics/pycbirrt)). If the planner supports a goal type natively, we delegate directly.

| Method | Goal type | What the planner receives |
|---|---|---|
| `plan_to_configuration(q)` | Single config | `goal=q` |
| `plan_to_configurations(qs)` | Multiple configs | `goal=qs` |
| `plan_to_pose(pose)` | EE pose | `goal_tsrs=[point_tsr]` — planner does IK |
| `plan_to_poses(poses)` | Multiple poses | `goal_tsrs=[point_tsr, ...]` — union |
| `plan_to_tsrs(goal_tsrs)` | TSR regions | `goal_tsrs=goal_tsrs` |

All methods accept optional `constraint_tsrs` for trajectory-wide path constraints, and `timeout` defaults come from `ArmConfig.planning_defaults`.

Any path can be time-parameterized with `arm.retime(path)`, which uses TOPP-RA with the arm's kinematic limits.

## IK Solver

Uses [EAIK](https://github.com/Jonte-Raab/EAIK) for analytical inverse kinematics. The `MuJoCoEAIKSolver` extracts joint axes (H) and position offsets (P) directly from the MuJoCo model — no DH parameters or frame calibration needed.

- **6-DOF** (e.g. UR5e): Direct analytical solve
- **7-DOF** (e.g. Franka): Lock one joint, discretize over its range, solve 6-DOF IK at each value

## Cartesian Control

Real-time Cartesian control moves the end-effector along a desired twist while respecting joint position and velocity limits. Rather than pseudoinverse + post-hoc clamping (which distorts motion direction), we solve a constrained QP at each timestep:

```math
min  ½ ‖J q̇ − v_d‖²_W + (λ/2)‖q̇‖²    s.t.  ℓ ≤ q̇ ≤ u
```

The bounds `ℓ`, `u` encode both velocity limits and position limits converted to velocity constraints, so joint limits are never violated. Singularities are handled implicitly by the damping term `λI`.

```python
from mj_manipulator import CartesianController

controller = CartesianController.from_arm(arm)

# Teleop: call from your 125 Hz control loop
result = controller.step(twist=np.array([0.05, 0, 0, 0, 0, 0]), dt=0.008)
print(result.achieved_fraction)   # 1.0 = full twist achieved
print(result.limiting_factor)     # None / "joint_limit" / "velocity"

# Small Cartesian plans: approach 5 cm along -z, stop on contact
result = controller.move(
    twist=np.array([0, 0, -0.05, 0, 0, 0]),
    dt=0.008, max_distance=0.05,
    stop_condition=lambda: checker.is_arm_in_collision(),
)

# Move to a target pose
result = controller.move_to(target_pose, dt=0.008, speed=0.05)
```

See [docs/cartesian-control.md](docs/cartesian-control.md) for the full derivation including twist weighting, projected gradient descent solver, convergence analysis, and comparison with MoveIt Servo.

## Grasp-Aware Collision

During manipulation, a grasped object must be treated as part of the robot: gripper-to-object contacts are expected, but arm-to-object and object-to-environment contacts indicate a collision. `CollisionChecker` handles this with software contact filtering — no MuJoCo collision group changes needed.

```python
from mj_manipulator import CollisionChecker, GraspManager

grasp_manager = GraspManager(model, data)

# After closing the gripper on a target (normal flow uses ctx.arm().grasp()):
grasp_manager.mark_grasped("can_0", arm="right")
grasp_manager.attach_object("can_0", "gripper/right_pad")

# Collision checker uses grasp state automatically:
checker = CollisionChecker(model, data, joint_names, grasp_manager=grasp_manager)
checker.is_valid(q)   # allows gripper↔can; rejects arm↔can, can↔env
```

See [docs/grasp-aware-collision.md](docs/grasp-aware-collision.md) for the filtering logic, live vs snapshot modes for parallel planning, and the complete grasp lifecycle.

## Force/Torque Sensing

Arms can expose a wrist F/T sensor via `ArmConfig`. In physics mode, MuJoCo populates `data.sensordata` each step with force and torque readings (with configurable noise).

```python
config = ArmConfig(
    ...,
    ft_force_sensor="ur5e/ft_sensor_force",
    ft_torque_sensor="ur5e/ft_sensor_torque",
)
arm = Arm(env, config)

# In a control loop:
wrench = arm.get_ft_wrench()  # [fx, fy, fz, tx, ty, tz]
if np.linalg.norm(wrench[:3]) > 10.0:
    print("Contact detected!")
```

## Teleop

Interactive control from a browser (SE(3) gizmo), VR controller, or joystick. Two input paths feed the same controller:

```python
from mj_manipulator.teleop import TeleopController, SafetyMode

controller = TeleopController(arm, ctx)
controller.activate()

# Pose input (browser gizmo, VR controller):
controller.set_target_pose(target_4x4)  # IK → step_cartesian

# Twist input (joystick, SpaceMouse):
controller.set_target_twist(np.array([0.05, 0, 0, 0, 0, 0]))  # CartesianController → step_cartesian

# In your control loop (~30 Hz):
state = controller.step()  # TRACKING, TRACKING_COLLISION, UNREACHABLE, or IDLE

# Safety modes
controller.safety_mode = SafetyMode.ALLOW   # move + flag collisions (default)
controller.safety_mode = SafetyMode.REJECT  # block colliding configs (real robot)

# Recording for ML data collection
controller.start_recording()
# ... teleop session ...
frames = controller.stop_recording()  # list of TeleopFrame
```

Thread-safe input methods for device callbacks. Works with SimContext (kinematic/physics) and HardwareContext (real robot).

## Architecture

```
mj_environment  →  mj_manipulator  →  geodude (UR5e + Robotiq)
                        │
                        ├── arms/          Arm factories + EAIK IK solver
                        ├── arm.py         Generic Arm class
                        ├── config.py      ArmConfig, KinematicLimits, PlanningDefaults
                        ├── protocols.py   IKSolver, Gripper, ExecutionContext contracts
                        ├── collision.py   Collision checking
                        ├── trajectory.py  Trajectory + TOPP-RA retiming
                        ├── executor.py    Kinematic/Physics executors
                        ├── cartesian.py   Cartesian (twist) control
                        ├── teleop.py      Teleop controller (pose + twist inputs)
                        └── grasp_manager.py  Grasp state tracking
```

Robot-specific code (joint names, limits, IK config) lives in `arms/<robot>.py`. The generic layer (`Arm`, protocols, executors) knows nothing about specific robots.

## Non-Arm Entities (Bases, Linear Actuators)

SimContext supports arbitrary controllable entities alongside arms. An entity is any object with `joint_qpos_indices`, `joint_qvel_indices`, `actuator_ids` (lists of ints), and optionally a `grasp_manager`. This is used for linear bases, mobile bases, or any actuated DOF that isn't an arm.

```python
# Entity must expose these properties:
# - joint_qpos_indices: list[int]
# - joint_qvel_indices: list[int]
# - actuator_ids: list[int]
# - grasp_manager (optional): for tracking attached objects

with SimContext(model, data, arms, entities={"left_base": base}) as ctx:
    # Base trajectories execute through the same path as arm trajectories
    base_traj = base.plan_to(0.3)
    ctx.execute(base_traj)  # physics continues during base motion

    # PlanResult can include both base and arm trajectories
    # (base executes first, then arm)
    ctx.execute(plan_result)
```

In physics mode, entity actuators are controlled alongside arm actuators each step — no actuator is left uncontrolled. In kinematic mode, entities use KinematicExecutor (same as arms).

## Adding a New Arm

`arms/ur5e.py` (6-DOF), `arms/franka.py` (7-DOF), and `arms/iiwa14.py` (7-DOF) are the complete references. For a runnable demo built on top of your arm, read `demos/franka_setup.py` (hand-integrated) or `demos/iiwa14_setup.py` (attach your own gripper).

### Step 0 — gather constants from the XML

Open your robot's MuJoCo XML and note:

- **Joint names** (e.g. `joint1 … joint7`) and their **limits** (`<joint range="...">`)
- **Home keyframe** if the model has one (`<keyframe><key name="home" qpos="..."/></keyframe>`)
- **Tip body name** — the last body in the kinematic chain (often `link7`, `flange`, `wrist_3_link`, `attachment_link`)
- Whether the model already has a named EE site on the tip body

You'll also need from the **manufacturer datasheet**:

- Joint **velocity limits** (rad/s per joint)
- Joint **acceleration limits** (rad/s² — often estimated; most datasheets don't publish them)

### Step 1 — add an EE site (if needed)

If the model doesn't have a named site on the tip body, add one via `MjSpec` before compiling.

```python
import mujoco

def add_my_robot_ee_site(spec: mujoco.MjSpec, site_name: str = "grasp_site") -> None:
    # Use spec.body(name) — it walks the whole tree. worldbody.find_child
    # only searches direct children and misses deeply-nested tip bodies.
    tip = spec.body("link7")   # adjust for your tip body name
    if tip is None:
        raise RuntimeError(f"No body named 'link7' in this spec")
    site = tip.add_site()
    site.name = site_name
    site.pos = [0.0, 0.0, 0.0]   # at flange; adjust if you want a TCP offset
```

See `add_franka_ee_site()` or `add_iiwa14_ee_site()` for full examples.

### Step 2 — (7-DOF only) find the EAIK locked joint

EAIK solves 6-DOF analytically. For 7-DOF, we lock one joint and discretize. Discover which joint to lock with a one-off script:

```python
from mj_manipulator.arm import Arm
from mj_manipulator.arms.eaik_solver import _extract_hp, find_locked_joint_index
from mj_manipulator.config import ArmConfig, KinematicLimits
import numpy as np

# Create a bare config just to extract H/P
config = ArmConfig(
    name="my_robot", entity_type="arm",
    joint_names=MY_ROBOT_JOINT_NAMES,
    kinematic_limits=KinematicLimits(velocity=np.ones(7), acceleration=np.ones(7)),
    ee_site="grasp_site",
)
arm = Arm(env, config)  # no IK yet
first_joint_body = env.model.jnt_bodyid[arm.joint_ids[0]]
base_body_id     = env.model.body_parentid[first_joint_body]
H, P, _ = _extract_hp(env.model, env.data, list(arm.joint_ids),
                      arm.joint_qpos_indices, arm.ee_site_id, base_body_id)
print(find_locked_joint_index(H, P))
```

**What if it returns `None`?** Your arm's kinematics don't fit any known EAIK family (no spherical wrist, no spherical base, no parallel pairs after locking). Options:

- Some arms (e.g. Kinova Gen3) are theoretically spherical but their MuJoCo XML has small residual offsets from URDF conversion. EAIK uses exact-equality checks, so a few mm of offset on the "wrong" axis breaks decomposition. You may be able to snap near-zero offsets to zero — see #126 for the discussion.
- Some arms (e.g. Flexiv Rizon 4) are legitimately non-analytical — joints don't line up in classic ways by design. You'll need numerical IK, which mj_manipulator doesn't ship today (see #127). Pick a different arm or wait for numerical IK support.

### Step 3 — create `arms/<robot>.py`

```python
import numpy as np
from mj_manipulator.arm import Arm
from mj_manipulator.arms.eaik_solver import MuJoCoEAIKSolver
from mj_manipulator.config import ArmConfig, KinematicLimits

MY_ROBOT_JOINT_NAMES = ["joint1", "joint2", ...]
MY_ROBOT_HOME        = np.array([0.0, 0.0, ...])
MY_ROBOT_VEL_LIMITS  = np.array([...]) * 0.5   # datasheet values, halved
MY_ROBOT_ACC_LIMITS  = np.array([...]) * 0.5
_MY_ROBOT_LOCKED_JOINT = 0  # from step 2 (7-DOF only; omit for 6-DOF)


def add_my_robot_gravcomp(spec):
    """Enable gravity compensation on the arm's kinematic subtree."""
    from mj_manipulator.arm import add_subtree_gravcomp
    add_subtree_gravcomp(spec, "base")   # adjust root body name


def create_my_robot_arm(env, *, ee_site="grasp_site", with_ik=True,
                       tcp_offset=None, gripper=None, grasp_manager=None):
    config = ArmConfig(
        name="my_robot", entity_type="arm",
        joint_names=list(MY_ROBOT_JOINT_NAMES),
        kinematic_limits=KinematicLimits(
            velocity=MY_ROBOT_VEL_LIMITS.copy(),
            acceleration=MY_ROBOT_ACC_LIMITS.copy(),
        ),
        ee_site=ee_site,
        tcp_offset=tcp_offset,
    )
    if not with_ik:
        return Arm(env, config, gripper=gripper, grasp_manager=grasp_manager)

    arm = Arm(env, config)
    first_joint_body = env.model.jnt_bodyid[arm.joint_ids[0]]
    base_body_id = env.model.body_parentid[first_joint_body]
    ik_solver = MuJoCoEAIKSolver(
        model=env.model, data=env.data,
        joint_ids=list(arm.joint_ids),
        joint_qpos_indices=arm.joint_qpos_indices,
        ee_site_id=arm.ee_site_id,
        base_body_id=base_body_id,
        joint_limits=arm.get_joint_limits(),
        fixed_joint_index=_MY_ROBOT_LOCKED_JOINT,   # 7-DOF only
    )
    return Arm(env, config, ik_solver=ik_solver, gripper=gripper, grasp_manager=grasp_manager)
```

### Step 4 — add tests

Copy `TestIIWA14Factory` / `TestIIWA14IK` / `TestAddIIWA14EeSite` from `tests/test_arms.py`. They verify: factory creates a valid `Arm`, FK-IK round-trip is accurate, solutions are within joint limits, the site helper adds to the right body.

### Step 5 — re-export

```python
# arms/__init__.py
from mj_manipulator.arms.my_robot import (
    create_my_robot_arm,
    add_my_robot_ee_site,
    add_my_robot_gravcomp,
)
```

### Step 6 — build a runnable demo (optional but recommended)

See `demos/iiwa14_setup.py` for the copy-template. Six steps:

1. Load your robot's menagerie scene (`menagerie_scene("my_robot")`)
2. Call your `add_my_robot_ee_site(spec)` and `add_my_robot_gravcomp(spec)`
3. Optionally add a worktop plate to the spec (for objects to sit on)
4. Optionally attach a gripper via `spec.attach(gripper_spec, prefix="gripper/", site=...)`.
   Menagerie ships the Robotiq 2F-85 (`<menagerie>/robotiq_2f85/2f85.xml`) which works
   with `mj_manipulator.grippers.RobotiqGripper` out of the box.
5. Build the environment and arm
6. Wrap in a `RobotBase` subclass that implements `get_worktop_pose()` and `reset(scene)`

Then wire the demo into `cli.py`'s `_ROBOTS` dict:

```python
_ROBOTS = {
    "franka": ("Franka", "mj_manipulator.demos.franka_setup", "build_franka_robot"),
    "iiwa14": ("iiwa14", "mj_manipulator.demos.iiwa14_setup", "build_iiwa14_robot"),
    "my_robot": ("MyRobot", "mj_manipulator.demos.my_robot_setup", "build_my_robot_robot"),
}
```

`python -m mj_manipulator --robot my_robot --scenario recycling` then runs the existing recycling scenario on your new arm — a portability test for your setup.

### Notes on grippers

If you attach a menagerie gripper and it drops objects during transit, check:

- **Position-spring vs constant-force actuator**: many menagerie grippers ship as position-springs where grip force weakens as fingers close (a real problem for compliant objects). See `fix_franka_grip_force` in `arms/franka.py` and `fix_robotiq_grip_force` in `grippers/robotiq.py` for how to rewrite the actuator.
- **Finger self-collision**: menagerie grippers often let the fingers interpenetrate on empty close, which breaks motion planning ("start in collision"). Exclude the finger↔finger pair via `spec.add_exclude()`. See `add_franka_finger_exclude` for the pattern.
- **Pad friction**: raise sliding friction to ~1.5 and torsional to ~0.05. See `add_franka_pad_friction`.

For adding a new gripper from scratch, see the next section.

## Adding a New Gripper

`grippers/franka.py` (integrated hand) and `grippers/robotiq.py` (attached 2F-85 or 2F-140) are the complete references. The iiwa 14 demo (`demos/iiwa14_setup.py`) shows how to attach a menagerie gripper to a bare arm via `MjSpec`. This section documents the full workflow the way we walked it for the 2F-85, including what goes wrong and how to diagnose it.

For the *why* behind the rules in this section — the math that makes the palm-placement rule matter, the affine-force equation behind the grip-dropout bug, and the empirical finding that pad area beats grip force for cylindrical objects — see **[`docs/grippers.md`](docs/grippers.md)**. Read the deep-dive once; refer back when debugging.

### The TSR frame convention (read this first)

Every parallel-jaw grasp in this codebase is computed relative to a canonical **ee_site** frame:

- **z** = approach direction (points from palm toward object)
- **y** = finger-opening axis (direction fingers separate)
- **x** = palm normal (right-hand rule: x = y × z)

The TSR library's `ParallelJawGripper.grasp_cylinder_side` / `grasp_box_face_*` / etc. generate poses where the ee_site sits at the **palm**, with **fingers extending forward by `FINGER_LENGTH`**. The critical assumption is that *everything on the gripper except the fingers is behind the palm* — if part of the housing extends past the palm along the approach axis, TSR grasps will try to drive the housing into the object.

If the gripper's XML doesn't have a site at the palm with this orientation, you add one via `MjSpec` before compiling.

### Step 0 — gather constants from the XML

Open the gripper's MuJoCo XML and note:

- **Mounting body name** — the root of the gripper's kinematic tree, the body where `spec.attach(...)` will graft it on. For a Robotiq, this is `base_mount`. For the Franka hand, it's `hand`.
- **Actuator name** and its `forcerange` / `gainprm` / `biasprm`. Menagerie grippers are often position actuators with a bias that couples grip force to finger gap — see Step 5 for the fix.
- **Joint names** for the finger mechanism (driver, coupler, follower, spring_link on Robotiq; finger_joint1/2 on Franka) and their ranges. Note whether qpos=0 is fully open or fully closed.
- **Pad/fingertip collision geoms** — where contact with the object happens.

### Step 1 — visualize + measure the gripper

Before writing any code, eyeball the gripper geometry against a test object:

```bash
uv run python scripts/visualize_grasps.py --gripper <name> --object can
```

You'll need to register the gripper in `scripts/visualize_grasps.py`'s `GRIPPERS` dict first. Start with a reasonable guess for `grasp_site_pos` / `grasp_site_quat` / `hand_type`. The viz teleports a bare gripper through TSR-sampled pre-grasp poses — you can see immediately whether the palm is at the right radial distance, whether fingers straddle the object, and (via the Collision panel) whether the housing intrudes on the object.

Then automate the check:

```bash
uv run python scripts/validate_gripper.py --gripper <name>
```

`validate_gripper.py` walks each collision geom's AABB along the approach axis, runs a deterministic 50-samples × N-templates sweep, and prints specific diagnostic messages when something's off ("housing extends 94 mm past grasp_site — move grasp_site forward by that amount").

### Step 2 — create a TSR `ParallelJawGripper` subclass

Add a class to `tsr/src/tsr/hands/parallel_jaw.py`:

```python
class MyGripper(ParallelJawGripper):
    """One-line description with the XML source."""

    FINGER_LENGTH = 0.059   # meters, palm → fingertip pad center
    MAX_APERTURE = 0.085    # meters, pad INNER face ↔ inner face at full open

    # Distance from the gripper's mounting-body origin to the TSR palm
    # along the approach axis. Callers placing a grasp_site in an
    # MjSpec should offset it by this value so the arm's ee_site lands
    # on the TSR palm, not inside the housing.
    PALM_OFFSET_FROM_MOUNT = 0.094

    def __init__(self):
        super().__init__(
            finger_length=self.FINGER_LENGTH,
            max_aperture=self.MAX_APERTURE,
        )
```

Critical details drawn from the 2F-85 experience:

- **`FINGER_LENGTH`** is palm-to-fingertip, where "palm" = the forward-most point of the *non-finger* structure. **Not** the distance from the mounting plate to the fingertip. If the gripper has a chunky housing, FL does not include the housing depth.
- **`MAX_APERTURE`** is the **inner-face** gap when fully open (i.e., the widest object that fits *between* the pads), not the outer-face distance. Using the outer distance silently allows TSR to propose grasps for objects that won't actually fit.
- **`PALM_OFFSET_FROM_MOUNT`** is a new constant we use to record the shift between the gripper's XML root (`base_mount`, `hand`, etc.) and the TSR palm. Callers read this when placing `grasp_site` so the value only lives in one place.

Register the class in `tsr/src/tsr/hands/__init__.py` and `tsr/src/tsr/hands/registry.py`.

Then extend `_get_hand` in `mj_manipulator/src/mj_manipulator/grasp_sources/prl_assets.py` so a `hand_type` string resolves to your class.

### Step 3 — place `grasp_site` in MjSpec

In your setup module (e.g. `demos/my_robot_setup.py`), after loading the gripper spec but before attaching it, add a `grasp_site` at `mounting_body + PALM_OFFSET_FROM_MOUNT` along the approach axis, oriented to match the TSR convention.

```python
from tsr.hands import MyGripper

gripper_spec = mujoco.MjSpec.from_file(str(gripper_xml))
base = gripper_spec.body("base_mount")   # or "hand", etc.
site = base.add_site()
site.name = "grasp_site"
site.pos = [0.0, 0.0, MyGripper.PALM_OFFSET_FROM_MOUNT]
site.quat = [0.7071, 0.0, 0.0, -0.7071]   # -90° about z for Robotiq-style
```

The rotation depends on the gripper's internal frame. Robotiq-family grippers in the menagerie have their fingers opening along the *x*-axis of `base_mount` — the −90° z-rotation maps TSR's expected +y opening axis to the gripper's actual +x. Franka's hand body already has z=approach, y=opening, so it needs identity rotation. When in doubt, open the viz (Step 1) — if the fingers don't visibly open along the "finger-opening" axis you expect, fix the rotation.

### Step 4 — gripper class + kinematic trajectory

For a Robotiq variant (2F-85 or similar), reuse the existing `RobotiqGripper` class and supply a variant-specific kinematic trajectory:

```bash
uv run python scripts/record_gripper_trajectory.py \
    --gripper-xml path/to/your_gripper.xml \
    --actuator-name fingers_actuator \
    --ctrl-open 0 --ctrl-closed 255 \
    --joints left_driver_joint right_driver_joint ...
```

Paste the resulting array into `mj_manipulator/src/mj_manipulator/grippers/_<variant>_trajectory.py` and pass it to `RobotiqGripper(..., trajectory=..., hand_type_override="<your_hand_type>")`.

For a non-Robotiq parallel jaw, subclass `_BaseGripper` and implement the gripper protocol. See `grippers/franka.py` for the shape.

### Step 5 — fix the grip force

Most menagerie grippers ship with a position actuator whose generalized force couples to finger gap:

```
force = gain[0] * ctrl + bias[0] + bias[1] * length + bias[2] * vel
```

For both the 2F-85 and the Franka hand in menagerie, `bias[1]` is large and negative, which means grip force → 0 exactly when the fingers reach the closed stop. If an object slips through, the gripper relaxes instead of re-closing. Symptoms: objects drop during transport, or empty-close states never settle.

The fix rewrites the actuator to produce a constant force:

- **Robotiq**: `fix_robotiq_grip_force(model, prefix="gripper/", target_tendon_force=15.0)`  
  Default 15 N tendon yields ~200–300 N pad force on the 2F-85 (middle of the real Robotiq 20–235 N spec).
- **Franka**: `fix_franka_grip_force(model, target_force=70.0)`  
  Default 70 N is the low end of the Franka Emika continuous-grip spec.

Call these *after* compilation (they mutate the compiled `MjModel`) and *before* the `Gripper` class is instantiated.

Benchmark harness: `/tmp/grip_force_bench.py` template — apply target force, step physics with a test cylinder between the pads, read `data.contact[].normal` force. Useful for tuning against a specific object set.

### Step 6 — pad friction and self-collision

Two more menagerie quirks that bite in practice:

- **Self-colliding fingers**: menagerie grippers often let the pads interpenetrate on empty close. The planner then reports "start in collision" for any empty-close state. Exclude the pair before compile:

  ```python
  exclude = spec.add_exclude()
  exclude.bodyname1 = "gripper/left_pad"     # or "gripper/left_finger"
  exclude.bodyname2 = "gripper/right_pad"
  ```

- **Pad friction**: rigid-body contact on a thin pad against a cylinder is nearly line contact — low friction means objects slide out easily even at high grip force. Bump sliding friction on pad collision geoms to ~1.5 (silicone-on-aluminum range) and torsional friction to ~0.05. See `add_franka_pad_friction` for the Franka recipe; a `add_robotiq_pad_friction` helper hasn't been written yet but would follow the same structure.

### Step 7 — validate

Re-run the validator:

```bash
uv run python scripts/validate_gripper.py --gripper <name>
```

It should print `RESULT: PASS ✓` — 0/300 grasps in collision, declared `FINGER_LENGTH` within ~1 mm of the measured pad tip, declared `MAX_APERTURE` within ~1 mm of the measured inner-face gap.

Also run the integration-level smoke test: in your setup module's scenario, pick up one object, move it to the drop destination, release, repeat a few times. The `GraspVerifier` will emit a `LOST` warning if any grasp slips — that's the ultimate "did we get the grip force/friction right" signal.

### Common failure modes

| Symptom | Likely cause | Where to look |
|---|---|---|
| Arm stops ~1–2 cm short of grasp target | FL too small, or ee_site at fingertip instead of palm | `visualize_grasps.py`, check grasp_site position |
| Housing penetrates object on mid/deep grasps | grasp_site inside the housing | `validate_gripper.py` flags this with a "move grasp_site forward" suggestion |
| Fingers don't close in kinematic mode | Wrong trajectory table (signs flipped for this variant) | Re-record with `record_gripper_trajectory.py` |
| Object drops during transport | Grip force too low OR empty-close bug | `fix_*_grip_force` helper |
| Planner reports "start in collision" after empty close | Fingers interpenetrating at close stop | `spec.add_exclude()` for finger↔finger |
| Object slips out under lateral acceleration | Pad friction too low | `add_franka_pad_friction`-style helper |
| `GraspVerifier` flips to LOST during transport | Load-signal baseline wrong, OR object really is slipping | Enable DEBUG logging on verifier; check baseline readings at `mark_grasped` |

### The tools, summarized

- **`scripts/visualize_grasps.py`** — interactive web viewer. Teleports a bare gripper through TSR-sampled poses; per-template collision indicator. Use first when eyeballing a new gripper.
- **`scripts/validate_gripper.py`** — deterministic 50-samples × N-templates collision sweep + AABB measurement. Automated pass/fail with specific diagnostic messages. Use after each geometry change; suitable for CI.
- **`scripts/record_gripper_trajectory.py`** — records the joint-qpos trajectory of an open→close physics rollout. Used to build the kinematic trajectory table that `RobotiqGripper` replays in no-physics mode.
- **`fix_*_grip_force` helpers** — rewrite the gripper actuator to produce a constant force independent of finger gap. One helper per gripper family (`fix_franka_grip_force` in `arms/franka.py`, `fix_robotiq_grip_force` in `grippers/robotiq.py`).

For the underlying math and worked examples, see [`docs/grippers.md`](docs/grippers.md).

## Behavior Trees

Optional `mj_manipulator.bt` subpackage provides py_trees leaf nodes for composing manipulation tasks as behavior trees. Install with `uv add "mj-manipulator[bt]"`.

**Leaf nodes:** `GenerateGrasps`, `GeneratePlaceTSRs`, `PlanToTSRs`, `PlanToConfig`, `Retime`, `Execute`, `Grasp`, `Release`, `CartesianMove`, `SafeRetract`, `Sync`, `CheckNotNearConfig`.

**Subtree builders:** `pickup(ns)`, `place(ns)` — flat sequences, no recovery. Recovery lives in the primitives layer (`mj_manipulator.primitives`).

```python
from mj_manipulator.bt import pickup
import py_trees

tree = pickup("/ur5e")
print(py_trees.display.ascii_tree(tree))
```

All nodes use namespaced blackboard keys (`{ns}/arm`, `{ns}/grasp_tsrs`, etc.) for multi-arm support. See `docs/behavior-trees.md` for the composition guide.

## Scenarios and Demos

A **scenario** is a Python module with a `scene = {...}` dict and optional user-facing functions:

```python
# my_task.py
"""One-line description."""

scene = {
    "objects": {"can": 5},                                   # pool of graspable types
    "fixtures": {"yellow_tote": [[-0.5, 0.0, 0.0]]},         # stationary objects
    # "spawn_count": 3,                                       # optional: random subset
}

def sort_all(robot):
    while robot.pickup():
        robot.place("yellow_tote")
        robot.go_home()
    robot.go_home()
```

The `mj_manipulator.scenarios` package handles discovery, loading, `robot` binding, and reset. Your function signatures take `robot` as the first argument; the scenario runner binds it via `functools.partial` so the IPython console calls `sort_all()` with no arguments.

### Portability

Scenarios are portable by convention, not by mandate:

- **Portable scenarios** use only `RobotBase` methods (`pickup`, `place`, `go_home`, `find_objects`, `holding`). The same `recycling.py` runs on a Franka, a UR5e, or a bimanual Geodude.
- **Robot-specific scenarios** can use anything on their `robot` instance (`robot._left_base.set_height(...)`, `robot._perception.refresh()`, etc.). Document the requirement in the docstring.

The framework doesn't enforce portability — write what the task needs.

### Running

```bash
python -m mj_manipulator                                  # Franka + scenario picker
python -m mj_manipulator --robot iiwa14                   # iiwa 14 + picker
python -m mj_manipulator --scenario recycling             # Franka + recycling
python -m mj_manipulator --robot iiwa14 --scenario recycling
python -m mj_manipulator --list-scenarios                 # list and exit
```

The same scenario file runs on both robots unchanged — a portability test for the scenario system.

Built-in scenarios live under `src/mj_manipulator/demos/`. Other workspace packages (like `geodude`) define their own scenarios using the same format — see `geodude/src/geodude/demos/`.

### Reference scripts

Short examples of individual APIs, at `demos/` in the repo root:

```bash
uv run python demos/ik_solver.py         # EAIK analytical IK
uv run python demos/arm_planning.py      # CBiRRT motion planning
uv run python demos/collision_check.py   # Collision checking modes
```

### Debug and study scripts

Diagnostic tools at `scripts/`:

```bash
# Teleport a bare gripper through TSR-sampled grasps for eyeballing —
# no arm, no IK, no collision checking. Fastest way to diagnose
# "gripper stops short" or "FINGER_LENGTH feels wrong".
uv run python scripts/visualize_grasps.py --gripper robotiq_2f85 --object can
uv run python scripts/visualize_grasps.py --gripper franka        --object spam_can

# Auto-test a gripper's TSR setup: runs a collision sweep (50 samples ×
# every template) and prints the per-template pass/fail rate. When a
# check fails, prints a concrete fix (grasp_site offset, FINGER_LENGTH
# value). Run after registering a new gripper, or in CI.
uv run python scripts/validate_gripper.py --gripper robotiq_2f85
uv run python scripts/validate_gripper.py --all

# Measure commanded-vs-actual joint tracking during a representative
# motion. Reports RMS/max error, peak lag, settling time.
uv run python scripts/trajectory_tracking_study.py --robot franka
uv run python scripts/trajectory_tracking_study.py --robot iiwa14

# Record a kinematic open→close joint trajectory for a new gripper.
# Paste the output into grippers/_robotiq_Xxx_trajectory.py.
uv run python scripts/record_gripper_trajectory.py --help
```

## Building Your Own Demo

You **import** mj_manipulator; you don't copy it. Your own package sits alongside `mj_manipulator` in the workspace (like `geodude` does):

```python
from mj_manipulator import RobotBase, SimContext, GraspManager
from mj_manipulator.arms.franka import create_franka_arm
from mj_manipulator.scenarios import WorktopPose
```

The template for wiring your own arm is [`src/mj_manipulator/demos/franka_setup.py`](src/mj_manipulator/demos/franka_setup.py). It's six commented steps:

1. Load the model (MjSpec + gravcomp injection)
2. Add a worktop surface (so the scenario system knows where to scatter objects)
3. Build the `Environment`
4. Construct arm, gripper, grasp manager
5. Home the robot (qpos → home, gripper open)
6. Wrap in a `RobotBase` subclass and apply the scene

To bring your own arm: copy that file into your own package, swap the Franka-specific bits (factory, gripper, home pose) for your own, adjust the worktop if needed. Everything else (planning, execution, grasping, BT nodes, scenarios) comes from mj_manipulator unchanged.
