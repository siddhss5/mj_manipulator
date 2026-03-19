# mj_manipulator

Generic MuJoCo manipulator control: planning, execution, grasping, and cartesian control for any robot arm.

## Supported Robots

Pre-built arm factories in `mj_manipulator.arms`:

- **UR5e** (6-DOF) — `create_ur5e_arm(env)`
- **Franka Emika Panda** (7-DOF) — `create_franka_arm(env)`

See [Adding a New Arm](#adding-a-new-arm) below.

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

All planning methods are thin pass-throughs to the underlying planner ([pycbirrt](https://github.com/siddhss5/pycbirrt)). If the planner supports a goal type natively, we delegate directly.

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
from mj_manipulator.cartesian import CartesianController, CartesianControlConfig

controller = CartesianController(arm, config=CartesianControlConfig())

# Move 5 cm/s in +x
result = controller.step(twist=np.array([0.05, 0, 0, 0, 0, 0]), dt=0.008)
print(result.achieved_fraction)   # 1.0 = full twist achieved
print(result.limiting_factor)     # None / "joint_limit" / "velocity"
```

See [docs/cartesian-control.md](docs/cartesian-control.md) for the full derivation including twist weighting, projected gradient descent solver, convergence analysis, and comparison with MoveIt Servo.

## Grasp-Aware Collision

During manipulation, a grasped object must be treated as part of the robot: gripper-to-object contacts are expected, but arm-to-object and object-to-environment contacts indicate a collision. `CollisionChecker` handles this with software contact filtering — no MuJoCo collision group changes needed.

```python
from mj_manipulator.collision import CollisionChecker
from mj_manipulator.grasp_manager import GraspManager, detect_grasped_object

grasp_manager = GraspManager(model, data)

# After closing gripper:
grasped = detect_grasped_object(model, data, gripper_body_names, candidate_objects=["can_0"])
if grasped:
    grasp_manager.mark_grasped(grasped, arm="right")
    grasp_manager.attach_object(grasped, "gripper/right_pad")

# Collision checker uses grasp state automatically:
checker = CollisionChecker(model, data, joint_names, grasp_manager=grasp_manager)
checker.is_valid(q)   # allows gripper↔can; rejects arm↔can, can↔env
```

See [docs/grasp-aware-collision.md](docs/grasp-aware-collision.md) for the filtering logic, live vs snapshot modes for parallel planning, and the complete grasp lifecycle.

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
                        └── grasp_manager.py  Grasp state tracking
```

Robot-specific code (joint names, limits, IK config) lives in `arms/<robot>.py`. The generic layer (`Arm`, protocols, executors) knows nothing about specific robots.

## Adding a New Arm

`arms/ur5e.py` (6-DOF) and `arms/franka.py` (7-DOF) are the complete references. The steps:

**1. Create `arms/<robot>.py`**

```python
import numpy as np
from mj_manipulator.arm import Arm
from mj_manipulator.arms.eaik_solver import MuJoCoEAIKSolver
from mj_manipulator.config import ArmConfig, KinematicLimits

MY_ROBOT_JOINT_NAMES = ["joint1", "joint2", ...]   # from your XML
MY_ROBOT_HOME        = np.array([0.0, 0.0, ...])
MY_ROBOT_VEL_LIMITS  = np.array([...]) * 0.5       # datasheet values, halved
MY_ROBOT_ACC_LIMITS  = np.array([...]) * 0.5

def create_my_robot_arm(env, *, ee_site="grasp_site", with_ik=True, ...):
    config = ArmConfig(
        name="my_robot", entity_type="arm",
        joint_names=list(MY_ROBOT_JOINT_NAMES),
        kinematic_limits=KinematicLimits(
            velocity=MY_ROBOT_VEL_LIMITS.copy(),
            acceleration=MY_ROBOT_ACC_LIMITS.copy(),
        ),
        ee_site=ee_site,
    )
    if not with_ik:
        return Arm(env, config)
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
        # fixed_joint_index=MY_ROBOT_LOCKED_JOINT,  # 7-DOF only — see step 2
    )
    return Arm(env, config, ik_solver=ik_solver)
```

**2. Find the locked joint (7-DOF only)**

Run this once as a one-off script to discover which joint to lock:

```python
from mj_manipulator.arms import find_locked_joint_index
from mj_manipulator.arms.eaik_solver import _extract_hp

arm = Arm(env, config)  # create without IK first
first_joint_body = env.model.jnt_bodyid[arm.joint_ids[0]]
base_body_id     = env.model.body_parentid[first_joint_body]
H, P, _          = _extract_hp(env.model, env.data, list(arm.joint_ids),
                                arm.joint_qpos_indices, arm.ee_site_id, base_body_id)
print(find_locked_joint_index(H, P))  # → hardcode this as MY_ROBOT_LOCKED_JOINT
```

Then pass `fixed_joint_index=MY_ROBOT_LOCKED_JOINT` to `MuJoCoEAIKSolver`.

**3. Add an EE site if the model doesn't have one**

Use `MjSpec` to add the site before creating the `Environment`:

```python
spec = mujoco.MjSpec.from_file("path/to/scene.xml")
hand = spec.worldbody.find_child("hand")  # adjust to your link name
site = hand.add_site()
site.name = "grasp_site"
site.pos = [0, 0, 0.1]   # at palm/flange; z = approach direction
model = spec.compile()
env = Environment.from_model(model)
```

See `add_franka_ee_site()` in `arms/franka.py` for the pattern.

**4. Add tests** — copy `TestUR5eFactory` / `TestUR5eIK` from `tests/test_arms.py`:
factory creates valid Arm, FK-IK round-trip, all solutions within joint limits.

**5. Re-export** — add to `arms/__init__.py`:
```python
from mj_manipulator.arms.my_robot import create_my_robot_arm
__all__ = [..., "create_my_robot_arm"]
```

## Demos

See [demos/README.md](demos/README.md) for the full list. The capstone demo runs both robots end-to-end:

```bash
cd mj_manipulator
uv run mjpython demos/recycling.py --robot both   # UR5e + Franka recycling demo
uv run mjpython demos/recycling.py --robot ur5e --headless
uv run python demos/ik_solver.py                  # EAIK analytical IK showcase
uv run python demos/arm_planning.py               # Motion planning with CBiRRT
```
