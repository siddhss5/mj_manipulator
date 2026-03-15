# mj_manipulator

Generic MuJoCo manipulator control: planning, execution, grasping, and cartesian control for any robot arm.

## Supported Robots

Pre-built arm factories in `mj_manipulator.arms`:

- **UR5e** (6-DOF) — `create_ur5e_arm(env)`
- **Franka Emika Panda** (7-DOF) — `create_franka_arm(env)`

Adding a new arm is straightforward — see `mj_manipulator/arms/__init__.py` for the recipe.

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

## Demos

See [demos/README.md](demos/README.md) for runnable examples with real MuJoCo models.

```bash
cd mj_manipulator
uv run python demos/ik_solver.py       # EAIK analytical IK showcase
uv run python demos/arm_planning.py    # Motion planning with CBiRRT
```
