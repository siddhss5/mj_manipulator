# Plan: Create `mj_manipulator` package

## Context

Geodude (10,508 lines) mixes generic MuJoCo arm control with UR5e/Robotiq-specific code. The lab has multiple robots (Franka, Xarm, UR5e). Extracting the generic manipulation layer into `mj_manipulator` lets any MuJoCo manipulator plan trajectories, execute them, grasp/ungrasp objects (with kinematic weld + collision filtering), do cartesian teleop, and run joint-based or cartesian policies — without reimplementing the simulation loop.

**Geodude becomes a thin robot configuration package** (~2,000 lines) that plugs UR5e + Robotiq + Vention into the generic framework.

## Core Insight: ExecutionContext as the Sim-to-Real Bridge

The central abstraction is `ExecutionContext` — a protocol that unifies simulation and hardware execution. All planning, primitives, cartesian control, and policies interact with the robot **exclusively** through this interface. Whether the implementation talks to MuJoCo or to real hardware is invisible to the caller.

```python
# The SAME code works in simulation and on real hardware:

# Simulation
with SimContext(env, arms={"franka": arm}, physics=True) as ctx:
    result = arm.plan_to_pose(target)
    ctx.execute(result)
    ctx.arm("franka").grasp("mug")

# Real robot — identical calling convention
with FrankaHardwareContext(robot_ip="192.168.1.1", arms={"franka": arm}) as ctx:
    result = arm.plan_to_pose(target)
    ctx.execute(result)
    ctx.arm("franka").grasp("mug")
```

The context provides three execution patterns:

1. **Batch trajectory execution** (plan → execute):
   `ctx.execute(result)` — runs pre-planned trajectories
2. **Streaming joint control** (policies):
   `ctx.step({"franka": q_target})` — one control cycle with joint targets
3. **Streaming cartesian control** (teleop, force-guided motion):
   `ctx.step_cartesian("franka", q_new, qd_new)` — reactive step with velocity feedforward

Each method has a clear meaning in every backend:

| Method | MuJoCo Kinematic | MuJoCo Physics | Real Hardware |
|---|---|---|---|
| `execute()` | Set qpos per waypoint | PD tracking with mj_step | Stream to trajectory controller |
| `step()` | Set qpos directly | Set target + mj_step | Send joint command, wait 1 cycle |
| `step_cartesian()` | Set qpos directly | Reactive lookahead + mj_step | Stream to low-level controller |
| `sync()` | mj_forward + viewer | mj_forward + viewer | Read sensors |
| `arm().grasp()` | Kinematic close + weld | Physics close + weld | Gripper close + feedback |

## Design Decisions (confirmed with user)

- **ExecutionContext is the central protocol** — all primitives, policies, and control loops depend only on this interface, never on SimContext directly. This is what makes sim-to-real seamless.
- **Depends on mj_environment** — forking is key for thread-safe planning and already well-implemented
- **Arm is concrete + injection** — one `Arm` class parameterized by config, injected IK solver, injected Gripper. No subclassing.
- **Primitives move to mj_manipulator** — with a `GraspSource` protocol so any robot that can provide grasps gets pickup/place
- **SimContext is one implementation** — mj_manipulator ships `SimContext` (kinematic + physics modes). Robot-specific packages provide `HardwareContext`.

## Package Structure

```
mj_manipulator/
  pyproject.toml
  src/mj_manipulator/
    __init__.py
    protocols.py        # ExecutionContext, ArmController, Gripper, IKSolver, GraspSource
    config.py           # ArmConfig, KinematicLimits, PlanningDefaults, PhysicsConfig
    trajectory.py       # Trajectory + TOPP-RA + linear trajectory
    planning.py         # PlanResult dataclass
    arm.py              # Generic Arm (FK, IK, planning, execution)
    adapters.py         # pycbirrt RobotModel/IKSolver/CollisionChecker adapters
    collision.py        # Unified grasp-aware CollisionChecker
    grasp_manager.py    # GraspManager + find_contacted_object
    cartesian.py        # QP solver, twist control, move_until_touch, execute_twist
    executor.py         # KinematicExecutor, PhysicsExecutor (sim-specific)
    controller.py       # PhysicsController (sim-specific, multi-arm physics stepping)
    context.py          # SimContext — the simulation implementation of ExecutionContext
    primitives.py       # pickup/place (depend on ExecutionContext, not SimContext)
  tests/
    ...
```

**Key dependency rule**: `primitives.py`, `cartesian.py`, and any policy code depend only on `ExecutionContext` (the protocol), never on `SimContext` (the implementation). This is what makes real robot execution possible without changing any manipulation logic.

## Core Protocols

### `ExecutionContext` — the sim-to-real bridge

```python
class ExecutionContext(Protocol):
    """Unified interface for robot execution (sim or real hardware)."""
    def execute(self, item: Trajectory | PlanResult) -> bool: ...
    def step(self, targets: dict[str, np.ndarray] | None = None) -> None: ...
    def step_cartesian(self, arm_name: str, position: np.ndarray,
                       velocity: np.ndarray | None = None) -> None: ...
    def sync(self) -> None: ...
    def is_running(self) -> bool: ...
    def arm(self, name: str) -> ArmController: ...
    control_dt: float  # property
```

### `ArmController` — per-arm grasp/release within a context

```python
class ArmController(Protocol):
    """Combines gripper actuation with grasp-manager bookkeeping."""
    def grasp(self, object_name: str) -> str | None: ...
    def release(self, object_name: str | None = None) -> None: ...
```

The grasp/release methods handle the **full pipeline**: actuate gripper, detect contact, update grasp manager (weld creation, collision-group updates). Whether this means physics simulation or real gripper feedback is invisible to the caller.

### `Gripper` — low-level gripper hardware

```python
class Gripper(Protocol):
    """Any gripper (Robotiq, Franka hand, suction, etc.)."""
    arm_name: str
    gripper_body_names: list[str]
    attachment_body: str          # Body objects weld to during kinematic grasp
    actuator_id: int | None
    ctrl_open: float
    ctrl_closed: float

    def kinematic_close(self, steps: int = 50) -> str | None: ...
    def kinematic_open(self) -> None: ...
    def get_actual_position(self) -> float: ...  # 0=open, 1=closed
    is_holding: bool
    held_object: str | None
    def set_candidate_objects(self, objects: list[str] | None) -> None: ...
```

### `GraspSource` and `IKSolver`

```python
class GraspSource(Protocol):
    """Provides grasps/placements for objects. Geodude's AffordanceRegistry implements this."""
    def get_grasps(self, object_name: str, hand_type: str) -> list[TSR]: ...
    def get_placements(self, destination: str, object_name: str) -> list[TSR]: ...
    def get_graspable_objects(self) -> list[str]: ...
    def get_place_destinations(self, object_name: str) -> list[str]: ...

class IKSolver(Protocol):
    """Mirrors pycbirrt's IKSolver protocol."""
    def solve(self, pose, q_init=None) -> list[np.ndarray]: ...
    def solve_valid(self, pose, q_init=None) -> list[np.ndarray]: ...
```

### Key: `Gripper.attachment_body`

This is the body that objects weld to during kinematic grasping. Currently hardcoded as `f"{side}_ur5e/gripper/right_follower"` in geodude's `execution.py`. Each gripper implementation provides its own (Robotiq follower link, Franka finger pad, suction cup tip, etc.).

## What Moves vs Stays

### Moves to mj_manipulator

| geodude module | → mj_manipulator module | Notes |
|---|---|---|
| `trajectory.py` (350) | `trajectory.py` | Verbatim |
| `planning.py` (58) | `planning.py` | Verbatim |
| `grasp_manager.py` (353) | `grasp_manager.py` | Parameterize finger detection |
| `collision.py` (652) | `collision.py` | Unify 3 classes → 1 (per #62) |
| `cartesian.py` (988) | `cartesian.py` | Replace `robot._active_context` with `context: ExecutionContext` param |
| `arm.py` generic parts (~800) | `arm.py` | FK, joint control, planning, execution |
| `arm.py` adapters (~200) | `adapters.py` | ArmRobotModel, ContextRobotModel, SimpleIKSolver |
| `executor.py` executors (~300) | `executor.py` | KinematicExecutor, PhysicsExecutor |
| `executor.py` controller (~600) | `controller.py` | Generalize RobotPhysicsController (dict of arms, not "left"/"right") |
| `execution.py` context (~400) | `context.py` | SimContext implements ExecutionContext protocol (arm registry, not hardcoded sides) |
| `primitives.py` (~843) | `primitives.py` | Depend on ExecutionContext protocol, use GraspSource for grasps |
| `config.py` generic parts (~400) | `config.py` | KinematicLimits, PlanningDefaults, ArmConfig, PhysicsConfig |
| NEW | `protocols.py` | ExecutionContext, ArmController, Gripper, GraspSource, IKSolver protocols |

### Stays in geodude

| Module | Lines (est.) | What remains |
|---|---|---|
| `config.py` | ~280 | GeodudConfig, VentionBaseConfig, VentionKinematicLimits, DebugConfig, UR5e defaults |
| `arm.py` | ~200 | UR5e IK solver factory (EAIK setup, base rotation, EE offset), F/T sensor utility |
| `gripper.py` | ~520 | Robotiq 2F-140 (implements Gripper protocol). Unchanged. |
| `robot.py` | ~500 | Geodude class: constructs arms/grippers/bases, named poses, wires everything |
| `vention_base.py` | ~371 | Vention linear actuator. Unchanged. |
| `affordances.py` | ~392 | AffordanceRegistry (implements GraspSource protocol) |
| `tsr_utils.py` | ~200 | Gripper frame compensation (shrinks after #69 TSR migration) |
| `__init__.py` | ~60 | Re-exports |
| **Total** | **~2,500** | **76% reduction from 10,508** |

### mj_manipulator estimated size: ~3,500 lines

Combined total: ~6,000 lines (vs 10,508 before). ~4,500 lines eliminated through dedup during extraction (collision unification, cartesian dedup, dead code removal, executor cleanup).

## Arm Constructor

```python
# mj_manipulator/arm.py
class Arm:
    def __init__(
        self,
        env: Environment,           # mj_environment — provides model, data, fork, sync
        config: ArmConfig,          # joint names, EE site, limits, etc.
        grasp_manager: GraspManager,
        gripper: Gripper | None = None,
        ik_solver: IKSolver | None = None,
        name: str = "",             # e.g., "left", "right", "franka"
    ): ...
```

Takes `env: Environment` (not raw model/data) since forking is essential for planning.

## How Geodude Uses mj_manipulator

```python
# geodude/robot.py
from mj_manipulator import Arm, GraspManager, ArmConfig
from geodude.gripper import Robotiq2F140
from geodude.ur5e import create_ur5e_ik_solver

class Geodude:
    def __init__(self, config, objects=None):
        self.env = Environment(...)
        self.grasp_manager = GraspManager(self.env.model, self.env.data)

        left_gripper = Robotiq2F140(self.env, "left", config.left_arm, self.grasp_manager)
        right_gripper = Robotiq2F140(self.env, "right", config.right_arm, self.grasp_manager)

        self.left_arm = Arm(
            env=self.env, config=config.left_arm,
            grasp_manager=self.grasp_manager,
            gripper=left_gripper,
            ik_solver=create_ur5e_ik_solver(self.env, config.left_arm),
            name="left",
        )
        # ... right_arm similarly
```

## How Franka Would Use mj_manipulator

```python
# franka_control/robot.py
from mj_manipulator import Arm, GraspManager, ArmConfig, KinematicLimits
from mj_manipulator.context import SimContext
from mj_manipulator.primitives import pickup
from mj_manipulator.cartesian import execute_twist

franka_config = ArmConfig(
    name="franka", entity_type="arm",
    joint_names=[f"franka/joint{i}" for i in range(1, 8)],
    ee_site="franka/ee_site",
    gripper_actuator="franka/gripper_actuator",
    gripper_bodies=["franka/left_finger", "franka/right_finger"],
    hand_type="franka_hand",
    kinematic_limits=KinematicLimits(
        velocity=np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]),
        acceleration=np.array([15, 7.5, 10, 12.5, 15, 20, 20]),
    ),
)

grasp_manager = GraspManager(env.model, env.data)
gripper = FrankaHand(env, franka_config, grasp_manager)  # implements Gripper protocol
arm = Arm(env=env, config=franka_config, grasp_manager=grasp_manager,
          gripper=gripper, ik_solver=FrankaIK(env), name="franka")

# --- Simulation (MuJoCo) ---
with SimContext(env, arms={"franka": arm}, physics=True) as ctx:
    # Plan + execute trajectory
    result = arm.plan_to_pose(target)
    ctx.execute(result)

    # Cartesian teleop
    execute_twist(arm, twist, context=ctx)

    # Run a joint policy
    while ctx.is_running():
        q_target = policy(arm.get_joint_positions())
        ctx.step({"franka": q_target})

    # Pickup (uses ExecutionContext protocol internally)
    pickup(arm, "mug", grasp_source=my_grasp_source, context=ctx)

# --- Real robot (IDENTICAL calling convention) ---
with FrankaHardwareContext(robot_ip="192.168.1.1", arms={"franka": arm}) as ctx:
    result = arm.plan_to_pose(target)
    ctx.execute(result)

    execute_twist(arm, twist, context=ctx)  # Same function, real robot

    pickup(arm, "mug", grasp_source=my_grasp_source, context=ctx)  # Same function
```

Notice that `pickup()`, `execute_twist()`, and the policy loop are **identical** — only the context constructor changes between sim and real.

## Kinematic Grasp Pipeline (critical for correctness)

When `pickup()` grasps an object in kinematic mode:

1. `gripper.kinematic_close()` → detects contact, returns object name
2. `grasp_manager.mark_grasped(object_name, arm_name)` → records grasp state
3. `grasp_manager.attach_object(object_name, gripper.attachment_body)` → computes & stores relative transform (weld)
4. All subsequent `collision_checker.is_valid(q)` calls filter out gripper↔object contacts
5. Every simulation step: `grasp_manager.update_attached_poses(data)` → moves object with gripper
6. On release: `grasp_manager.detach_object()` + `mark_released()` → object stays in place, collisions re-enabled

`gripper.attachment_body` is the key generic interface — each gripper implementation specifies which body objects attach to.

## PhysicsController (generalized from RobotPhysicsController)

Current `RobotPhysicsController` hardcodes "left"/"right" arms. Generic version uses a dict:

```python
class PhysicsController:
    def __init__(self, env, arms: dict[str, Arm], grippers: dict[str, Gripper],
                 physics_config: PhysicsConfig, viewer=None): ...

    def step(self, targets: dict[str, np.ndarray] | None = None) -> None:
        """Step all arms toward their targets."""

    def step_cartesian(self, arm_name: str, position: np.ndarray,
                       velocity: np.ndarray | None = None) -> None:
        """Reactive step for cartesian control."""

    def execute(self, arm_name: str, trajectory: Trajectory) -> None: ...
    def close_gripper(self, arm_name: str) -> str | None: ...
    def open_gripper(self, arm_name: str) -> None: ...
```

## SimContext (implements ExecutionContext for MuJoCo)

```python
class SimContext:
    """Simulation implementation of ExecutionContext.

    Supports two modes:
    - Kinematic: perfect tracking, no physics (fast for planning visualization)
    - Physics: MuJoCo physics stepping with PD control (realistic execution)
    """
    def __init__(self, env, arms: dict[str, Arm],
                 physics: bool = False, viewer: bool = True,
                 physics_config: PhysicsConfig | None = None): ...
```

Takes `arms: dict[str, Arm]` instead of assuming `robot.left_arm`/`robot.right_arm`. Camera setup becomes optional/configurable rather than hardcoded Geodude angles.

## HardwareContext (robot-specific packages provide this)

```python
# Example: geodude provides UR5e hardware context
class UR5eHardwareContext:
    """Hardware implementation of ExecutionContext for UR5e via RTDE.

    Same interface as SimContext — all primitives and policies work unchanged.
    """
    def __init__(self, robot_ip: str, arms: dict[str, Arm],
                 control_rate: float = 500.0): ...

    def execute(self, item) -> bool:
        # Stream trajectory to UR RTDE servoj interface
        ...

    def step(self, targets=None) -> None:
        # Send joint command via RTDE, wait one control cycle
        ...

    def step_cartesian(self, arm_name, position, velocity=None) -> None:
        # Stream to RTDE speedj or servoj with small lookahead
        ...

    def sync(self) -> None:
        # Read joint encoders, F/T sensors, gripper state via RTDE
        ...

    def arm(self, name) -> HardwareArmController:
        # Returns controller that talks to Robotiq gripper via Modbus
        ...
```

**The key**: primitives like `pickup()` take `context: ExecutionContext` (the protocol), so they work with both `SimContext` and `UR5eHardwareContext` without any code changes.

## Migration Strategy (incremental, never breaks geodude)

### Phase 1: Scaffold + pure data types + core protocols ✅
- Create `mj_manipulator/` with pyproject.toml
- Move: `trajectory.py`, `planning.py`, `config.py` (generic parts)
- New: `protocols.py` with **ExecutionContext**, **ArmController**, Gripper, IKSolver, GraspSource
- ExecutionContext protocol defined early so all subsequent phases depend on the protocol, not implementations
- **Test**: 51 tests passing, including mock ExecutionContext/ArmController tests demonstrating the sim-to-real pattern

### Phase 2: Grasp management + collision
- Move: `grasp_manager.py`, unified `collision.py` (does #62 during extraction)
- Update geodude imports
- **Test**: geodude tests pass

### Phase 3: Executors + cartesian control
- Move: `KinematicExecutor`, `PhysicsExecutor`, cartesian functions
- Break `arm.robot._active_context` dependency → functions take `context: ExecutionContext`
- All cartesian functions depend on the protocol, not SimContext
- Add thin compat wrappers in geodude during transition
- **Test**: geodude tests pass

### Phase 4: Arm class
- Create `mj_manipulator.Arm` (generic, takes `env: Environment`)
- Geodude's Arm becomes a thin construction wrapper that builds mj_manipulator.Arm with UR5e config + EAIK solver + Robotiq gripper
- **Test**: all planning/execution tests pass

### Phase 5: Controller + SimContext (the simulation implementation)
- Generalize `RobotPhysicsController` → `PhysicsController` (arm dict, not left/right)
- Generalize `SimContext` → implements `ExecutionContext` protocol, parameterized by arm registry
- Verify: primitives/cartesian depend only on `ExecutionContext` protocol, not `SimContext`
- **Test**: physics-mode tests pass, mock `ExecutionContext` tests demonstrate protocol independence

### Phase 6: Primitives
- Move `primitives.py` — depends on `ExecutionContext` + `GraspSource` protocols
- `pickup(arm, obj, grasp_source, context)` where `context: ExecutionContext`
- Geodude's `AffordanceRegistry` implements `GraspSource`
- **Test**: pickup/place tests pass with mock `ExecutionContext` (no MuJoCo needed)

### Phase 7: Cleanup
- Remove geodude shims/re-exports
- Delete moved files from geodude
- Update geodude's `__init__.py`
- Run full test suite

### Phase 8: Validate
- Create minimal Franka example using mj_manipulator + mujoco_menagerie model
- Verify: plan trajectory, execute, cartesian teleop, grasp/release all work

## Cleanup Issues Disposition

| Issue | Action |
|---|---|
| **#61** (dead code) | Do first, before extraction. Pure deletion, reduces noise. |
| **#62** (collision unify) | Fold into Phase 2 — unification happens during extraction |
| **#63** (arm.py decomp) | Fold into Phase 4 — splitting into generic Arm + UR5e parts IS the decomposition |
| **#64** (planning dispatch) | Fold into Phase 4 — simplified during Arm extraction |
| **#65** (cartesian dedup) | Fold into Phase 3 — dedup during extraction |
| **#66** (executor internals) | Fold into Phase 5 — controller generalization |
| **#67** (config/data extern) | Do separately — gripper trajectory extraction is geodude-internal |
| **#68** (small modules) | Mostly absorbed — planning.py merges, parallel.py deleted, primitives move |
| **#69** (TSR migration) | Do separately — orthogonal to extraction |
| **#70** (robot.py pass-throughs) | Absorbed — robot.py shrinks naturally |

## Verification: Tests vs Demos

### Tests (`tests/`) — correctness, CI-friendly

Automated, run with `uv run pytest tests/ -v`. No robot models, no viewer, no GPU. Use mocks and pure logic. Every phase adds tests.

| Phase | Tests |
|---|---|
| **1** | Trajectory (construction, TOPP-RA, interpolation), PlanResult, Config (UR5e + Franka), Protocol satisfaction (isinstance), ExecutionContext mock lifecycle |
| **2** | GraspManager (mark/release, attach/detach, collision groups), CollisionChecker (is_valid with mock model, grasp filtering) |
| **3** | Cartesian QP solver (twist → joint velocities, joint limit constraints), executor interface (mock-based) |
| **4** | Arm (FK, joint read/write, plan_to with mock IK + collision checker) |
| **5** | SimContext (trajectory routing, step/step_cartesian, arm controller creation), PhysicsController (multi-arm target management) |
| **6** | Primitives (pickup/place with mock ExecutionContext + mock GraspSource — no MuJoCo needed) |
| **7** | Full geodude regression: `cd ../geodude && uv run pytest tests/ -v` |

### Demos (`demos/`) — integration, real robot models

Standalone scripts, run directly with `python demos/...`. Load real MuJoCo models, may open viewer. Show the framework working end-to-end with actual robots. Serve as documentation for users adopting the framework.

| Phase | Demo | What it shows |
|---|---|---|
| **2** | `demos/collision_check.py` | Load UR5e + Franka models, check collisions at various configs |
| **4** | `demos/plan_trajectory.py` | Plan + visualize trajectories for both UR5e and Franka |
| **5** | `demos/sim_context.py` | Open viewer, execute trajectory, cartesian step, grasp/release (both arms) |
| **6** | `demos/pickup_place.py` | Full pickup/place with both robots |
| **8** | `demos/franka_e2e.py` | End-to-end Franka: plan, execute, grasp, cartesian teleop, policy loop |

Key differences:
- **Tests** answer "is the code correct?" — run in CI, mock everything external
- **Demos** answer "does it work with my robot?" — load real models, show visual results, serve as examples

### Commands

```bash
# After every phase:
cd mj_manipulator && uv run pytest tests/ -v

# After Phase 7+:
cd ../geodude && uv run pytest tests/ -v

# Run a specific demo:
cd mj_manipulator && uv run python demos/sim_context.py
```

## Workflow

- This plan lives as a WIP PR in the mj_manipulator repo
- Each phase is a separate commit (or small set of commits)
- PR description tracks progress with checkboxes
- Tests and demos are committed alongside code

## Key Files to Modify

**geodude (read before modifying):**
- `src/geodude/arm.py` (2091 lines) — split into generic + UR5e
- `src/geodude/collision.py` (652 lines) — unify + move
- `src/geodude/cartesian.py` (988 lines) — break context dep + move
- `src/geodude/executor.py` (1066 lines) — split executors vs controller
- `src/geodude/execution.py` (539 lines) — generalize context
- `src/geodude/primitives.py` (843 lines) — add GraspSource protocol + move
- `src/geodude/config.py` (680 lines) — split generic vs geodude-specific
- `src/geodude/grasp_manager.py` (353 lines) — parameterize + move
- `src/geodude/trajectory.py` (350 lines) — move
- `src/geodude/gripper.py` (520 lines) — stays, implements Gripper protocol
- `src/geodude/robot.py` (882 lines) — slim down to construction + wiring
- `src/geodude/affordances.py` (392 lines) — stays, implements GraspSource

**pycbirrt (read-only reference):**
- `src/pycbirrt/interfaces/robot_model.py` — RobotModel protocol
- `src/pycbirrt/interfaces/ik_solver.py` — IKSolver protocol
- `src/pycbirrt/interfaces/collision_checker.py` — CollisionChecker protocol

**mj_environment (read-only reference):**
- `mj_environment/environment.py` — Environment.fork(), model, data
