# Changelog

## v2.0.0 (2026-04-17)

104 commits since v1.0.0. Major additions to grasp verification, multi-arm control, perception, and gripper integration — plus a second bundled robot (KUKA iiwa 14) and diagnostic tooling for the gripper integration story.

### New features

- **GraspVerifier + LoadSignal protocol** — signal-agnostic grasp verification with a sticky IDLE → HOLDING → LOST state machine. Drives grasp health from whatever sensors the arm actually has (gripper position, wrist F/T, joint torques). Replaces contact-inspection post-checks that only worked in simulation. New public types: `GraspVerifier`, `GraspState`, `VerifierParams`, `VerifierFacts`, `LoadSignal`, `GripperPositionSignal`, `WristFTSignal`, `JointTorqueSignal`. (#93, #98)

- **OwnershipRegistry** — per-arm ownership and abort tracking for concurrent multi-arm control (teleop, trajectory, gripper). Prevents two operations from fighting over the same arm. New public types: `OwnershipRegistry`, `OwnerKind`. (#53, #59)

- **ExecutionContext protocol** — solidified interface for `SimContext` and (future) `HardwareContext`. Same code works whether running in MuJoCo or on a real robot.

- **PerceptionService protocol + SimPerceptionService** — sim/hardware-agnostic object pose queries with mock detection → alias resolution → tracker → environment update pipeline. (#175)

- **Behavior tree nodes** — `mj_manipulator.bt` subpackage with py_trees leaf nodes for manipulation tasks. Nodes: `GenerateGrasps`, `GeneratePlaceTSRs`, `PlanToTSRs`, `PlanToConfig`, `Retime`, `Execute`, `Grasp`, `Release`, `CartesianMove`, `SafeRetract`, `Sync`, `CheckNotNearConfig`. Subtree builders: `pickup(ns)`, `place(ns)`. (#104)

- **Primitives layer** — `pickup`, `place`, `go_home` with centralized recovery. (#63, #76, #78, #79)

- **E-Stop system** — `request_abort` / `clear_abort` with `force_release_all`, persistent until user releases. CLI commands and prompt indicator. (#118, #119)

- **Fork-based planning** — `Arm.create_planner(planning_env=fork)` for side-effect-free planning. (#120)

- **Gravity compensation helpers** — `add_ur5e_gravcomp`, `add_franka_gravcomp`, `add_iiwa14_gravcomp` MjSpec injection. Generic `add_subtree_gravcomp` walker. (#c61788d, #080a6f7)

- **TeleopController** — unified teleop with pose/twist inputs, safety modes, and per-arm preemption. Position step derived from velocity limits. (#47, #70, #105)

- **Wrist F/T sensor support** — `get_ft_wrench_world()`, `tare_ft()`, NaN in kinematic mode, physics-settle gating. (#27, #35, #36, #37, #38)

- **KUKA iiwa 14 + Robotiq 2F-85 demo** — second bundled robot demonstrating the "attach your own gripper" workflow via MjSpec. Velocity/acceleration limits from KUKA datasheet. Same recycling scenario runs unchanged on both robots. (#125, #128)

- **Gripper diagnostic tools** — `scripts/visualize_grasps.py` (interactive disembodied-hand viewer with per-template collision indicator) and `scripts/validate_gripper.py` (deterministic collision sweep with automatic fix suggestions). (#128)

- **`fix_robotiq_grip_force`** — rewrites the menagerie Robotiq actuator for constant grip force independent of finger gap. Companion to `fix_franka_grip_force`. (#128)

- **`docs/grippers.md`** — practitioner deep-dive on the palm–housing distinction, the menagerie actuator bug, pad area vs grip force, and diagnostic workflow. (#128)

### Bug fixes

- **CartesianController physics mode** — `step()` re-read `data.qpos` every cycle; in physics mode the PD-lagged state prevented the target from accumulating, so the arm barely moved or drifted in the wrong direction. Fix: internal `_q_ref` that integrates forward independently. (#84, #134)

- **Franka hand palm offset** — `add_franka_ee_site` placed `grasp_site` at the finger-joint origin, but the `hand` body extends 17 mm past it along the approach axis. Deep-grasp templates drove the collar into the object (33% of samples in collision). Fix: `grasp_site` at `hand + [0, 0, 0.0753]`, `FrankaHand.FINGER_LENGTH` 0.054 → 0.037. (#129, #131)

- **Robotiq 2F-85 palm inside housing** — `grasp_site` at `base_mount` origin buried the TSR "palm" inside the 94 mm-deep housing. 4 of 6 side-grasp templates were 100% in collision. Fix: `grasp_site` at `base_mount + [0, 0, 0.094]`, `Robotiq2F85.FINGER_LENGTH` 0.129 → 0.059, `MAX_APERTURE` 0.098 → 0.085. (#128)

- **Robotiq grip force dropout** — both 2F-85 and 2F-140 menagerie actuators have a length-coupled bias that sends grip force → 0 at full close. Fix: `fix_robotiq_grip_force` kills the coupling and bumps tendon force to 15 N (~300 N pad on 2F-85). (#128)

- **GraspVerifier weld cleanup** — `_transition_to_lost` now detaches the physics weld so subsequent plans don't see a stale attachment and report "start in collision". (#128)

- **safe_retract rewritten** — IK-based Cartesian path with abort predicate replaces the broken `CartesianController.move` path. Collision-checked at every waypoint. (#81, #6b21511)

- **validate_gripper --all graceful skip** — removed geodude_assets dependency from mj_manipulator's gripper registry; grippers whose XML resolver fails are now skipped with a diagnostic message. (#130, #133)

### Breaking changes

- `safe_retract` no longer uses `CartesianController.move`; it's now IK-based via `plan_cartesian_path`. Signature unchanged but internal behaviour differs. (#121)

- `detect_grasped_object` (bilateral contact heuristic) is deprecated in favour of `GraspVerifier`. Still present but logged as a warning. (#80, #83)

- Teleop position step now derived from `arm.config.kinematic_limits.velocity * dt`, no longer a separate `max_joint_step` constant. (#105)

- Safety layer from #106 was added then reverted — no velocity/acceleration clamping in execution. (#106, revert #c5fb5dc)

### Documentation

- **README** — overhauled "Adding a New Arm" section from real-world iiwa 14 experience (9 friction points fixed). New "Adding a New Gripper" section with 7-step workflow, TSR frame convention, common-failure-modes table.

- **`docs/grippers.md`** — 6-section deep-dive on palm placement, actuator force, pad area vs grip force, diagnostic workflow, worked examples, frame convention reference.

- **`docs/cartesian-control.md`** — QP formulation reference.

- **`docs/behavior-trees.md`** — composition guide for BT nodes.

### Internal

- 406 tests (was 285 in v1.0.0)
- Arm velocity/acceleration limits corrected with cited datasheet sources for UR5e, Franka, iiwa 14
- Kinematic trajectory recording helper (`scripts/record_gripper_trajectory.py`)
- Trajectory tracking study script (`scripts/trajectory_tracking_study.py`)
- Open-source prep: MIT relicense, SPDX headers, issue/PR templates (#48)

### Known limitations

- `CartesianController` in physics mode works correctly now (#84 fix) but accumulates drift if the physical state diverges far from `_q_ref` (e.g., external collision). Call `reset()` before changing direction.
- Franka hand aperture is tight for 66 mm cans (~5 mm clearance per side). (#132)
- Bilateral contact fallback (`detect_grasped_object`) is deprecated but not yet removed. (#80, #83)
