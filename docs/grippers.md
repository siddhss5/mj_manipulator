# Gripper integration: insights and diagnostics

A companion to the README's ["Adding a New Gripper"](../README.md#adding-a-new-gripper) step-by-step guide. That section tells you *what to do*. This document captures *why* — the two recurring classes of bug we've hit when integrating menagerie grippers, the math behind them, and the empirical findings that guided the fixes.

Read this if you're debugging a grasp that "looks fine but drops objects", "stops a centimetre short of the target", or "collides with the object before the fingers even close". Or read it when adding a new gripper for the first time, to develop the mental model that will save you the debugging cycle we went through.

## 1. The palm–housing distinction

### The assumption TSR makes

Every parallel-jaw grasp template in `tsr.hands.ParallelJawGripper` is built on one implicit assumption:

> Everything on the gripper except the fingers is *behind* the palm along the approach axis.

Concretely, for a side grasp of a cylinder, the template places the ee_site (palm) at radial distance `ro` from the cylinder axis:

```
ro = cylinder_radius + FINGER_LENGTH − d     (for grasp depth d ∈ [r, min(FL, 2r) − ε])
```

At the **shallowest** depth `d = r`: palm at `FL` from the axis, fingertip at the axis.
At the **deepest** depth `d ≈ min(FL, 2r)`: palm very close to the surface, fingertip past it.

At any given depth, the TSR palm sits at `ro` from the axis and approach direction points inward. **Behind the palm is assumed free space**; the finger mechanism alone is assumed to occupy the space between palm and fingertip.

### How this assumption fails

If any part of the gripper's non-finger structure (the motor housing, the driver linkage, the base plate) extends *past* the ee_site along the approach axis, the assumption is violated. As `d` grows, the palm moves toward the object, and at some depth the housing enters the object's radial footprint. Collision.

That's exactly what happened with the Robotiq 2F-85. Its `base` body — the chunky housing that contains the driver motor — extends ~94 mm past `base_mount` along the approach axis. Initial setup placed `grasp_site` at `base_mount`, so TSR's computed "palm distance" counted the housing as free space. At mid- and deep-grasp depths the palm was 99–115 mm from the can center, but the housing reached inward to within 11–21 mm of the axis — well inside the 33 mm can surface.

### The fix: palm = forward edge of housing

The palm for TSR purposes is the **forward-most point of the non-finger structure**, not the mounting-plate origin. Two things shift together:

1. **`grasp_site` placement**: offset it from the mounting body so it sits at the forward edge of the housing.
2. **`FINGER_LENGTH`**: measure from that new palm position to the fingertip, not from the mounting plate.

These are paired — shifting the palm forward without reducing `FINGER_LENGTH` would make the arm drive the fingertip past the object.

For the 2F-85:

| | Before | After |
|---|---|---|
| `grasp_site.pos` | `[0, 0, 0]` on `base_mount` | `[0, 0, 0.094]` on `base_mount` |
| `FINGER_LENGTH` | `0.129` m | `0.059` m |
| Collision rate (side-grasp × can) | 200/300 = 66.7% | 0/300 = 0.0% |

The arm's flange position in the world is unchanged — we moved the *label* "palm" forward by 94 mm *and* shrank `FL` by 94 mm; the fingertip's world pose is identical. What changed is that TSR no longer thinks the 94 mm of housing is free finger-extension.

### Pattern: `PALM_OFFSET_FROM_MOUNT` class constant

To keep the mounting-body → palm offset in one place, each `ParallelJawGripper` subclass exposes a class-level constant:

```python
class Robotiq2F85(ParallelJawGripper):
    FINGER_LENGTH = 0.059
    MAX_APERTURE = 0.085
    PALM_OFFSET_FROM_BASE_MOUNT = 0.094
```

Callers placing `grasp_site` in an `MjSpec` read from this constant:

```python
from tsr.hands import Robotiq2F85
site.pos = [0.0, 0.0, Robotiq2F85.PALM_OFFSET_FROM_BASE_MOUNT]
```

If the gripper's XML already has a `grasp_site` baked in (like the geodude_assets 2F-140 at `z=0.100`), the XML constant and the TSR `FINGER_LENGTH` must be consistent by construction — no external offset needed.

### Detection

`scripts/validate_gripper.py` walks each collision geom's AABB along the approach axis. If any non-finger geom has `approach_max > 0` (i.e., extends past `grasp_site` toward the object), *and* the collision sweep fails, it prints a concrete fix:

```
Diagnosis: the colliding geom(s) extend 16.9 mm forward of grasp_site.
TSR assumes everything except fingers is behind the palm — that's violated here.

Suggested fix:
  1. Move grasp_site forward by 0.017 m along the approach axis.
  2. Reduce FINGER_LENGTH: 0.054 → 0.037 m.
```

Some geoms can extend past `grasp_site` without causing failure (e.g., the 2F-85's spring-link arms extend +22 mm past the palm but sit way outside the object in the opening axis, so they don't collide against a can). The validator reports this as an informational note when the sweep *passes*, rather than a failure.

## 2. The menagerie position-actuator bug

### The actuator

Menagerie grippers commonly use a position actuator shaped like:

```
force = gain[0] · ctrl + bias[0] + bias[1] · length + bias[2] · vel
```

Where `length` is either the actuator tendon length (Robotiq 2-finger via `tendon="split"`) or the joint position (Franka via direct joint actuation). With menagerie defaults:

| Gripper | gain[0] | bias[0] | bias[1] | bias[2] | forcerange | ctrlrange |
|---|---|---|---|---|---|---|
| Robotiq 2F-85/2F-140 | 0.3137 | 0 | −100 | −10 | [−5, +5] | [0, 255] |
| Franka hand | (scaled) | some | negative | (damping) | (bounded) | [0, 255] |

For the Robotiq case, at `ctrl = 255` (full-close command):

| `length` | `0.3137 · 255` | `−100 · length` | Unclamped F | Clamped F (±5) |
|---|---|---|---|---|
| 0.00 (fully open) | 80 | 0 | 80 | **+5** |
| 0.40 (on a 66 mm can) | 80 | −40 | 40 | **+5** |
| 0.75 (nearly closed) | 80 | −75 | 5 | **+5** |
| 0.80 (fully closed) | 80 | −80 | **0** | **0** |

The intent was *position control*: `ctrl` sets a target position, the actuator drives toward it, force goes to zero when it arrives. That's correct for a free-moving pointer. For a gripper holding an object, the object blocks the fingers before they reach `length = 0.80`, so the `forcerange` clamp kicks in and the gripper squeezes at the clamp value. Fine.

### How this fails

If the object *slips through the jaws*, the fingers travel past their object-blocked position, `length` grows, and unclamped force drops below the clamp. At `length = 0.80` (empty full-close), force is exactly zero. The gripper relaxes instead of re-closing.

If the object then rolls back between the fingers, the gripper *has no restoring force* — it sits open, finger-apart, with zero closing torque. Cans fall out. Empty-close states never settle.

The second problem is the peak force itself. The `forcerange="-5 5"` clamp caps tendon force at 5 N, which on the 2F-85 and 2F-140 yields roughly 100 N and 90 N of pad force respectively (measured; see benchmark below). The real Robotiq 2F-85 spec is 20–235 N. We're at the low end, leaving no headroom for transport accelerations.

### The fix

Rewrite the actuator for constant force independent of `length`:

- `bias[0] = −target_force`    (at `ctrl = 0`, output `-target_force`: open)
- `bias[1] = 0`                (kill length coupling)
- `gain[0] = 2 · target_force / ctrl_max`    (at `ctrl = ctrl_max`, output `+target_force`: close)
- `forcerange = ±1.2 · target_force`          (headroom for velocity damping transients)
- `bias[2] *= |new_bias[0] / old_bias[1]|`    (preserve damping ratio)

Two helpers implement this:

- `fix_franka_grip_force(model, target_force=70.0)` in [`arms/franka.py`](../src/mj_manipulator/arms/franka.py)
- `fix_robotiq_grip_force(model, prefix, target_tendon_force=15.0)` in [`grippers/robotiq.py`](../src/mj_manipulator/grippers/robotiq.py)

Call them *after* `spec.compile()` and *before* instantiating the `Gripper` class — they mutate the compiled `MjModel`.

### Units: tendon force vs pad force

For the Franka, `target_force` is the direct force on the finger joint — no mechanical reduction. Pick 70 N to match the real Franka continuous-grip spec.

For the Robotiq, `target_tendon_force` is the *tendon* force, which gets distributed via the 4-bar linkage to the pads. The mechanical advantage is roughly **20×** on both 2F-85 and 2F-140 (measured empirically; see next section). So:

| `target_tendon_force` | 2F-85 pad force | 2F-140 pad force |
|---|---|---|
| 5 N (stock clamp) | ~100 N | ~90 N |
| 15 N (default in `fix_robotiq_grip_force`) | ~300 N | ~195 N |
| 25 N | ~500 N | ~325 N |

300 N pad force on the 2F-85 is middle-of-spec for the real hardware. 500 N would exceed max spec but is fine for sim against rigid cans.

## 3. The empirical finding: pad area beats grip force

When the 2F-85 grip fix landed, the expectation was "more grip force = holds better". The benchmark results said something different.

`/tmp/grip_force_bench.py` — standalone scene with a 66 mm can between the pads, close at `ctrl = 255`, read contact normal force after physics settles:

| Gripper | Fix | Tendon N | Pad sum N | Advantage | Contacts |
|---|---|---|---|---|---|
| 2F-85 | stock | 5.0 | 100 | 20× | 6 |
| 2F-85 | fixed (15 N) | 15.1 | 300 | 20× | 6 |
| 2F-140 | stock | 5.0 | 90 | 18× | 4 |
| 2F-140 | fixed (15 N) | 14.7 | 195 | 13× | 4 |

Two surprises:

1. **Stock grip force is nearly identical** between the two grippers (100 N vs 90 N pad force). Mechanical advantage is similar (~20×). The "2F-140 holds cans that the 2F-85 drops" behaviour **is not a force effect**.

2. **Pad geometry matters more than pad force for holding cylinders.** The 2F-140's pads are 65 × 27 mm; the 2F-85's are 22 × 8 mm. That's ~10× the surface area, producing much wider contact strips against the can's curved side and much more resistance to tangential perturbation (arm acceleration, object inertia during lift).

### Consequences

- The 2F-85 needs both the grip-force fix *and* larger pads (or higher-friction pads) to match 2F-140's reliability. For now we have the force fix only; an `add_robotiq_pad_friction` helper analogous to [`add_franka_pad_friction`](../src/mj_manipulator/arms/franka.py) is a reasonable follow-up.
- The 2F-140 has the same actuator bug as the 2F-85 (force → 0 at full close). It's latent because the larger pads mean cans never slip enough to trigger it. Fixing anyway for hygiene is tracked as [personalrobotics/geodude#189](https://github.com/personalrobotics/geodude/issues/189).

## 4. Diagnostic workflow

The three tools in `scripts/` target different phases of integration:

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │  visualize_grasps.py  →  validate_gripper.py  →  /tmp/grip_force_bench.py │
 │  (eyeball geometry)      (automated pass/fail)    (tune force)            │
 └──────────────────────────────────────────────────────────────────────┘
```

### `visualize_grasps.py` — interactive

Web viewer (viser) that teleports a bare gripper through TSR-sampled pre-grasp poses against a target object. Per-template cycling, live collision indicator (gripper vs target; intra-gripper contacts ignored), per-template hit-rate stats. Use **first** when eyeballing a new gripper — "does it look like a reasonable grasp pose?" before investigating IK or planning.

```bash
uv run python scripts/visualize_grasps.py --gripper <name> --object can
```

### `validate_gripper.py` — automated

Deterministic 50-samples × N-templates collision sweep plus AABB measurement. Exits non-zero on failure; prints specific fix suggestions. Suitable for CI.

```bash
uv run python scripts/validate_gripper.py --gripper <name>
uv run python scripts/validate_gripper.py --all       # every registered gripper
```

Run after each geometry change. Don't declare a gripper "done" without this passing.

### Grip-force benchmark (ad-hoc)

No committed script — the template lives at `/tmp/grip_force_bench.py` and shows how to stand up a minimal scene (bare gripper, test cylinder between pads, ctrl=255, physics loop, read `data.contact[].force`). Useful when:

- Tuning `target_force` / `target_tendon_force` for a specific object set.
- Measuring mechanical advantage when adding a non-Robotiq, non-Franka gripper.

## 5. Worked examples

### Robotiq 2F-85 — the housing-inside-palm case

Symptom: 4 of 6 side-grasp templates generated 100%-colliding pre-grasps against a can. The arm would plan fine for shallow grasps, then fail at mid/deep with "start in collision" errors from the planner.

Root cause: `grasp_site` at `base_mount` origin with `FINGER_LENGTH = 0.129`. The 2F-85's `base` body extends 94 mm past `base_mount` — the housing was in the ee_site's +z (approach) direction.

Fix: palm at `base_mount + [0, 0, 0.094]`, `FINGER_LENGTH = 0.059`, `MAX_APERTURE = 0.085` (inner-face gap, not outer). Validator goes from 200/300 collisions → 0/300. See [`demos/iiwa14_setup.py`](../src/mj_manipulator/demos/iiwa14_setup.py) for the canonical attach.

### Robotiq 2F-140 — the well-calibrated case

The geodude_assets 2F-140 XML bakes `grasp_site` at `base_mount + [0, 0, 0.100]` directly, which is ~6 mm past the housing's forward edge. `FINGER_LENGTH = 0.114` from that palm to the pad center. Validator passes out of the box. Lesson: if the gripper XML already has a `grasp_site`, read its position carefully before assuming `FINGER_LENGTH` corresponds to your own measurement.

### Franka hand — the latent small-offset case

Symptom: 33 % of side-grasp samples in collision, but only at the deepest depth. Shallow and mid grasps pass. The production recycling demo still works reliably because the planner's collision check silently filters out infeasible TSR proposals.

Root cause: same pattern as 2F-85 but with a smaller offset — the `hand` body extends 17 mm past the finger-joint origin (the XML position of `grasp_site`).

Fix: shift `grasp_site` to `hand + [0, 0, 0.0754]` (was `0.0584`), `FINGER_LENGTH = 0.037` (was `0.054`). Tracked as [personalrobotics/mj_manipulator#129](https://github.com/personalrobotics/mj_manipulator/issues/129) — latent-bug cleanup.

### Menagerie position-actuator bug, universally

Every menagerie gripper we've looked at (Franka hand, 2F-85, 2F-140) has a position actuator with `bias[1] != 0`. For the Franka it's the joint-position coupling; for the Robotiqs it's the tendon-length coupling. All share the same behaviour: grip force drops to zero at full close.

We fix it in the two demos we ship (`fix_franka_grip_force` in the Franka demo, `fix_robotiq_grip_force` in the iiwa14 demo). The 2F-140 in geodude is still on stock actuator but hasn't manifested the bug because of pad geometry — fixed for hygiene as a follow-up.

## 6. Frame convention reference

The canonical TSR ee_site frame:

```
                y  (finger-opening axis)
                ↑
                │   ──o──  left pad
                │         
    palm ─ ee ──┼───── approach direction → z
                │         
                │   ──o──  right pad
                ↓
               −y
```

- **+z** points from the palm toward the object (approach).
- **+y** points from right pad to left pad (opening direction — the "open more" direction for the left pad).
- **+x** = y × z, i.e., the palm's outward normal (right-hand rule).

Two failure modes on rotation:

- **Fingers open in the wrong axis**: the gripper's XML orientation has fingers spreading along its own x-axis (Robotiq) or some other. The TSR thinks fingers open along +y. Fix: set `grasp_site.quat` to rotate the gripper axes to match TSR's convention. Robotiq family: `[0.7071, 0, 0, -0.7071]` (−90° about z). Franka hand: identity (its frame already matches).
- **Approach direction reversed**: the gripper's internal +z points away from the fingertip rather than toward it. Fix: add a 180° rotation about the opening axis. Rare — most menagerie grippers orient the fingertip along +z.

When in doubt, run `visualize_grasps.py` — if the fingers don't visibly open along the "finger-opening" axis you expect, adjust the quaternion.
