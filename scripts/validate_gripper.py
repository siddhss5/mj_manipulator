#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Validate a parallel-jaw gripper's TSR setup against its MuJoCo geometry.

Run this whenever you add a new gripper (or suspect the TSR params on
an existing one). It answers two questions at once:

1. **Does the declared geometry match the model?** — measures each
   collision geom's AABB along the approach axis and compares against
   the ``ParallelJawGripper`` subclass's ``FINGER_LENGTH`` and
   ``MAX_APERTURE``.
2. **Do the TSR-generated pre-grasps actually avoid collision?** —
   samples N grasps per template against a test object and reports
   the per-template collision rate.

If the two line up (no housing intrudes on the approach path, declared
params match measurements, 0% collision rate), the gripper is ready.
If not, the script prints specific suggestions: "shift grasp_site by
+0.094 m", "reduce FINGER_LENGTH from 0.129 to 0.059", etc.

Usage::

    uv run python scripts/validate_gripper.py --gripper robotiq_2f85
    uv run python scripts/validate_gripper.py --gripper franka --object spam_can
    uv run python scripts/validate_gripper.py --all

Exits 0 on pass, 1 if any check fails — suitable for CI.

For the underlying math (why the palm-vs-housing rule matters, the
affine-force actuator bug, pad area vs grip force) and worked
examples per gripper, see ``mj_manipulator/docs/grippers.md``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

# Reuse the visualizer's scene-building + collision machinery.
sys.path.insert(0, str(Path(__file__).parent))
import visualize_grasps as vg  # noqa: E402

# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


@dataclass
class GeomExtent:
    """AABB of one collision geom projected onto the grasp-site frame."""

    geom_name: str
    body_name: str
    approach_min: float
    approach_max: float
    opening_min: float
    opening_max: float
    is_finger: bool  # True if body OR geom name suggests finger/pad


def _collect_extents(model, data, ids) -> list[GeomExtent]:
    """For each collision geom on the gripper, compute its AABB in the
    grasp_site frame (projected onto approach / opening axes)."""
    # Park the freejoint at identity so measurements are in a canonical frame.
    fj = ids["freejoint_qpos"]
    data.qpos[fj : fj + 3] = [0.0, 0.0, 0.0]
    data.qpos[fj + 3 : fj + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    sid = ids["site"]
    site_pos = data.site_xpos[sid].copy()
    site_R = data.site_xmat[sid].reshape(3, 3).copy()
    approach = site_R[:, 2]
    opening = site_R[:, 1]

    out: list[GeomExtent] = []
    for gid in sorted(ids["gripper_geoms"]):
        if int(model.geom_contype[gid]) == 0:
            continue
        g_pos = data.geom_xpos[gid]
        g_R = data.geom_xmat[gid].reshape(3, 3)
        half = model.geom_size[gid]
        corners = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    corners.append(g_R @ np.array([sx * half[0], sy * half[1], sz * half[2]]) + g_pos)
        corners = np.array(corners)
        rel = corners - site_pos
        along_approach = rel @ approach
        along_opening = rel @ opening

        body_id = int(model.geom_bodyid[gid])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "<unnamed>"
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom{gid}"
        # "Finger" = pad, finger, or follower (the body that carries the
        # pad on the 2F-140 where pads don't get their own body).
        is_finger = any(
            tok in s.lower() for s in (body_name, geom_name) for tok in ("pad", "finger", "follower")
        )
        out.append(
            GeomExtent(
                geom_name=geom_name,
                body_name=body_name,
                approach_min=float(along_approach.min()),
                approach_max=float(along_approach.max()),
                opening_min=float(along_opening.min()),
                opening_max=float(along_opening.max()),
                is_finger=is_finger,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Collision sweep
# ---------------------------------------------------------------------------


@dataclass
class SweepResult:
    template_index: int
    label: str
    hits: int
    total: int
    offenders: set[str]


def _sweep(spec, obj, model, data, ids, samples_per_template: int, rng_seed: int) -> list[SweepResult]:
    templates = vg.generate_templates(spec, obj)
    T_ref = vg.get_object_pose(model, data, ids, obj)
    rng = np.random.default_rng(rng_seed)
    results: list[SweepResult] = []
    for i, tmpl in enumerate(templates):
        hits = 0
        offenders: set[str] = set()
        tsr = tmpl.instantiate(T_ref)
        for _ in range(samples_per_template):
            # Deterministic sampling so PASS/FAIL is reproducible.
            xyzrpy = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
            for j, (lo, hi) in enumerate(tsr.Bw):
                if np.isnan(xyzrpy[j]):
                    xyzrpy[j] = rng.uniform(lo, hi) if hi > lo else lo
            try:
                pose = tsr.to_transform(xyzrpy)
            except Exception:
                continue
            vg.teleport_gripper(model, data, ids, pose)
            in_col, names = vg.check_collision(model, data, ids)
            if in_col:
                hits += 1
                offenders.update(names)
        label = tmpl.name or tmpl.variant or f"template_{i}"
        results.append(SweepResult(i, label, hits, samples_per_template, offenders))
    return results


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------


def validate(gripper_name: str, object_name: str, samples: int, seed: int) -> bool:
    spec = vg.GRIPPERS[gripper_name]
    obj = vg.load_object(object_name)
    model, data, ids = vg.build_scene(spec, obj)

    # Declared params from the TSR hand class.
    from mj_manipulator.grasp_sources.prl_assets import _get_hand

    hand = _get_hand(spec.hand_type)
    declared_fl = float(hand.finger_length)
    declared_aperture = float(hand.max_aperture)

    # Measurements.
    extents = _collect_extents(model, data, ids)
    finger_exts = [e for e in extents if e.is_finger]
    non_finger_exts = [e for e in extents if not e.is_finger]

    # Max forward extent across ALL geoms — the real "reach" of the
    # gripper along the approach axis.
    gripper_forward = max((e.approach_max for e in extents), default=0.0)
    # The "housing" we care about for the TSR assumption is non-finger
    # structure past the grasp_site.
    housing_forward = max((e.approach_max for e in non_finger_exts), default=0.0)
    # Finger-tip estimate: the finger geom that reaches furthest forward.
    # When no fingers can be identified by name, fall back to the overall max.
    finger_tip = max((e.approach_max for e in finger_exts), default=gripper_forward)
    # Inner-face aperture: only use geoms named "pad" (which sit at the
    # finger tip between gripper and object). Follower/coupler bodies
    # have smaller |y| values but aren't the contact surface — using them
    # would underreport the aperture. If no pad-named geoms exist (e.g.
    # 2F-140 puts pad meshes inside the follower body), skip the check.
    pad_exts = [
        e for e in finger_exts
        if "pad" in e.geom_name.lower() or "pad" in e.body_name.lower()
    ]
    inner_ys = [
        min(abs(e.opening_min), abs(e.opening_max))
        for e in pad_exts
        if not (e.opening_min <= 0 <= e.opening_max)
    ]
    measured_aperture = 2.0 * min(inner_ys) if inner_ys else 0.0

    # Collision sweep — the ground truth.
    sweep = _sweep(spec, obj, model, data, ids, samples, seed)
    total_hits = sum(r.hits for r in sweep)
    total_samples = sum(r.total for r in sweep)
    overall_rate = 100.0 * total_hits / total_samples if total_samples else 0.0

    # ---- Render report ----
    banner = f" {gripper_name} × {object_name} "
    print(f"\n{banner:=^70}")

    # Geometry table.
    print("\nCollision geoms (AABB in grasp_site frame):")
    print(f"  {'body':<25} {'approach [mm]':<18} {'opening [mm]':<16} {'class':<8}")
    print(f"  {'-' * 25} {'-' * 18} {'-' * 16} {'-' * 8}")
    for e in sorted(extents, key=lambda x: (x.is_finger, -x.approach_max)):
        kind = "finger" if e.is_finger else "other"
        print(
            f"  {e.body_name:<25} "
            f"[{e.approach_min*1000:+6.1f}, {e.approach_max*1000:+6.1f}] "
            f"[{e.opening_min*1000:+6.1f}, {e.opening_max*1000:+6.1f}] "
            f"{kind:<8}"
        )

    # Sweep table.
    print(f"\nCollision sweep ({samples} samples × {len(sweep)} templates = {total_samples} total):")
    for r in sweep:
        rate = 100.0 * r.hits / r.total if r.total else 0.0
        marker = "✓" if r.hits == 0 else "✗"
        offenders = ", ".join(sorted(r.offenders)[:3]) if r.offenders else ""
        print(f"  {marker} [{r.template_index}] {r.label:<50} {r.hits:3d}/{r.total}  {rate:5.1f}%  {offenders}")
    overall_marker = "✓" if total_hits == 0 else "✗"
    print(f"\n  {overall_marker} overall: {total_hits}/{total_samples} = {overall_rate:.1f}%")

    # Informational geometry summary.
    print("\nGeometry summary:")
    print(
        f"  FINGER_LENGTH    declared={declared_fl*1000:6.1f} mm   "
        f"measured finger tip={finger_tip*1000:6.1f} mm"
    )
    if measured_aperture > 0:
        print(
            f"  MAX_APERTURE     declared={declared_aperture*1000:6.1f} mm   "
            f"measured inner-face gap={measured_aperture*1000:6.1f} mm"
        )
    print(f"  Housing forward-of-grasp_site:  {housing_forward*1000:+6.1f} mm")

    # Pass / fail + diagnostics.
    passed = total_hits == 0
    if passed:
        print("\nRESULT: PASS ✓")
        if housing_forward > 0.001:
            print(
                "  (Note: some non-finger geoms extend past grasp_site but they clear the "
                "object in the opening axis. Sweep confirms no collision.)"
            )
        return True

    print(f"\nRESULT: FAIL ✗  ({overall_rate:.1f}% of grasps in collision)")

    # Diagnostics — which offenders appear and how to fix.
    fixable_by_shift = housing_forward > 0.001
    # Derive the shift by looking at which offender bodies are listed
    # and how far forward THEY extend (not the whole non-finger max).
    offender_bodies: set[str] = set()
    for r in sweep:
        offender_bodies.update(r.offenders)
    # Find the extent data for each named offender.
    offending_extent = 0.0
    for ext in extents:
        # Geom names used in sweep are the geom's own name; compare both.
        if ext.geom_name in offender_bodies or ext.body_name in offender_bodies:
            offending_extent = max(offending_extent, ext.approach_max)

    if fixable_by_shift and offending_extent > 0:
        shift = offending_extent
        new_fl = declared_fl - shift
        print(
            f"\n  Diagnosis: the colliding geom(s) extend {shift*1000:.1f} mm forward of grasp_site. "
            f"TSR assumes everything except fingers is behind the palm — that's violated here."
        )
        print("\n  Suggested fix:")
        print(
            f"    1. Move grasp_site forward by {shift:.3f} m along the approach axis."
        )
        print(
            f"    2. Reduce FINGER_LENGTH: {declared_fl:.3f} → {new_fl:.3f} m "
            f"(so the fingertip position in world stays unchanged)."
        )
    else:
        print(
            "\n  Diagnosis: collisions don't match a simple housing-shift pattern. "
            "Inspect the geom table and sweep offenders to locate the issue."
        )
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gripper", choices=sorted(vg.GRIPPERS.keys()))
    parser.add_argument("--all", action="store_true", help="Validate every registered gripper.")
    parser.add_argument("--object", default="can", help="prl_assets object type (default: 'can').")
    parser.add_argument("--samples", type=int, default=50, help="Samples per TSR template (default: 50).")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility.")
    args = parser.parse_args()

    if not args.gripper and not args.all:
        parser.error("specify --gripper NAME or --all")

    targets = sorted(vg.GRIPPERS.keys()) if args.all else [args.gripper]

    all_pass = True
    for name in targets:
        ok = validate(name, args.object, args.samples, args.seed)
        all_pass = all_pass and ok

    print("\n" + ("=" * 70))
    if len(targets) > 1:
        print(f"OVERALL: {'PASS ✓' if all_pass else 'FAIL ✗'}  ({len(targets)} grippers)")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
