#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Head-to-head benchmark: EAIK (analytical) vs mink (numerical) IK.

Generates random reachable poses for each arm, solves with both solvers,
and compares solve time, solution count, FK accuracy, and solution
diversity. Optionally runs full CBiRRT planning episodes to measure
end-to-end planning time.

Usage::

    uv run python scripts/benchmark_ik.py
    uv run python scripts/benchmark_ik.py --arms ur5e,franka --n-poses 100
    uv run python scripts/benchmark_ik.py --with-planning --n-plans 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# Data classes for results
# ---------------------------------------------------------------------------


@dataclass
class IKCallResult:
    solve_time_ms: float
    n_solutions: int
    fk_errors_mm: list[float]  # FK position error for each solution
    converged: bool  # at least one solution found


@dataclass
class IKBenchResult:
    arm: str
    solver: str
    n_poses: int
    success_rate: float  # fraction of poses with ≥1 solution
    mean_solutions: float
    solve_time_mean_ms: float
    solve_time_p50_ms: float
    solve_time_p95_ms: float
    solve_time_max_ms: float
    fk_error_mean_mm: float
    fk_error_max_mm: float
    diversity_mean_rad: float  # mean pairwise joint-space distance


@dataclass
class PlanBenchResult:
    arm: str
    solver: str
    n_queries: int
    success_rate: float
    plan_time_mean_s: float
    plan_time_p95_s: float
    ik_calls_mean: int
    path_length_mean_rad: float


@dataclass
class BenchmarkReport:
    ik_results: list[IKBenchResult] = field(default_factory=list)
    plan_results: list[PlanBenchResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Arm builders
# ---------------------------------------------------------------------------


def _build_ur5e():
    from mj_environment import Environment

    from mj_manipulator.arms.ur5e import UR5E_HOME, add_ur5e_gravcomp, create_ur5e_arm
    from mj_manipulator.menagerie import menagerie_scene

    spec = mujoco.MjSpec.from_file(str(menagerie_scene("universal_robots_ur5e")))
    add_ur5e_gravcomp(spec)
    env = Environment.from_model(spec.compile())
    arm_eaik = create_ur5e_arm(env, with_ik=True)
    arm_bare = create_ur5e_arm(env, with_ik=False)
    return arm_eaik, arm_bare, np.array(UR5E_HOME), "attachment_site"


def _build_franka():
    from mj_environment import Environment

    from mj_manipulator.arms.franka import FRANKA_HOME, add_franka_ee_site, add_franka_gravcomp, create_franka_arm
    from mj_manipulator.menagerie import menagerie_scene

    spec = mujoco.MjSpec.from_file(str(menagerie_scene("franka_emika_panda")))
    add_franka_ee_site(spec)
    add_franka_gravcomp(spec)
    env = Environment.from_model(spec.compile())
    arm_eaik = create_franka_arm(env, with_ik=True)
    arm_bare = create_franka_arm(env, with_ik=False)
    return arm_eaik, arm_bare, np.array(FRANKA_HOME), "grasp_site"


def _build_iiwa14():
    from mj_environment import Environment

    from mj_manipulator.arms.iiwa14 import IIWA14_HOME, add_iiwa14_ee_site, add_iiwa14_gravcomp, create_iiwa14_arm
    from mj_manipulator.menagerie import menagerie_scene

    spec = mujoco.MjSpec.from_file(str(menagerie_scene("kuka_iiwa_14")))
    add_iiwa14_ee_site(spec)
    add_iiwa14_gravcomp(spec)
    env = Environment.from_model(spec.compile())
    arm_eaik = create_iiwa14_arm(env, with_ik=True)
    arm_bare = create_iiwa14_arm(env, with_ik=False)
    return arm_eaik, arm_bare, np.array(IIWA14_HOME), "grasp_site"


ARM_BUILDERS = {
    "ur5e": _build_ur5e,
    "franka": _build_franka,
    "iiwa14": _build_iiwa14,
}


# ---------------------------------------------------------------------------
# Pose generation
# ---------------------------------------------------------------------------


def generate_reachable_poses(arm, home: np.ndarray, n: int, seed: int) -> list[np.ndarray]:
    """Generate reachable poses by random FK within ±1 rad of home."""
    rng = np.random.default_rng(seed)
    q_lower, q_upper = arm.get_joint_limits()
    poses = []
    for _ in range(n):
        q = home + rng.uniform(-1.0, 1.0, len(home))
        q = np.clip(q, q_lower, q_upper)
        for i, idx in enumerate(arm.joint_qpos_indices):
            arm.env.data.qpos[idx] = q[i]
        mujoco.mj_forward(arm.env.model, arm.env.data)
        poses.append(arm.get_ee_pose().copy())
    return poses


# ---------------------------------------------------------------------------
# IK micro-benchmark
# ---------------------------------------------------------------------------


def _fk_errors(arm, solutions: list[np.ndarray], target: np.ndarray) -> list[float]:
    errors = []
    for q in solutions:
        for i, idx in enumerate(arm.joint_qpos_indices):
            arm.env.data.qpos[idx] = q[i]
        mujoco.mj_forward(arm.env.model, arm.env.data)
        ee = arm.get_ee_pose()
        errors.append(float(np.linalg.norm(ee[:3, 3] - target[:3, 3])) * 1000)
    return errors


def _pairwise_diversity(solutions: list[np.ndarray]) -> float:
    if len(solutions) < 2:
        return 0.0
    dists = []
    for i in range(len(solutions)):
        for j in range(i + 1, len(solutions)):
            dists.append(float(np.linalg.norm(solutions[i] - solutions[j])))
    return float(np.mean(dists))


def benchmark_ik(
    arm_name: str,
    solver_name: str,
    solver,
    arm,
    poses: list[np.ndarray],
    home: np.ndarray,
) -> IKBenchResult:
    """Run IK solve on each pose and collect statistics."""
    results: list[IKCallResult] = []
    all_errors: list[float] = []
    diversities: list[float] = []

    for pose in poses:
        t0 = time.perf_counter()
        solutions = solver.solve_valid(pose, q_init=home)
        elapsed = (time.perf_counter() - t0) * 1000

        errors = _fk_errors(arm, solutions, pose) if solutions else []
        all_errors.extend(errors)
        diversities.append(_pairwise_diversity(solutions))

        results.append(
            IKCallResult(
                solve_time_ms=elapsed,
                n_solutions=len(solutions),
                fk_errors_mm=errors,
                converged=len(solutions) > 0,
            )
        )

    times = [r.solve_time_ms for r in results]
    n_sols = [r.n_solutions for r in results]
    success = sum(1 for r in results if r.converged) / len(results)

    return IKBenchResult(
        arm=arm_name,
        solver=solver_name,
        n_poses=len(poses),
        success_rate=success,
        mean_solutions=float(np.mean(n_sols)),
        solve_time_mean_ms=float(np.mean(times)),
        solve_time_p50_ms=float(np.percentile(times, 50)),
        solve_time_p95_ms=float(np.percentile(times, 95)),
        solve_time_max_ms=float(np.max(times)),
        fk_error_mean_mm=float(np.mean(all_errors)) if all_errors else float("nan"),
        fk_error_max_mm=float(np.max(all_errors)) if all_errors else float("nan"),
        diversity_mean_rad=float(np.mean(diversities)),
    )


# ---------------------------------------------------------------------------
# Full planning benchmark
# ---------------------------------------------------------------------------


class _CountingIKSolver:
    """Wrapper that counts solve_valid calls."""

    def __init__(self, inner):
        self._inner = inner
        self.call_count = 0

    def solve(self, pose, q_init=None, **kw):
        return self._inner.solve(pose, q_init=q_init, **kw)

    def solve_valid(self, pose, q_init=None, **kw):
        self.call_count += 1
        return self._inner.solve_valid(pose, q_init=q_init, **kw)


def benchmark_planning(
    arm_name: str,
    solver_name: str,
    arm,
    solver,
    home: np.ndarray,
    n_queries: int,
    seed: int,
) -> PlanBenchResult:
    """Run full CBiRRT planning with the given IK solver."""
    rng = np.random.default_rng(seed)
    q_lower, q_upper = arm.get_joint_limits()

    times = []
    successes = 0
    ik_counts = []
    path_lengths = []

    for i in range(n_queries):
        # Random reachable target
        q_target = home + rng.uniform(-1.0, 1.0, len(home))
        q_target = np.clip(q_target, q_lower, q_upper)
        for j, idx in enumerate(arm.joint_qpos_indices):
            arm.env.data.qpos[idx] = q_target[j]
        mujoco.mj_forward(arm.env.model, arm.env.data)
        target_pose = arm.get_ee_pose().copy()

        # Reset to home
        for j, idx in enumerate(arm.joint_qpos_indices):
            arm.env.data.qpos[idx] = home[j]
        mujoco.mj_forward(arm.env.model, arm.env.data)

        # Swap IK solver temporarily
        counting = _CountingIKSolver(solver)
        original_solver = arm.ik_solver
        arm.ik_solver = counting

        t0 = time.perf_counter()
        result = arm.plan_to_pose(target_pose, timeout=5.0)
        elapsed = time.perf_counter() - t0

        arm.ik_solver = original_solver

        times.append(elapsed)
        ik_counts.append(counting.call_count)

        if result is not None:
            successes += 1
            # result is list[np.ndarray] — a path of joint configs
            length = sum(float(np.linalg.norm(result[i + 1] - result[i])) for i in range(len(result) - 1))
            path_lengths.append(length)

    return PlanBenchResult(
        arm=arm_name,
        solver=solver_name,
        n_queries=n_queries,
        success_rate=successes / n_queries if n_queries > 0 else 0.0,
        plan_time_mean_s=float(np.mean(times)),
        plan_time_p95_s=float(np.percentile(times, 95)) if times else 0.0,
        ik_calls_mean=int(np.mean(ik_counts)) if ik_counts else 0,
        path_length_mean_rad=float(np.mean(path_lengths)) if path_lengths else 0.0,
    )


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_ik_table(results: list[IKBenchResult]) -> None:
    hdr = (
        f"{'arm':<10} {'solver':<8} {'success':>8} {'sols/pose':>10} "
        f"{'mean ms':>8} {'p50 ms':>7} {'p95 ms':>7} {'max ms':>7} "
        f"{'err mm':>7} {'max mm':>7} {'div rad':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r.arm:<10} {r.solver:<8} {r.success_rate:>7.0%} {r.mean_solutions:>10.1f} "
            f"{r.solve_time_mean_ms:>8.1f} {r.solve_time_p50_ms:>7.1f} {r.solve_time_p95_ms:>7.1f} "
            f"{r.solve_time_max_ms:>7.1f} {r.fk_error_mean_mm:>7.2f} {r.fk_error_max_mm:>7.2f} "
            f"{r.diversity_mean_rad:>8.3f}"
        )


def print_plan_table(results: list[PlanBenchResult]) -> None:
    hdr = f"{'arm':<10} {'solver':<8} {'success':>8} {'mean s':>7} {'p95 s':>6} {'IK calls':>9} {'path rad':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r.arm:<10} {r.solver:<8} {r.success_rate:>7.0%} {r.plan_time_mean_s:>7.2f} "
            f"{r.plan_time_p95_s:>6.2f} {r.ik_calls_mean:>9d} {r.path_length_mean_rad:>9.2f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", default="ur5e,franka,iiwa14", help="Comma-separated arm names.")
    parser.add_argument("--n-poses", type=int, default=50, help="Number of random poses for IK benchmark.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument("--with-planning", action="store_true", help="Also run full CBiRRT planning benchmark.")
    parser.add_argument("--n-plans", type=int, default=10, help="Number of planning queries per arm×solver.")
    parser.add_argument("--output", type=Path, help="Write JSON results to this file.")
    args = parser.parse_args()

    arm_names = [a.strip() for a in args.arms.split(",")]
    report = BenchmarkReport()

    print("=" * 80)
    print("Phase A: IK micro-benchmark")
    print("=" * 80)

    for arm_name in arm_names:
        if arm_name not in ARM_BUILDERS:
            print(f"Unknown arm: {arm_name}", file=sys.stderr)
            continue

        print(f"\nBuilding {arm_name}...", flush=True)
        arm_eaik, arm_bare, home, ee_frame = ARM_BUILDERS[arm_name]()

        # Generate poses once (shared between solvers)
        poses = generate_reachable_poses(arm_bare, home, args.n_poses, args.seed)
        print(f"Generated {len(poses)} reachable poses.", flush=True)

        # EAIK
        eaik_solver = arm_eaik.ik_solver
        print("  Running EAIK...", flush=True)
        eaik_result = benchmark_ik(arm_name, "eaik", eaik_solver, arm_bare, poses, home)
        report.ik_results.append(eaik_result)

        # Mink
        from mj_manipulator.arms.mink_solver import make_mink_solver

        mink_solver = make_mink_solver(arm_bare)
        print("  Running mink...", flush=True)
        mink_result = benchmark_ik(arm_name, "mink", mink_solver, arm_bare, poses, home)
        report.ik_results.append(mink_result)

    print("\n")
    print_ik_table(report.ik_results)

    if args.with_planning:
        print("\n" + "=" * 80)
        print("Phase B: Full CBiRRT planning benchmark")
        print("=" * 80)

        for arm_name in arm_names:
            if arm_name not in ARM_BUILDERS:
                continue

            print(f"\nBuilding {arm_name}...", flush=True)
            arm_eaik, arm_bare, home, ee_frame = ARM_BUILDERS[arm_name]()

            from mj_manipulator.arms.mink_solver import make_mink_solver

            mink_solver = make_mink_solver(arm_bare)

            print("  Planning with EAIK...", flush=True)
            eaik_plan = benchmark_planning(
                arm_name, "eaik", arm_eaik, arm_eaik.ik_solver, home, args.n_plans, args.seed
            )
            report.plan_results.append(eaik_plan)

            print("  Planning with mink...", flush=True)
            mink_plan = benchmark_planning(arm_name, "mink", arm_bare, mink_solver, home, args.n_plans, args.seed)
            report.plan_results.append(mink_plan)

        print("\n")
        print_plan_table(report.plan_results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)
        print(f"\nResults written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
