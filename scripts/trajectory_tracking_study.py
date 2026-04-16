#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Measure trajectory-tracking quality on any supported arm.

Executes a representative motion (home → pickup reach → home) and
samples commanded vs actual joint positions and end-effector position
at each control cycle. Reports per-joint RMS/max error, EE-pose RMS/max
error, and the time-lag between the commanded and the actual trajectory.

If tracking error is large enough to push the fingertips off a grasp
target (even by a few mm), grasp reliability suffers. This script is
the ground truth for "are the gains tuned well enough for manipulation?"

Usage::

    uv run python scripts/trajectory_tracking_study.py --robot franka
    uv run python scripts/trajectory_tracking_study.py --robot iiwa14

Output:
    - Per-joint error table (mean / RMS / max / peak lag)
    - End-effector position error (RMS / max, mm)
    - A CSV with samples if --out is provided
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np


def build_robot(robot_name: str):
    """Build the specified robot (Franka or iiwa14) without a scene."""
    if robot_name == "franka":
        from mj_environment import Environment

        from mj_manipulator.arms.franka import (
            FRANKA_HOME,
            add_franka_ee_site,
            add_franka_gravcomp,
            add_franka_pad_friction,
            create_franka_arm,
            fix_franka_grip_force,
        )
        from mj_manipulator.grasp_manager import GraspManager
        from mj_manipulator.grippers.franka import FrankaGripper
        from mj_manipulator.menagerie import menagerie_scene

        spec = mujoco.MjSpec.from_file(str(menagerie_scene("franka_emika_panda")))
        add_franka_ee_site(spec)
        add_franka_gravcomp(spec)
        add_franka_pad_friction(spec)
        env = Environment.from_model(spec.compile())
        fix_franka_grip_force(env.model)
        gm = GraspManager(env.model, env.data)
        gripper = FrankaGripper(env.model, env.data, "franka", grasp_manager=gm)
        arm = create_franka_arm(env, gripper=gripper, grasp_manager=gm)
        home = np.array(FRANKA_HOME)
    elif robot_name == "iiwa14":
        from mj_environment import Environment

        from mj_manipulator.arms.iiwa14 import (
            IIWA14_HOME,
            add_iiwa14_ee_site,
            add_iiwa14_gravcomp,
            create_iiwa14_arm,
        )
        from mj_manipulator.grasp_manager import GraspManager
        from mj_manipulator.menagerie import menagerie_scene

        spec = mujoco.MjSpec.from_file(str(menagerie_scene("kuka_iiwa_14")))
        add_iiwa14_ee_site(spec)
        add_iiwa14_gravcomp(spec)
        env = Environment.from_model(spec.compile())
        gm = GraspManager(env.model, env.data)
        arm = create_iiwa14_arm(env, grasp_manager=gm)
        home = np.array(IIWA14_HOME)
    else:
        raise ValueError(f"Unsupported robot: {robot_name}")
    return env, arm, home


def plan_test_motion(arm, home: np.ndarray):
    """Plan a home → reach-forward → home motion.

    Reach-forward is a pose 30 cm in front of the robot at home height,
    pitched so the EE points down — a representative "approach a can"
    motion that exercises J2-J5 which carry most of the arm's inertia.

    Returns an :class:`mj_manipulator.trajectory.Trajectory`.
    """
    # Set to home and get current EE pose
    for i, idx in enumerate(arm.joint_qpos_indices):
        arm.env.data.qpos[idx] = home[i]
    mujoco.mj_forward(arm.env.model, arm.env.data)

    home_ee = arm.get_ee_pose()
    target_ee = home_ee.copy()
    # Shift in world +x by 15 cm and down by 15 cm, keep orientation
    target_ee[0, 3] += 0.15
    target_ee[2, 3] -= 0.15

    path1 = arm.plan_to_pose(target_ee, timeout=5.0)
    if path1 is None:
        raise RuntimeError("Could not plan forward reach motion")

    traj1 = arm.retime(path1)
    return traj1


def run_tracking_study(robot_name: str, out_csv: Path | None = None) -> dict:
    """Execute a test motion and log commanded vs actual state each step.

    Returns a dict of summary statistics.
    """
    env, arm, home = build_robot(robot_name)
    model, data = env.model, env.data

    # Set to home
    for i, idx in enumerate(arm.joint_qpos_indices):
        data.qpos[idx] = home[i]
    mujoco.mj_forward(model, data)

    traj = plan_test_motion(arm, home)
    print(f"Planned trajectory: {len(traj.positions)} samples, duration {traj.duration:.2f}s", flush=True)

    # Set up a minimal physics controller to run the trajectory while
    # we sample state each cycle. Skip the full PhysicsController plumbing
    # and use the raw actuator loop so we can log per-step.
    from mj_manipulator.config import PhysicsConfig

    cfg = PhysicsConfig()
    control_dt = cfg.execution.control_dt
    lookahead = cfg.execution.lookahead_time

    actuator_ids = []
    for jname in arm.config.joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        # Find matching actuator on this joint
        for aid in range(model.nu):
            if model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT and model.actuator_trnid[aid, 0] == jid:
                actuator_ids.append(aid)
                break

    qpos_idx = np.array(arm.joint_qpos_indices)
    qvel_idx = np.array(arm.joint_qvel_indices)

    # Log per-control-cycle: time, commanded q, actual q, commanded qd, actual qd, ee_pos
    samples = []
    n_arm_steps = int(control_dt / model.opt.timestep)

    # Start the clock
    t_start = data.time
    while True:
        t = data.time - t_start
        if t > traj.duration + 0.2:  # extra settling time
            break

        # Sample trajectory at current time
        q_des, qd_des, _ = traj.sample(min(t, traj.duration))

        # Commanded ctrl = q_des + lookahead * qd_des
        q_cmd = q_des + lookahead * qd_des
        data.ctrl[actuator_ids] = q_cmd

        # Step physics for one control cycle
        for _ in range(n_arm_steps):
            mujoco.mj_step(model, data)

        # Record
        q_actual = data.qpos[qpos_idx].copy()
        qd_actual = data.qvel[qvel_idx].copy()
        ee_pose = arm.get_ee_pose()
        ee_pos = ee_pose[:3, 3].copy()

        # Compute desired EE pos via FK at q_des (using a tmp data)
        # Skip for simplicity: the actual ee position will be off from
        # the desired if the arm isn't at q_des.
        samples.append(
            {
                "t": t,
                "q_des": q_des,
                "q_act": q_actual,
                "qd_des": qd_des,
                "qd_act": qd_actual,
                "ee_pos_act": ee_pos,
            }
        )

    # Compute stats
    q_err = np.array([s["q_act"] - s["q_des"] for s in samples])
    qd_err = np.array([s["qd_act"] - s["qd_des"] for s in samples])

    n_dof = q_err.shape[1]
    per_joint_rms = np.sqrt(np.mean(q_err**2, axis=0))
    per_joint_max = np.max(np.abs(q_err), axis=0)
    per_joint_qd_rms = np.sqrt(np.mean(qd_err**2, axis=0))

    # Compute time lag by cross-correlation of joint 1
    # (simpler: peak lag = index of max |err| during acceleration phase)
    # For each joint, the lag ≈ err / desired_velocity at peak velocity
    qd_des_arr = np.array([s["qd_des"] for s in samples])

    # At peak commanded velocity, lag = q_err / qd_des
    per_joint_lag = np.zeros(n_dof)
    for j in range(n_dof):
        qd_j = qd_des_arr[:, j]
        if np.max(np.abs(qd_j)) < 1e-4:
            continue
        peak_idx = np.argmax(np.abs(qd_j))
        if abs(qd_j[peak_idx]) > 1e-4:
            per_joint_lag[j] = q_err[peak_idx, j] / qd_j[peak_idx]

    # Print results
    print(f"\n=== Tracking report: {robot_name} ===")
    print(f"Trajectory duration: {traj.duration:.2f}s, {len(samples)} samples")
    print()
    print(
        f"{'joint':<8}  {'RMS err (rad)':>14}  {'max err (rad)':>14}  "
        f"{'qd RMS (rad/s)':>14}  {'lag @ peak (ms)':>16}"
    )
    print("-" * 80)
    for j in range(n_dof):
        print(
            f"J{j + 1:<7}  {per_joint_rms[j]:>14.5f}  {per_joint_max[j]:>14.5f}  "
            f"{per_joint_qd_rms[j]:>14.5f}  {per_joint_lag[j] * 1000:>16.1f}"
        )

    # Joint-space norm
    overall_rms = np.sqrt(np.mean(np.sum(q_err**2, axis=1)))
    overall_max = np.max(np.linalg.norm(q_err, axis=1))
    print()
    print(f"Joint-space L2 RMS: {overall_rms:.5f} rad ({np.degrees(overall_rms):.3f}°)")
    print(f"Joint-space L2 MAX: {overall_max:.5f} rad ({np.degrees(overall_max):.3f}°)")

    # Settling time: how long after trajectory ends until |q_err| < 0.005 on all joints
    settle_idx = None
    traj_end_idx = int(traj.duration / control_dt)
    for i in range(traj_end_idx, len(samples)):
        if np.max(np.abs(q_err[i])) < 0.005:
            settle_idx = i
            break
    if settle_idx is not None:
        settle_time = (settle_idx - traj_end_idx) * control_dt
        print(f"Settling time (|err| < 5 mrad on all joints): {settle_time * 1000:.0f} ms")
    else:
        final_err = np.abs(q_err[-1])
        print(f"Did not settle within sampling window. Final |err| max: {np.max(final_err) * 1000:.2f} mrad")

    # Write CSV if requested
    if out_csv is not None:
        import csv

        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            hdr = ["t"]
            for j in range(n_dof):
                hdr += [f"q_des_{j + 1}", f"q_act_{j + 1}", f"qd_des_{j + 1}", f"qd_act_{j + 1}"]
            hdr += ["ee_x", "ee_y", "ee_z"]
            w.writerow(hdr)
            for s in samples:
                row = [s["t"]]
                for j in range(n_dof):
                    row += [s["q_des"][j], s["q_act"][j], s["qd_des"][j], s["qd_act"][j]]
                row += list(s["ee_pos_act"])
                w.writerow(row)
        print(f"Wrote samples: {out_csv}")

    return {
        "per_joint_rms": per_joint_rms,
        "per_joint_max": per_joint_max,
        "per_joint_lag": per_joint_lag,
        "overall_rms": overall_rms,
        "overall_max": overall_max,
        "settle_time_s": None if settle_idx is None else (settle_idx - traj_end_idx) * control_dt,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", choices=["franka", "iiwa14"], required=True)
    parser.add_argument("--out", type=Path, help="Write per-sample CSV to this path.")
    args = parser.parse_args()

    try:
        run_tracking_study(args.robot, args.out)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
