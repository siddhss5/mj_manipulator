# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tick timing profiler for diagnosing jitter.

Measures per-tick duration breakdown: advance, teleop, physics step,
total. Run with pytest -s to see the timing report.
"""

import time

import numpy as np
import pytest
from conftest import MockArm, make_trajectory

from mj_manipulator.config import PhysicsExecutionConfig
from mj_manipulator.event_loop import PhysicsEventLoop
from mj_manipulator.physics_controller import PhysicsController


@pytest.fixture
def setup(model_and_data):
    model, data = model_and_data
    arm = MockArm("test_arm", model, data)
    loop = PhysicsEventLoop()
    ctrl = PhysicsController(
        model,
        data,
        {"test_arm": arm},
        config=PhysicsExecutionConfig(control_dt=0.002),
    )
    loop.set_controller(ctrl)
    return loop, ctrl


def test_tick_timing_profile(setup):
    """Profile per-tick costs during trajectory execution.

    Run with: uv run pytest tests/test_tick_timing.py -s
    """
    loop, ctrl = setup

    positions = np.array([[i * 0.01, i * 0.01] for i in range(100)])
    traj = make_trajectory(positions, entity="test_arm")
    future = ctrl.start_trajectory("test_arm", traj)

    tick_times = []
    advance_times = []
    step_times = []

    for _ in range(200):
        if future.done():
            break

        t0 = time.perf_counter()

        # Measure advance
        ta = time.perf_counter()
        ctrl.advance_all()
        advance_times.append(time.perf_counter() - ta)

        # Measure physics step
        ts = time.perf_counter()
        ctrl.step()
        step_times.append(time.perf_counter() - ts)

        tick_times.append(time.perf_counter() - t0)

    tick_us = np.array(tick_times) * 1e6
    adv_us = np.array(advance_times) * 1e6
    step_us = np.array(step_times) * 1e6

    print("\n--- Tick timing (microseconds) ---")
    print(f"{'Component':<15} {'mean':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8}")
    for name, arr in [("advance", adv_us), ("step", step_us), ("total", tick_us)]:
        print(
            f"{name:<15} {np.mean(arr):8.1f} {np.median(arr):8.1f} "
            f"{np.percentile(arr, 95):8.1f} {np.percentile(arr, 99):8.1f} "
            f"{np.max(arr):8.1f}"
        )

    control_dt_us = ctrl.control_dt * 1e6
    print(f"\ncontrol_dt: {control_dt_us:.0f} µs")
    print(f"ticks over budget (>{control_dt_us:.0f} µs): {np.sum(tick_us > control_dt_us)}/{len(tick_us)}")

    # Tick should comfortably fit within control_dt
    assert np.percentile(tick_us, 95) < control_dt_us, (
        f"p95 tick ({np.percentile(tick_us, 95):.0f} µs) exceeds "
        f"control_dt ({control_dt_us:.0f} µs)"
    )
