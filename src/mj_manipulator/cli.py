# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""CLI entry point for ``python -m mj_manipulator``.

Runs a scenario on one of the bundled robots. Scenarios live as
Python modules under :mod:`mj_manipulator.demos` — each defines a
``scene`` dict and optional user-facing functions. See
:mod:`mj_manipulator.scenarios` for the protocol.

Supported robots (``--robot``):

- ``franka`` (default): Franka Panda with its built-in hand.
- ``iiwa14``: KUKA LBR iiwa 14 with a Robotiq 2F-85 attached.

Usage::

    python -m mj_manipulator                               # Franka + picker
    python -m mj_manipulator --robot iiwa14                # iiwa14 + picker
    python -m mj_manipulator --scenario recycling          # run directly
    python -m mj_manipulator --robot iiwa14 --scenario recycling
    python -m mj_manipulator --list-scenarios              # print and exit
    python -m mj_manipulator --no-physics --no-viser       # flags compose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mj_manipulator import scenarios

# Directory where this package's scenarios live. Users can add their own
# scenarios by placing Python modules here (or pass --scenario <path>).
_SCENARIO_DIR = Path(__file__).parent / "demos"


_ROBOTS = {
    "franka": ("Franka", "mj_manipulator.demos.franka_setup", "build_franka_robot"),
    "iiwa14": ("iiwa14", "mj_manipulator.demos.iiwa14_setup", "build_iiwa14_robot"),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mj_manipulator",
        description="Run a manipulation scenario on a bundled robot.",
    )
    parser.add_argument(
        "--robot",
        choices=sorted(_ROBOTS.keys()),
        default="franka",
        help="Which robot to load (default: franka).",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Scenario name (e.g. 'recycling') or path to a .py file. If omitted, shows a picker.",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List available scenarios and exit.",
    )
    parser.add_argument("--no-physics", action="store_true", help="Kinematic mode (no physics stepping)")
    parser.add_argument("--no-viser", action="store_true", help="Disable viser web viewer")
    args = parser.parse_args()

    if args.list_scenarios:
        _list_scenarios()
        return

    # Resolve the scenario module.
    scenario_module = _resolve_scenario(args.scenario)
    if scenario_module is None:
        sys.exit(0)

    scene = getattr(scenario_module, "scene", None) or {}
    scenario_name = scenario_module.__name__

    # Dispatch to the chosen robot's builder.
    import importlib

    display_name, builder_module, builder_fn = _ROBOTS[args.robot]
    print(f"\nLoading {display_name} with scenario '{scenario_name}'...", flush=True)
    module = importlib.import_module(builder_module)
    build_fn = getattr(module, builder_fn)
    robot = build_fn(scene)

    from mj_manipulator.console import start_console

    user_fns = scenarios.get_user_functions(scenario_module, robot)
    extra_ns = dict(user_fns)
    extra_ns["reset"] = lambda: robot.reset(scene)

    start_console(
        robot,
        physics=not args.no_physics,
        viser=not args.no_viser,
        robot_name=display_name,
        extra_ns=extra_ns,
    )


def _resolve_scenario(name_or_path: str | None):
    """Pick a scenario interactively or load the given name."""
    if name_or_path is None:
        return scenarios.choose_interactive([_SCENARIO_DIR])
    try:
        return scenarios.load(name_or_path, search_dirs=[_SCENARIO_DIR])
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def _list_scenarios() -> None:
    """Print discovered scenarios with their descriptions."""
    found = scenarios.discover([_SCENARIO_DIR])
    if not found:
        print("No scenarios found.")
        return
    print("\nAvailable scenarios:\n")
    for name, path in found.items():
        print(f"  {name:20s} — {scenarios.describe(path)}")
    print()
