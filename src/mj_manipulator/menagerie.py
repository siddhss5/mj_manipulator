"""Utilities for locating the MuJoCo Menagerie robot model collection."""

from __future__ import annotations

import os
from pathlib import Path

_MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie"

_SEARCH_PATHS = [
    # Env var override (highest priority)
    lambda: Path(os.environ["MUJOCO_MENAGERIE_PATH"]) if "MUJOCO_MENAGERIE_PATH" in os.environ else None,
    # Sibling of this package's repo root (our workspace convention)
    lambda: Path(__file__).resolve().parents[3] / "mujoco_menagerie",
    # Home directory
    lambda: Path.home() / "mujoco_menagerie",
]


def find_menagerie() -> Path:
    """Return the path to the mujoco_menagerie directory.

    Search order:
      1. ``MUJOCO_MENAGERIE_PATH`` environment variable
      2. ``../mujoco_menagerie`` relative to this package's repo root
      3. ``~/mujoco_menagerie``

    Raises:
        FileNotFoundError: If the menagerie is not found in any location,
            with instructions for obtaining it.
    """
    for candidate_fn in _SEARCH_PATHS:
        path = candidate_fn()
        if path is not None and path.is_dir():
            return path

    raise FileNotFoundError(
        "mujoco_menagerie not found. Get it from:\n"
        f"  {_MENAGERIE_URL}\n\n"
        "Then either:\n"
        "  • Clone it next to this repo:  git clone "
        f"{_MENAGERIE_URL}\n"
        "  • Or set the env var:          "
        "export MUJOCO_MENAGERIE_PATH=/path/to/mujoco_menagerie"
    )


def menagerie_scene(robot_dir: str) -> Path:
    """Return the path to scene.xml for a named menagerie robot.

    Args:
        robot_dir: Directory name inside mujoco_menagerie, e.g.
            ``"universal_robots_ur5e"`` or ``"franka_emika_panda"``.

    Returns:
        Path to ``<menagerie>/<robot_dir>/scene.xml``.
    """
    return find_menagerie() / robot_dir / "scene.xml"
