# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Status HUD overlay for the viser browser viewer.

Shows per-arm status: force, held object, and current action.
Works with any robot that implements ManipulationRobot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import viser
from mj_viser import MujocoViewer, PanelBase

from mj_manipulator.ownership import OwnerKind

if TYPE_CHECKING:
    pass


class StatusHud(PanelBase):
    """Compact status overlay on the 3D viewport.

    Reads live state from the robot each frame. Actions are set by
    primitives and auto-cleared after a timeout.
    """

    ACTION_TIMEOUT = 5.0  # seconds before action text clears

    def __init__(self, robot, mode: str = "") -> None:
        self._robot = robot
        self._mode = mode
        self._actions: dict[str, tuple[str, float]] = {}  # arm → (text, timestamp)

    def name(self) -> str:
        return "StatusHud"

    def set_action(self, arm_name: str, text: str) -> None:
        """Set the current action text for an arm."""
        import time

        self._actions[arm_name] = (text, time.monotonic())

    def clear_action(self, arm_name: str) -> None:
        """Clear the action text for an arm."""
        self._actions.pop(arm_name, None)

    def setup(self, gui: viser.GuiApi, viewer: MujocoViewer) -> None:
        self._viewer = viewer
        viewer.set_hud("status", self._build_status(), "bottom-left")

    def on_sync(self, viewer: MujocoViewer) -> None:
        viewer.set_hud("status", self._build_status(), "bottom-left")

    def _build_status(self) -> str:
        import time

        robot = self._robot
        now = time.monotonic()
        parts = []

        for arm_name, arm in robot.arms.items():
            # Label: full name for single-arm, first letter for multi-arm
            if len(robot.arms) == 1:
                label = arm_name
            else:
                label = arm_name[0].upper()

            # F/T magnitude (skip if no sensor)
            force_str = ""
            if arm.has_ft_sensor:
                wrench = arm.get_ft_wrench()
                if not np.isnan(wrench[0]):
                    force_mag = float(np.linalg.norm(wrench[:3]))
                    force_str = f"[{force_mag:.0f}N] "

            # Held object
            held = robot.grasp_manager.get_grasped_by(arm_name)
            held_str = held[0] if held else ""

            # Current action: teleop overrides, timed actions expire
            action = ""
            ctx = getattr(robot, "_active_context", None)
            if ctx is not None and hasattr(ctx, "ownership") and ctx.ownership is not None:
                kind, _ = ctx.ownership.owner_of(arm_name)
                if kind == OwnerKind.TELEOP:
                    action = "teleop"
                elif kind == OwnerKind.TRAJECTORY:
                    action = "executing"

            if not action and arm_name in self._actions:
                text, ts = self._actions[arm_name]
                if now - ts < self.ACTION_TIMEOUT:
                    action = text
                else:
                    del self._actions[arm_name]

            # Build arm status
            status = f"<b>{label}</b>: {force_str}"
            if held_str:
                status += held_str
            if action:
                if held_str:
                    status += f" | {action}"
                else:
                    status += action

            parts.append(status)

        line = " &nbsp;&nbsp; ".join(parts)
        if self._mode:
            line += f" &nbsp;&nbsp; {self._mode}"
        return line
