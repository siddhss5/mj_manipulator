# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Generic IPython console for any ManipulationRobot.

Provides an interactive shell with physics simulation, viser viewer,
teleop panels, and manipulation primitives — no robot-specific code.

Usage::

    from mj_manipulator.console import start_console

    robot = MyRobot(objects={"mug": 1})
    start_console(robot, physics=True, viser=True)
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def start_console(
    robot,
    *,
    physics: bool = True,
    viser: bool = True,
    headless: bool = False,
    robot_name: str = "Robot",
    extra_ns: dict | None = None,
    panel_setup: Callable | None = None,
) -> None:
    """Launch an interactive IPython console for a manipulation robot.

    Sets up: physics event loop, viser viewer with teleop panels,
    pickup/place/go_home in the namespace.

    Args:
        robot: ManipulationRobot instance.
        physics: If True, simulate physics. If False, kinematic mode.
        viser: If True, launch a viser web viewer.
        headless: If True, no viewer at all.
        robot_name: Display name for the prompt (e.g. "Geodude").
        extra_ns: Additional entries for the IPython namespace.
        panel_setup: Optional callback(gui, viewer, robot, event_loop, tabs)
            to add robot-specific panels (chat, sensors, HUD, etc.).
    """
    import numpy as np
    from IPython.terminal.embed import InteractiveShellEmbed
    from IPython.terminal.prompts import Prompts, Token

    from mj_manipulator.primitives import go_home, pickup, place

    mode = "physics" if physics else "kinematic"

    # -- E-Stop / Resume (shared by CLI and browser) ----------------------------
    _estop_btn = None
    _resume_btn = None

    def estop():
        """E-Stop: halt all execution. Call release_estop() to clear."""
        robot.request_abort()
        if _estop_btn is not None:
            _estop_btn.visible = False
            _resume_btn.visible = True

    def release_estop():
        """Release E-Stop and allow execution again."""
        robot.clear_abort()
        if _estop_btn is not None:
            _estop_btn.visible = True
            _resume_btn.visible = False

    # -- Build namespace -------------------------------------------------------
    user_ns: dict = {
        "robot": robot,
        "np": np,
        "pickup": lambda target=None, **kw: pickup(robot, target, **kw),
        "place": lambda dest=None, **kw: place(robot, dest, **kw),
        "go_home": lambda **kw: go_home(robot, **kw),
        "estop": estop,
        "release_estop": release_estop,
    }
    if extra_ns:
        user_ns.update(extra_ns)

    # -- Banner ----------------------------------------------------------------
    n_arms = len(robot.arms)
    arm_str = f"{n_arms} arm{'s' if n_arms > 1 else ''}"
    viewer_str = " | viser" if viser else ""
    banner = f"\n{'=' * 60}\n  {robot_name} [{mode}] | {arm_str}{viewer_str}\n"
    if viser:
        banner += "  Browser: http://localhost:8080\n"
    banner += (
        f"{'=' * 60}\n\n"
        f"  pickup('object')  — pick up an object\n"
        f"  place('dest')     — place held object\n"
        f"  go_home()         — return to ready\n"
        f"  estop()           — E-Stop: halt everything\n"
        f"  release_estop()   — release E-Stop\n"
        f"  robot.<tab>       — tab completion\n"
    )

    # -- Viser viewer ----------------------------------------------------------
    viser_viewer = None
    tabs = None
    if viser:
        import viser as _viser
        from mj_viser import MujocoViewer

        viser_viewer = MujocoViewer(
            robot.model,
            robot.data,
            label=robot_name,
            show_sim_controls=False,
            show_visibility=False,
        )

        gui = viser_viewer._server.gui

        # E-Stop / Resume — above tabs so they're always visible
        _estop_btn = gui.add_button("E-Stop", color="red", icon=_viser.Icon.ALERT_OCTAGON)
        _resume_btn = gui.add_button("Release E-Stop", color="green", icon=_viser.Icon.PLAYER_PLAY, visible=False)

        _estop_btn.on_click(lambda _: estop())
        _resume_btn.on_click(lambda _: release_estop())

        tabs = gui.add_tab_group()

        viser_viewer.launch_passive(open_browser=False)
        print("  Viser viewer: http://localhost:8080")

    # -- Event loop ------------------------------------------------------------
    from mj_manipulator.event_loop import PhysicsEventLoop

    event_loop = PhysicsEventLoop()

    sim_viewer = viser_viewer if viser else None
    show_viewer = not headless and not viser
    with robot.sim(physics=physics, headless=show_viewer, viewer=sim_viewer, event_loop=event_loop) as ctx:
        user_ns["ctx"] = ctx

        # Event loop is wired to the controller in SimContext.__enter__

        # -- Teleop panels (needs ctx) -----------------------------------------
        if viser and viser_viewer is not None and tabs is not None:
            from mj_viser import TeleopPanel

            from mj_manipulator.teleop import TeleopController

            gui = viser_viewer._server.gui
            with tabs.add_tab("Teleop"):
                for arm_name, arm in robot.arms.items():
                    controller = TeleopController(arm, ctx)
                    gripper_prefix = ""
                    if arm.gripper and hasattr(arm.gripper, "gripper_body_names"):
                        # Derive prefix from first gripper body name
                        names = arm.gripper.gripper_body_names
                        if names:
                            prefix = names[0].rsplit("/", 1)[0] + "/"
                            gripper_prefix = prefix

                    panel = TeleopPanel(
                        arm=arm,
                        controller=controller,
                        model=robot.model,
                        data=robot.data,
                        gripper_body_prefix=gripper_prefix,
                        arm_label=arm_name.title(),
                        abort_fn=robot.is_abort_requested,
                        clear_abort_fn=robot.clear_abort,
                        request_abort_fn=robot.request_abort,
                        event_loop=event_loop,
                        ownership=ctx.ownership,
                    )
                    panel.setup(gui, viser_viewer)
                    viser_viewer._panels.append(panel)

        # -- Status HUD ------------------------------------------------------------
        if viser and viser_viewer is not None:
            from mj_manipulator.status_hud import StatusHud

            status_hud = StatusHud(robot, mode)
            robot._status_hud = status_hud
            viser_viewer._panels.append(status_hud)
            status_hud.setup(viser_viewer._server.gui, viser_viewer)

        # -- Robot-specific panels via callback --------------------------------
        if panel_setup is not None and viser_viewer is not None and tabs is not None:
            panel_setup(
                gui=viser_viewer._server.gui,
                viewer=viser_viewer,
                robot=robot,
                event_loop=event_loop,
                tabs=tabs,
            )

        # -- IPython shell -----------------------------------------------------
        class _Prompts(Prompts):
            def in_prompt_tokens(self, cli=None):
                stopped = " E-STOPPED" if robot.is_abort_requested() else ""
                return [
                    (Token.Prompt, f"{robot_name} [{mode}]{stopped} [{self.shell.execution_count}]: "),
                ]

            def out_prompt_tokens(self, cli=None):
                return [
                    (Token.OutPrompt, f"Out[{self.shell.execution_count}]: "),
                ]

        shell = InteractiveShellEmbed(
            banner1=banner,
            user_ns=user_ns,
            colors="neutral",
        )
        shell.prompts = _Prompts(shell)

        # -- Inputhook (drives event loop while REPL waits for input) ----------
        control_dt = ctx.control_dt

        def _inputhook(context):
            t_next = time.monotonic() + control_dt
            while not context.input_is_ready():
                now = time.monotonic()
                if now >= t_next:
                    event_loop.tick()
                    t_next = now + control_dt
                else:
                    time.sleep(min(t_next - now, 0.001))

        shell._inputhook = _inputhook

        shell()
