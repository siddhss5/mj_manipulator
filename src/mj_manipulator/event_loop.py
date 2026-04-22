# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Single-threaded MuJoCo event loop.

MuJoCo is not thread-safe — concurrent mj_step/mj_forward calls segfault.
This module provides a PhysicsEventLoop that ensures all MuJoCo access
happens on one thread. Other threads submit work via Futures.

Design follows the game loop / update method pattern: tick() is the single
owner of the step. Trajectory runners and teleop controllers are target
providers that write to arm state each tick. One step per cycle with all
arms' targets applied. Works identically for physics and kinematic modes.

Usage (from console.py)::

    loop = PhysicsEventLoop()
    # ... pass loop to SimContext ...

    def inputhook(context):
        while not context.input_is_ready():
            loop.tick()
            time.sleep(1 / 60)

    shell._inputhook = inputhook
"""

from __future__ import annotations

import logging
import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from mj_manipulator.controller import Controller

logger = logging.getLogger(__name__)


@dataclass
class _Command:
    """A unit of work to execute on the physics thread."""

    fn: Callable[[], Any]
    future: Future = field(default_factory=Future)


class PhysicsEventLoop:
    """Single-threaded MuJoCo event loop.

    All mj_step/mj_forward calls must happen on the thread that created
    this object (the "owner" thread). Other threads call submit() or go
    through SimContext methods which dispatch automatically.

    The owner thread calls tick() in a loop (typically an IPython inputhook).
    Each tick:

    1. Advances active trajectory runners (writes targets)
    2. Processes queued commands (fast — e.g. starting a trajectory runner)
    3. Steps active teleop controllers (writes targets)
    4. Calls controller.step() — one control cycle with all arms
    """

    def __init__(self) -> None:
        self._queue: queue.SimpleQueue[_Command] = queue.SimpleQueue()
        self._owner_thread: int = threading.get_ident()
        self._teleop_entries: list[tuple[Any, Any]] = []  # (controller, panel)
        self._teleop_lock = threading.Lock()  # protects _teleop_entries
        self._controller: Controller | None = None
        self._in_tick: bool = False  # reentrancy guard

    # -- Controller setup (called from SimContext.__enter__) ------------------

    def set_controller(self, controller: Controller | None) -> None:
        """Set the Controller that tick() drives.

        When set, tick() calls controller.advance_all() and controller.step()
        each cycle. When None, tick() is a no-op.
        """
        self._controller = controller

    # -- Public API (any thread) ---------------------------------------------

    def submit(self, fn: Callable[[], Any]) -> Future:
        """Submit work to the physics thread. Returns a Future."""
        cmd = _Command(fn=fn)
        self._queue.put(cmd)
        return cmd.future

    def run_on_physics_thread(self, fn: Callable[[], Any]) -> Any:
        """Run fn on the physics thread.

        If already on the physics thread, call directly.
        If on another thread, submit and block until complete.
        """
        if threading.get_ident() == self._owner_thread:
            return fn()
        fut = self.submit(fn)
        return fut.result()

    # -- Teleop registration (called from viser callbacks) -------------------

    def register_teleop(self, controller: Any, panel: Any = None) -> None:
        """Register a teleop controller to be stepped each tick.

        Thread-safe — may be called from viser callbacks.
        """
        with self._teleop_lock:
            self._teleop_entries.append((controller, panel))

    def unregister_teleop(self, controller: Any) -> None:
        """Remove a teleop controller from the tick loop.

        Thread-safe — may be called from viser callbacks.
        """
        with self._teleop_lock:
            self._teleop_entries = [(c, p) for c, p in self._teleop_entries if c is not controller]

    def _deactivate_all_teleop(self) -> None:
        """Deactivate all teleop controllers and reset their panels."""
        with self._teleop_lock:
            entries = list(self._teleop_entries)
            self._teleop_entries.clear()
        for controller, panel in entries:
            try:
                controller.deactivate()
            except Exception:
                pass
            if panel is not None:
                try:
                    panel._on_teleop_error()  # resets gizmo/button/status
                except Exception:
                    pass

    # -- Main loop (owner thread only) ---------------------------------------

    def tick(self) -> None:
        """Process one event loop cycle. Called from the inputhook.

        Requires a controller to be set via :meth:`set_controller`.
        Each tick:

        1. Advance active trajectory runners (write targets)
        2. Process all queued commands (fast — start runners, etc.)
        3. Step active teleop controllers (write targets)
        4. controller.step() — one control cycle with all arms
        """
        if self._in_tick:
            return  # prevent recursion (e.g. teleop → step_cartesian → tick)
        if self._controller is None:
            return
        self._in_tick = True
        try:
            self._tick_driven()
        finally:
            self._in_tick = False

    def _tick_driven(self) -> None:
        """Controller-driven tick: runners provide targets, controller steps."""
        # 1. Advance active trajectory runners (writes targets)
        #    Runs BEFORE commands so that abort flags set by preempt()
        #    are seen before _do_activate clears them.
        self._controller.advance_all()

        # 2. Process ALL queued commands (they're fast — start runners, activate teleop)
        while True:
            try:
                cmd = self._queue.get_nowait()
            except queue.Empty:
                break
            try:
                result = cmd.fn()
                cmd.future.set_result(result)
            except Exception as e:
                cmd.future.set_exception(e)

        # 3. Step active teleop controllers (writes targets, no step)
        with self._teleop_lock:
            entries = list(self._teleop_entries)
        for controller, panel in entries:
            if controller.is_active:
                try:
                    state = controller.step()
                    if panel is not None:
                        panel._update_status(state)
                except Exception as e:
                    logger.warning("Teleop step error: %s", e)
                    controller.deactivate()
                    if panel is not None:
                        panel._on_teleop_error()

        # 4. Single control step for ALL arms
        # (Controller.step → _apply_targets_and_step handles mode-specific logic)
        self._controller.step()
