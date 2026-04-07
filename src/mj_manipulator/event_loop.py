# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Single-threaded MuJoCo event loop.

MuJoCo is not thread-safe — concurrent mj_step/mj_forward calls segfault.
This module provides a PhysicsEventLoop that ensures all MuJoCo access
happens on one thread. Other threads submit work via Futures.

Design follows the game loop / update method pattern: tick() is the single
owner of the physics step. Trajectory runners and teleop controllers are
target providers that write to arm state each tick. One mj_step per cycle
with all arms' targets applied.

Usage (from geodude console.py)::

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
    from mj_manipulator.physics_controller import PhysicsController

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

    1. Processes queued commands (fast — e.g. starting a trajectory runner)
    2. Advances active trajectory runners (writes targets)
    3. Steps active teleop controllers (writes targets)
    4. Calls controller.step() — ONE mj_step with all arms
    5. Syncs the viewer
    6. Falls back to idle step if nothing else ran
    """

    def __init__(self) -> None:
        self._queue: queue.SimpleQueue[_Command] = queue.SimpleQueue()
        self._owner_thread: int = threading.get_ident()
        self._teleop_entries: list[tuple[Any, Any]] = []  # (controller, panel)
        self._teleop_lock = threading.Lock()  # protects _teleop_entries
        self._controller: PhysicsController | None = None
        self._idle_step_fn: Callable[[], None] | None = None
        self._viewer_sync_fn: Callable[[], None] | None = None

    # -- Controller setup (called from SimContext.__enter__) ------------------

    def set_controller(self, controller: PhysicsController | None) -> None:
        """Set the PhysicsController that tick() drives.

        When set, tick() calls controller.advance_all() and controller.step()
        each cycle. When None, falls back to idle_step_fn.
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

        When a PhysicsController is set (tick-driven mode):

        1. Process all queued commands (fast — start runners, etc.)
        2. Advance active trajectory runners (write targets)
        3. Step active teleop controllers (write targets)
        4. controller.step() — ONE mj_step with all arms
        5. Sync viewer

        When no controller is set (legacy mode):

        1. Process one queued command (may block — e.g. execute)
        2. Step active teleop controllers
        3. Idle physics step
        """
        if self._controller is not None:
            self._tick_driven()
        else:
            self._tick_legacy()

    def _tick_driven(self) -> None:
        """Tick-driven mode: controller owns physics, runners provide targets."""
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

        # 3. Step active teleop controllers (writes targets, no mj_step)
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

        # 4. Single physics step for ALL arms
        # (PhysicsController.step → _step_physics handles throttled viewer sync)
        self._controller.step()

    def _tick_legacy(self) -> None:
        """Legacy mode: backwards compatible with blocking execute()."""
        stepped = False

        # 1. Process one queued command (may block for seconds — e.g. execute)
        try:
            cmd = self._queue.get_nowait()
        except queue.Empty:
            pass
        else:
            self._deactivate_all_teleop()
            try:
                result = cmd.fn()
                cmd.future.set_result(result)
            except Exception as e:
                cmd.future.set_exception(e)
            stepped = True
            return

        # 2. Step active teleop controllers
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
                stepped = True

        if stepped and self._viewer_sync_fn is not None:
            try:
                self._viewer_sync_fn()
            except Exception:
                pass
            return

        # 3. Idle physics step (gravity, contacts, F/T sensors)
        if self._idle_step_fn is not None:
            try:
                self._idle_step_fn()
            except Exception:
                pass
