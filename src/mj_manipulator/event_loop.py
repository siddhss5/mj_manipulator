# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Single-threaded MuJoCo event loop.

MuJoCo is not thread-safe — concurrent mj_step/mj_forward calls segfault.
This module provides a PhysicsEventLoop that ensures all MuJoCo access
happens on one thread. Other threads submit work via Futures.

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
    pass

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
    Each tick drains the command queue, steps active teleop controllers,
    and runs an idle physics step if nothing else stepped.
    """

    def __init__(self) -> None:
        self._queue: queue.SimpleQueue[_Command] = queue.SimpleQueue()
        self._owner_thread: int = threading.get_ident()
        self._teleop_entries: list[tuple[Any, Any]] = []  # (controller, panel)
        self._teleop_lock = threading.Lock()  # protects _teleop_entries
        self._idle_step_fn: Callable[[], None] | None = None
        self._viewer_sync_fn: Callable[[], None] | None = None
        self._executing_entity: str | None = None  # arm currently in execute()

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

    def is_executing(self, entity: str | None = None) -> bool:
        """Check if an entity is currently executing a trajectory.

        Args:
            entity: Arm name to check, or None to check if anything is executing.
        """
        if entity is None:
            return self._executing_entity is not None
        return self._executing_entity == entity

    def deactivate_teleop(self, entity: str | None = None) -> None:
        """Deactivate teleop controllers and reset their panels.

        Args:
            entity: Arm/entity name to deactivate (e.g. "left", "right").
                If None, deactivates all controllers.
        """
        with self._teleop_lock:
            if entity is None:
                to_deactivate = list(self._teleop_entries)
                self._teleop_entries.clear()
            else:
                to_deactivate = [
                    (c, p) for c, p in self._teleop_entries if c._arm.config.name == entity
                ]
                self._teleop_entries = [
                    (c, p) for c, p in self._teleop_entries if c._arm.config.name != entity
                ]
        for controller, panel in to_deactivate:
            try:
                controller.deactivate()
            except Exception:
                pass
            if panel is not None:
                try:
                    panel._on_teleop_error()
                except Exception:
                    pass

    # -- Main loop (owner thread only) ---------------------------------------

    def tick(self) -> None:
        """Process one event loop cycle. Called from the inputhook.

        Priority order:
        1. Queued commands (chat trajectory, etc.) — at most one per tick
        2. Active teleop controllers
        3. Idle physics step (hold position)
        """
        stepped = False

        # 1. Process one queued command (may block for seconds — e.g. execute)
        try:
            cmd = self._queue.get_nowait()
        except queue.Empty:
            pass
        else:
            # Deactivate all teleop controllers before running a command —
            # the command (trajectory, grasp, etc.) needs exclusive control.
            self.deactivate_teleop()
            try:
                result = cmd.fn()
                cmd.future.set_result(result)
            except Exception as e:
                cmd.future.set_exception(e)
            stepped = True
            # Return after a blocking command so the inputhook can check
            # input_is_ready() before processing more.
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

        # Sync viewer after teleop steps
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
