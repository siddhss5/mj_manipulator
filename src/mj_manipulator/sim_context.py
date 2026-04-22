# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Simulation execution context for MuJoCo.

Implements the ExecutionContext protocol for MuJoCo simulation, with both
kinematic (perfect tracking) and physics (realistic PD control) modes.

For real hardware, a separate HardwareContext would implement the same
ExecutionContext protocol using RTDE, ROS, libfranka, etc.

Usage::

    arms = {"ur5e": arm}
    with SimContext(model, data, arms, physics=True) as ctx:
        path = arm.plan_to_pose(target)
        traj = arm.retime(path)
        ctx.execute(traj)

        ctx.arm("ur5e").grasp("mug")

        while ctx.is_running():
            ctx.step({"ur5e": policy(arm.get_joint_positions())})
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from mj_manipulator.arm import Arm
    from mj_manipulator.config import PhysicsConfig
    from mj_manipulator.controller import Controller
    from mj_manipulator.event_loop import PhysicsEventLoop
    from mj_manipulator.ownership import OwnershipRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SimArmController
# ---------------------------------------------------------------------------


class SimArmController:
    """ArmController for simulation — grasp/release in physics or kinematic mode.

    In physics mode, delegates gripper actuation to PhysicsController for
    realistic contact-based grasp detection. In kinematic mode, uses the
    Gripper's kinematic_close/kinematic_open directly.

    In both modes, updates the GraspManager for collision-aware planning
    and creates kinematic attachments so grasped objects move with the gripper.

    This is the simulation implementation of the ArmController protocol.
    On real hardware, a HardwareArmController would use real gripper feedback
    (F/T sensing, gripper position) instead of MuJoCo contacts.
    """

    def __init__(self, arm: Arm, context: SimContext):
        self._arm = arm
        self._context = context

    def grasp(self, object_name: str | None = None) -> str | None:
        """Close gripper and attempt to grasp an object.

        In physics mode: gradually closes the gripper with contact detection.
        In kinematic mode: directly closes and assumes grasp succeeds.

        Args:
            object_name: Name of the object to grasp, or None to grasp
                whatever is between the fingers.

        Returns:
            Name of grasped object if successful, None otherwise.
        """
        gripper = self._arm.gripper
        if gripper is None:
            return None

        candidates = [object_name] if object_name else None
        gripper.set_candidate_objects(candidates)
        arm_name = self._arm.config.name

        # Tare the wrist F/T right before closing: the reading at this
        # moment reflects gripper weight + any existing load. Once the
        # object is held, subsequent get_ft_wrench() readings reflect
        # the object's weight alone, which is what GraspVerifier's
        # WristFTSignal compares against its post-grasp baseline. Skip
        # when the arm has no F/T sensor or F/T isn't meaningful
        # (kinematic mode).
        if self._arm.has_ft_sensor and self._arm.ft_valid:
            self._arm.tare_ft()

        # Resolve which object we think we're grasping. For the BT path
        # this is always the target name. For the nameless interactive
        # path (REPL / teleop), we fall back to find_contacted_object —
        # a simple contact-count heuristic that's sim-only by design.
        # On hardware the nameless path will be identified by the
        # PerceptionService after close (see HardwarePerceptionService
        # in mj_manipulator_ros). The bookkeeping below (grasp_manager
        # and verifier) sets up sim's kinematic weld; on hardware
        # there's no weld and the held-object pose comes from FK.
        target = object_name
        if target is None:
            from mj_manipulator.grasp_manager import find_contacted_object

            # Close the gripper first so the post-close contact state
            # reflects what we grabbed.
            self._run_close(candidates)
            target = find_contacted_object(
                self._arm.env.model,
                self._arm.env.data,
                gripper.gripper_body_names,
            )
            if target is None:
                logger.info("Grasp (no target): no object detected between fingers")
                self._context.sync()
                return None
        else:
            self._run_close(candidates)

        # Record the grasp in bookkeeping (kinematic weld) and in the
        # verifier (baseline capture + HOLDING state).
        #
        # The verifier enters HOLDING *inside its settling window* —
        # drop-detection is suppressed for the first ``settling_ticks``
        # ticks to give the actuator time to settle into its final
        # grip configuration. Real verification runs on whatever the
        # next physics cycle is, which is the next motion's
        # ``ctx.execute`` call. For the BT path this is always a
        # Sync/LiftBase that immediately runs through the event
        # loop's normal tick pump, so the settling window elapses
        # before any consumer reads ``is_held`` as authoritative.
        #
        # We deliberately do *not* pump extra physics ticks here to
        # \"confirm\" the grasp synchronously: doing so blocks the
        # caller thread and races with the event loop's own tick
        # schedule, producing (a) visible lag in every downstream
        # movement and (b) false positives when the verifier runs
        # its first verification inside the actuator-integration
        # transient that follows close_gripper. See the investigation
        # recorded in the recycling-demo diagnostic run where
        # empty-close → pos=1.000 but a healthy can-close → pos=0.46
        # with perfectly stable post-settle readings — the readings
        # themselves were fine, the synchronous tick pump was reading
        # them during the transient settling window.
        if self._arm.grasp_manager is not None:
            self._arm.grasp_manager.mark_grasped(target, arm_name)
            self._arm.grasp_manager.attach_object(target, gripper.attachment_body)

        if gripper.grasp_verifier is not None:
            gripper.grasp_verifier.mark_grasped(target)

        logger.info("Grasped %s with %s arm", target, arm_name)
        self._context.sync()
        return target

    def _run_close(self, candidates: list[str] | None) -> None:
        """Close the gripper via the controller (physics or kinematic)."""
        arm_name = self._arm.config.name
        self._context._controller.close_gripper(arm_name, candidate_objects=candidates)

    def release(self, object_name: str | None = None) -> None:
        """Open gripper and release held object(s).

        Args:
            object_name: Specific object to release, or None for all
                objects held by this arm.
        """
        gripper = self._arm.gripper
        if gripper is None:
            return

        arm_name = self._arm.config.name
        gm = self._arm.grasp_manager

        # Determine which objects to release
        if object_name is not None:
            objects = [object_name]
        elif gm is not None:
            objects = list(gm.get_grasped_by(arm_name))
        else:
            objects = []

        # Update grasp manager first (before opening gripper)
        if gm is not None:
            for obj in objects:
                gm.mark_released(obj)
                gm.detach_object(obj)

        # Clear the verifier's baseline so is_held returns False after
        # the release. Release is a hard ground truth — we know we're
        # letting go, no need to consult sensors.
        if gripper.grasp_verifier is not None:
            gripper.grasp_verifier.mark_released()

        self._context._controller.open_gripper(arm_name)

        self._context.sync()


# ---------------------------------------------------------------------------
# SimContext
# ---------------------------------------------------------------------------


class SimContext:
    """Simulation execution context — implements ExecutionContext.

    Provides unified trajectory execution, streaming joint/cartesian control,
    and grasp/release operations in MuJoCo simulation. Supports both kinematic
    (perfect tracking, no dynamics) and physics (PD control with settling).

    This is the simulation implementation of the ExecutionContext protocol.
    For real hardware, a HardwareContext implements the same protocol using
    the robot's driver (RTDE, ROS, libfranka, etc.).

    Three execution patterns::

        # 1. Batch trajectory execution
        path = arm.plan_to_pose(target)
        traj = arm.retime(path)
        ctx.execute(traj)

        # 2. Streaming joint control
        while ctx.is_running():
            ctx.step({"ur5e": policy(arm.get_joint_positions())})

        # 3. Streaming cartesian control
        while ctx.is_running():
            ctx.step_cartesian("ur5e", q_new, qd_new)

    Args:
        model: MuJoCo model.
        data: MuJoCo data (modified during execution).
        arms: Dict mapping arm names to Arm instances.
        physics: If True, use physics simulation. If False, kinematic mode.
        headless: If True, no viewer is created.
        viewer: Optional pre-existing MuJoCo viewer.
        physics_config: Combined PhysicsConfig (execution + gripper + recovery).
        initial_positions: Optional per-arm initial joint positions.
        viewer_fps: Target viewer refresh rate in Hz.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        arms: dict[str, Arm],
        *,
        physics: bool = True,
        headless: bool = False,
        viewer=None,
        physics_config: PhysicsConfig | None = None,
        initial_positions: dict[str, np.ndarray] | None = None,
        viewer_fps: float = 30.0,
        entities: dict[str, object] | None = None,
        abort_fn: object | None = None,
        event_loop: PhysicsEventLoop | None = None,
    ):
        self._model = model
        self._data = data
        self._arms = arms
        self._entities = entities or {}
        self._physics = physics
        self._headless = headless
        self._viewer = viewer
        self._physics_config = physics_config
        self._initial_positions = initial_positions
        self._abort_fn = abort_fn
        self._viewer_fps = viewer_fps
        self._event_loop = event_loop

        self._controller: Controller | None = None
        self._executors: dict[str, object] = {}
        self._arm_controllers: dict[str, SimArmController] = {}
        self._owns_viewer = False
        self._last_viewer_sync = 0.0
        self._viewer_sync_interval = 0.0  # set in __enter__
        self._ownership: OwnershipRegistry | None = None

    @property
    def ownership(self) -> OwnershipRegistry | None:
        """Arm ownership registry, available when event loop is set."""
        return self._ownership

    def __enter__(self) -> SimContext:
        """Enter context: create viewer and set up executors."""
        # Create viewer if needed
        if self._viewer is None and not self._headless:
            self._viewer = mujoco.viewer.launch_passive(
                self._model,
                self._data,
                show_left_ui=False,
                show_right_ui=False,
            )
            self._owns_viewer = True

            # Apply model's camera defaults (launch_passive overrides them)
            self._viewer.cam.azimuth = self._model.vis.global_.azimuth
            self._viewer.cam.elevation = self._model.vis.global_.elevation
            self._viewer.cam.distance = self._model.stat.extent * 1.5
            self._viewer.cam.lookat[:] = self._model.stat.center

        # Viewer sync interval
        if self._viewer_fps <= 0 or self._viewer_fps == float("inf"):
            viewer_sync_interval = 0.0
        else:
            viewer_sync_interval = 1.0 / self._viewer_fps
        self._viewer_sync_interval = viewer_sync_interval

        if self._physics:
            self._setup_physics(viewer_sync_interval)
        else:
            self._setup_kinematic(viewer_sync_interval)

        # Mark F/T sensors as valid in physics mode (always valid on hardware)
        for arm in self._arms.values():
            arm.ft_valid = self._physics

        # Wire event loop to controller (both physics and kinematic modes)
        if self._event_loop is not None:
            self._event_loop.set_controller(self._controller)

            # Create ownership registry for all arms + entities
            from mj_manipulator.ownership import OwnershipRegistry

            all_names = list(self._arms.keys()) + list(self._entities.keys())
            self._ownership = OwnershipRegistry(all_names)

        self.sync()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context: clean up viewer and executors."""
        if self._event_loop is not None:
            self._event_loop.set_controller(None)
        if self._owns_viewer and self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self._controller = None
        self._executors.clear()
        self._arm_controllers.clear()
        self._ownership = None
        return False

    def _make_drop_abort(self) -> Callable[[], bool] | None:
        """Build an abort predicate that fires when any arm's verifier
        transitions HOLDING → LOST during a trajectory.

        Snapshots which arms are currently HOLDING at the start of the
        trajectory. The returned predicate returns True if any of those
        arms is no longer HOLDING on a subsequent check — meaning the
        verifier saw a load collapse mid-motion and flipped to LOST.
        That's the \"we dropped it during transport\" case.

        Returns None if no arm has a verifier, or no arm is currently
        HOLDING — either way there's nothing to monitor. Skipping the
        predicate avoids per-tick function-call overhead in the vastly
        more common case where no grasp is active.
        """
        holding_verifiers = []
        for arm in self._arms.values():
            gripper = arm.gripper
            if gripper is None or gripper.grasp_verifier is None:
                continue
            if gripper.grasp_verifier.is_held:
                holding_verifiers.append(gripper.grasp_verifier)

        if not holding_verifiers:
            return None

        def _drop_check() -> bool:
            for v in holding_verifiers:
                if not v.is_held:
                    return True
            return False

        return _drop_check

    # -- ExecutionContext protocol -------------------------------------------

    def execute(
        self,
        item: object,
        *,
        abort_fn: Callable[[], bool] | None = None,
    ) -> bool:
        """Execute a trajectory or plan result.

        In tick-driven mode (event loop + controller), trajectories run
        as non-blocking runners: the caller blocks on a Future while
        tick() advances the runner each cycle. Physics keeps stepping
        and teleop on other arms continues working.

        If teleop is active on the target arm, it is deactivated first.
        This is the per-arm equivalent of the old _deactivate_all_teleop().

        In legacy mode (no event loop, or kinematic), execution is
        synchronous and blocking.

        Args:
            item: Trajectory or PlanResult to execute.
            abort_fn: Optional per-call abort predicate. Runner stops the
                next control cycle after this returns True. Used by
                collision-aware primitives (e.g. safe_retract) to halt
                mid-trajectory when a new contact appears. Composes with
                (does not replace) the ownership-registry abort: either
                one returning True stops the trajectory.

        Returns:
            True if execution completed successfully.
        """
        if self._event_loop is not None and self._controller is not None:
            return self._execute_tick_driven(item, abort_fn=abort_fn)

        return self._execute_impl(item, abort_fn=abort_fn)

    def _execute_tick_driven(
        self,
        item: object,
        *,
        abort_fn: Callable[[], bool] | None = None,
    ) -> bool:
        """Execute via non-blocking trajectory runners.

        Two modes depending on which thread calls:

        - **Owner thread** (IPython command): starts the runner, then pumps
          tick() ourselves until it completes. This keeps physics stepping
          and teleop on other arms alive.

        - **Background thread** (chat): starts the runner via submit, then
          blocks on Future. The inputhook pumps tick() on the owner thread.
        """
        from mj_manipulator.planning import PlanResult
        from mj_manipulator.trajectory import Trajectory

        if isinstance(item, PlanResult):
            trajectories = item.trajectories
        elif isinstance(item, Trajectory):
            trajectories = [item]
        else:
            raise TypeError(f"Cannot execute {type(item)}")

        on_owner_thread = threading.get_ident() == self._event_loop._owner_thread

        for traj in trajectories:
            entity = traj.entity
            if entity is None:
                raise ValueError("Trajectory has no entity set")

            # Build per-arm/entity abort function.
            # The caller-supplied ``abort_fn`` composes with (does not
            # replace) the ownership abort and the context-level abort:
            # any of the three returning True halts the trajectory at
            # the next control cycle.
            caller_abort = abort_fn  # shadow the outer parameter into closure
            runner_abort_fn: Callable[[], bool] | None = None

            if self._ownership is not None:
                from mj_manipulator.ownership import OwnerKind

                kind, _ = self._ownership.owner_of(entity)
                if kind != OwnerKind.IDLE:
                    # Arm is owned by another controller (teleop, etc.).
                    # Don't fight it — the user or another system has
                    # explicit control. Return False so the caller knows
                    # execution didn't happen.
                    logger.info(
                        "Cannot execute on %s: owned by %s",
                        entity,
                        kind.value,
                    )
                    return False
                self._ownership.acquire(entity, OwnerKind.TRAJECTORY, traj)

                def _make_abort_fn(e=entity, ca=caller_abort):
                    reg = self._ownership
                    ctx_abort = self._abort_fn
                    drop_check = self._make_drop_abort()

                    def _abort() -> bool:
                        if reg is not None and reg.is_aborted(e):
                            return True
                        if ctx_abort is not None and ctx_abort():
                            return True
                        if drop_check is not None and drop_check():
                            return True
                        if ca is not None and ca():
                            return True
                        return False

                    return _abort

                runner_abort_fn = _make_abort_fn()
            else:
                # No ownership registry — still compose context + caller + drop
                ctx_abort = self._abort_fn
                drop_check = self._make_drop_abort()
                if caller_abort is not None or ctx_abort is not None or drop_check is not None:

                    def _abort(ca=caller_abort, cx=ctx_abort, dc=drop_check) -> bool:
                        if cx is not None and cx():
                            return True
                        if dc is not None and dc():
                            return True
                        if ca is not None and ca():
                            return True
                        return False

                    runner_abort_fn = _abort

            try:
                if on_owner_thread:
                    # We ARE the tick pump — start runner and drive tick() directly
                    future = self._controller.start_trajectory(entity, traj, runner_abort_fn)
                    control_dt = self._controller.control_dt
                    realtime = self._controller.viewer is not None
                    t_next = time.monotonic() + control_dt
                    while not future.done():
                        self._event_loop.tick()
                        if realtime:
                            now = time.monotonic()
                            if t_next > now:
                                time.sleep(t_next - now)
                            t_next = now + control_dt
                    if not future.result():
                        return False
                else:
                    # Background thread — submit start, block on future while
                    # the inputhook pumps tick() on the owner thread
                    runner_future: Future[bool] = Future()

                    def _start(t=traj, af=runner_abort_fn, rf=runner_future):
                        try:
                            f = self._controller.start_trajectory(t.entity, t, af)
                            f.add_done_callback(lambda done_f: rf.set_result(done_f.result()))
                        except Exception as e:
                            rf.set_exception(e)

                    self._event_loop.submit(_start)
                    if not runner_future.result():
                        return False
            finally:
                # Release ownership (unless preempted — owner already changed)
                if self._ownership is not None:
                    kind, owner = self._ownership.owner_of(entity)
                    if owner is traj:
                        self._ownership.release(entity, traj)

        return True

    def _execute_impl(
        self,
        item: object,
        *,
        abort_fn: Callable[[], bool] | None = None,
    ) -> bool:
        """Execute implementation — synchronous, always runs on the physics thread."""
        from mj_manipulator.planning import PlanResult
        from mj_manipulator.trajectory import Trajectory

        # In the kinematic/legacy path, abort checks happen between
        # trajectories only (not between waypoints). That's acceptable
        # because kinematic mode is fast and primarily used for planning/
        # testing; the tick-driven physics path does per-waypoint checks.
        def _should_abort() -> bool:
            if self._abort_fn is not None and self._abort_fn():
                return True
            if abort_fn is not None and abort_fn():
                return True
            return False

        if isinstance(item, PlanResult):
            for traj in item.trajectories:
                if _should_abort():
                    return False
                if not self._execute_trajectory(traj):
                    return False
            return True
        elif isinstance(item, Trajectory):
            if _should_abort():
                return False
            return self._execute_trajectory(item)
        else:
            raise TypeError(f"Cannot execute {type(item)}")

    def step(self, targets: dict[str, np.ndarray] | None = None) -> None:
        """Advance one control cycle with optional joint targets.

        Arms not in ``targets`` hold their current position. In physics
        mode, steps MuJoCo dynamics. In kinematic mode, sets qpos directly.

        Args:
            targets: Dict mapping arm names to target joint positions.
                None means hold all arms at current positions.
        """
        if self._event_loop is not None:
            self._event_loop.run_on_physics_thread(lambda: self._step_impl(targets))
        else:
            self._step_impl(targets)

    def _step_impl(self, targets: dict[str, np.ndarray] | None = None) -> None:
        if self._controller is not None:
            if targets:
                for name, q in targets.items():
                    self._controller.set_arm_target(name, q)
            self._controller.step()
        else:
            # Fallback: no controller (shouldn't happen in normal usage)
            mujoco.mj_forward(self._model, self._data)
            self._throttled_viewer_sync()

    def step_cartesian(
        self,
        arm_name: str,
        position: np.ndarray,
        velocity: np.ndarray | None = None,
    ) -> None:
        """Advance one control cycle with a cartesian-space joint target.

        In tick-driven mode (event loop + controller), sets targets only —
        tick() handles the physics step. In other modes, steps physics
        immediately.

        Args:
            arm_name: Which arm to control.
            position: Target joint positions (from IK / QP solver output).
            velocity: Target joint velocities for feedforward (optional).
        """
        position = np.asarray(position)

        if self._event_loop is not None and self._controller is not None:
            # Tick-driven: set targets, let tick() do the step.
            # On the owner thread (e.g. CartesianMove inside a BT tree),
            # we must pump tick() ourselves — nobody else will.
            self._event_loop.run_on_physics_thread(lambda: self._set_reactive_target(arm_name, position, velocity))
            if threading.get_ident() == self._event_loop._owner_thread:
                self._event_loop.tick()
        else:
            self._step_cartesian_impl(arm_name, position, velocity)

    def _set_reactive_target(self, arm_name: str, position: np.ndarray, velocity: np.ndarray | None) -> None:
        """Set arm target with reactive lookahead. Does NOT step physics."""
        state = self._controller._arms[arm_name]
        state.target_position = np.asarray(position).copy()
        state.target_velocity = (
            np.asarray(velocity).copy() if velocity is not None else np.zeros(len(state.actuator_ids))
        )
        state.lookahead = 2.0 * self._controller.control_dt

    def _step_cartesian_impl(self, arm_name: str, position: np.ndarray, velocity: np.ndarray | None) -> None:
        if self._controller is not None:
            self._controller.step_reactive(arm_name, position, velocity)
        else:
            # Fallback: no controller (shouldn't happen in normal usage)
            mujoco.mj_forward(self._model, self._data)
            self._throttled_viewer_sync()

    def set_arm_target(
        self,
        arm_name: str,
        position: np.ndarray,
        velocity: np.ndarray | None = None,
    ) -> None:
        """Set arm target without stepping physics.

        For use in tick-driven mode where the event loop owns the step.

        Args:
            arm_name: Which arm to set targets for.
            position: Target joint positions.
            velocity: Target joint velocities for feedforward.
        """
        if self._controller is not None:
            self._controller.set_arm_target(arm_name, position, velocity)
        # When no controller, this is a no-op (shouldn't happen in normal usage)

    def sync(self) -> None:
        """Synchronize state with simulation (mj_forward + viewer sync)."""
        if self._event_loop is not None:
            self._event_loop.run_on_physics_thread(self._sync_impl)
        else:
            self._sync_impl()

    def _sync_impl(self) -> None:
        mujoco.mj_forward(self._model, self._data)
        if self._viewer is not None:
            self._viewer.sync()

    def hold(self) -> None:
        """Update all controller targets to match current joint positions.

        Call after externally modifying qpos (e.g. keyframe reset) to
        prevent the controller from violently correcting to stale targets.
        """
        if self._controller is not None:
            self._controller.hold_all()

    def reset_state(self) -> None:
        """Deactivate teleop, release grasps, abort runners, hold at current qpos.

        Call after modifying ``data.qpos`` (e.g. ``mj_resetDataKeyframe``)
        so the controller, event loop, ownership registry, grasp manager,
        and grasp verifiers all agree on the new state.

        This is the safe way to reset during interactive sessions — it
        ensures teleop gizmos are hidden, running trajectories are stopped,
        held objects are released, and the controller won't fight the new
        positions.

        Typical usage::

            mujoco.mj_resetDataKeyframe(model, data, key_id)
            ctx.reset_state()
        """
        # 1. Deactivate all teleop controllers and reset their panels
        if self._event_loop is not None:
            self._event_loop._deactivate_all_teleop()

        # 2. Clear ownership (abort any running trajectories, release all arms)
        if self._ownership is not None:
            self._ownership.abort_all()
            self._ownership.clear_all()

        # 3. Release all grasps: clear grasp manager bookkeeping, detach
        #    kinematic welds, and reset grasp verifiers to IDLE.
        for arm in self._arms.values():
            arm_name = arm.config.name
            gm = arm.grasp_manager
            if gm is not None:
                for obj in list(gm.get_grasped_by(arm_name)):
                    gm.mark_released(obj)
                    gm.detach_object(obj)

            gripper = arm.gripper
            if gripper is not None:
                if gripper.grasp_verifier is not None:
                    gripper.grasp_verifier.mark_released()

        # 4. Defer hold to the next tick — the caller may modify qpos
        #    after this call (e.g. setup_scene, base height changes).
        #    Whatever qpos exists at tick-time gets captured.
        if self._controller is not None:
            self._controller.request_hold()

        # 5. Sync viewer to show the new state
        self.sync()

    def reset_to_keyframe(self, keyframe: str) -> None:
        """Reset MuJoCo state to a named keyframe and clean up.

        Single entry point for resetting — handles the MuJoCo state reset
        AND all the cleanup (teleop, grasps, ownership, controller targets).
        Robots should call this instead of ``mj_resetDataKeyframe`` +
        ``reset_state()`` separately.

        Args:
            keyframe: MuJoCo keyframe name (must exist in the model).

        Raises:
            ValueError: If the keyframe is not found in the model.
        """
        key_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
        if key_id == -1:
            raise ValueError(f"Keyframe '{keyframe}' not found in model")
        mujoco.mj_resetDataKeyframe(self._model, self._data, key_id)
        self.reset_state()

    def is_running(self) -> bool:
        """Check if context is still active.

        Returns True in headless mode. Otherwise, True while the viewer
        window is open.
        """
        if self._viewer is None:
            return True
        return self._viewer.is_running()

    def arm(self, name: str) -> SimArmController:
        """Get per-arm controller for grasp/release operations.

        Args:
            name: Arm identifier (must match a key in the ``arms`` dict).

        Returns:
            SimArmController for the specified arm.
        """
        if name not in self._arms:
            raise ValueError(f"Unknown arm: {name}")

        if name not in self._arm_controllers:
            self._arm_controllers[name] = SimArmController(
                self._arms[name],
                self,
            )
        return self._arm_controllers[name]

    @property
    def control_dt(self) -> float:
        """Control timestep in seconds.

        Physics mode: from PhysicsExecutionConfig (default 0.008s = 125 Hz).
        Kinematic mode: 0.004s (250 Hz, suitable for visualization).
        """
        if self._controller is not None:
            return self._controller.control_dt
        return 0.004

    @property
    def viewer(self) -> mujoco.viewer.Handle | None:
        """MuJoCo viewer, or None in headless mode."""
        return self._viewer

    def _throttled_viewer_sync(self) -> None:
        """Sync viewer if present, throttled to viewer_fps."""
        if self._viewer is not None:
            now = time.time()
            if now - self._last_viewer_sync >= self._viewer_sync_interval:
                self._viewer.sync()
                self._last_viewer_sync = now

    # -- Internal helpers ----------------------------------------------------

    def _deactivate_teleop_for_item(self, item: object) -> None:
        """Deactivate teleop on all arms referenced by a trajectory/plan."""
        if self._ownership is None:
            return
        from mj_manipulator.ownership import OwnerKind
        from mj_manipulator.planning import PlanResult
        from mj_manipulator.trajectory import Trajectory

        if isinstance(item, PlanResult):
            entities = {t.entity for t in item.trajectories if t.entity}
        elif isinstance(item, Trajectory):
            entities = {item.entity} if item.entity else set()
        else:
            return

        for entity in entities:
            kind, _ = self._ownership.owner_of(entity)
            if kind == OwnerKind.TELEOP:
                self._deactivate_teleop_for(entity)

    def _deactivate_teleop_for(self, entity: str) -> None:
        """Deactivate any teleop controller owning this arm.

        Finds the teleop controller registered for this entity in the event
        loop, deactivates it, unregisters it, and releases ownership.
        """
        if self._event_loop is None or self._ownership is None:
            return

        from mj_manipulator.ownership import OwnerKind

        kind, owner = self._ownership.owner_of(entity)
        if kind != OwnerKind.TELEOP:
            return

        # Find and remove the matching teleop entry
        with self._event_loop._teleop_lock:
            remaining = []
            for controller, panel in self._event_loop._teleop_entries:
                if controller is owner:
                    try:
                        controller.deactivate()
                    except Exception:
                        pass
                    if panel is not None:
                        try:
                            panel._on_teleop_error()
                        except Exception:
                            pass
                else:
                    remaining.append((controller, panel))
            self._event_loop._teleop_entries = remaining

        self._ownership.release(entity, owner)

    # -- Internal setup -----------------------------------------------------

    def _setup_physics(self, viewer_sync_interval: float) -> None:
        """Create PhysicsController and per-arm/entity executor wrappers."""
        from mj_manipulator.physics_controller import PhysicsController

        exec_config = None
        gripper_config = None
        if self._physics_config is not None:
            exec_config = self._physics_config.execution
            gripper_config = self._physics_config.gripper

        self._controller = PhysicsController(
            self._model,
            self._data,
            self._arms,
            config=exec_config,
            gripper_config=gripper_config,
            viewer=self._viewer,
            viewer_sync_interval=viewer_sync_interval,
            initial_positions=self._initial_positions,
            entities=self._entities,
            abort_fn=self._abort_fn,
        )

        for name in self._arms:
            self._executors[name] = self._controller.get_executor(name)

        for name in self._entities:
            self._executors[name] = self._controller.get_entity_executor(name)

        # Settle physics so sensors (F/T, contacts) are immediately valid
        for _ in range(500):
            mujoco.mj_step(self._model, self._data)

    def _setup_kinematic(self, viewer_sync_interval: float) -> None:
        """Create KinematicController and per-arm executor wrappers."""
        from mj_manipulator.kinematic_controller import KinematicController

        self._controller = KinematicController(
            self._model,
            self._data,
            self._arms,
            viewer=self._viewer,
            viewer_sync_interval=viewer_sync_interval,
            initial_positions=self._initial_positions,
            entities=self._entities,
            abort_fn=self._abort_fn,
        )

        # Executor wrappers for the no-event-loop legacy path
        for name in self._arms:
            self._executors[name] = self._controller.get_executor(name)

        for name in self._entities:
            self._executors[name] = self._controller.get_entity_executor(name)

    def _execute_trajectory(self, trajectory) -> bool:
        """Route a single trajectory to its executor."""
        entity = trajectory.entity
        if entity is None:
            raise ValueError("Trajectory has no entity set")

        if entity in self._executors:
            return self._executors[entity].execute(trajectory)

        raise ValueError(f"No executor for entity: {entity}")
