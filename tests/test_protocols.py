# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for protocol definitions.

Verifies that the protocols are well-formed and that example implementations
satisfy them correctly (using runtime_checkable).
"""

import numpy as np

from mj_manipulator.protocols import (
    ArmController,
    ExecutionContext,
    GraspSource,
    Gripper,
    IKSolver,
)


class MockArmController:
    """Minimal arm controller for protocol testing.

    Simulates the grasp/release lifecycle that both SimArmController
    and a future HardwareArmController would implement.
    """

    def __init__(self):
        self._grasped = None

    def grasp(self, object_name: str) -> str | None:
        self._grasped = object_name
        return object_name

    def release(self, object_name: str | None = None) -> None:
        self._grasped = None


class MockExecutionContext:
    """Minimal execution context for protocol testing.

    Demonstrates the interface that both SimContext (MuJoCo) and
    HardwareContext (real robot) must implement.
    """

    def __init__(self):
        self._running = True
        self._executed = []
        self._arms = {"test_arm": MockArmController()}
        self._last_step_targets = None
        self._last_cartesian = None

    def execute(self, item) -> bool:
        self._executed.append(item)
        return True

    def step(self, targets=None) -> None:
        self._last_step_targets = targets

    def step_cartesian(self, arm_name, position, velocity=None) -> None:
        self._last_cartesian = (arm_name, position, velocity)

    def sync(self) -> None:
        pass

    def is_running(self) -> bool:
        return self._running

    def arm(self, name: str) -> MockArmController:
        return self._arms[name]

    @property
    def control_dt(self) -> float:
        return 0.002  # 500Hz, typical for UR RTDE


class MockGripper:
    """Minimal gripper implementation for protocol testing."""

    def __init__(self):
        self._holding = False
        self._held = None

    @property
    def arm_name(self) -> str:
        return "test_arm"

    @property
    def gripper_body_names(self) -> list[str]:
        return ["gripper/left_finger", "gripper/right_finger"]

    @property
    def attachment_body(self) -> str:
        return "gripper/right_finger"

    @property
    def actuator_id(self) -> int | None:
        return 0

    @property
    def ctrl_open(self) -> float:
        return 0.0

    @property
    def ctrl_closed(self) -> float:
        return 255.0

    @property
    def is_holding(self) -> bool:
        return self._holding

    @property
    def held_object(self) -> str | None:
        return self._held

    def set_candidate_objects(self, objects: list[str] | None) -> None:
        pass

    def kinematic_close(self, steps: int = 50) -> str | None:
        self._holding = True
        self._held = "test_object"
        return "test_object"

    def kinematic_open(self) -> None:
        self._holding = False
        self._held = None

    def get_actual_position(self) -> float:
        return 0.0 if not self._holding else 0.7


class MockIKSolver:
    """Minimal IK solver for protocol testing."""

    def solve(self, pose, q_init=None):
        return [np.zeros(6)]

    def solve_valid(self, pose, q_init=None):
        return [np.zeros(6)]


class MockGraspSource:
    """Minimal grasp source for protocol testing."""

    def get_grasps(self, object_name, hand_type):
        return []

    def get_placements(self, destination, object_name):
        return []

    def get_graspable_objects(self):
        return ["mug", "can"]

    def get_place_destinations(self, object_name):
        return ["table", "bin"]


class TestGripperProtocol:
    """Tests for the Gripper protocol."""

    def test_mock_satisfies_protocol(self):
        """MockGripper satisfies the Gripper protocol."""
        gripper = MockGripper()
        assert isinstance(gripper, Gripper)

    def test_gripper_lifecycle(self):
        """Basic open/close lifecycle works."""
        gripper = MockGripper()
        assert not gripper.is_holding
        assert gripper.held_object is None

        result = gripper.kinematic_close()
        assert result == "test_object"
        assert gripper.is_holding
        assert gripper.held_object == "test_object"

        gripper.kinematic_open()
        assert not gripper.is_holding

    def test_attachment_body(self):
        """attachment_body is accessible."""
        gripper = MockGripper()
        assert gripper.attachment_body == "gripper/right_finger"


class TestIKSolverProtocol:
    """Tests for the IKSolver protocol."""

    def test_mock_satisfies_protocol(self):
        """MockIKSolver satisfies the IKSolver protocol."""
        solver = MockIKSolver()
        assert isinstance(solver, IKSolver)

    def test_solve_returns_list(self):
        """solve() returns a list of configurations."""
        solver = MockIKSolver()
        solutions = solver.solve(np.eye(4))
        assert isinstance(solutions, list)
        assert len(solutions) == 1
        assert solutions[0].shape == (6,)


class TestGraspSourceProtocol:
    """Tests for the GraspSource protocol."""

    def test_mock_satisfies_protocol(self):
        """MockGraspSource satisfies the GraspSource protocol."""
        source = MockGraspSource()
        assert isinstance(source, GraspSource)

    def test_get_graspable_objects(self):
        """Can query available objects."""
        source = MockGraspSource()
        objects = source.get_graspable_objects()
        assert "mug" in objects
        assert "can" in objects

    def test_get_place_destinations(self):
        """Can query placement destinations."""
        source = MockGraspSource()
        destinations = source.get_place_destinations("mug")
        assert "table" in destinations


class TestExecutionContextProtocol:
    """Tests for the ExecutionContext protocol — the sim-to-real bridge."""

    def test_mock_satisfies_protocol(self):
        """MockExecutionContext satisfies the ExecutionContext protocol."""
        ctx = MockExecutionContext()
        assert isinstance(ctx, ExecutionContext)

    def test_execute_returns_bool(self):
        """execute() returns success/failure."""
        ctx = MockExecutionContext()
        assert ctx.execute("fake_trajectory") is True
        assert len(ctx._executed) == 1

    def test_step_accepts_targets(self):
        """step() accepts joint targets dict."""
        ctx = MockExecutionContext()
        targets = {"test_arm": np.array([0.1, 0.2, 0.3])}
        ctx.step(targets)
        assert ctx._last_step_targets is targets

    def test_step_no_targets_holds_position(self):
        """step(None) means hold all arms at current position."""
        ctx = MockExecutionContext()
        ctx.step(None)
        assert ctx._last_step_targets is None

    def test_step_cartesian(self):
        """step_cartesian() accepts position + optional velocity."""
        ctx = MockExecutionContext()
        pos = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        vel = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
        ctx.step_cartesian("test_arm", pos, vel)
        assert ctx._last_cartesian[0] == "test_arm"
        np.testing.assert_array_equal(ctx._last_cartesian[1], pos)
        np.testing.assert_array_equal(ctx._last_cartesian[2], vel)

    def test_is_running(self):
        """is_running() reflects context state."""
        ctx = MockExecutionContext()
        assert ctx.is_running()
        ctx._running = False
        assert not ctx.is_running()

    def test_control_dt(self):
        """control_dt is accessible."""
        ctx = MockExecutionContext()
        assert ctx.control_dt == 0.002

    def test_arm_returns_controller(self):
        """arm() returns an ArmController."""
        ctx = MockExecutionContext()
        controller = ctx.arm("test_arm")
        assert isinstance(controller, ArmController)


class TestArmControllerProtocol:
    """Tests for the ArmController protocol."""

    def test_mock_satisfies_protocol(self):
        """MockArmController satisfies the ArmController protocol."""
        ctrl = MockArmController()
        assert isinstance(ctrl, ArmController)

    def test_grasp_release_lifecycle(self):
        """Grasp/release lifecycle works through the protocol."""
        ctrl = MockArmController()
        assert ctrl._grasped is None

        result = ctrl.grasp("mug")
        assert result == "mug"
        assert ctrl._grasped == "mug"

        ctrl.release("mug")
        assert ctrl._grasped is None

    def test_same_code_sim_and_hardware(self):
        """Demonstrate that caller code is identical regardless of backend.

        This is the key insight: primitives and policies interact only with
        the ExecutionContext protocol. The same function works for sim and
        real hardware.
        """

        def execute_pick(ctx: ExecutionContext, arm_name: str, object_name: str):
            """Example primitive that works in sim or on real hardware."""
            # Plan and execute approach (trajectory would come from planner)
            ctx.execute("approach_trajectory")
            # Grasp
            result = ctx.arm(arm_name).grasp(object_name)
            # Execute lift
            if result is not None:
                ctx.execute("lift_trajectory")
            return result

        # Works with mock (stands in for SimContext or HardwareContext)
        ctx = MockExecutionContext()
        result = execute_pick(ctx, "test_arm", "can")
        assert result == "can"
        assert len(ctx._executed) == 2  # approach + lift
