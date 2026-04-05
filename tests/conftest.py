# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Shared test fixtures for mj_manipulator.

Provides a minimal 2-DOF MuJoCo arm model, MockArm, and helpers
used by test_executor, test_physics_controller, and test_sim_context.
"""

from dataclasses import dataclass

import mujoco
import numpy as np
import pytest

from mj_manipulator.trajectory import Trajectory

# Minimal model: 2-joint arm with position actuators + free-body box
ARM_XML = """
<mujoco model="test_arm">
  <option timestep="0.002"/>
  <worldbody>
    <body name="link1" pos="0 0 0.5">
      <joint name="joint1" type="hinge" axis="0 0 1"/>
      <geom type="capsule" size="0.04" fromto="0 0 0 0.3 0 0"/>
      <body name="link2" pos="0.3 0 0">
        <joint name="joint2" type="hinge" axis="0 0 1"/>
        <geom type="capsule" size="0.04" fromto="0 0 0 0.3 0 0"/>
      </body>
    </body>
    <body name="box" pos="0.5 0 0.5">
      <joint name="box_free" type="free"/>
      <geom type="box" size="0.03 0.03 0.03"/>
    </body>
  </worldbody>
  <actuator>
    <position name="act1" joint="joint1" kp="100"/>
    <position name="act2" joint="joint2" kp="100"/>
  </actuator>
</mujoco>
"""

JOINT_NAMES = ["joint1", "joint2"]


@dataclass
class MockConfig:
    name: str


class MockArm:
    """Minimal arm-like object for testing PhysicsController / SimContext."""

    def __init__(self, name, model, data):
        self.config = MockConfig(name=name)

        j1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint1")
        j2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint2")
        self.joint_qpos_indices = [model.jnt_qposadr[j1], model.jnt_qposadr[j2]]
        self.joint_qvel_indices = [model.jnt_dofadr[j1], model.jnt_dofadr[j2]]

        a1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act1")
        a2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act2")
        self.actuator_ids = [a1, a2]

        self.dof = 2
        self.gripper = None
        self.grasp_manager = None
        self._data = data

    def get_joint_positions(self):
        return np.array([self._data.qpos[idx] for idx in self.joint_qpos_indices])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model_and_data():
    model = mujoco.MjModel.from_xml_string(ARM_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


@pytest.fixture
def mock_arm(model_and_data):
    model, data = model_and_data
    return MockArm("test_arm", model, data)


@pytest.fixture
def joint_qpos_indices(model_and_data):
    model, _ = model_and_data
    return [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in JOINT_NAMES]


@pytest.fixture
def actuator_ids(model_and_data):
    model, _ = model_and_data
    return [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ["act1", "act2"]]


def make_trajectory(
    positions: np.ndarray,
    entity: str | None = None,
) -> Trajectory:
    """Create a test trajectory from positions array."""
    return Trajectory(
        timestamps=np.linspace(0, 1, len(positions)),
        positions=positions,
        velocities=np.zeros_like(positions),
        accelerations=np.zeros_like(positions),
        joint_names=list(JOINT_NAMES),
        entity=entity,
    )
