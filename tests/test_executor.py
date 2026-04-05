# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for KinematicExecutor and PhysicsExecutor.

Uses shared test fixtures from conftest.py (model_and_data, joint_qpos_indices, actuator_ids).
"""

import numpy as np
import pytest
from conftest import make_trajectory

from mj_manipulator.executor import KinematicExecutor, PhysicsExecutor
from mj_manipulator.grasp_manager import GraspManager
from mj_manipulator.trajectory import Trajectory


class TestKinematicExecutor:
    def test_constructs(self, model_and_data, joint_qpos_indices):
        model, data = model_and_data
        ex = KinematicExecutor(model, data, joint_qpos_indices)
        assert ex is not None

    def test_execute_trajectory(self, model_and_data, joint_qpos_indices):
        model, data = model_and_data
        ex = KinematicExecutor(model, data, joint_qpos_indices, control_dt=0.0)
        positions = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.1],
                [0.2, 0.2],
                [0.3, 0.3],
            ]
        )
        traj = make_trajectory(positions)
        result = ex.execute(traj)
        assert result is True

        # Final position should match trajectory end
        for i, idx in enumerate(joint_qpos_indices):
            assert abs(data.qpos[idx] - 0.3) < 1e-6

    def test_execute_dof_mismatch_raises(self, model_and_data, joint_qpos_indices):
        model, data = model_and_data
        ex = KinematicExecutor(model, data, joint_qpos_indices)
        positions = np.array([[0.0, 0.0, 0.0]])  # 3 DOF, but arm is 2
        traj = Trajectory(
            timestamps=np.array([0.0]),
            positions=positions,
            velocities=np.zeros_like(positions),
            accelerations=np.zeros_like(positions),
            joint_names=["j1", "j2", "j3"],
        )
        with pytest.raises(ValueError, match="DOF"):
            ex.execute(traj)

    def test_set_position(self, model_and_data, joint_qpos_indices):
        model, data = model_and_data
        ex = KinematicExecutor(model, data, joint_qpos_indices)
        ex.set_position(np.array([0.5, -0.5]))
        for i, idx in enumerate(joint_qpos_indices):
            expected = [0.5, -0.5][i]
            assert abs(data.qpos[idx] - expected) < 1e-6

    def test_step_applies_target(self, model_and_data, joint_qpos_indices):
        model, data = model_and_data
        ex = KinematicExecutor(model, data, joint_qpos_indices)
        ex.set_position(np.array([1.0, -1.0]))
        # Overwrite qpos to something different
        for idx in joint_qpos_indices:
            data.qpos[idx] = 0.0
        # Step should restore target
        ex.step()
        for i, idx in enumerate(joint_qpos_indices):
            expected = [1.0, -1.0][i]
            assert abs(data.qpos[idx] - expected) < 1e-6

    def test_with_grasp_manager(self, model_and_data, joint_qpos_indices):
        model, data = model_and_data
        gm = GraspManager(model, data)
        ex = KinematicExecutor(model, data, joint_qpos_indices, grasp_manager=gm)
        # Should not crash even with grasp manager attached
        ex.set_position(np.array([0.1, 0.1]))
        ex.step()


class TestPhysicsExecutor:
    def test_constructs(self, model_and_data, joint_qpos_indices, actuator_ids):
        model, data = model_and_data
        ex = PhysicsExecutor(model, data, joint_qpos_indices, actuator_ids)
        assert ex is not None
        assert ex.steps_per_control == 4  # 0.008 / 0.002

    def test_set_target_and_step(self, model_and_data, joint_qpos_indices, actuator_ids):
        model, data = model_and_data
        ex = PhysicsExecutor(model, data, joint_qpos_indices, actuator_ids, control_dt=0.002)
        ex.set_target(np.array([0.5, -0.5]))
        ex.step()
        # Physics should have stepped — position won't be exact but
        # actuators should be commanding the target
        assert data.ctrl[actuator_ids[0]] != 0.0

    def test_hold(self, model_and_data, joint_qpos_indices, actuator_ids):
        model, data = model_and_data
        ex = PhysicsExecutor(model, data, joint_qpos_indices, actuator_ids)
        # Set some position in data
        for idx in joint_qpos_indices:
            data.qpos[idx] = 0.42
        ex.hold()
        np.testing.assert_allclose(ex.target_position, [0.42, 0.42])

    def test_execute_trajectory(self, model_and_data, joint_qpos_indices, actuator_ids):
        model, data = model_and_data
        ex = PhysicsExecutor(model, data, joint_qpos_indices, actuator_ids, control_dt=0.0)
        positions = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.1],
                [0.2, 0.2],
            ]
        )
        traj = make_trajectory(positions)
        result = ex.execute(traj)
        assert result is True

    def test_get_position(self, model_and_data, joint_qpos_indices, actuator_ids):
        model, data = model_and_data
        ex = PhysicsExecutor(model, data, joint_qpos_indices, actuator_ids)
        pos = ex.get_position()
        assert pos.shape == (2,)

    def test_get_velocity(self, model_and_data, joint_qpos_indices, actuator_ids):
        model, data = model_and_data
        ex = PhysicsExecutor(model, data, joint_qpos_indices, actuator_ids)
        vel = ex.get_velocity()
        assert vel.shape == (2,)

    def test_tracking_error(self, model_and_data, joint_qpos_indices, actuator_ids):
        model, data = model_and_data
        ex = PhysicsExecutor(model, data, joint_qpos_indices, actuator_ids)
        ex.set_target(np.array([1.0, 1.0]))
        error = ex.get_tracking_error()
        assert error.shape == (2,)
        # Error should be target - current
        np.testing.assert_allclose(error, [1.0, 1.0] - ex.get_position())

    def test_lookahead_time(self, model_and_data, joint_qpos_indices, actuator_ids):
        model, data = model_and_data
        ex = PhysicsExecutor(
            model,
            data,
            joint_qpos_indices,
            actuator_ids,
            lookahead_time=0.2,
        )
        ex.set_target(np.array([1.0, 0.5]), velocity=np.array([0.5, 0.25]))
        ex.step()
        # Command should be position + lookahead * velocity
        expected_cmd = np.array([1.0 + 0.2 * 0.5, 0.5 + 0.2 * 0.25])
        np.testing.assert_allclose(
            [data.ctrl[aid] for aid in actuator_ids],
            expected_cmd,
        )
