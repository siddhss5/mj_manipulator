"""Tests for generic configuration classes."""

import numpy as np
import pytest

from mj_manipulator.config import (
    ArmConfig,
    EntityConfig,
    GripperPhysicsConfig,
    KinematicLimits,
    PhysicsConfig,
    PhysicsExecutionConfig,
    PlanningDefaults,
    RecoveryConfig,
)


# -- Robot-specific config fixtures (UR5e and Franka) --
# These demonstrate that ArmConfig works for any robot.


def ur5e_kinematic_limits(vel_scale: float = 0.5, acc_scale: float = 0.5):
    """UR5e limits from datasheet."""
    return KinematicLimits(
        velocity=np.array([3.14, 3.14, 3.14, 6.28, 6.28, 6.28]) * vel_scale,
        acceleration=np.array([2.5, 2.5, 2.5, 5.0, 5.0, 5.0]) * acc_scale,
    )


def franka_kinematic_limits():
    """Franka Emika Panda limits from datasheet."""
    return KinematicLimits(
        velocity=np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]),
        acceleration=np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0]),
    )


def ur5e_arm_config(side: str = "left") -> ArmConfig:
    """Create UR5e arm config."""
    return ArmConfig(
        name=f"{side}_arm",
        entity_type="arm",
        joint_names=[f"{side}_ur5e/{j}" for j in [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        ]],
        kinematic_limits=ur5e_kinematic_limits(),
        ee_site=f"{side}_ur5e/gripper_attachment_site",
    )


def franka_arm_config() -> ArmConfig:
    """Create Franka Panda arm config."""
    return ArmConfig(
        name="franka",
        entity_type="arm",
        joint_names=[f"panda/joint{i}" for i in range(1, 8)],
        kinematic_limits=franka_kinematic_limits(),
        ee_site="panda/attachment_site",
    )


class TestKinematicLimits:
    """Tests for KinematicLimits."""

    def test_ur5e_limits(self):
        """UR5e limits have 6 DOF."""
        limits = ur5e_kinematic_limits()
        assert len(limits.velocity) == 6
        assert len(limits.acceleration) == 6

    def test_franka_limits(self):
        """Franka limits have 7 DOF."""
        limits = franka_kinematic_limits()
        assert len(limits.velocity) == 7
        assert len(limits.acceleration) == 7

    def test_scaling(self):
        """Velocity scaling works."""
        full = ur5e_kinematic_limits(vel_scale=1.0)
        half = ur5e_kinematic_limits(vel_scale=0.5)
        np.testing.assert_allclose(half.velocity, full.velocity * 0.5)


class TestArmConfig:
    """Tests for ArmConfig."""

    def test_ur5e_config(self):
        """UR5e config has correct structure."""
        config = ur5e_arm_config("left")
        assert config.name == "left_arm"
        assert config.entity_type == "arm"
        assert len(config.joint_names) == 6
        assert config.ee_site == "left_ur5e/gripper_attachment_site"
        assert config.tcp_offset is None

    def test_franka_config(self):
        """Franka config has correct structure."""
        config = franka_arm_config()
        assert config.name == "franka"
        assert config.entity_type == "arm"
        assert len(config.joint_names) == 7
        assert config.ee_site == "panda/attachment_site"
        assert config.tcp_offset is None

    def test_tcp_offset(self):
        """tcp_offset stores SE3 transform from ee_site to TCP."""
        offset = np.eye(4)
        offset[2, 3] = 0.1034  # 10.34cm along Z (Franka fingertip)
        config = ArmConfig(
            name="franka",
            entity_type="arm",
            joint_names=[f"joint{i}" for i in range(1, 8)],
            kinematic_limits=franka_kinematic_limits(),
            ee_site="ee_site",
            tcp_offset=offset,
        )
        assert config.tcp_offset is not None
        np.testing.assert_allclose(config.tcp_offset[2, 3], 0.1034)

    def test_entity_type_forced(self):
        """entity_type is always set to 'arm' by __post_init__."""
        config = ArmConfig(
            name="test",
            entity_type="wrong",  # Should be overridden
            joint_names=["j1"],
            kinematic_limits=KinematicLimits(
                velocity=np.array([1.0]), acceleration=np.array([1.0])
            ),
        )
        assert config.entity_type == "arm"

    def test_planning_defaults(self):
        """Planning defaults are accessible."""
        config = ur5e_arm_config()
        assert config.planning_defaults.timeout == 30.0
        assert config.planning_defaults.max_iterations == 5000


class TestPlanningDefaults:
    """Tests for PlanningDefaults."""

    def test_default_values(self):
        """Default constructor gives standard values."""
        defaults = PlanningDefaults()
        assert defaults.timeout == 30.0
        assert defaults.step_size == 0.1

    def test_fast_preset(self):
        """Fast preset has shorter timeout."""
        fast = PlanningDefaults.fast()
        assert fast.timeout == 10.0
        assert fast.max_iterations == 2000

    def test_thorough_preset(self):
        """Thorough preset has longer timeout."""
        thorough = PlanningDefaults.thorough()
        assert thorough.timeout == 60.0
        assert thorough.max_iterations == 10000


class TestPhysicsConfig:
    """Tests for PhysicsConfig and sub-configs."""

    def test_default_construction(self):
        """Default PhysicsConfig constructs all sub-configs."""
        config = PhysicsConfig()
        assert config.execution.control_dt == 0.008
        assert config.gripper.close_steps == 200
        assert config.recovery.retract_height == 0.15

    def test_tight_execution(self):
        """Tight execution preset has smaller tolerances."""
        tight = PhysicsExecutionConfig.tight()
        assert tight.position_tolerance == 0.02
