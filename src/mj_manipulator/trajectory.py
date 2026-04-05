# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Trajectory representation and time-optimal retiming using TOPP-RA."""

from dataclasses import dataclass

import numpy as np
import toppra
import toppra.algorithm as algo
import toppra.constraint as constraint


@dataclass
class Trajectory:
    """Time-parameterized robot trajectory.

    Stores positions, velocities, accelerations at discrete timestamps.
    Compatible with both MuJoCo simulation and real robot execution.

    The trajectory is dense (waypoints at control frequency) to avoid
    needing spline evaluation in the control loop.

    Entity information (entity name and joint_names) enables hardware
    deployment by identifying which joints to command.
    """

    timestamps: np.ndarray  # (N,) seconds from start
    positions: np.ndarray  # (N, dof) joint positions in radians
    velocities: np.ndarray  # (N, dof) joint velocities in rad/s
    accelerations: np.ndarray  # (N, dof) joint accelerations in rad/s²
    entity: str | None = None  # Entity name: "left_arm", "right_base", etc.
    joint_names: list[str] | None = None  # MuJoCo joint names for validation

    def __post_init__(self):
        """Validate trajectory dimensions."""
        n_waypoints = len(self.timestamps)
        if self.positions.shape[0] != n_waypoints:
            raise ValueError(f"Position shape {self.positions.shape} doesn't match timestamps length {n_waypoints}")
        if self.velocities.shape[0] != n_waypoints:
            raise ValueError(f"Velocity shape {self.velocities.shape} doesn't match timestamps length {n_waypoints}")
        if self.accelerations.shape[0] != n_waypoints:
            raise ValueError(
                f"Acceleration shape {self.accelerations.shape} doesn't match timestamps length {n_waypoints}"
            )

        if self.positions.shape[1] != self.velocities.shape[1]:
            raise ValueError(
                f"DOF mismatch: positions {self.positions.shape[1]} vs velocities {self.velocities.shape[1]}"
            )

        # Validate joint_names length matches DOF if provided
        if self.joint_names is not None:
            if len(self.joint_names) != self.positions.shape[1]:
                raise ValueError(
                    f"joint_names length {len(self.joint_names)} doesn't match DOF {self.positions.shape[1]}"
                )

    @property
    def duration(self) -> float:
        """Total duration of trajectory in seconds."""
        return float(self.timestamps[-1])

    @property
    def dof(self) -> int:
        """Degrees of freedom (number of joints)."""
        return self.positions.shape[1]

    @property
    def num_waypoints(self) -> int:
        """Number of waypoints in trajectory."""
        return len(self.timestamps)

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate position, velocity, acceleration at time t.

        Args:
            t: Time in seconds (clamped to [0, duration])

        Returns:
            Tuple of (position, velocity, acceleration) arrays
        """
        t = np.clip(t, 0.0, self.duration)

        idx = np.searchsorted(self.timestamps, t)

        if idx == 0:
            return self.positions[0], self.velocities[0], self.accelerations[0]
        if idx >= len(self.timestamps):
            return self.positions[-1], self.velocities[-1], self.accelerations[-1]

        t0, t1 = self.timestamps[idx - 1], self.timestamps[idx]
        alpha = (t - t0) / (t1 - t0)

        pos = (1 - alpha) * self.positions[idx - 1] + alpha * self.positions[idx]
        vel = (1 - alpha) * self.velocities[idx - 1] + alpha * self.velocities[idx]
        acc = (1 - alpha) * self.accelerations[idx - 1] + alpha * self.accelerations[idx]

        return pos, vel, acc

    @classmethod
    def from_path(
        cls,
        path: list[np.ndarray],
        vel_limits: np.ndarray,
        acc_limits: np.ndarray,
        control_dt: float = 0.008,
        entity: str | None = None,
        joint_names: list[str] | None = None,
    ) -> "Trajectory":
        """Create time-optimal trajectory from geometric path using TOPP-RA.

        Args:
            path: List of waypoint configurations (joint angles in radians)
            vel_limits: Joint velocity limits in rad/s (shape: (dof,))
            acc_limits: Joint acceleration limits in rad/s² (shape: (dof,))
            control_dt: Control timestep in seconds (default: 125 Hz)
            entity: Entity name for hardware deployment
            joint_names: MuJoCo joint names for validation

        Returns:
            Trajectory with time-optimal parameterization respecting limits

        Raises:
            ValueError: If path is empty or has inconsistent dimensions
            RuntimeError: If TOPP-RA fails to find a valid parameterization
        """
        if not path:
            raise ValueError("Path cannot be empty")

        path_array = np.array(path)
        if path_array.ndim != 2:
            raise ValueError(f"Path must be 2D array, got shape {path_array.shape}")

        dof = path_array.shape[1]
        if len(vel_limits) != dof:
            raise ValueError(f"Velocity limits dimension {len(vel_limits)} doesn't match path DOF {dof}")
        if len(acc_limits) != dof:
            raise ValueError(f"Acceleration limits dimension {len(acc_limits)} doesn't match path DOF {dof}")

        # Remove consecutive duplicate waypoints
        filtered_path = [path_array[0]]
        for i in range(1, len(path_array)):
            if not np.allclose(path_array[i], filtered_path[-1], atol=1e-10):
                filtered_path.append(path_array[i])

        path_array = np.array(filtered_path)

        # Trivial trajectory if already at goal
        if len(path_array) < 2:
            return cls(
                timestamps=np.array([0.0]),
                positions=path_array,
                velocities=np.zeros_like(path_array),
                accelerations=np.zeros_like(path_array),
                entity=entity,
                joint_names=joint_names,
            )

        path_positions = toppra.SplineInterpolator(np.linspace(0, 1, len(path_array)), path_array)

        vel_limits_minmax = np.stack((-vel_limits, vel_limits)).T
        acc_limits_minmax = np.stack((-acc_limits, acc_limits)).T

        pc_vel = constraint.JointVelocityConstraint(vel_limits_minmax)
        pc_acc = constraint.JointAccelerationConstraint(acc_limits_minmax)

        instance = algo.TOPPRA(
            [pc_vel, pc_acc],
            path_positions,
            parametrizer="ParametrizeConstAccel",
        )

        jnt_traj = instance.compute_trajectory()

        if jnt_traj is None:
            raise RuntimeError(
                "TOPP-RA failed to find valid trajectory. Path may violate velocity or acceleration constraints."
            )

        duration = jnt_traj.duration
        timestamps = np.arange(0.0, duration, control_dt)
        if not np.isclose(timestamps[-1], duration, rtol=0.0, atol=1e-8):
            timestamps = np.append(timestamps, duration)

        positions = jnt_traj(timestamps)
        velocities = jnt_traj(timestamps, 1)
        accelerations = jnt_traj(timestamps, 2)

        return cls(
            timestamps=timestamps,
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            entity=entity,
            joint_names=joint_names,
        )


def create_linear_trajectory(
    start: float,
    end: float,
    vel_limit: float,
    acc_limit: float,
    control_dt: float = 0.008,
    entity: str | None = None,
    joint_names: list[str] | None = None,
) -> Trajectory:
    """Generate trapezoidal velocity profile for 1D linear motion.

    Three phases: acceleration, cruise (if distance allows), deceleration.
    For short distances, becomes triangular (no cruise phase).

    Args:
        start: Starting position in meters
        end: Target position in meters
        vel_limit: Maximum velocity in m/s
        acc_limit: Maximum acceleration in m/s²
        control_dt: Control timestep in seconds (default: 125 Hz)
        entity: Entity name for hardware deployment
        joint_names: MuJoCo joint names for validation

    Returns:
        Trajectory object with 1 DOF
    """
    distance = abs(end - start)
    direction = 1.0 if end > start else -1.0

    if distance < 1e-8:
        return Trajectory(
            timestamps=np.array([0.0]),
            positions=np.array([[start]]),
            velocities=np.array([[0.0]]),
            accelerations=np.array([[0.0]]),
            entity=entity,
            joint_names=joint_names,
        )

    t_accel = vel_limit / acc_limit
    d_accel = 0.5 * acc_limit * t_accel**2

    if 2 * d_accel <= distance:
        # Trapezoidal profile
        d_cruise = distance - 2 * d_accel
        t_cruise = d_cruise / vel_limit
        t_total = 2 * t_accel + t_cruise
    else:
        # Triangular profile
        t_accel = np.sqrt(distance / acc_limit)
        vel_limit = acc_limit * t_accel
        d_accel = 0.5 * distance
        t_cruise = 0.0
        t_total = 2 * t_accel

    timestamps = np.arange(0.0, t_total, control_dt)
    if not np.isclose(timestamps[-1], t_total, atol=1e-8):
        timestamps = np.append(timestamps, t_total)

    positions = []
    velocities = []
    accelerations = []

    for t in timestamps:
        if t <= t_accel:
            p = start + direction * 0.5 * acc_limit * t**2
            v = direction * acc_limit * t
            a = direction * acc_limit
        elif t <= t_accel + t_cruise:
            t_in_cruise = t - t_accel
            p = start + direction * (d_accel + vel_limit * t_in_cruise)
            v = direction * vel_limit
            a = 0.0
        else:
            t_remaining = t_total - t
            p = end - direction * 0.5 * acc_limit * t_remaining**2
            v = direction * acc_limit * t_remaining
            a = -direction * acc_limit

        positions.append([p])
        velocities.append([v])
        accelerations.append([a])

    return Trajectory(
        timestamps=timestamps,
        positions=np.array(positions),
        velocities=np.array(velocities),
        accelerations=np.array(accelerations),
        entity=entity,
        joint_names=joint_names,
    )
