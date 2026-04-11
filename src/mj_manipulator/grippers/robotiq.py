# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Robotiq 2F-140 gripper implementation.

Supports both physics-mode (actuator-driven) and kinematic-mode (trajectory
replay) operation. The 4-bar linkage geometry is replayed from a pre-recorded
physics trajectory for kinematic mode, ensuring correct joint coupling.

Usage:
    from mj_manipulator.grippers.robotiq import RobotiqGripper

    gripper = RobotiqGripper(model, data, "ur5e", prefix="right_ur5e/gripper/")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

from mj_manipulator.grippers._base import _BaseGripper

if TYPE_CHECKING:
    from mj_manipulator.grasp_manager import GraspManager

# ---------------------------------------------------------------------------
# Robotiq 2F-140 constants
# ---------------------------------------------------------------------------

# Gripper joint names (suffixes, without prefix). Order matches trajectory columns.
_JOINT_SUFFIXES = [
    "left_coupler_joint",
    "left_driver_joint",
    "left_follower_joint",
    "left_spring_link_joint",
    "right_coupler_joint",
    "right_driver_joint",
    "right_follower_joint",
    "right_spring_link_joint",
]

# Body name suffixes for contact detection and collision filtering.
_BODY_SUFFIXES = [
    "base_mount",
    "base",
    "right_driver",
    "right_coupler",
    "right_spring_link",
    "right_follower",
    "right_pad",
    "left_driver",
    "left_coupler",
    "left_spring_link",
    "left_follower",
    "left_pad",
]

# Attachment body for kinematic tracking. Using base_mount (gripper center)
# rather than a finger pad — base_mount doesn't move when fingers close,
# so the attachment offset is stable regardless of finger position.
_ATTACHMENT_BODY_SUFFIX = "base_mount"

# Pre-recorded gripper joint trajectory from physics simulation.
# Shape: (101, 8) — 101 waypoints from open (t=0) to closed (t=1), 8 joints.
# Joint order matches _JOINT_SUFFIXES above.
_GRIPPER_TRAJECTORY = np.array(
    [
        [-0.000622, -0.025613, 0.032463, -0.027563, -0.000610, -0.025589, 0.032481, -0.027446],
        [-0.000622, -0.028316, 0.035389, -0.030293, -0.000609, -0.028292, 0.035394, -0.030171],
        [-0.000621, -0.032757, 0.040187, -0.034784, -0.000609, -0.032733, 0.040171, -0.034654],
        [-0.000621, -0.038365, 0.046238, -0.040459, -0.000608, -0.038340, 0.046197, -0.040319],
        [-0.000620, -0.044755, 0.053128, -0.046928, -0.000608, -0.044731, 0.053058, -0.046777],
        [-0.000619, -0.051671, 0.060579, -0.053929, -0.000607, -0.051646, 0.060479, -0.053766],
        [-0.000619, -0.058939, 0.068403, -0.061287, -0.000606, -0.058914, 0.068272, -0.061112],
        [-0.000618, -0.066444, 0.076475, -0.068885, -0.000606, -0.066419, 0.076312, -0.068697],
        [-0.000618, -0.074108, 0.084712, -0.076643, -0.000605, -0.074083, 0.084518, -0.076442],
        [-0.000617, -0.081879, 0.093057, -0.084508, -0.000604, -0.081854, 0.092831, -0.084294],
        [-0.000616, -0.089721, 0.101472, -0.092445, -0.000603, -0.089696, 0.101216, -0.092217],
        [-0.000616, -0.097612, 0.109932, -0.100429, -0.000603, -0.097586, 0.109646, -0.100189],
        [-0.000615, -0.105534, 0.118421, -0.108446, -0.000602, -0.105508, 0.118105, -0.108192],
        [-0.000614, -0.113479, 0.126926, -0.116483, -0.000601, -0.113453, 0.126582, -0.116216],
        [-0.000614, -0.121438, 0.135441, -0.124535, -0.000601, -0.121411, 0.135069, -0.124255],
        [-0.000613, -0.129407, 0.143961, -0.132595, -0.000600, -0.129380, 0.143561, -0.132302],
        [-0.000612, -0.137382, 0.152481, -0.140662, -0.000599, -0.137355, 0.152055, -0.140355],
        [-0.000612, -0.145362, 0.161001, -0.148732, -0.000598, -0.145335, 0.160548, -0.148412],
        [-0.000611, -0.153345, 0.169517, -0.156804, -0.000598, -0.153318, 0.169039, -0.156472],
        [-0.000611, -0.161330, 0.178031, -0.164877, -0.000597, -0.161302, 0.177528, -0.164532],
        [-0.000610, -0.169316, 0.186540, -0.172951, -0.000596, -0.169289, 0.186013, -0.172593],
        [-0.000609, -0.177303, 0.195044, -0.181025, -0.000596, -0.177276, 0.194495, -0.180655],
        [-0.000609, -0.185291, 0.203544, -0.189100, -0.000595, -0.185263, 0.202972, -0.188716],
        [-0.000608, -0.193279, 0.212040, -0.197173, -0.000594, -0.193251, 0.211445, -0.196777],
        [-0.000607, -0.201268, 0.220530, -0.205247, -0.000593, -0.201240, 0.219914, -0.204838],
        [-0.000607, -0.209257, 0.229015, -0.213320, -0.000593, -0.209229, 0.228379, -0.212898],
        [-0.000606, -0.217246, 0.237496, -0.221392, -0.000592, -0.217218, 0.236839, -0.220958],
        [-0.000606, -0.225235, 0.245972, -0.229464, -0.000591, -0.225207, 0.245296, -0.229017],
        [-0.000605, -0.233224, 0.254443, -0.237535, -0.000591, -0.233196, 0.253748, -0.237076],
        [-0.000604, -0.241214, 0.262909, -0.245605, -0.000590, -0.241185, 0.262196, -0.245134],
        [-0.000604, -0.249203, 0.271371, -0.253675, -0.000589, -0.249174, 0.270640, -0.253192],
        [-0.000603, -0.257193, 0.279828, -0.261745, -0.000589, -0.257164, 0.279080, -0.261249],
        [-0.000602, -0.265182, 0.288280, -0.269814, -0.000588, -0.265153, 0.287517, -0.269305],
        [-0.000602, -0.273172, 0.296728, -0.277882, -0.000587, -0.273143, 0.295949, -0.277361],
        [-0.000601, -0.281162, 0.305172, -0.285950, -0.000586, -0.281132, 0.304377, -0.285416],
        [-0.000601, -0.289151, 0.313612, -0.294017, -0.000586, -0.289122, 0.312802, -0.293471],
        [-0.000600, -0.297141, 0.322047, -0.302084, -0.000585, -0.297112, 0.321223, -0.301526],
        [-0.000599, -0.305131, 0.330478, -0.310150, -0.000584, -0.305101, 0.329641, -0.309579],
        [-0.000599, -0.313121, 0.338905, -0.318215, -0.000584, -0.313091, 0.338055, -0.317633],
        [-0.000598, -0.321111, 0.347327, -0.326280, -0.000583, -0.321081, 0.346466, -0.325685],
        [-0.000597, -0.329101, 0.355746, -0.334345, -0.000582, -0.329070, 0.354873, -0.333738],
        [-0.000597, -0.337091, 0.364161, -0.342409, -0.000582, -0.337060, 0.363276, -0.341790],
        [-0.000596, -0.345081, 0.372572, -0.350473, -0.000581, -0.345050, 0.371677, -0.349841],
        [-0.000596, -0.353071, 0.380979, -0.358536, -0.000580, -0.353040, 0.380074, -0.357892],
        [-0.000595, -0.361061, 0.389382, -0.366599, -0.000580, -0.361030, 0.388468, -0.365942],
        [-0.000594, -0.369051, 0.397782, -0.374661, -0.000579, -0.369020, 0.396859, -0.373992],
        [-0.000594, -0.377041, 0.406178, -0.382723, -0.000578, -0.377010, 0.405246, -0.382042],
        [-0.000593, -0.385031, 0.414570, -0.390784, -0.000578, -0.385000, 0.413631, -0.390091],
        [-0.000592, -0.393021, 0.422959, -0.398845, -0.000577, -0.392990, 0.422012, -0.398140],
        [-0.000592, -0.401011, 0.431344, -0.406906, -0.000576, -0.400980, 0.430391, -0.406188],
        [-0.000591, -0.409002, 0.439725, -0.414966, -0.000576, -0.408970, 0.438766, -0.414236],
        [-0.000590, -0.416992, 0.448103, -0.423025, -0.000575, -0.416961, 0.447139, -0.422283],
        [-0.000590, -0.424982, 0.456478, -0.431085, -0.000574, -0.424951, 0.455509, -0.430330],
        [-0.000589, -0.432973, 0.464850, -0.439144, -0.000574, -0.432941, 0.463876, -0.438377],
        [-0.000589, -0.440963, 0.473218, -0.447202, -0.000573, -0.440931, 0.472240, -0.446423],
        [-0.000588, -0.448953, 0.481582, -0.455261, -0.000572, -0.448922, 0.480602, -0.454469],
        [-0.000587, -0.456944, 0.489944, -0.463318, -0.000571, -0.456912, 0.488960, -0.462515],
        [-0.000587, -0.464934, 0.498302, -0.471376, -0.000571, -0.464902, 0.497316, -0.470560],
        [-0.000586, -0.472925, 0.506657, -0.479433, -0.000570, -0.472893, 0.505670, -0.478605],
        [-0.000585, -0.480915, 0.515009, -0.487490, -0.000569, -0.480883, 0.514021, -0.486649],
        [-0.000585, -0.488906, 0.523358, -0.495547, -0.000569, -0.488873, 0.522369, -0.494693],
        [-0.000584, -0.496896, 0.531704, -0.503603, -0.000568, -0.496864, 0.530715, -0.502737],
        [-0.000583, -0.504887, 0.540047, -0.511659, -0.000567, -0.504854, 0.539059, -0.510781],
        [-0.000583, -0.512877, 0.548387, -0.519714, -0.000567, -0.512845, 0.547400, -0.518824],
        [-0.000582, -0.520868, 0.556723, -0.527770, -0.000566, -0.520835, 0.555738, -0.526867],
        [-0.000581, -0.528859, 0.565057, -0.535825, -0.000565, -0.528826, 0.564074, -0.534909],
        [-0.000581, -0.536849, 0.573388, -0.543880, -0.000565, -0.536817, 0.572408, -0.542951],
        [-0.000580, -0.544840, 0.581716, -0.551934, -0.000564, -0.544807, 0.580740, -0.550993],
        [-0.000579, -0.552831, 0.590041, -0.559989, -0.000563, -0.552798, 0.589069, -0.559035],
        [-0.000579, -0.560821, 0.598364, -0.568043, -0.000562, -0.560788, 0.597396, -0.567076],
        [-0.000578, -0.568812, 0.606683, -0.576096, -0.000562, -0.568779, 0.605720, -0.575117],
        [-0.000577, -0.576803, 0.615000, -0.584150, -0.000561, -0.576770, 0.614043, -0.583158],
        [-0.000576, -0.584794, 0.623314, -0.592203, -0.000560, -0.584760, 0.622363, -0.591198],
        [-0.000576, -0.592784, 0.631625, -0.600256, -0.000560, -0.592751, 0.630681, -0.599238],
        [-0.000575, -0.600775, 0.639934, -0.608309, -0.000559, -0.600742, 0.638998, -0.607278],
        [-0.000574, -0.608766, 0.648239, -0.616362, -0.000558, -0.608733, 0.647311, -0.615318],
        [-0.000574, -0.616757, 0.656542, -0.624414, -0.000558, -0.616724, 0.655623, -0.623357],
        [-0.000573, -0.624748, 0.664843, -0.632467, -0.000557, -0.624714, 0.663933, -0.631396],
        [-0.000572, -0.632739, 0.673141, -0.640519, -0.000556, -0.632705, 0.672241, -0.639435],
        [-0.000571, -0.640730, 0.681436, -0.648571, -0.000555, -0.640696, 0.680547, -0.647474],
        [-0.000571, -0.648721, 0.689728, -0.656622, -0.000555, -0.648687, 0.688851, -0.655512],
        [-0.000570, -0.656712, 0.698018, -0.664674, -0.000554, -0.656678, 0.697152, -0.663550],
        [-0.000569, -0.664703, 0.706306, -0.672726, -0.000553, -0.664669, 0.705452, -0.671588],
        [-0.000568, -0.672694, 0.714590, -0.680777, -0.000552, -0.672660, 0.713750, -0.679626],
        [-0.000568, -0.680685, 0.722873, -0.688828, -0.000552, -0.680651, 0.722046, -0.687663],
        [-0.000567, -0.688676, 0.731152, -0.696879, -0.000551, -0.688642, 0.730340, -0.695701],
        [-0.000566, -0.696667, 0.739429, -0.704930, -0.000550, -0.696633, 0.738633, -0.703738],
        [-0.000565, -0.704658, 0.747704, -0.712981, -0.000549, -0.704624, 0.746923, -0.711775],
        [-0.000565, -0.712649, 0.755976, -0.721031, -0.000549, -0.712615, 0.755212, -0.719811],
        [-0.000564, -0.720640, 0.764246, -0.729082, -0.000548, -0.720606, 0.763498, -0.727848],
        [-0.000563, -0.728631, 0.772513, -0.737133, -0.000547, -0.728597, 0.771783, -0.735884],
        [-0.000562, -0.736622, 0.780777, -0.745183, -0.000546, -0.736588, 0.780066, -0.743920],
        [-0.000561, -0.744613, 0.789039, -0.753233, -0.000546, -0.744579, 0.788348, -0.751956],
        [-0.000561, -0.752604, 0.797299, -0.761284, -0.000545, -0.752570, 0.796627, -0.759992],
        [-0.000560, -0.760596, 0.805556, -0.769334, -0.000544, -0.760561, 0.804905, -0.768028],
        [-0.000559, -0.768587, 0.813810, -0.777384, -0.000543, -0.768553, 0.813181, -0.776063],
        [-0.000558, -0.776578, 0.822062, -0.785434, -0.000542, -0.776544, 0.821455, -0.784099],
        [-0.000557, -0.784569, 0.830312, -0.793484, -0.000542, -0.784535, 0.829728, -0.792134],
        [-0.000515, -0.792415, 0.838404, -0.801223, -0.000540, -0.792402, 0.837864, -0.800002],
        [-0.000416, -0.797107, 0.843477, -0.805558, -0.000418, -0.797098, 0.842906, -0.804302],
        [-0.000420, -0.800270, 0.846809, -0.808623, -0.000422, -0.800264, 0.846252, -0.807362],
    ]
)


# ---------------------------------------------------------------------------
# RobotiqGripper
# ---------------------------------------------------------------------------


class RobotiqGripper(_BaseGripper):
    """Robotiq 2F-140 parallel jaw gripper.

    Supports the Robotiq 2F-140 4-bar linkage gripper used with UR5e arms.
    In kinematic mode, replays a pre-recorded physics trajectory to maintain
    correct linkage geometry. In physics mode, the PhysicsController drives
    the tendon actuator directly.

    The ``prefix`` parameter handles namespacing in multi-robot scenes.
    For a standalone Robotiq model, use ``prefix=""``. For a prefixed model
    (e.g., geodude's ``right_ur5e/gripper/``), pass the full prefix.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        arm_name: Which arm this gripper belongs to.
        prefix: MuJoCo name prefix for all gripper elements.
        grasp_manager: Optional grasp state tracker.
    """

    hand_type: str = "robotiq"
    # The Robotiq 2F-140 has 14 cm of finger travel, so fully-closed
    # can in principle still hold an extremely thin object — but in
    # practice the objects we grasp are much wider than the
    # ``empty_position_threshold`` resolution (default 2% = 2.8 mm on
    # the 140 mm travel). Treating fully-closed as \"empty\" lets the
    # :class:`~mj_manipulator.grasp_verifier.GraspVerifier`'s
    # decisive-negative branch fire immediately when the gripper
    # closed on nothing, which is a crisp, noise-free, motion-
    # independent signal — better than trying to derive the same
    # verdict from F/T readings that bounce around during transport.
    # See personalrobotics/geodude#173 and personalrobotics/mj_manipulator#98
    # for the full story.
    empty_at_fully_closed: bool = True

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        arm_name: str,
        prefix: str = "",
        grasp_manager: GraspManager | None = None,
    ):
        # Resolve actuator
        actuator_name = f"{prefix}fingers_actuator"

        # Resolve body names
        body_names = [f"{prefix}{s}" for s in _BODY_SUFFIXES]

        # Attachment body (finger pad that contacts objects)
        attachment_body = f"{prefix}{_ATTACHMENT_BODY_SUFFIX}"

        super().__init__(
            model=model,
            data=data,
            arm_name=arm_name,
            actuator_name=actuator_name,
            gripper_body_names=body_names,
            attachment_body=attachment_body,
            ctrl_open=0.0,
            ctrl_closed=255.0,
            grasp_manager=grasp_manager,
        )

        # Resolve gripper joint qpos indices for kinematic control
        indices = []
        for suffix in _JOINT_SUFFIXES:
            full_name = f"{prefix}{suffix}"
            joint_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                full_name,
            )
            if joint_id != -1:
                indices.append(model.jnt_qposadr[joint_id])
        self._joint_qpos_indices = np.array(indices, dtype=int)

    def set_kinematic_position(self, t: float) -> None:
        """Set gripper to interpolated position kinematically.

        Uses the pre-recorded 4-bar linkage trajectory to ensure correct
        geometry at any position.

        Args:
            t: Position from 0.0 (open) to 1.0 (closed).
        """
        if len(self._joint_qpos_indices) == 0:
            return

        t = np.clip(t, 0.0, 1.0)
        n = len(_GRIPPER_TRAJECTORY) - 1
        idx = t * n
        idx_low = int(idx)
        idx_high = min(idx_low + 1, n)
        alpha = idx - idx_low

        joint_positions = (1 - alpha) * _GRIPPER_TRAJECTORY[idx_low] + alpha * _GRIPPER_TRAJECTORY[idx_high]

        self._data.qpos[self._joint_qpos_indices] = joint_positions
        mujoco.mj_forward(self._model, self._data)

    def _apply_kinematic_position(self, t: float) -> None:
        self.set_kinematic_position(t)

    def get_actual_position(self) -> float:
        """Get actual gripper position (0=open, 1=closed).

        Reads the driver joint position and maps it to [0, 1] by comparing
        to the open/closed trajectory endpoints.
        """
        if len(self._joint_qpos_indices) == 0:
            return 0.0

        # Use driver joint (index 1) as reference — it has the most range.
        driver_idx = 1
        driver_open = _GRIPPER_TRAJECTORY[0, driver_idx]
        driver_closed = _GRIPPER_TRAJECTORY[-1, driver_idx]

        if abs(driver_closed - driver_open) < 1e-6:
            return 0.0

        driver_pos = self._data.qpos[self._joint_qpos_indices[driver_idx]]
        t = (driver_pos - driver_open) / (driver_closed - driver_open)
        return float(np.clip(t, 0.0, 1.0))
