# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for MjSpec-level gravcomp helpers.

Covers:

- ``add_subtree_gravcomp`` as a generic primitive:
    - Returns correct body count (Franka = 11, UR5e = 7)
    - Sets gravcomp=1 on the expected set of bodies (via compile + read
      back from MjModel)
    - Raises ValueError with a helpful message when the root body name
      is missing
    - Is idempotent on repeated calls and overlapping subtrees

- ``add_franka_gravcomp`` and ``add_ur5e_gravcomp`` wrappers:
    - Delegate correctly to ``add_subtree_gravcomp``
    - Compiled model's ``qfrc_gravcomp`` matches ``qfrc_bias`` at the
      home pose (the property that actually matters for sim-to-real
      parity: gravity is fully compensated)

Tests skip automatically if the mujoco_menagerie is not available.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures — skip the whole module if menagerie is not available
# ---------------------------------------------------------------------------


try:
    from mj_manipulator.menagerie import menagerie_scene
except ImportError:  # pragma: no cover
    pytest.skip("mj_manipulator.menagerie not available", allow_module_level=True)


@pytest.fixture
def franka_spec():
    try:
        scene = menagerie_scene("franka_emika_panda")
    except FileNotFoundError:
        pytest.skip("mujoco_menagerie/franka_emika_panda not found")
    return mujoco.MjSpec.from_file(str(scene))


@pytest.fixture
def ur5e_spec():
    try:
        scene = menagerie_scene("universal_robots_ur5e")
    except FileNotFoundError:
        pytest.skip("mujoco_menagerie/universal_robots_ur5e not found")
    return mujoco.MjSpec.from_file(str(scene))


# ---------------------------------------------------------------------------
# add_subtree_gravcomp — generic walker
# ---------------------------------------------------------------------------


# Body sets that the walker should touch for each menagerie arm. These
# are the ground truth that the old hardcoded per-arm helpers curated
# by hand. Asserting exact set equality catches both "walker missed a
# body" and "walker grabbed something extra."
_FRANKA_EXPECTED_BODIES = {
    "link0",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "link7",
    "hand",
    "left_finger",
    "right_finger",
}

_UR5E_EXPECTED_BODIES = {
    "base",
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
}


def _gravcomp_body_set(model: mujoco.MjModel) -> set[str]:
    """Return the set of body names with gravcomp > 0 in a compiled model."""
    out = set()
    for bid in range(model.nbody):
        if model.body_gravcomp[bid] > 0:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
            if name:
                out.add(name)
    return out


class TestAddSubtreeGravcomp:
    """Unit tests for the generic subtree walker."""

    def test_franka_returns_expected_count(self, franka_spec):
        from mj_manipulator.arm import add_subtree_gravcomp

        count = add_subtree_gravcomp(franka_spec, "link0")
        assert count == len(_FRANKA_EXPECTED_BODIES)

    def test_franka_touches_expected_body_set(self, franka_spec):
        from mj_manipulator.arm import add_subtree_gravcomp

        add_subtree_gravcomp(franka_spec, "link0")
        model = franka_spec.compile()
        assert _gravcomp_body_set(model) == _FRANKA_EXPECTED_BODIES

    def test_ur5e_returns_expected_count(self, ur5e_spec):
        from mj_manipulator.arm import add_subtree_gravcomp

        count = add_subtree_gravcomp(ur5e_spec, "base")
        assert count == len(_UR5E_EXPECTED_BODIES)

    def test_ur5e_touches_expected_body_set(self, ur5e_spec):
        from mj_manipulator.arm import add_subtree_gravcomp

        add_subtree_gravcomp(ur5e_spec, "base")
        model = ur5e_spec.compile()
        assert _gravcomp_body_set(model) == _UR5E_EXPECTED_BODIES

    def test_missing_root_raises(self, franka_spec):
        from mj_manipulator.arm import add_subtree_gravcomp

        with pytest.raises(ValueError) as exc_info:
            add_subtree_gravcomp(franka_spec, "totally_not_a_body")
        msg = str(exc_info.value)
        assert "totally_not_a_body" in msg
        # Error message lists top-level world-body children so the
        # user can see what's actually available.
        assert "link0" in msg

    def test_idempotent_repeated_calls(self, franka_spec):
        from mj_manipulator.arm import add_subtree_gravcomp

        # Two calls on the same spec — second should not error and
        # should not change the set of gravcomp'd bodies.
        count1 = add_subtree_gravcomp(franka_spec, "link0")
        count2 = add_subtree_gravcomp(franka_spec, "link0")
        assert count1 == count2 == len(_FRANKA_EXPECTED_BODIES)
        model = franka_spec.compile()
        assert _gravcomp_body_set(model) == _FRANKA_EXPECTED_BODIES

    def test_overlapping_subtrees_are_idempotent(self, franka_spec):
        from mj_manipulator.arm import add_subtree_gravcomp

        # Touching link0 (root) then link1 (child of link0) should
        # visit link1's subtree twice but end in the same state.
        add_subtree_gravcomp(franka_spec, "link0")
        add_subtree_gravcomp(franka_spec, "link1")
        model = franka_spec.compile()
        assert _gravcomp_body_set(model) == _FRANKA_EXPECTED_BODIES

    def test_does_not_touch_bodies_outside_subtree(self, franka_spec):
        """Adding an extra body under worldbody (not under link0) should
        not be affected by a call that targets link0."""
        from mj_manipulator.arm import add_subtree_gravcomp

        # Add a table body as a sibling of link0 under worldbody
        table = franka_spec.worldbody.add_body()
        table.name = "table"
        g = table.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [0.5, 0.5, 0.05]
        g.pos = [0, 0, -0.05]

        add_subtree_gravcomp(franka_spec, "link0")
        model = franka_spec.compile()
        gc_bodies = _gravcomp_body_set(model)

        # link0 subtree should have gravcomp
        assert "link0" in gc_bodies
        # The table we added should NOT have gravcomp — the walker
        # was scoped to the link0 subtree.
        assert "table" not in gc_bodies


# ---------------------------------------------------------------------------
# Per-arm wrapper delegation
# ---------------------------------------------------------------------------


class TestPerArmWrappers:
    """Tests that the per-arm helpers delegate to the walker correctly
    and produce models where gravity is fully compensated at home."""

    def test_franka_wrapper_matches_walker_body_set(self, franka_spec):
        from mj_manipulator.arms.franka import add_franka_gravcomp

        add_franka_gravcomp(franka_spec)
        model = franka_spec.compile()
        assert _gravcomp_body_set(model) == _FRANKA_EXPECTED_BODIES

    def test_ur5e_wrapper_matches_walker_body_set(self, ur5e_spec):
        from mj_manipulator.arms.ur5e import add_ur5e_gravcomp

        add_ur5e_gravcomp(ur5e_spec)
        model = ur5e_spec.compile()
        assert _gravcomp_body_set(model) == _UR5E_EXPECTED_BODIES

    def test_franka_wrapper_qfrc_gravcomp_matches_qfrc_bias_at_home(self, franka_spec):
        """The property that actually matters: after add_franka_gravcomp
        + compile, the gravity term in qfrc_bias is fully canceled by
        qfrc_gravcomp at the home pose, so the PD loop doesn't have to
        fight gravity.
        """
        from mj_manipulator.arms.franka import FRANKA_HOME, add_franka_gravcomp

        add_franka_gravcomp(franka_spec)
        model = franka_spec.compile()
        data = mujoco.MjData(model)

        # Set home qpos on joint1-joint7
        for i in range(7):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i + 1}")
            data.qpos[model.jnt_qposadr[jid]] = FRANKA_HOME[i]
        mujoco.mj_forward(model, data)

        # Residual = gravity_torque - gravcomp_torque ≈ 0
        max_residual = float(np.max(np.abs(data.qfrc_bias - data.qfrc_gravcomp)))
        assert max_residual < 1e-6, f"Franka gravity not fully compensated: residual={max_residual}"

    def test_ur5e_wrapper_qfrc_gravcomp_matches_qfrc_bias_at_home(self, ur5e_spec):
        """Same regression check for UR5e."""
        from mj_manipulator.arms.ur5e import UR5E_HOME, add_ur5e_gravcomp

        add_ur5e_gravcomp(ur5e_spec)
        model = ur5e_spec.compile()
        data = mujoco.MjData(model)

        joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        for i, jname in enumerate(joint_names):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            data.qpos[model.jnt_qposadr[jid]] = UR5E_HOME[i]
        mujoco.mj_forward(model, data)

        max_residual = float(np.max(np.abs(data.qfrc_bias - data.qfrc_gravcomp)))
        assert max_residual < 1e-6, f"UR5e gravity not fully compensated: residual={max_residual}"
