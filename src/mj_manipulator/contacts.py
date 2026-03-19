"""Shared contact iteration utility for MuJoCo.

Not part of the public API.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import mujoco


def iter_contacts(
    model: mujoco.MjModel, data: mujoco.MjData
) -> Iterator[tuple[int, int, Any]]:
    """Iterate over active MuJoCo contacts.

    Yields:
        (body_id_1, body_id_2, contact) for each active contact, where
        body_id_1 and body_id_2 are the body IDs of the two geoms in contact.
    """
    for i in range(data.ncon):
        contact = data.contact[i]
        yield (
            model.geom_bodyid[contact.geom1],
            model.geom_bodyid[contact.geom2],
            contact,
        )
