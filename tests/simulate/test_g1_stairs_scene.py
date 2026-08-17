"""Behavior contract for the independent G1 manual staircase scene."""

from __future__ import annotations

from pathlib import Path

import mujoco
import pytest

ROOT = Path(__file__).parents[2]
G1_XML_ROOT = ROOT / "src/assets/robots/unitree_g1/xmls"
FLAT_SCENE = G1_XML_ROOT / "scene_g1.xml"
STAIRS_SCENE = G1_XML_ROOT / "scene_g1_stairs.xml"
STAIR_NAMES = ("stair_level_1", "stair_level_2", "stair_level_3")


def _geom_box(model: mujoco.MjModel, name: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert geom_id >= 0, f"missing geom {name}"
    position = tuple(float(value) for value in model.geom_pos[geom_id])
    half_size = tuple(float(value) for value in model.geom_size[geom_id])
    return position, half_size


def test_stair_scene_loads_existing_g1_with_three_risers_at_30cm_pitch() -> None:
    """A missing include or wrong stair transform would invalidate the manual trial."""
    flat_model = mujoco.MjModel.from_xml_path(str(FLAT_SCENE))
    stairs_model = mujoco.MjModel.from_xml_path(str(STAIRS_SCENE))

    assert stairs_model.nq == flat_model.nq
    assert stairs_model.nv == flat_model.nv
    assert stairs_model.nu == flat_model.nu == 29
    assert stairs_model.nsensordata == flat_model.nsensordata
    assert all(
        mujoco.mj_name2id(flat_model, mujoco.mjtObj.mjOBJ_GEOM, name) == -1
        for name in STAIR_NAMES
    )

    boxes = [_geom_box(stairs_model, name) for name in STAIR_NAMES]
    leading_edges = [position[0] - size[0] for position, size in boxes]
    rear_edges = [position[0] + size[0] for position, size in boxes]
    top_surfaces = [position[2] + size[2] for position, size in boxes]
    half_widths = [size[1] for _, size in boxes]

    assert leading_edges == pytest.approx((0.75, 1.05, 1.35))
    assert rear_edges == pytest.approx((2.25, 2.25, 2.25))
    assert top_surfaces == pytest.approx((0.08, 0.16, 0.24))
    assert half_widths == pytest.approx((0.60, 0.60, 0.60))
    assert [
        leading_edges[index + 1] - leading_edges[index]
        for index in range(len(leading_edges) - 1)
    ] == pytest.approx((0.30, 0.30))
