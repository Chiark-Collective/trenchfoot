# ABOUTME: Tests for SDF readiness: normal orientation and metadata export.
# ABOUTME: Ensures meshes are suitable for signed distance field computation.
"""
Tests verifying that generated meshes are ready for SDF computation.

These tests validate:
1. Normal orientation follows "into void" convention
2. SDF metadata is correctly exported
3. Trench opening polygon matches actual geometry
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PKG_ROOT = ROOT / "packages"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))
VENVDIR = ROOT / ".venv"
if VENVDIR.exists():
    for candidate in (VENVDIR / "lib").glob("python*/site-packages"):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            break

from trenchfoot.generate_scenarios import (
    ScenarioDefinition,
    default_scenarios,
    generate_scenarios,
)
from trenchfoot.trench_scene_generator_v3 import (
    scene_spec_from_dict,
    generate_surface_mesh,
)


def _compute_face_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Compute per-face normals for a triangle mesh."""
    p0 = V[F[:, 0]]
    p1 = V[F[:, 1]]
    p2 = V[F[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)
    # Normalize
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-10)  # Avoid division by zero
    return normals / lengths


def _minimal_spec_dict() -> dict:
    """Minimal trench spec for testing."""
    return {
        "path_xy": [[0.0, 0.0], [3.0, 0.0]],
        "width": 1.0,
        "depth": 1.2,
        "wall_slope": 0.1,
        "ground": {"z0": 0.0, "slope": [0.0, 0.0], "size_margin": 2.0},
        "pipes": [],
        "boxes": [],
        "spheres": [],
        "noise": {"enable": False},
    }


class TestFloorNormalOrientation:
    """Tests for trench floor normal orientation."""

    def test_floor_normals_point_up_minimal(self):
        """Floor normals should point UP (+z) for correct SDF sign."""
        spec = scene_spec_from_dict(_minimal_spec_dict())
        result = generate_surface_mesh(spec, make_preview=False)

        V, F = result.groups["trench_bottom"]
        normals = _compute_face_normals(V, F)

        # All floor normals should have positive z component
        z_components = normals[:, 2]
        down_count = np.sum(z_components < 0)

        assert down_count == 0, (
            f"Floor has {down_count}/{len(F)} faces with downward normals. "
            f"Floor normals should all point UP (+z) into the void."
        )

        # Check they're strongly upward (nz > 0.5 for mostly-horizontal faces)
        weak_up_count = np.sum(z_components < 0.5)
        assert weak_up_count == 0, (
            f"Floor has {weak_up_count}/{len(F)} faces with weak upward normals (nz < 0.5). "
            f"Floor should be mostly horizontal."
        )

    @pytest.mark.parametrize("scenario_name", [
        "S01_straight_vwalls",
        "S02_straight_sloped",
        "S03_angled_corner",
        "S04_curved_bend",
        "S05_pipe_cluster",
        "S06_complex_layout",
        "S07_circular_well",
    ])
    def test_floor_normals_point_up_all_scenarios(self, scenario_name):
        """Floor normals should point UP (+z) for all standard scenarios."""
        scenarios = default_scenarios()
        scenario = next((s for s in scenarios if s.name == scenario_name), None)
        if scenario is None:
            pytest.skip(f"{scenario_name} not available")

        spec = scene_spec_from_dict(scenario.spec)
        result = generate_surface_mesh(spec, make_preview=False)

        V, F = result.groups["trench_bottom"]
        normals = _compute_face_normals(V, F)

        z_components = normals[:, 2]
        down_count = np.sum(z_components < 0)

        assert down_count == 0, (
            f"{scenario_name}: Floor has {down_count}/{len(F)} faces with downward normals. "
            f"All floor normals should point UP (+z)."
        )


class TestWallNormalOrientation:
    """Tests for trench wall normal orientation."""

    def test_wall_normals_point_into_void(self):
        """Wall normals should point INTO the trench void (toward centerline)."""
        spec = scene_spec_from_dict(_minimal_spec_dict())
        result = generate_surface_mesh(spec, make_preview=False)

        V, F = result.groups["trench_walls"]
        normals = _compute_face_normals(V, F)

        # For a straight trench along x-axis (y=0), walls are on either side.
        # The trench centerline is at y=0.
        # Left wall (y < 0) should have normals pointing +y (into trench)
        # Right wall (y > 0) should have normals pointing -y (into trench)

        # Compute face centroids
        centroids = (V[F[:, 0]] + V[F[:, 1]] + V[F[:, 2]]) / 3

        # Check each face: direction from centroid to centerline should align with normal
        errors = 0
        for i, (centroid, normal) in enumerate(zip(centroids, normals)):
            # Direction toward centerline (y=0)
            toward_center_y = -np.sign(centroid[1]) if abs(centroid[1]) > 0.01 else 0

            # For wall faces, the y-component of normal should match direction toward center
            if abs(normal[1]) > 0.3:  # Face has significant y-normal component
                if np.sign(normal[1]) != toward_center_y:
                    errors += 1

        assert errors == 0, (
            f"Wall has {errors}/{len(F)} faces with normals pointing away from trench void. "
            f"Wall normals should point inward toward the trench centerline."
        )

    @pytest.mark.parametrize("scenario_name", [
        "S01_straight_vwalls",
        "S02_straight_sloped",
        "S03_angled_corner",
    ])
    def test_wall_normals_mostly_horizontal(self, scenario_name):
        """Wall normals should be mostly horizontal (small z component)."""
        scenarios = default_scenarios()
        scenario = next((s for s in scenarios if s.name == scenario_name), None)
        if scenario is None:
            pytest.skip(f"{scenario_name} not available")

        spec = scene_spec_from_dict(scenario.spec)
        result = generate_surface_mesh(spec, make_preview=False)

        V, F = result.groups["trench_walls"]
        normals = _compute_face_normals(V, F)

        # Wall faces should have mostly horizontal normals
        # Allow some z component for sloped walls but should be < 0.7
        z_components = np.abs(normals[:, 2])
        vertical_wall_ratio = np.mean(z_components < 0.7)

        assert vertical_wall_ratio > 0.8, (
            f"{scenario_name}: Only {vertical_wall_ratio*100:.1f}% of wall faces have |nz| < 0.7. "
            f"Wall faces should be mostly vertical (horizontal normals)."
        )


class TestGroundNormalOrientation:
    """Tests for ground surface normal orientation."""

    def test_ground_normals_point_up(self):
        """Ground surface normals should point UP (+z)."""
        spec = scene_spec_from_dict(_minimal_spec_dict())
        result = generate_surface_mesh(spec, make_preview=False)

        V, F = result.groups["ground_surface"]
        normals = _compute_face_normals(V, F)

        z_components = normals[:, 2]
        down_count = np.sum(z_components < 0)

        assert down_count == 0, (
            f"Ground has {down_count}/{len(F)} faces with downward normals. "
            f"Ground normals should all point UP (+z)."
        )


class TestSDFMetadata:
    """Tests for SDF metadata export."""

    def test_metadata_exported(self, tmp_path):
        """SDF metadata JSON should be exported alongside mesh."""
        spec = scene_spec_from_dict(_minimal_spec_dict())
        result = generate_surface_mesh(spec, make_preview=False)
        files = result.persist(tmp_path)

        assert files.sdf_metadata_path is not None, "sdf_metadata_path should be set"
        assert files.sdf_metadata_path.exists(), "sdf_metadata.json should exist"

    def test_metadata_has_required_fields(self, tmp_path):
        """SDF metadata should contain all required fields."""
        spec = scene_spec_from_dict(_minimal_spec_dict())
        result = generate_surface_mesh(spec, make_preview=False)
        files = result.persist(tmp_path)

        with files.sdf_metadata_path.open() as f:
            data = json.load(f)

        assert "sdf_metadata" in data, "Root key 'sdf_metadata' missing"
        meta = data["sdf_metadata"]

        required_fields = ["version", "normal_convention", "geometry_type", "trench_opening"]
        for field in required_fields:
            assert field in meta, f"Required field '{field}' missing from metadata"

        assert meta["version"] == "2.0"
        assert meta["normal_convention"] == "into_void"
        assert meta["geometry_type"] == "open_trench"

    def test_metadata_trench_opening_structure(self, tmp_path):
        """Trench opening should have proper polygon structure."""
        spec = scene_spec_from_dict(_minimal_spec_dict())
        result = generate_surface_mesh(spec, make_preview=False)
        files = result.persist(tmp_path)

        with files.sdf_metadata_path.open() as f:
            data = json.load(f)

        opening = data["sdf_metadata"]["trench_opening"]
        assert opening["type"] == "polygon"
        assert "vertices_xy" in opening
        assert "z_level" in opening

        vertices = opening["vertices_xy"]
        assert len(vertices) >= 4, "Trench opening should have at least 4 vertices"

        # Each vertex should be [x, y]
        for v in vertices:
            assert len(v) == 2, f"Vertex should be [x, y], got {v}"

    def test_metadata_polygon_bounds_match_trench(self, tmp_path):
        """Trench opening polygon should match actual trench geometry bounds."""
        spec_dict = _minimal_spec_dict()
        spec = scene_spec_from_dict(spec_dict)
        result = generate_surface_mesh(spec, make_preview=False)
        files = result.persist(tmp_path)

        with files.sdf_metadata_path.open() as f:
            data = json.load(f)

        opening = data["sdf_metadata"]["trench_opening"]
        vertices = np.array(opening["vertices_xy"])

        # The trench path goes from [0,0] to [3,0] with width 1.0
        # The metadata uses the base half-width (0.5), which represents the
        # footprint of the trench floor, not the sloped width at z=0.
        expected_half_width = 0.5

        # Check x range covers path (0 to 3)
        x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
        assert x_min <= 0.1, f"Opening x_min ({x_min}) should be near 0"
        assert x_max >= 2.9, f"Opening x_max ({x_max}) should be near 3"

        # Check y range matches half-width
        y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
        assert abs(y_min + expected_half_width) < 0.1, (
            f"Opening y_min ({y_min}) should be near {-expected_half_width}"
        )
        assert abs(y_max - expected_half_width) < 0.1, (
            f"Opening y_max ({y_max}) should be near {expected_half_width}"
        )

    def test_metadata_surface_groups_present(self, tmp_path):
        """Surface groups should list all mesh groups with properties."""
        spec = scene_spec_from_dict(_minimal_spec_dict())
        result = generate_surface_mesh(spec, make_preview=False)
        files = result.persist(tmp_path)

        with files.sdf_metadata_path.open() as f:
            data = json.load(f)

        groups = data["sdf_metadata"]["surface_groups"]

        expected_groups = ["trench_bottom", "trench_walls", "ground_surface"]
        for group in expected_groups:
            assert group in groups, f"Expected surface group '{group}' in metadata"
            assert "normal_direction" in groups[group]
            assert "surface_type" in groups[group]

        # Verify correct normal conventions documented
        assert groups["trench_bottom"]["normal_direction"] == "up"
        assert groups["trench_walls"]["normal_direction"] == "inward"
        assert groups["ground_surface"]["normal_direction"] == "up"

    def test_metadata_embedded_objects_count(self, tmp_path):
        """Embedded objects count should match spec."""
        spec_dict = _minimal_spec_dict()
        # Note: PipeSpec uses s_center, SphereSpec uses s
        spec_dict["pipes"] = [
            {"radius": 0.1, "length": 1.0, "angle_deg": 0, "s_center": 0.5, "z": -0.6, "offset_u": 0},
            {"radius": 0.05, "length": 0.8, "angle_deg": 90, "s_center": 0.3, "z": -0.4, "offset_u": 0},
        ]
        spec_dict["spheres"] = [{"radius": 0.15, "s": 0.7, "z": -0.5}]

        spec = scene_spec_from_dict(spec_dict)
        result = generate_surface_mesh(spec, make_preview=False)
        files = result.persist(tmp_path)

        with files.sdf_metadata_path.open() as f:
            data = json.load(f)

        embedded = data["sdf_metadata"]["embedded_objects"]
        assert embedded["pipes"] == 2
        assert embedded["spheres"] == 1
        assert embedded["boxes"] == 0

    def test_metadata_l_shaped_polygon_is_non_convex(self, tmp_path):
        """L-shaped trench opening polygon should be non-convex (not use ConvexHull).

        The L-shape has an inner corner that ConvexHull would skip. This test
        verifies that the boundary extraction correctly captures the inner corner.
        """
        # Create an L-shaped path
        spec_dict = _minimal_spec_dict()
        spec_dict["path_xy"] = [
            [0.0, 0.0],   # Start
            [3.0, 0.0],   # Go right
            [3.0, 2.0],   # Turn up
        ]
        spec_dict["width"] = 1.0
        spec_dict["depth"] = 1.2

        spec = scene_spec_from_dict(spec_dict)
        result = generate_surface_mesh(spec, make_preview=False)
        files = result.persist(tmp_path)

        with files.sdf_metadata_path.open() as f:
            data = json.load(f)

        opening = data["sdf_metadata"]["trench_opening"]
        vertices = np.array(opening["vertices_xy"])

        # An L-shaped trench should have 6 vertices (correct boundary extraction)
        # ConvexHull would give 4 vertices (skipping the inner corner)
        # Note: The L-shape has vertices at:
        #   - left side (x ~ 0)
        #   - inner corner of L (around y ~ 0.5 for horizontal arm)
        #   - right/top end (x ~ 3.5, y ~ 2.0)
        assert len(vertices) >= 6, (
            f"L-shaped polygon has {len(vertices)} vertices, expected >= 6. "
            "This might indicate ConvexHull is being used instead of boundary extraction."
        )

        # Verify the polygon covers the full L-shape bounds
        x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
        y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()

        # X should cover the horizontal arm (roughly 0 to 3.5 with width=1)
        assert x_min < 0.5, f"L-shape x_min ({x_min}) should be < 0.5"
        assert x_max > 2.5, f"L-shape x_max ({x_max}) should be > 2.5"

        # Y should cover the vertical arm (roughly -0.5 to 2 with width=1)
        assert y_min < 0.5, f"L-shape y_min ({y_min}) should be < 0.5"
        assert y_max > 1.5, f"L-shape y_max ({y_max}) should be > 1.5"

        # Critical: Verify the inner corner vertices are present
        # The inner corner of the L is around x=2.5, y=0.5 (inner corner)
        # ConvexHull would skip these vertices and go directly from (0, 0.5) to (3.5, 2)
        inner_corner_found = False
        for v in vertices:
            # Look for a vertex at the inner corner (around x=2.5, y between 0 and 1)
            if 2.0 < v[0] < 3.0 and 0.0 < v[1] < 1.0:
                inner_corner_found = True
                break

        assert inner_corner_found, (
            f"Inner corner vertex not found in L-shaped polygon. "
            f"Vertices: {vertices.tolist()}. "
            "This suggests ConvexHull is being used instead of boundary extraction."
        )


class TestCircularWellNormals:
    """Specific tests for circular well (S07) normal orientation."""

    def test_s07_floor_normals_all_upward(self):
        """S07 circular well floor should have 100% upward normals."""
        scenarios = default_scenarios()
        s07 = next((s for s in scenarios if s.name == "S07_circular_well"), None)
        if s07 is None:
            pytest.skip("S07_circular_well not available")

        spec = scene_spec_from_dict(s07.spec)
        result = generate_surface_mesh(spec, make_preview=False)

        V, F = result.groups["trench_bottom"]
        normals = _compute_face_normals(V, F)

        z_components = normals[:, 2]
        down_count = np.sum(z_components < 0)
        up_pct = 100 * (1 - down_count / len(F))

        assert down_count == 0, (
            f"S07 circular well has {down_count}/{len(F)} floor faces with downward normals "
            f"({up_pct:.1f}% upward). Should be 100% upward."
        )

    def test_s07_annulus_ground_normals_upward(self):
        """S07 ground annulus should have upward normals."""
        scenarios = default_scenarios()
        s07 = next((s for s in scenarios if s.name == "S07_circular_well"), None)
        if s07 is None:
            pytest.skip("S07_circular_well not available")

        spec = scene_spec_from_dict(s07.spec)
        result = generate_surface_mesh(spec, make_preview=False)

        V, F = result.groups["ground_surface"]
        normals = _compute_face_normals(V, F)

        z_components = normals[:, 2]
        down_count = np.sum(z_components < 0)

        assert down_count == 0, (
            f"S07 ground annulus has {down_count}/{len(F)} faces with downward normals. "
            f"Ground should all point UP."
        )
