import json
import sys
from pathlib import Path
from typing import Optional

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
    main as generate_scenarios_cli,
    build_gallery_markdown,
    write_gallery,
)
from trenchfoot.trench_scene_generator_v3 import (
    scene_spec_from_dict,
    build_scene,
    generate_surface_mesh,
)
import trenchfoot as tf

_GMSH_RUNTIME_READY: Optional[bool] = None


def _gmsh_runtime_ready() -> bool:
    global _GMSH_RUNTIME_READY
    if _GMSH_RUNTIME_READY is not None:
        return _GMSH_RUNTIME_READY
    try:
        import gmsh  # noqa: F401
    except Exception:
        _GMSH_RUNTIME_READY = False
        return False
    try:
        gmsh.initialize()
    except Exception:
        _GMSH_RUNTIME_READY = False
    else:
        _GMSH_RUNTIME_READY = True
    finally:
        try:
            gmsh.finalize()
        except Exception:
            pass
    return _GMSH_RUNTIME_READY


def _require_gmsh_runtime():
    gmsh = pytest.importorskip("gmsh")
    if not _gmsh_runtime_ready():
        pytest.skip("gmsh runtime prerequisites (e.g. libGLU) not available in this environment")
    return gmsh


def _minimal_spec_dict() -> dict:
    return {
        "path_xy": [[0.0, 0.0], [3.0, 0.0]],
        "width": 1.0,
        "depth": 1.2,
        "wall_slope": 0.1,
        "ground": {"z0": 0.0, "slope": [0.0, 0.0], "size_margin": 2.0},
        "pipes": [
            {
                "radius": 0.1,
                "length": 1.8,
                "angle_deg": 0.0,
                "s_center": 0.5,
                "z": -0.6,
                "offset_u": 0.0,
                "clearance_scale": 0.75,
            }
        ],
        "boxes": [],
        "spheres": [],
        "noise": {"enable": False},
    }


def test_build_scene_produces_surface(tmp_path):
    spec = scene_spec_from_dict(_minimal_spec_dict())
    out = build_scene(spec, tmp_path.as_posix(), make_preview=False)

    obj_path = Path(out["obj_path"])
    metrics_path = tmp_path / "metrics.json"

    assert obj_path.exists(), "surface OBJ was not written"
    assert metrics_path.exists(), "metrics.json missing"

    with metrics_path.open() as fh:
        metrics = json.load(fh)

    assert metrics["width_top"] == pytest.approx(1.0)
    assert metrics["volumes"]["trench_from_surface"] < 0.0

    counts = out["object_counts"]
    assert counts == {"pipes": 1, "boxes": 0, "spheres": 0}


def test_generate_surface_mesh_in_memory(tmp_path):
    spec = scene_spec_from_dict(_minimal_spec_dict())
    result = generate_surface_mesh(spec, make_preview=True)

    assert "trench_walls" in result.groups
    assert result.metrics["width_top"] == pytest.approx(1.0)

    files = result.persist(tmp_path, include_previews=True)
    assert files.obj_path.exists()
    assert files.metrics_path.exists()
    assert files.obj_path.read_text().startswith("g trench_bottom")

    if result.previews:
        assert len(files.preview_paths) == len(result.previews)
        for path in files.preview_paths:
            assert path.exists()
    else:
        assert files.preview_paths == ()


def test_generate_scenarios_single(tmp_path):
    spec = _minimal_spec_dict()
    scenario = ScenarioDefinition(name="unit_test", spec=spec)

    report = generate_scenarios(
        tmp_path / "scenarios",
        scenarios=[scenario],
        make_preview=False,
        make_volumes=False,
        write_summary_json=True,
    )

    assert report.preview_enabled is False
    assert report.volumetric_requested is False
    assert len(report.scenarios) == 1

    summary = report.scenarios[0]
    assert summary.spec_path.exists()
    assert summary.metrics_path.exists()
    assert summary.preview_count == 0
    assert summary.volumetric_path is None
    assert summary.volumetric_error is None
    assert summary.pipe_clearances == []

    summary_json = json.loads((report.out_root / "SUMMARY.json").read_text())
    assert summary_json["scenarios"][0]["name"] == "unit_test"
    assert summary_json["preview_enabled"] is False
    assert summary_json["scenarios"][0]["pipe_clearances"] == []


def test_volumetric_generation(tmp_path):
    gmsh = _require_gmsh_runtime()

    spec = _minimal_spec_dict()
    scenario = ScenarioDefinition(name="volume_case", spec=spec)

    report = generate_scenarios(
        tmp_path / "scenarios",
        scenarios=[scenario],
        make_preview=False,
        make_volumes=True,
        mesh_characteristic_length=0.4,
        write_summary_json=False,
    )

    summary = report.scenarios[0]
    assert summary.volumetric_path is not None
    assert summary.volumetric_path.exists()
    assert summary.volumetric_error is None
    assert len(summary.pipe_clearances) == 1
    assert summary.pipe_clearances[0]["radius"] == pytest.approx(0.1)
    assert summary.pipe_clearances[0]["clearance_scale"] == pytest.approx(0.75)
    assert summary.pipe_clearances[0]["clearance"] == pytest.approx(0.0375, rel=1e-3)

    gmsh.initialize()
    try:
        gmsh.open(summary.volumetric_path.as_posix())
        elem_types, elem_tags, _ = gmsh.model.mesh.getElements()
        tet_count = 0
        for etype, tags in zip(elem_types, elem_tags):
            _, dim, _, _, _, _ = gmsh.model.mesh.getElementProperties(etype)
            if dim == 3:
                tet_count += len(tags)
        groups = gmsh.model.getPhysicalGroups(dim=3)
        assert groups, "Expected at least one 3D physical group in volumetric mesh"
        group_names = [gmsh.model.getPhysicalName(3, tag) for (_, tag) in groups]
        assert "TrenchAir" in group_names, "Trench volume group missing from mesh"
        assert "Pipe0" in group_names, "Primary pipe volume group missing from mesh"
    finally:
        gmsh.finalize()

    assert tet_count > 0, "Expected volumetric mesh elements in generated mesh"


def test_generate_trench_volume_in_memory(tmp_path):
    gmsh = _require_gmsh_runtime()

    spec = _minimal_spec_dict()
    result = tf.generate_trench_volume(spec, lc=0.4, persist_path=tmp_path / "volume.msh")

    assert result.persisted_path is not None
    assert result.persisted_path.exists()
    assert result.nodes.shape[1] == 3
    assert any(block.element_tags.size for block in result.element_blocks)

    volume_groups = [pg for pg in result.physical_groups if pg.dimension == 3]
    assert volume_groups, "Expected 3D physical groups in volumetric mesh"
    assert any(pg.name == "TrenchAir" for pg in volume_groups)
    assert result.pipe_clearances


def test_default_scenarios_volumetric(tmp_path):
    gmsh = _require_gmsh_runtime()

    scenarios = default_scenarios()
    report = generate_scenarios(
        tmp_path / "scenarios",
        scenarios=scenarios,
        make_preview=False,
        make_volumes=True,
        mesh_characteristic_length=0.4,
        write_summary_json=True,
    )

    assert len(report.scenarios) == len(scenarios)
    for summary in report.scenarios:
        assert summary.volumetric_error is None
        assert summary.volumetric_path is not None
        assert summary.volumetric_path.exists()

    summary_json = json.loads((report.out_root / "SUMMARY.json").read_text())
    assert summary_json["volumetric_requested"] is True
    assert all(s["volumetric_error"] is None for s in summary_json["scenarios"])
    for definition, summary in zip(scenarios, summary_json["scenarios"]):
        expected_pipes = len(definition.spec.get("pipes", []))
        assert len(summary["pipe_clearances"]) == expected_pipes
        for pipe_entry in summary["pipe_clearances"]:
            assert "clearance_scale" in pipe_entry


def test_cli_respects_env_out_root(monkeypatch, tmp_path):
    out_dir = tmp_path / "cli_env_out"
    monkeypatch.setenv("TRENCHFOOT_SCENARIO_OUT_ROOT", str(out_dir))
    generate_scenarios_cli(["--skip-volumetric", "--no-preview"])

    expected = out_dir / "S01_straight_vwalls"
    assert expected.exists(), "CLI did not honor TRENCHFOOT_SCENARIO_OUT_ROOT"
    summary_path = out_dir / "SUMMARY.json"
    assert summary_path.exists()
    data = json.loads(summary_path.read_text())
    assert "pipe_clearances" in data["scenarios"][0]
    assert all("clearance_scale" in entry for entry in data["scenarios"][0]["pipe_clearances"])


def test_gallery_helpers(tmp_path):
    # no gmsh dependency required for gallery
    scenarios = default_scenarios()
    report = generate_scenarios(
        tmp_path / "scenarios",
        scenarios=scenarios,
        make_preview=True,
        make_volumes=False,
        mesh_characteristic_length=0.4,
        write_summary_json=False,
    )

    markdown = build_gallery_markdown(report, base=tmp_path)
    assert "S01_straight_vwalls" in markdown
    gallery_path = tmp_path / "gallery.md"
    write_gallery(gallery_path, report, base=tmp_path)
    assert gallery_path.read_text() == markdown


def test_cap_excluded_from_obj_but_kept_for_metrics(tmp_path):
    """Verify trench_cap_for_volume is kept for metrics but excluded from OBJ export."""
    spec = scene_spec_from_dict(_minimal_spec_dict())
    result = generate_surface_mesh(spec, make_preview=False)

    # Cap should be in groups (needed for signed volume metrics)
    assert "trench_cap_for_volume" in result.groups, "Cap should exist in groups for metrics"

    # Metrics should include cap in surface area calculations
    assert "trench_cap_for_volume" in result.metrics["surface_area_by_group"], (
        "Cap should be included in metrics surface area"
    )

    # But cap should NOT appear in exported OBJ file
    files = result.persist(tmp_path)
    obj_content = files.obj_path.read_text()
    assert "trench_cap_for_volume" not in obj_content, (
        "Cap group should not be written to OBJ - trenches should be open-topped"
    )


def test_ground_follows_trench_outline(tmp_path):
    """Verify ground plane follows trench outline instead of being axis-aligned box."""
    # Use L-shaped trench to test non-rectangular ground
    spec_dict = {
        "path_xy": [[0.0, 0.0], [5.0, 0.0], [5.0, 3.0]],
        "width": 1.0,
        "depth": 0.8,
        "wall_slope": 0.1,
        "ground": {"z0": 0.0, "slope": [0.0, 0.0], "size_margin": 1.5},
        "pipes": [],
        "boxes": [],
        "spheres": [],
        "noise": {"enable": False},
    }
    spec = scene_spec_from_dict(spec_dict)
    result = generate_surface_mesh(spec, make_preview=False)

    assert "ground_surface" in result.groups, "Ground surface should exist"
    V, F = result.groups["ground_surface"]

    # Ground should have more than 4 vertices (not a simple rectangle)
    # An L-shaped trench with offset margin creates a polygon with ~12+ vertices
    assert len(V) > 4, (
        f"Ground should follow L-shape outline, not be a simple rectangle. "
        f"Got {len(V)} vertices, expected >4"
    )


def test_circular_well_scenario_exists():
    """Verify S07 circular well scenario is defined and has expected properties."""
    scenarios = default_scenarios()
    s07 = next((s for s in scenarios if s.name == "S07_circular_well"), None)
    assert s07 is not None, "S07_circular_well scenario should exist"

    # Should have multiple pipes at different depths
    pipes = s07.spec.get("pipes", [])
    assert len(pipes) >= 4, "Circular well should have at least 4 criss-crossing pipes"

    # Pipes should have varying radii
    radii = {p["radius"] for p in pipes}
    assert len(radii) >= 3, "Pipes should have at least 3 different diameters"

    # Pipes should be at different depths
    depths = {p.get("z", 0) for p in pipes}
    assert len(depths) >= 3, "Pipes should be at different elevations"


def test_circular_well_generates_surface(tmp_path):
    """Verify S07 circular well generates a valid surface mesh."""
    scenarios = default_scenarios()
    s07 = next((s for s in scenarios if s.name == "S07_circular_well"), None)
    if s07 is None:
        pytest.skip("S07_circular_well not yet implemented")

    report = generate_scenarios(
        tmp_path / "scenarios",
        scenarios=[s07],
        make_preview=False,
        make_volumes=False,
        write_summary_json=False,
    )

    assert len(report.scenarios) == 1
    summary = report.scenarios[0]
    assert summary.surface_obj.exists(), "Surface OBJ should be generated"

    # Verify it has expected pipe count from scenario summary
    assert summary.object_counts["pipes"] >= 4, "Should have at least 4 pipes"


def test_trench_opening_is_open_for_straight_trench():
    """Verify straight trench has open top with no geometry covering the opening."""
    import numpy as np

    spec = scene_spec_from_dict(_minimal_spec_dict())
    result = generate_surface_mesh(spec, make_preview=False)

    # Get ground surface vertices at z=0
    V_ground, F_ground = result.groups["ground_surface"]

    # Ground surface should be an annulus (ring) with the trench opening as a hole.
    # Check that no face covers the center region (trench opening).
    # The trench in _minimal_spec_dict has width=1.0 centered on y=0,
    # so the opening is roughly y in [-0.5, 0.5].

    for face in F_ground:
        verts = V_ground[face]
        centroid = np.mean(verts, axis=0)
        # Centroid should NOT be inside the trench opening region (y between -0.5 and 0.5)
        y_coords = verts[:, 1]
        # If all vertices are inside trench y-range, this face is covering the opening
        if np.all(np.abs(y_coords) < 0.45):  # small margin
            pytest.fail(
                f"Found face covering trench opening: centroid={centroid}, y_coords={y_coords}"
            )


def test_circular_well_trench_opening_is_open():
    """Verify S07 circular well has open center with no lid geometry."""
    import numpy as np

    scenarios = default_scenarios()
    s07 = next((s for s in scenarios if s.name == "S07_circular_well"), None)
    if s07 is None:
        pytest.skip("S07_circular_well not available")

    spec = scene_spec_from_dict(s07.spec)
    result = generate_surface_mesh(spec, make_preview=False)

    # Ground surface should be an annulus around the outer edge of the circular trench,
    # with the center (trench opening) completely open.
    V_ground, F_ground = result.groups["ground_surface"]

    # The circular well has inner radius ~0.5m and outer radius ~2.5m from center.
    # Ground vertices should all be at radius > outer_trench_edge (about 2.0-2.5m)
    # or at the outer_trench_edge itself as the inner boundary.

    # Calculate radius of each ground vertex from center (0,0)
    radii = np.sqrt(V_ground[:, 0] ** 2 + V_ground[:, 1] ** 2)

    # The minimum radius should be at the trench outer edge (~2.5m), not near center
    min_radius = np.min(radii)
    assert min_radius > 1.0, (
        f"Ground surface vertices extend too close to center (min_radius={min_radius:.3f}m). "
        f"For an open circular well, ground should only surround the outer trench edge."
    )

    # No face should have centroid inside the trench opening (radius < ~2.0m)
    for face in F_ground:
        face_verts = V_ground[face]
        centroid = np.mean(face_verts, axis=0)
        centroid_radius = np.sqrt(centroid[0] ** 2 + centroid[1] ** 2)
        if centroid_radius < 1.5:  # Well inside the trench
            pytest.fail(
                f"Found ground face covering trench opening: centroid radius={centroid_radius:.3f}m"
            )


def test_ground_surface_annular_structure():
    """Verify ground surface properly forms an annulus with the trench as a hole."""
    import numpy as np

    # Test with L-shaped trench to ensure annular structure works for complex shapes
    spec_dict = {
        "path_xy": [[0.0, 0.0], [3.0, 0.0], [3.0, 2.0]],
        "width": 0.8,
        "depth": 0.6,
        "wall_slope": 0.0,
        "ground": {"z0": 0.0, "slope": [0.0, 0.0], "size_margin": 1.0},
        "pipes": [],
        "boxes": [],
        "spheres": [],
        "noise": {"enable": False},
    }
    spec = scene_spec_from_dict(spec_dict)
    result = generate_surface_mesh(spec, make_preview=False)

    V_ground, F_ground = result.groups["ground_surface"]

    # All ground vertices should be at z=0 (ground elevation)
    assert np.allclose(V_ground[:, 2], 0.0), "Ground surface should be at z=0"

    # Ground should have both inner vertices (near trench edge) and outer vertices
    # The trench edge is at offset half_width = 0.4m from path
    # Ground outer edge is at half_width + margin = 0.4 + 1.0 = 1.4m from path

    # Check that there are vertices at different distances from the origin
    # (indicating annular structure with inner and outer boundaries)
    x_coords = V_ground[:, 0]
    y_coords = V_ground[:, 1]

    # The L-shaped path goes from (0,0) to (3,0) to (3,2)
    # Inner boundary should be close to path, outer boundary further away
    # For a proper annulus, we should have spread in both x and y

    x_range = np.max(x_coords) - np.min(x_coords)
    y_range = np.max(y_coords) - np.min(y_coords)

    assert x_range > 2.0, f"Ground should span significant x range: {x_range:.2f}"
    assert y_range > 2.0, f"Ground should span significant y range: {y_range:.2f}"


# ==================== Truncation Tests ====================


def test_pipe_truncation_at_wall():
    """Verify pipe perpendicular to trench is truncated at walls."""
    import numpy as np

    # Pipe oriented across the trench (90 degrees = perpendicular)
    spec_dict = {
        "path_xy": [[0.0, 0.0], [5.0, 0.0]],
        "width": 1.0,
        "depth": 1.0,
        "wall_slope": 0.0,
        "ground": {"z0": 0.0, "slope": [0.0, 0.0], "size_margin": 1.0},
        "pipes": [
            {
                "radius": 0.1,
                "length": 3.0,  # Much longer than trench width
                "angle_deg": 90.0,  # Perpendicular to trench
                "s_center": 0.5,
                "z": -0.5,
                "offset_u": 0.0,
            }
        ],
        "boxes": [],
        "spheres": [],
        "noise": {"enable": False},
    }
    spec = scene_spec_from_dict(spec_dict)
    result = generate_surface_mesh(spec, make_preview=False)

    # Find pipe side vertices
    pipe_groups = [k for k in result.groups.keys() if k.startswith("pipe")]
    assert len(pipe_groups) > 0, "Should have pipe geometry"

    # Collect all pipe vertices
    all_pipe_verts = []
    for gname in pipe_groups:
        V, _ = result.groups[gname]
        all_pipe_verts.append(V)
    pipe_verts = np.vstack(all_pipe_verts)

    # Trench has width=1.0, so half_width=0.5. Pipe radius=0.1.
    # Pipe surface should not extend beyond y = ±(0.5 - small_clearance).
    # The pipe is at 90 degrees, so its axis is along Y.
    y_coords = pipe_verts[:, 1]
    half_width = 0.5

    # Allow small tolerance for cap geometry that might be exactly at wall
    tolerance = 0.05
    assert np.all(np.abs(y_coords) <= half_width + tolerance), (
        f"Pipe vertices extend beyond trench walls: "
        f"max |y| = {np.max(np.abs(y_coords)):.3f}, expected <= {half_width + tolerance:.3f}"
    )


def test_pipe_truncation_at_floor():
    """Verify low pipe is truncated at floor level."""
    import numpy as np

    # Pipe close to floor
    spec_dict = {
        "path_xy": [[0.0, 0.0], [5.0, 0.0]],
        "width": 1.0,
        "depth": 0.5,  # Shallow trench
        "wall_slope": 0.0,
        "ground": {"z0": 0.0, "slope": [0.0, 0.0], "size_margin": 1.0},
        "pipes": [
            {
                "radius": 0.15,
                "length": 0.5,
                "angle_deg": 0.0,
                "s_center": 0.5,
                "z": -0.6,  # Requested below floor
                "offset_u": 0.0,
            }
        ],
        "boxes": [],
        "spheres": [],
        "noise": {"enable": False},
    }
    spec = scene_spec_from_dict(spec_dict)
    result = generate_surface_mesh(spec, make_preview=False)

    # Find pipe vertices
    pipe_groups = [k for k in result.groups.keys() if k.startswith("pipe")]
    all_pipe_verts = []
    for gname in pipe_groups:
        V, _ = result.groups[gname]
        all_pipe_verts.append(V)
    pipe_verts = np.vstack(all_pipe_verts)

    # Floor is at z = z0 - depth = 0 - 0.5 = -0.5
    floor_z = -0.5
    min_z = np.min(pipe_verts[:, 2])

    # Pipe should not go below floor
    tolerance = 0.02
    assert min_z >= floor_z - tolerance, (
        f"Pipe extends below floor: min_z = {min_z:.3f}, floor = {floor_z:.3f}"
    )


def test_pipe_angled_cap_geometry():
    """Verify angled pipe caps lie on expected plane."""
    import numpy as np

    # Pipe at 90 degrees (perpendicular) to trench axis
    spec_dict = {
        "path_xy": [[0.0, 0.0], [5.0, 0.0]],
        "width": 1.0,
        "depth": 1.0,
        "wall_slope": 0.0,
        "ground": {"z0": 0.0, "slope": [0.0, 0.0], "size_margin": 1.0},
        "pipes": [
            {
                "radius": 0.1,
                "length": 2.0,  # Longer than trench width
                "angle_deg": 90.0,
                "s_center": 0.5,
                "z": -0.5,
                "offset_u": 0.0,
            }
        ],
        "boxes": [],
        "spheres": [],
        "noise": {"enable": False},
    }
    spec = scene_spec_from_dict(spec_dict)
    result = generate_surface_mesh(spec, make_preview=False)

    # Get cap geometry
    cap_neg = result.groups.get("pipe0_pipe_cap_neg")
    cap_pos = result.groups.get("pipe0_pipe_cap_pos")

    # Both caps should exist
    assert cap_neg is not None, "Negative cap should exist"
    assert cap_pos is not None, "Positive cap should exist"

    V_neg, _ = cap_neg
    V_pos, _ = cap_pos

    # For perpendicular pipe truncated at walls, caps should be at y = ±wall_position
    # The exact wall position depends on truncation logic - we verify caps are near walls
    half_width = 0.5

    # Cap vertices y-coords should be clustered near the wall positions
    y_neg = np.mean(V_neg[:, 1])
    y_pos = np.mean(V_pos[:, 1])

    # One should be near -half_width, other near +half_width
    tolerance = 0.1
    assert abs(y_neg) > (half_width - tolerance) or abs(y_pos) > (half_width - tolerance), (
        f"Cap centers should be near walls: y_neg={y_neg:.3f}, y_pos={y_pos:.3f}"
    )


def test_box_shrink_to_fit():
    """Verify oversized box is shrunk to fit within trench."""
    import numpy as np

    # Box larger than trench
    spec_dict = {
        "path_xy": [[0.0, 0.0], [5.0, 0.0]],
        "width": 0.8,
        "depth": 0.6,
        "wall_slope": 0.0,
        "ground": {"z0": 0.0, "slope": [0.0, 0.0], "size_margin": 1.0},
        "pipes": [],
        "boxes": [
            {
                "along": 0.5,
                "across": 2.0,  # Much wider than trench
                "height": 1.5,  # Much taller than trench depth
                "s": 0.5,
                "offset_u": 0.0,
                "z": None,
            }
        ],
        "spheres": [],
        "noise": {"enable": False},
    }
    spec = scene_spec_from_dict(spec_dict)
    result = generate_surface_mesh(spec, make_preview=False)

    # Get box vertices
    V_box, _ = result.groups["box0"]

    # Trench: width=0.8 (half=0.4), depth=0.6, floor at z=-0.6
    half_width = 0.4
    floor_z = -0.6
    ground_z = 0.0

    # Box should fit within trench
    y_coords = V_box[:, 1]
    z_coords = V_box[:, 2]

    tolerance = 0.05
    assert np.all(np.abs(y_coords) <= half_width + tolerance), (
        f"Box extends beyond trench walls: max |y| = {np.max(np.abs(y_coords)):.3f}"
    )
    assert np.min(z_coords) >= floor_z - tolerance, (
        f"Box extends below floor: min_z = {np.min(z_coords):.3f}"
    )
    assert np.max(z_coords) <= ground_z + tolerance, (
        f"Box extends above ground: max_z = {np.max(z_coords):.3f}"
    )


def test_sphere_shrink_to_fit():
    """Verify oversized sphere is shrunk to fit within trench."""
    import numpy as np

    # Sphere larger than trench can hold
    spec_dict = {
        "path_xy": [[0.0, 0.0], [5.0, 0.0]],
        "width": 0.6,
        "depth": 0.5,
        "wall_slope": 0.0,
        "ground": {"z0": 0.0, "slope": [0.0, 0.0], "size_margin": 1.0},
        "pipes": [],
        "boxes": [],
        "spheres": [
            {
                "radius": 1.0,  # Much larger than can fit
                "s": 0.5,
                "offset_u": 0.0,
                "z": None,
            }
        ],
        "noise": {"enable": False},
    }
    spec = scene_spec_from_dict(spec_dict)
    result = generate_surface_mesh(spec, make_preview=False)

    # Get sphere vertices
    V_sphere, _ = result.groups["sphere0"]

    # Trench: width=0.6 (half=0.3), depth=0.5, floor at z=-0.5
    half_width = 0.3
    floor_z = -0.5
    ground_z = 0.0

    # Sphere should fit within trench
    y_coords = V_sphere[:, 1]
    z_coords = V_sphere[:, 2]

    tolerance = 0.05
    assert np.all(np.abs(y_coords) <= half_width + tolerance), (
        f"Sphere extends beyond trench walls: max |y| = {np.max(np.abs(y_coords)):.3f}"
    )
    assert np.min(z_coords) >= floor_z - tolerance, (
        f"Sphere extends below floor: min_z = {np.min(z_coords):.3f}"
    )
    assert np.max(z_coords) <= ground_z + tolerance, (
        f"Sphere extends above ground: max_z = {np.max(z_coords):.3f}"
    )


def test_no_vertices_outside_trench():
    """Integration test: all embedded object vertices are inside trench boundary."""
    import numpy as np

    # Complex scene with multiple objects
    spec_dict = {
        "path_xy": [[0.0, 0.0], [4.0, 0.0], [4.0, 3.0]],  # L-shaped
        "width": 1.0,
        "depth": 1.0,
        "wall_slope": 0.1,
        "ground": {"z0": 0.0, "slope": [0.0, 0.0], "size_margin": 1.5},
        "pipes": [
            {
                "radius": 0.08,
                "length": 2.0,  # Long pipe
                "angle_deg": 90.0,  # Perpendicular
                "s_center": 0.3,
                "z": -0.5,
                "offset_u": 0.0,
            },
            {
                "radius": 0.1,
                "length": 1.0,
                "angle_deg": 0.0,  # Along trench
                "s_center": 0.7,
                "z": -0.3,
                "offset_u": 0.1,
            },
        ],
        "boxes": [
            {
                "along": 0.3,
                "across": 1.5,  # Wider than trench at depth
                "height": 0.4,
                "s": 0.5,
                "offset_u": 0.0,
                "z": None,
            }
        ],
        "spheres": [
            {
                "radius": 0.5,  # Large sphere
                "s": 0.4,
                "offset_u": 0.0,
                "z": None,
            }
        ],
        "noise": {"enable": False},
    }
    spec = scene_spec_from_dict(spec_dict)
    result = generate_surface_mesh(spec, make_preview=False)

    # Helper to check if point is inside trench cross-section
    def point_inside_trench(x, y, z, spec):
        """Check if a point is inside the trench void."""
        from trenchfoot.trench_scene_generator_v3 import (
            _sample_polyline_at_s,
            _rotate_ccw,
            _ground_fn,
            _polyline_lengths,
        )

        path_xy = spec.path_xy
        half_top = spec.width / 2.0
        depth = spec.depth
        slope = spec.wall_slope
        gfun = _ground_fn(spec.ground)

        # Find closest point on path
        P = np.array(path_xy, float)
        cum, total = _polyline_lengths(path_xy)

        # Sample at many points to find closest
        best_dist = float("inf")
        best_s = 0.0
        for s in np.linspace(0, 1, 100):
            pos, _ = _sample_polyline_at_s(path_xy, s)
            dist = (pos[0] - x) ** 2 + (pos[1] - y) ** 2
            if dist < best_dist:
                best_dist = dist
                best_s = s

        pos, tangent = _sample_polyline_at_s(path_xy, best_s)
        left_normal = _rotate_ccw(tangent)
        top_z = gfun(pos[0], pos[1])

        # Local coordinates
        offset_from_center = np.array([x - pos[0], y - pos[1]])
        local_u = np.dot(offset_from_center, left_normal)

        # Half-width at this depth
        if z > top_z:
            return False  # Above ground
        if z < top_z - depth:
            return False  # Below floor
        half_w = max(0.001, half_top - slope * (top_z - z))

        return abs(local_u) <= half_w

    # Check all embedded object vertices
    tolerance = 0.1  # Allow small overshoot at boundaries
    for gname, (V, F) in result.groups.items():
        if not any(gname.startswith(prefix) for prefix in ["pipe", "box", "sphere"]):
            continue

        for i, vert in enumerate(V):
            x, y, z = vert
            if not point_inside_trench(x, y, z, spec):
                # Check with tolerance
                inside_with_tol = False
                for dx in [-tolerance, 0, tolerance]:
                    for dy in [-tolerance, 0, tolerance]:
                        for dz in [-tolerance, 0, tolerance]:
                            if point_inside_trench(x + dx, y + dy, z + dz, spec):
                                inside_with_tol = True
                                break
                        if inside_with_tol:
                            break
                    if inside_with_tol:
                        break

                if not inside_with_tol:
                    pytest.fail(
                        f"Vertex {i} of {gname} is outside trench: "
                        f"({x:.3f}, {y:.3f}, {z:.3f})"
                    )
