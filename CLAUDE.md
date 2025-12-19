# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Trenchfoot is a synthetic trench mesh generator that produces surface meshes (OBJ) and volumetric meshes (via Gmsh). It generates semi-realistic trench scenes with embedded pipes, boxes, and spheres, useful for creating synthetic datasets.

## Commands

```bash
# Install with uv (development)
uv pip install -e ".[dev,preview,mesher,viz]"

# Run tests
uv run pytest -rs

# Run a single test
uv run pytest tests/test_trenchfoot_generation.py::test_build_scene_produces_surface -v

# Generate scenario previews (writes to TRENCHFOOT_SCENARIO_OUT_ROOT or packages/trenchfoot/scenarios/)
uv run python -m trenchfoot.generate_scenarios --preview --skip-volumetric

# Generate with volumetric meshes (requires gmsh)
uv run python -m trenchfoot.generate_scenarios --volumetric --lc 0.4

# Generate gallery markdown
uv run python -m trenchfoot.generate_scenarios --preview --skip-volumetric --gallery docs/scenario_gallery.md

# Interactive Plotly mesh viewer (requires viz extra)
trenchfoot-plot path/to/trench_scene.obj --open
```

Set `TRENCHFOOT_SCENARIO_OUT_ROOT=/tmp/trench-previews` to keep generated assets out of the repository.

## Architecture

The package lives in `packages/trenchfoot/` with these core modules:

- **`trench_scene_generator_v3.py`**: Surface mesh generator. Produces OBJ files with polyline trenches, sloped walls, ground planes, and embedded geometry (pipes/boxes/spheres). Handles noise application and preview rendering. Key types: `SceneSpec`, `SurfaceMeshResult`.

- **`gmsh_sloped_trench_mesher.py`**: Volumetric mesher using Gmsh. Creates tetrahedral meshes with physical groups (`TrenchAir`, `Pipe0`, etc.). Has adaptive clearance logic that scales guard bands based on pipe radius and wall slope. Key types: `VolumeMeshResult`.

- **`generate_scenarios.py`**: CLI and scenario runner. Defines preset scenarios (S01-S06) and orchestrates both surface and volumetric generation. Handles gallery markdown output and SUMMARY.json diagnostics.

- **`__init__.py`**: Public API exports. Gracefully handles missing gmsh dependency.

## Scene Specification

Trenches are defined by JSON specs with:
- `path_xy`: 2D polyline defining trench centerline
- `width`, `depth`, `wall_slope`: Cross-section parameters
- `ground`: Elevation, slope, and margin for ground plane
- `pipes`, `boxes`, `spheres`: Embedded objects positioned along the trench
- `noise`: Perlin-style surface perturbation settings

Objects use `s` or `s_center` (0-1 arc-length parameter) to position along the trench path.

## Optional Dependencies

- `[preview]` - matplotlib for PNG previews
- `[mesher]` - gmsh for volumetric mesh generation
- `[viz]` - plotly for interactive HTML viewers
- `[dev]` - pytest

## Testing Notes

Tests skip volumetric tests if gmsh runtime prerequisites (e.g., libGLU) are unavailable. The `_require_gmsh_runtime()` helper handles this gracefully.
