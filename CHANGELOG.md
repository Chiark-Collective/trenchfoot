# Changelog

## [0.4.3] - 2026-01-30

### Fixed
- Regenerated bundled scenario meshes with truncation fix (0.4.2 shipped stale OBJ files)

## [0.4.2] - 2026-01-29

### Added
- **Embedded object truncation**: Pipes, boxes, and spheres that extend beyond trench boundaries are now automatically truncated or shrunk to fit within the trench void:
  - **Pipes**: Truncated at walls/floor with angled elliptical caps matching the intersection plane
  - **Boxes**: Dimensions shrunk to fit within available space
  - **Spheres**: Radius reduced to fit within available space
- **New helper functions** for trench boundary computation:
  - `_find_trench_frame_at_xy()`: Find local trench coordinate frame at any XY position
  - `_compute_pipe_truncation()`: Binary search for where pipe exits trench
  - `_compute_cap_plane_at_truncation()`: Determine wall/floor plane at intersection
  - `_compute_box_fit()`: Calculate shrunk box dimensions
  - `_compute_sphere_fit()`: Calculate shrunk sphere radius
- **Extended `make_cylinder()`**: Now supports asymmetric extents (`neg_extent`, `pos_extent`) and angled cap planes (`cap_plane_neg`, `cap_plane_pos`)
- **New tests** for truncation functionality:
  - `test_pipe_truncation_at_wall`
  - `test_pipe_truncation_at_floor`
  - `test_pipe_angled_cap_geometry`
  - `test_box_shrink_to_fit`
  - `test_sphere_shrink_to_fit`
  - `test_no_vertices_outside_trench`

### Fixed
- Long pipes in S07 circular well no longer protrude through trench walls
- Oversized embedded objects no longer extend outside trench geometry

## [0.4.1] - 2026-01-28

### Fixed
- **Non-convex polygon extraction**: `_build_sdf_metadata()` now correctly extracts boundary polygons for non-convex shapes (L-shaped, U-shaped trenches) using boundary edge chaining instead of ConvexHull. ConvexHull was incorrectly filling in concave regions, causing sign computation errors at trench edges.
- **sdf_metadata version bumped to 2.0**: Updated version to reflect the corrected polygon format.

### Added
- **`_extract_boundary_polygon()` helper**: New geometry helper that extracts ordered boundary vertices from triangulated meshes by finding boundary edges (edges appearing in only one face) and chaining them together.
- **L-shaped polygon test**: Added `test_metadata_l_shaped_polygon_is_non_convex` to verify correct handling of non-convex trench shapes.

## [0.4.0] - 2026-01-26

### Added
- **Makefile**: Added `make dump-meshes`, `make dump-meshes-volumetric`, and `make clean-meshes` targets for local mesh inspection
- **Inner column lid for circular wells**: S07 and other closed-path trenches now have a proper ground surface capping the central column
- **`fill_interior` option for GroundSpec**: Closed-path trenches can optionally fill the interior island with ground surface (used in S06)
- **`inner_column_lid` render color**: Gray color matching ground surface

### Fixed
- **S03 box-pipe intersection**: Reduced box size to create clearance from nearby pipe
- **S04 diagonal pipe clipping**: Adjusted pipe offset to prevent intersection with inner corner
- **S05 pipe-box intersection**: Moved box position and offset to avoid pipe collision
- **S06 ground overhang at open ends**: Closed the loop path to use proper annulus triangulation
- **S07 pipe positions**: Repositioned all pipes with tangential orientation and negative offset to avoid crossing the central opening
- **Circular path seam artifact**: `_offset_closed_polyline` now removes duplicate closing point that caused triangular protrusion on inner walls
- **`.gitignore`**: Added `data/` directory for generated mesh inspection

### Changed
- **S07 pipe configuration**: All pipes now tangential (angle_deg=0) with offset_u=-0.4, lengths increased to 7.0-8.0 for wall penetration

## [0.3.1] - 2026-01-08

### Added
- **Pre-generated `sdf_metadata.json` files**: All 7 scenarios now include static `sdf_metadata.json` files with version 2.0, enabling the HybridTrenchSDF pipeline in survi without regeneration.
- **Ground truth isosurface visualizations**: Each scenario includes `isosurface_truth.html` showing the mesh isosurface for visual verification.

## [0.3.0] - 2026-01-07

### Added
- **SDF metadata export**: Each scenario now exports `sdf_metadata.json` alongside the OBJ mesh, containing:
  - Normal convention ("into_void")
  - Geometry type ("open_trench")
  - Trench opening polygon with z-level
  - Surface group annotations (floor, walls, ground)
  - Embedded object counts
- **SDF readiness tests**: 21 new tests in `tests/test_sdf_readiness.py` validating:
  - Floor normals point up (+z)
  - Wall normals point inward toward trench centerline
  - Ground surface normals point up (+z)
  - Metadata consistency with mesh geometry

### Fixed
- **Normal orientation consistency**: All floor and ground surfaces now have upward-pointing normals (+z). Previously, some faces (particularly on S07 circular well ground surface) had incorrect winding order causing downward normals, which broke SDF sign computation in downstream consumers.
- **`_ensure_upward_normals()` post-processing**: Added helper function that flips face winding order for any horizontal faces with downward normals, applied to all ground surface generation paths.

## [0.2.7] - 2025-12-24

### Fixed
- **Ground end fill**: Fixed gaps at trench ends by only extending the outer ring (ground boundary) while keeping the inner ring (trench opening) at its original position. The arc-length triangulation now correctly fills the end areas with ground geometry.

## [0.2.6] - 2025-12-24

### Fixed
- **Ground end extensions**: Replaced semicircular end caps with flat rectangular extensions. Both inner and outer rings are now extended identically, ensuring proper triangulation for L-shaped, U-shaped, and other complex trench paths.

## [0.2.5] - 2025-12-24

### Fixed
- **Ground end caps geometry**: Fixed semicircular end caps that were incorrectly offset along the tangent, causing them to overlap the trench opening. Caps are now properly centered at path endpoints.

## [0.2.4] - 2025-12-24

### Added
- **Ground end caps**: For open-path trenches, the ground surface now extends past the trench endpoints with semicircular caps, providing a buffer of ground at each end instead of terminating abruptly.

## [0.2.3] - 2025-12-20

### Fixed
- **Closed path offset direction bug**: `_offset_closed_polyline` was using CCW rotation for normals, causing positive offsets to go inward instead of outward. Fixed to use CW rotation for correct outward-pointing normals on CCW polygons.
- **Circular well ground surface**: Removed incorrect center island from closed path ground surfaces. For circular wells, only the outer ground ring exists, leaving the trench opening completely open.

### Added
- **Open-topped trench tests**: New tests to verify trenches are truly open (no geometry covering the trench opening):
  - `test_trench_opening_is_open_for_straight_trench`
  - `test_circular_well_trench_opening_is_open`
  - `test_ground_surface_annular_structure`

## [0.2.2] - 2025-12-20

### Fixed
- **Closed path handling for S07 circular well**: Explicitly close the circular path by repeating the first point. The previous heuristic was incorrectly detecting L-shaped and U-shaped paths as closed.
- **Volumetric meshing for closed paths**: Added proper annular geometry support to the gmsh mesher for closed circular paths with outer and inner walls.

## [0.2.1] - 2025-12-20

### Fixed
- **Annular triangulation bug**: Ground plane was incorrectly filling in the trench opening with triangles instead of leaving it as a hole. Replaced bridge+ear-clipping algorithm with proper annular triangulation that "zips" around both polygon boundaries.

## [0.2.0] - 2025-12-19

### Added
- **S07 circular well scenario**: Deep cylindrical well with 4 criss-crossing pipes at different elevations and diameters
- **Offset polygon ground plane**: Ground surface now follows trench outline instead of axis-aligned bounding box, reducing wasted space for L-shaped, U-shaped, and curved trenches
- **Open-topped trenches**: Trench cap (`trench_cap_for_volume`) is kept internally for metrics calculations but excluded from OBJ export and preview renders
- New tests for cap exclusion, offset ground, and circular well scenario

### Changed
- **Shallower trench depths**: All scenarios adjusted to be less extreme (S01: 0.6m, S02: 0.9m, S03: 1.1m, S04: 1.2m, S05: 0.7m, S06: 0.85m)
- Reduced ground `size_margin` values to work better with offset polygon ground
- Pipe z-positions adjusted proportionally to fit within shallower trenches

### Fixed
- Ground plane no longer wastes space on flat areas away from the trench path

## [0.1.1] - 2025-10-30

- CI hardening for PyPI token handling
- Initial PyPI release

## [0.1.0] - 2025-10-30

- Initial release with surface and volumetric mesh generation
- Scenarios S01-S06 with increasing complexity
- Plotly HTML viewer support
- Python SDK with `generate_surface_mesh()` and `generate_trench_volume()`
