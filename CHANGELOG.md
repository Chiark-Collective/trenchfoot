# Changelog

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
