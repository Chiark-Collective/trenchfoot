# Trenchfoot

Surface and volumetric trench mesh generator with shipped presets, Plotly previews, and a lightweight Python SDK.

## Install

```bash
pip install trenchfoot
```

Want volumetrics or visualisations? Install extras as needed:
- `pip install "trenchfoot[mesher]"` for gmsh-powered volume meshes.
- `pip install "trenchfoot[preview]"` for matplotlib snapshot renders.
- `pip install "trenchfoot[viz]"` for Plotly HTML viewers.

## Scenario Gallery

Color key: trench surfaces use warm soil tones; embedded geometry is colour-coded per group.

| Scenario | Top | Side | Oblique |
| --- | --- | --- | --- |
| S01_straight_vwalls | ![S01 top](packages/trenchfoot/scenarios/S01_straight_vwalls/preview_top.png) | ![S01 side](packages/trenchfoot/scenarios/S01_straight_vwalls/preview_side.png) | ![S01 oblique](packages/trenchfoot/scenarios/S01_straight_vwalls/preview_oblique.png) |
| S02_straight_slope_pipe | ![S02 top](packages/trenchfoot/scenarios/S02_straight_slope_pipe/preview_top.png) | ![S02 side](packages/trenchfoot/scenarios/S02_straight_slope_pipe/preview_side.png) | ![S02 oblique](packages/trenchfoot/scenarios/S02_straight_slope_pipe/preview_oblique.png) |
| S03_L_slope_two_pipes_box | ![S03 top](packages/trenchfoot/scenarios/S03_L_slope_two_pipes_box/preview_top.png) | ![S03 side](packages/trenchfoot/scenarios/S03_L_slope_two_pipes_box/preview_side.png) | ![S03 oblique](packages/trenchfoot/scenarios/S03_L_slope_two_pipes_box/preview_oblique.png) |
| S04_U_slope_multi_noise | ![S04 top](packages/trenchfoot/scenarios/S04_U_slope_multi_noise/preview_top.png) | ![S04 side](packages/trenchfoot/scenarios/S04_U_slope_multi_noise/preview_side.png) | ![S04 oblique](packages/trenchfoot/scenarios/S04_U_slope_multi_noise/preview_oblique.png) |
| S05_wide_slope_pair | ![S05 top](packages/trenchfoot/scenarios/S05_wide_slope_pair/preview_top.png) | ![S05 side](packages/trenchfoot/scenarios/S05_wide_slope_pair/preview_side.png) | ![S05 oblique](packages/trenchfoot/scenarios/S05_wide_slope_pair/preview_oblique.png) |
| S06_bumpy_wide_loop | ![S06 top](packages/trenchfoot/scenarios/S06_bumpy_wide_loop/preview_top.png) | ![S06 side](packages/trenchfoot/scenarios/S06_bumpy_wide_loop/preview_side.png) | ![S06 oblique](packages/trenchfoot/scenarios/S06_bumpy_wide_loop/preview_oblique.png) |
| S07_circular_well | ![S07 top](packages/trenchfoot/scenarios/S07_circular_well/preview_top.png) | ![S07 side](packages/trenchfoot/scenarios/S07_circular_well/preview_side.png) | ![S07 oblique](packages/trenchfoot/scenarios/S07_circular_well/preview_oblique.png) |

### S07 circular well preset

A deep cylindrical well with criss-crossing pipes at different elevations:

```json
{
  "path_xy": "<<32-vertex circle approximation, radius=1.5>>",
  "width": 2.0,
  "depth": 2.5,
  "wall_slope": 0.05,
  "ground": {"z0": 0.0, "slope": [0.0, 0.0], "size_margin": 2.0},
  "pipes": [
    {"radius": 0.20, "length": 4.0, "angle_deg": 0, "s_center": 0.25, "z": -0.5},
    {"radius": 0.15, "length": 3.5, "angle_deg": 45, "s_center": 0.5, "z": -1.2},
    {"radius": 0.10, "length": 3.0, "angle_deg": -60, "s_center": 0.75, "z": -1.8},
    {"radius": 0.12, "length": 3.2, "angle_deg": 90, "s_center": 0.0, "z": -2.2}
  ],
  "spheres": [{"radius": 0.25, "s": 0.4, "z": -1.5}],
  "noise": {"enable": true, "amplitude": 0.02, "corr_length": 0.4, "octaves": 2, "gain": 0.5}
}
```

## CLI quick start

```bash
trenchfoot-generate --help
trenchfoot-generate --preview --skip-volumetric --gallery docs/scenario_gallery.md
trenchfoot-plot packages/trenchfoot/scenarios/S05_wide_slope_pair/trench_scene.obj --open
```

Set `TRENCHFOOT_SCENARIO_OUT_ROOT=/tmp/trench-previews` (or another writable path) to keep generated assets out of your checkout.

## Python API

```python
from trenchfoot import scene_spec_from_dict, generate_surface_mesh, generate_trench_volume, gmsh_available

spec_dict = {
    "path_xy": [[0.0, 0.0], [5.0, 0.0]],
    "width": 1.0,
    "depth": 1.2,
    "pipes": [{"radius": 0.1, "length": 1.8, "angle_deg": 0.0, "s_center": 0.5}],
    "boxes": [],
    "spheres": [],
    "noise": {"enable": False},
}

scene = scene_spec_from_dict(spec_dict)
surface = generate_surface_mesh(scene, make_preview=True)
surface.persist("./surface")

if gmsh_available():
    volume = generate_trench_volume(spec_dict, lc=0.4, persist_path="./volume/trench_volume.msh")
```

`SurfaceMeshResult` keeps per-group faces, metrics, and optional preview PNG bytes; call `.persist(...)` when you need files. `VolumeMeshResult` exposes node coordinates, elements, and physical groups while still letting you stay in memory.

## Testing

```bash
pytest -rs
```

The suite exercises each preset (surface + volumetric), the gallery helpers, and the SDK smoke paths.
