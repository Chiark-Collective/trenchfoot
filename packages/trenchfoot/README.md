# Trenchfoot: synthetic trench generator

This bundle generates semi-realistic, synthetic mesh datasets of trenches with pipes and objects **inside** them, cut into a consistent **ground surface**.

**Included:**
- `trench_scene_generator_v3.py` — surface generator (polyline trenches, **sloped walls**, **ground plane**, pipes/boxes/spheres, noise, **multi‑angle previews**).
- `gmsh_sloped_trench_mesher.py` — **volumetric** mesher (Gmsh) with ground-aware sloped‑wall loft and conformal pipes.
- `generate_scenarios.py` — creates 4 scenarios from simple → complex (and generates OBJ/metrics/previews).
- `scene_spec_example.json` — example with a `ground` section.
- `Dockerfile` — recommended for volumetric meshing.

---

## Quick start

### Surface (OBJ)
```bash
python trench_scene_generator_v3.py --spec scene_spec_example.json --out ./out --preview
# outputs: trench_scene.obj, metrics.json, preview_top.png, preview_side.png, preview_oblique.png
```

### Volumetric (Gmsh)
```bash
pip install gmsh meshio numpy
python gmsh_sloped_trench_mesher.py --spec scene_spec_example.json --out ./vol --lc 0.3
# outputs: vol/trench_volume.msh with Physical Volumes: "TrenchAir", "Pipe*"
```

### Docker (recommended for volumetric)
```bash
# build
docker build -t trench-mesher .
# run volumetric mesher for Scenario 3
docker run --rm -v $PWD:/work trench-mesher   python gmsh_sloped_trench_mesher.py   --spec scenarios/S03_L_slope_two_pipes_box/scene.json   --out /work/vol --lc 0.3
```

---

## Scene JSON schema (added **ground**)

```jsonc
{
  "path_xy": [[0,0],[6,0],[6,4]],
  "width": 1.2,
  "depth": 1.8,
  "wall_slope": 0.15,
  "ground_margin": 1.0,             // legacy; still used if you disable ground_surface
  "ground": {                        // NEW: consistent ground plane
    "z0": 0.0,                      // base elevation
    "slope": [0.0, 0.0],            // dz/dx, dz/dy (0,0 = flat)
    "size_margin": 4.0              // how far the rectangular ground_surface extends past trench
  },
  "pipes": [
    {"radius": 0.15, "length": 8.0, "angle_deg": 0, "s_center": 0.35, "z": -1.0, "offset_u": 0.0}
  ],
  "boxes": [{"along": 0.8, "across": 0.5, "height": 0.4, "s": 0.55, "offset_u": 0.0}],
  "spheres": [{"radius": 0.25, "s": 0.85, "offset_u": -0.2}],
  "noise": {
    "enable": true, "amplitude": 0.01, "corr_length": 0.5,
    "octaves": 2, "gain": 0.5, "seed": 7,
    "apply_to": ["trench_walls","trench_bottom"]
  }
}
```

### Behavior
- The trench **top ring** lies on the ground plane (`z = z0 + sx*x + sy*y`), bottom ring at `ground(x,y) - depth`.
- Objects (pipes/boxes/spheres) are **clamped** so they fit entirely inside the sloped cross‑section at their depth (with small clearance).
- Previews: `preview_top.png`, `preview_side.png`, `preview_oblique.png`.

**Tip:** To keep the legacy narrow ground strips (instead of a single rectangular `ground_surface`), set `"ground": {"size_margin": 0}`.

---

## Python SDK

Import the package to work in memory without touching disk:

```python
import json

from trenchfoot import (
    SceneSpec,
    scene_spec_from_dict,
    generate_surface_mesh,
    generate_trench_volume,
    gmsh_available,
)

with open("scene_spec_example.json", "r", encoding="utf-8") as fh:
    spec_dict = json.load(fh)

scene = scene_spec_from_dict(spec_dict)
surface = generate_surface_mesh(scene)
surface.persist("./out/surface")

if gmsh_available():
    volume = generate_trench_volume(spec_dict, lc=0.3, persist_path="./out/vol/trench_volume.msh")
    print("Physical groups:", [pg.name for pg in volume.physical_groups])
```

`SurfaceMeshResult.groups` contains the per-object vertex/face arrays (useful for direct NumPy workflows), while `VolumeMeshResult` exposes raw nodes/elements plus the adaptive clearance diagnostics consumed by the CLI.

## Scenarios

Run:
```bash
python -m trenchfoot.generate_scenarios
# Creates ./scenarios/S01..S04 with OBJ/metrics + previews
```
You can directly use those `scene.json` files with the volumetric mesher.

### Tips
- Set `TRENCHFOOT_SCENARIO_OUT_ROOT=/path/to/scratch` before running the CLI to keep regenerated assets out of the repository tree; override with `--out` when needed.
- Each volumetric run records per-pipe clearance diagnostics in `SUMMARY.json["scenarios"][i]["pipe_clearances"]`. The mesher adapts the guard band based on the pipe radius and wall slope (minimum 50 mm) and warns or aborts before Gmsh hits a PLC failure.
- Add an optional `"clearance_scale"` to individual pipe specs to tighten/loosen the adaptive guard band (values > 1.0 widen the buffer; < 1.0 allow closer geometry when safe).
- Previews require `matplotlib`. If missing, install it or run `uv pip install matplotlib`.
- Run `uv run pytest -rs` to exercise the surface generator, the volumetric mesher (including all default presets), and the CLI entry point.
- The scenario CLI accepts `--gallery path.md` to emit a Markdown table of previews, `--scratch` to spill results into a temporary directory (handy for CI), and `--include-prebuilt` to copy the shipped assets into the output directory before regeneration.
- Preview renders now use color-coded materials (trench surfaces in copper/beige, pipes/boxes/spheres in distinct accent palettes) for quicker visual inspection.
- Install `plotly` via the `[viz]` extra and use `trenchfoot-plot path/to/trench_scene.obj --open` to generate interactive HTML mesh previews.

---

© 2025-10-03 Synthetic Trench Bundle
