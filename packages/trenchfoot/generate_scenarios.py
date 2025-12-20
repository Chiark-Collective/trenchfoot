#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_scenarios.py

Produce the bundled trench scenarios from simple → complex using the surface generator,
and optionally run the volumetric mesher (via Gmsh) when available.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[1]


def _ensure_repo_on_path() -> None:
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

try:
    from .trench_scene_generator_v3 import (
        SceneSpec,
        build_scene,
        scene_spec_from_dict,
    )
except ImportError:  # pragma: no cover - direct script invocation
    _ensure_repo_on_path()
    from trenchfoot.trench_scene_generator_v3 import (  # type: ignore
        SceneSpec,
        build_scene,
        scene_spec_from_dict,
    )

_gmsh_mesher = None
_gmsh_import_error: Optional[Exception] = None


def _load_gmsh_mesher() -> None:
    global _gmsh_mesher, _gmsh_import_error
    if _gmsh_mesher is not None or _gmsh_import_error is not None:
        return
    try:
        from . import gmsh_sloped_trench_mesher as _mesher  # type: ignore
    except Exception:
        _ensure_repo_on_path()
        try:
            import trenchfoot.gmsh_sloped_trench_mesher as _mesher  # type: ignore
        except Exception as exc:
            _gmsh_import_error = exc
            return
    _gmsh_mesher = _mesher


@dataclass(frozen=True)
class ScenarioDefinition:
    """Name + spec dictionary for generating a scenario."""

    name: str
    spec: Dict[str, Any]


@dataclass
class ScenarioSummary:
    """Outputs captured for a single generated scenario."""

    name: str
    directory: Path
    spec_path: Path
    surface_obj: Path
    metrics_path: Path
    preview_paths: List[Path]
    preview_count: int
    object_counts: Dict[str, int]
    footprint_top: float
    footprint_bottom: float
    trench_from_surface: float
    volumetric_path: Optional[Path]
    volumetric_lc: Optional[float]
    volumetric_error: Optional[str]
    pipe_clearances: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "directory": str(self.directory),
            "spec_path": str(self.spec_path),
            "surface_obj": str(self.surface_obj),
            "metrics_path": str(self.metrics_path),
            "preview_paths": [str(p) for p in self.preview_paths],
            "preview_count": self.preview_count,
            "object_counts": dict(self.object_counts),
            "footprint_top": self.footprint_top,
            "footprint_bottom": self.footprint_bottom,
            "trench_from_surface": self.trench_from_surface,
            "volumetric_path": str(self.volumetric_path) if self.volumetric_path else None,
            "volumetric_lc": self.volumetric_lc,
            "volumetric_error": self.volumetric_error,
            "pipe_clearances": self.pipe_clearances,
        }


@dataclass
class RunReport:
    """Aggregate report describing a generate_scenarios run."""

    out_root: Path
    preview_enabled: bool
    volumetric_requested: bool
    volumetric_available: bool
    mesh_characteristic_length: float
    scenarios: List[ScenarioSummary]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "out_root": str(self.out_root),
            "preview_enabled": self.preview_enabled,
            "volumetric_requested": self.volumetric_requested,
            "volumetric_available": self.volumetric_available,
            "mesh_characteristic_length": self.mesh_characteristic_length,
            "scenarios": [sc.to_dict() for sc in self.scenarios],
        }


def gmsh_available() -> bool:
    """Return True if the gmsh-based mesher can be imported."""
    _load_gmsh_mesher()
    return _gmsh_mesher is not None


def _generate_circular_path(
    center: tuple[float, float], radius: float, n_vertices: int = 32
) -> List[List[float]]:
    """Generate vertices for a closed circular polyline.

    The path is explicitly closed by repeating the first point at the end.
    This ensures proper handling as a closed loop in mesh generation.
    """
    import math
    cx, cy = center
    angles = [2 * math.pi * i / n_vertices for i in range(n_vertices)]
    points = [[cx + radius * math.cos(a), cy + radius * math.sin(a)] for a in angles]
    # Close the path by repeating the first point
    points.append(points[0].copy())
    return points


def default_scenarios() -> List[ScenarioDefinition]:
    """Built-in scenario presets with shallower depths and offset polygon ground."""
    return [
        ScenarioDefinition(
            "S01_straight_vwalls",
            {
                "path_xy": [[0, 0], [5, 0]],
                "width": 1.0,
                "depth": 0.6,
                "wall_slope": 0.0,
                "ground_margin": 0.5,
                "ground": {"z0": 0.0, "slope": [0.0, 0.0], "size_margin": 1.0},
                "pipes": [],
                "boxes": [],
                "spheres": [],
                "noise": {"enable": False},
            },
        ),
        ScenarioDefinition(
            "S02_straight_slope_pipe",
            {
                "path_xy": [[0, 0], [6, 0]],
                "width": 1.2,
                "depth": 0.9,
                "wall_slope": 0.2,
                "ground_margin": 0.5,
                "ground": {"z0": 0.0, "slope": [0.0, 0.0], "size_margin": 1.0},
                "pipes": [
                    {
                        "radius": 0.15,
                        "length": 7.0,
                        "angle_deg": 0,
                        "s_center": 0.5,
                        "z": -0.45,
                        "offset_u": 0.0,
                    }
                ],
                "boxes": [],
                "spheres": [],
                "noise": {"enable": False},
            },
        ),
        ScenarioDefinition(
            "S03_L_slope_two_pipes_box",
            {
                "path_xy": [[0, 0], [6, 0], [6, 4]],
                "width": 1.2,
                "depth": 1.1,
                "wall_slope": 0.15,
                "ground_margin": 1.0,
                "ground": {"z0": 0.0, "slope": [0.0, 0.0], "size_margin": 1.2},
                "pipes": [
                    {
                        "radius": 0.15,
                        "length": 8.0,
                        "angle_deg": 0,
                        "s_center": 0.35,
                        "z": -0.6,
                        "offset_u": 0.0,
                    },
                    {
                        "radius": 0.10,
                        "length": 5.0,
                        "angle_deg": 90,
                        "s_center": 0.75,
                        "z": -0.55,
                        "offset_u": 0.2,
                    },
                ],
                "boxes": [
                    {
                        "along": 0.8,
                        "across": 0.5,
                        "height": 0.4,
                        "s": 0.55,
                        "offset_u": 0.0,
                    }
                ],
                "spheres": [],
                "noise": {
                    "enable": True,
                    "amplitude": 0.01,
                    "corr_length": 0.5,
                    "octaves": 2,
                    "gain": 0.5,
                    "seed": 7,
                    "apply_to": ["trench_walls", "trench_bottom"],
                },
            },
        ),
        ScenarioDefinition(
            "S04_U_slope_multi_noise",
            {
                "path_xy": [[0, 0], [6, 0], [6, 4], [0, 4]],
                "width": 1.4,
                "depth": 1.2,
                "wall_slope": 0.25,
                "ground_margin": 1.2,
                "ground": {"z0": 0.0, "slope": [0.0, 0.0], "size_margin": 1.5},
                "pipes": [
                    {
                        "radius": 0.18,
                        "length": 9.0,
                        "angle_deg": 0,
                        "s_center": 0.25,
                        "z": -0.6,
                        "offset_u": 0.0,
                    },
                    {
                        "radius": 0.10,
                        "length": 5.0,
                        "angle_deg": 45,
                        "s_center": 0.55,
                        "z": -0.65,
                        "offset_u": -0.2,
                    },
                    {
                        "radius": 0.08,
                        "length": 3.5,
                        "angle_deg": -60,
                        "s_center": 0.75,
                        "z": -0.8,
                        "offset_u": 0.25,
                    },
                ],
                "boxes": [],
                "spheres": [{"radius": 0.25, "s": 0.85, "offset_u": -0.2, "z": -0.7}],
                "noise": {
                    "enable": True,
                    "amplitude": 0.02,
                    "corr_length": 0.6,
                    "octaves": 2,
                    "gain": 0.6,
                    "seed": 13,
                    "apply_to": ["trench_walls", "trench_bottom", "pipe*_pipe_side"],
                },
            },
        ),
        ScenarioDefinition(
            "S05_wide_slope_pair",
            {
                "path_xy": [[0, 0], [9, 0], [9, 3]],
                "width": 2.4,
                "depth": 0.7,
                "wall_slope": 0.08,
                "ground_margin": 1.5,
                "ground": {"z0": 0.0, "slope": [0.02, -0.015], "size_margin": 1.5},
                "pipes": [
                    {
                        "radius": 0.2,
                        "length": 5.5,
                        "angle_deg": 10,
                        "s_center": 0.35,
                        "z": -0.4,
                        "offset_u": 0.3,
                        "clearance_scale": 0.9,
                    },
                    {
                        "radius": 0.16,
                        "length": 4.2,
                        "angle_deg": -15,
                        "s_center": 0.7,
                        "z": -0.45,
                        "offset_u": -0.4,
                        "clearance_scale": 1.1,
                    },
                ],
                "boxes": [
                    {
                        "along": 1.2,
                        "across": 0.9,
                        "height": 0.35,
                        "s": 0.55,
                        "offset_u": -0.25,
                        "z": -0.35,
                    }
                ],
                "spheres": [],
                "noise": {
                    "enable": True,
                    "amplitude": 0.015,
                    "corr_length": 0.8,
                    "octaves": 3,
                    "gain": 0.45,
                    "seed": 17,
                    "apply_to": ["trench_walls", "trench_bottom"],
                },
            },
        ),
        ScenarioDefinition(
            "S06_bumpy_wide_loop",
            {
                "path_xy": [[0, 0], [4, -1], [8, 0], [8, 5], [2, 5], [-1, 2]],
                "width": 2.6,
                "depth": 0.85,
                "wall_slope": 0.12,
                "ground_margin": 2.0,
                "ground": {"z0": 0.2, "slope": [0.015, 0.03], "size_margin": 1.8},
                "pipes": [
                    {
                        "radius": 0.18,
                        "length": 6.0,
                        "angle_deg": 35,
                        "s_center": 0.3,
                        "z": -0.55,
                        "offset_u": 0.35,
                        "clearance_scale": 1.0,
                    },
                    {
                        "radius": 0.14,
                        "length": 4.8,
                        "angle_deg": -40,
                        "s_center": 0.6,
                        "z": -0.6,
                        "offset_u": -0.45,
                        "clearance_scale": 0.85,
                    },
                ],
                "boxes": [],
                "spheres": [
                    {"radius": 0.3, "s": 0.82, "offset_u": 0.3, "z": -0.4}
                ],
                "noise": {
                    "enable": True,
                    "amplitude": 0.035,
                    "corr_length": 0.5,
                    "octaves": 4,
                    "gain": 0.55,
                    "seed": 29,
                    "apply_to": ["trench_walls", "trench_bottom", "pipe*_pipe_side"],
                },
            },
        ),
        ScenarioDefinition(
            "S07_circular_well",
            {
                "path_xy": _generate_circular_path((0.0, 0.0), radius=1.5, n_vertices=32),
                "width": 2.0,
                "depth": 2.5,
                "wall_slope": 0.05,
                "ground_margin": 1.0,
                "ground": {"z0": 0.0, "slope": [0.0, 0.0], "size_margin": 2.0},
                "pipes": [
                    # Upper pipe - large diameter, horizontal
                    {
                        "radius": 0.20,
                        "length": 4.0,
                        "angle_deg": 0,
                        "s_center": 0.25,
                        "z": -0.5,
                        "offset_u": 0.0,
                    },
                    # Middle pipe - medium, angled
                    {
                        "radius": 0.15,
                        "length": 3.5,
                        "angle_deg": 45,
                        "s_center": 0.5,
                        "z": -1.2,
                        "offset_u": 0.1,
                    },
                    # Lower pipe - small, opposite angle
                    {
                        "radius": 0.10,
                        "length": 3.0,
                        "angle_deg": -60,
                        "s_center": 0.75,
                        "z": -1.8,
                        "offset_u": -0.15,
                    },
                    # Deep pipe - crossing at bottom
                    {
                        "radius": 0.12,
                        "length": 3.2,
                        "angle_deg": 90,
                        "s_center": 0.0,
                        "z": -2.2,
                        "offset_u": 0.0,
                    },
                ],
                "boxes": [],
                "spheres": [{"radius": 0.25, "s": 0.4, "offset_u": 0.0, "z": -1.5}],
                "noise": {
                    "enable": True,
                    "amplitude": 0.02,
                    "corr_length": 0.4,
                    "octaves": 2,
                    "gain": 0.5,
                    "seed": 37,
                    "apply_to": ["trench_walls", "trench_bottom"],
                },
            },
        ),
    ]


def _ensure_spec_written(path: Path, spec: Dict[str, Any]) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(spec, fh, indent=2)


def _build_surface(spec: SceneSpec, out_dir: Path, make_preview: bool) -> Dict[str, Any]:
    return build_scene(spec, str(out_dir), make_preview=make_preview)


def _build_volume(spec: Dict[str, Any], out_dir: Path, lc: float) -> tuple[Optional[Path], Optional[str], List[Dict[str, Any]]]:
    _load_gmsh_mesher()
    if _gmsh_mesher is None:
        reason = str(_gmsh_import_error) if _gmsh_import_error else "gmsh_not_available"
        return None, reason, []
    out_dir.mkdir(parents=True, exist_ok=True)
    msh_path = out_dir / "trench_volume.msh"
    clearance_data: List[Dict[str, Any]] = []

    try:
        if hasattr(_gmsh_mesher, "generate_trench_volume"):
            result = _gmsh_mesher.generate_trench_volume(
                spec,
                lc=lc,
                persist_path=str(msh_path),
            )
            clearance_data.extend(
                [dict(entry) for entry in result.pipe_clearances]
                if isinstance(result.pipe_clearances, list)
                else []
            )
            persisted = result.persisted_path if result.persisted_path is not None else Path(str(msh_path))
        else:
            _gmsh_mesher.build_trench_volume_from_spec(
                spec, lc=lc, out_msh=str(msh_path)
            )
            persisted = Path(str(msh_path))
    except Exception as exc:  # pragma: no cover - gmsh failure
        return None, str(exc), clearance_data

    final_path = Path(persisted)
    return final_path, None, clearance_data


def generate_scenarios(
    out_root: Path | str,
    scenarios: Optional[Sequence[ScenarioDefinition]] = None,
    *,
    make_preview: bool = True,
    make_volumes: bool = True,
    mesh_characteristic_length: float = 0.3,
    write_summary_json: bool = True,
) -> RunReport:
    """
    Generate trench scenarios, producing surface meshes (+previews) and optional volumetric meshes.
    Returns a RunReport describing the run.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    scenario_defs = list(scenarios) if scenarios is not None else default_scenarios()
    results: List[ScenarioSummary] = []
    gmsh_ok = gmsh_available() if make_volumes else False

    for definition in scenario_defs:
        scen_dir = out_root / definition.name
        scen_dir.mkdir(parents=True, exist_ok=True)

        spec_path = scen_dir / "scene.json"
        _ensure_spec_written(spec_path, definition.spec)

        scene_spec = scene_spec_from_dict(definition.spec)
        surface_out = _build_surface(scene_spec, scen_dir, make_preview=make_preview)

        volumetric_path: Optional[Path] = None
        volumetric_error: Optional[str] = None
        pipe_clearances: List[Dict[str, Any]] = []
        if gmsh_ok:
            vol_dir = scen_dir / "volumetric"
            volumetric_path, volumetric_error, pipe_clearances = _build_volume(
                definition.spec, vol_dir, mesh_characteristic_length
            )
            if volumetric_error:
                print(
                    f"[volumetric] {definition.name} failed: {volumetric_error}",
                    file=sys.stderr,
                )

        summary = ScenarioSummary(
            name=definition.name,
            directory=scen_dir,
            spec_path=spec_path,
            surface_obj=Path(surface_out["obj_path"]),
            metrics_path=scen_dir / "metrics.json",
            preview_paths=[Path(p) for p in surface_out["previews"]],
            preview_count=len(surface_out["previews"]),
            object_counts=dict(surface_out["object_counts"]),
            footprint_top=float(surface_out["metrics"]["footprint_area_top"]),
            footprint_bottom=float(surface_out["metrics"]["footprint_area_bottom"]),
            trench_from_surface=float(surface_out["metrics"]["volumes"]["trench_from_surface"]),
            volumetric_path=volumetric_path,
            volumetric_lc=mesh_characteristic_length if volumetric_path else None,
            volumetric_error=volumetric_error,
            pipe_clearances=pipe_clearances,
        )
        results.append(summary)

    report = RunReport(
        out_root=out_root,
        preview_enabled=make_preview,
        volumetric_requested=make_volumes,
        volumetric_available=gmsh_ok,
        mesh_characteristic_length=mesh_characteristic_length,
        scenarios=results,
    )

    if write_summary_json:
        summary_path = out_root / "SUMMARY.json"
        with summary_path.open("w") as fh:
            json.dump(report.to_dict(), fh, indent=2)

    return report


def _relative_path(path: Path, base: Path) -> str:
    path_abs = path.resolve()
    base_abs = base.resolve()
    rel = os.path.relpath(path_abs, start=base_abs)
    return Path(rel).as_posix()


def _preview_mapping(paths: List[Path]) -> Dict[str, Optional[Path]]:
    lookup = {"top": None, "side": None, "oblique": None}
    for p in paths:
        name = p.name
        if "preview_top" in name:
            lookup["top"] = p
        elif "preview_side" in name:
            lookup["side"] = p
        elif "preview_oblique" in name:
            lookup["oblique"] = p
    return lookup


def build_gallery_markdown(report: RunReport, base: Path = _ROOT) -> str:
    lines = ["| Scenario | Top | Side | Oblique |", "| --- | --- | --- | --- |"]
    for scenario in report.scenarios:
        previews = _preview_mapping(scenario.preview_paths)

        def cell(which: str) -> str:
            path = previews[which]
            if path is None:
                return "_(missing)_"
            rel = _relative_path(path, base)
            return f"![{scenario.name} {which}]({rel})"

        lines.append(
            f"| {scenario.name} | {cell('top')} | {cell('side')} | {cell('oblique')} |"
        )
    return "\n".join(lines) + "\n"


def write_gallery(markdown_path: Path, report: RunReport, base: Optional[Path] = None) -> None:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    actual_base = base if base is not None else markdown_path.parent
    markdown_path.write_text(build_gallery_markdown(report, base=actual_base))


def _format_table(report: RunReport) -> str:
    header = f"{'Scenario':<28} {'Surface OBJ':<40} {'Previews':<9} {'Volumetric':<12}"
    rows = [header, "-" * len(header)]
    for scenario in report.scenarios:
        preview_label = f"{scenario.preview_count}"
        if report.preview_enabled and scenario.preview_count == 0:
            preview_label = "0 (matplotlib?)"
        vol_label = "disabled"
        if report.volumetric_requested:
            if scenario.volumetric_path:
                vol_label = "ok"
            elif scenario.volumetric_error:
                vol_label = "fail"
            elif report.volumetric_available:
                vol_label = "skipped"
            else:
                vol_label = "skipped"
        rows.append(
            f"{scenario.name:<28} "
            f"{scenario.surface_obj.name:<40} "
            f"{preview_label:<9} "
            f"{vol_label:<12}"
        )
    return "\n".join(rows)


def main(argv: Optional[Sequence[str]] = None) -> None:
    env_out_root = os.environ.get("TRENCHFOOT_SCENARIO_OUT_ROOT")
    default_out_root = (
        Path(env_out_root).expanduser()
        if env_out_root
        else Path(__file__).resolve().parent / "scenarios"
    )
    parser = argparse.ArgumentParser(description="Generate the bundled trench scenarios.")
    parser.add_argument(
        "--out",
        dest="out_root",
        default=default_out_root,
        type=Path,
        help=(
            "Destination directory for scenario outputs "
            "(default: packages/trenchfoot/scenarios or $TRENCHFOOT_SCENARIO_OUT_ROOT)"
        ),
    )
    parser.add_argument(
        "--preview",
        dest="make_preview",
        action="store_true",
        help="Enable preview rendering.",
    )
    parser.add_argument(
        "--no-preview",
        dest="make_preview",
        action="store_false",
        help="Disable preview rendering.",
    )
    parser.add_argument(
        "--volumetric",
        dest="make_volumes",
        action="store_true",
        help="Enable volumetric meshing when gmsh is available.",
    )
    parser.add_argument(
        "--skip-volumetric",
        dest="make_volumes",
        action="store_false",
        help="Skip volumetric meshing even if gmsh is installed.",
    )
    parser.add_argument(
        "--lc",
        dest="lc",
        type=float,
        default=0.3,
        help="Characteristic mesh length passed to gmsh (default: 0.3).",
    )
    parser.add_argument(
        "--gallery",
        dest="gallery",
        type=Path,
        help="Write a Markdown gallery of scenario previews to this path.",
    )
    parser.add_argument(
        "--scratch",
        dest="scratch",
        action="store_true",
        help="Generate scenarios in a temporary directory to keep the repository tree clean.",
    )
    parser.add_argument(
        "--include-prebuilt",
        dest="include_prebuilt",
        action="store_true",
        help="Copy the shipped scenario assets into the output directory before regeneration.",
    )
    parser.set_defaults(make_preview=True, make_volumes=True)
    args = parser.parse_args(argv)

    if args.scratch:
        tmp_dir = Path(tempfile.mkdtemp(prefix="trenchfoot_scenarios_"))
        print(f"[trenchfoot] Using scratch directory: {tmp_dir}", file=sys.stderr)
        args.out_root = tmp_dir

    if args.include_prebuilt:
        source = (_ROOT / "scenarios")
        if source.exists():
            shutil.copytree(source, args.out_root, dirs_exist_ok=True)

    report = generate_scenarios(
        out_root=args.out_root,
        make_preview=args.make_preview,
        make_volumes=args.make_volumes,
        mesh_characteristic_length=args.lc,
        write_summary_json=True,
    )

    print(_format_table(report))
    print()
    print(json.dumps(report.to_dict(), indent=2))

    if args.gallery:
        write_gallery(args.gallery, report)
        print(f"[trenchfoot] Wrote gallery to {args.gallery}", file=sys.stderr)


if __name__ == "__main__":
    main()
