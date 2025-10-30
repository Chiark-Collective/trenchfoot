"""Trenchfoot public API."""

from __future__ import annotations

from typing import Any, Dict

from .trench_scene_generator_v3 import (
    BoxSpec,
    GroundSpec,
    NoiseSpec,
    PipeSpec,
    SceneSpec,
    SphereSpec,
    SurfaceMeshFiles,
    SurfaceMeshResult,
    build_scene,
    generate_surface_mesh,
    load_scene_spec_from_json,
    scene_spec_from_dict,
)
from .generate_scenarios import (
    RunReport,
    ScenarioDefinition,
    default_scenarios,
    generate_scenarios,
    gmsh_available,
)

try:  # optional gmsh dependency
    from .gmsh_sloped_trench_mesher import (
        VolumeMeshResult,
        build_trench_volume_from_spec,
        generate_trench_volume,
    )
except Exception as _gmsh_exc:  # pragma: no cover - triggered when gmsh unavailable

    def generate_trench_volume(*args: Any, **kwargs: Dict[str, Any]) -> "VolumeMeshResult":
        raise ImportError("gmsh is required for volumetric mesh generation") from _gmsh_exc

    def build_trench_volume_from_spec(*args: Any, **kwargs: Dict[str, Any]) -> str:
        raise ImportError("gmsh is required for volumetric mesh generation") from _gmsh_exc

    VolumeMeshResult = None  # type: ignore


__all__ = [
    "BoxSpec",
    "GroundSpec",
    "NoiseSpec",
    "PipeSpec",
    "SceneSpec",
    "SphereSpec",
    "SurfaceMeshFiles",
    "SurfaceMeshResult",
    "build_scene",
    "generate_surface_mesh",
    "load_scene_spec_from_json",
    "scene_spec_from_dict",
    "RunReport",
    "ScenarioDefinition",
    "default_scenarios",
    "generate_scenarios",
    "gmsh_available",
    "VolumeMeshResult",
    "generate_trench_volume",
    "build_trench_volume_from_spec",
]
