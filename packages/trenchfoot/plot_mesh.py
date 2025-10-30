"""
Generate Plotly-based interactive HTML visualisations for trench meshes.
"""
from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from pathlib import Path
from typing import Optional

from .render_colors import color_for_group, opacity_for_group
from .trench_scene_generator_v3 import parse_obj_groups

try:  # Optional dependency
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - handled at runtime
    go = None


def _ensure_plotly_available() -> None:
    if go is None:
        raise RuntimeError(
            "plotly is required for this command. Install with 'pip install trenchfoot[viz]'"
        )


def _mesh_traces_from_obj(obj_path: Path) -> list:
    import numpy as np

    V, faces_by_group = parse_obj_groups(str(obj_path))
    traces = []
    for name, faces in faces_by_group.items():
        if faces.size == 0:
            continue
        faces = np.asarray(faces, dtype=int)
        trace = go.Mesh3d(
            x=V[:, 0],
            y=V[:, 1],
            z=V[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=color_for_group(name),
            opacity=opacity_for_group(name),
            name=name,
            flatshading=True,
            lighting=dict(ambient=0.5, diffuse=0.7, specular=0.1),
            showscale=False,
        )
        traces.append(trace)
    return traces


def _figure_for_mesh(path: Path) -> "go.Figure":
    suffix = path.suffix.lower()
    if suffix == ".obj":
        traces = _mesh_traces_from_obj(path)
    else:
        raise RuntimeError(f"Unsupported mesh format '{suffix}'. Only OBJ is currently supported.")

    if not traces:
        raise RuntimeError("No faces found in mesh; nothing to render.")

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis=dict(title="X", backgroundcolor="rgb(245,245,245)", showgrid=False),
            yaxis=dict(title="Y", backgroundcolor="rgb(245,245,245)", showgrid=False),
            zaxis=dict(title="Z", backgroundcolor="rgb(245,245,245)", showgrid=False),
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate a Plotly HTML visualisation for a trench mesh (OBJ).")
    parser.add_argument("input", type=Path, help="Path to trench_scene.obj (or other supported mesh).")
    parser.add_argument(
        "--out",
        dest="out_path",
        type=Path,
        help="Destination HTML file (defaults to <input>.plot.html).",
    )
    parser.add_argument(
        "--title",
        dest="title",
        type=str,
        help="Optional figure title (defaults to input file name).",
    )
    parser.add_argument(
        "--open",
        dest="open_browser",
        action="store_true",
        help="Open the resulting HTML in your default browser after writing.",
    )
    args = parser.parse_args(argv)

    _ensure_plotly_available()

    mesh_path = args.input.expanduser().resolve()
    if not mesh_path.exists():
        parser.error(f"Mesh not found: {mesh_path}")

    fig = _figure_for_mesh(mesh_path)
    if args.title:
        fig.update_layout(title=args.title)
    else:
        fig.update_layout(title=mesh_path.name)

    out_path = args.out_path or mesh_path.with_suffix(".plot.html")
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.write_html(str(out_path), auto_open=False, include_plotlyjs="cdn")
    print(f"Wrote Plotly visualisation to {out_path}")

    if args.open_browser:
        webbrowser.open(out_path.as_uri())


if __name__ == "__main__":  # pragma: no cover
    main()
