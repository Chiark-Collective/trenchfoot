#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trench_scene_generator_v3.py
Surface generator with:
  * Polyline trench (L/U/...)
  * Sloped walls (bottom width = max(width - 2*slope*depth, epsilon))
  * Ground surface: consistent ground plane z = z0 + sx*x + sy*y
  * Pipes/boxes/spheres correctly oriented and clamped inside trench
  * Optional vertex-normal noise
  * Multi-angle previews: top / side / oblique
CLI:
  python trench_scene_generator_v3.py --spec scene.json --out ./out --preview
Outputs:
  - trench_scene.obj, metrics.json
  - preview_top.png, preview_side.png, preview_oblique.png (if --preview)
"""
from __future__ import annotations

import io
import os, json, math, argparse
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from .render_colors import color_for_group, opacity_for_group

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except Exception:
    plt = None
    Poly3DCollection = None

# Groups kept for internal metrics but excluded from OBJ export and previews
_INTERNAL_GROUPS = frozenset({"trench_cap_for_volume", "inner_column_lid"})

# ---------------- Geometry helpers ----------------

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0: return v
    return v / n

def _rotate_cw(v: np.ndarray) -> np.ndarray:
    return np.array([v[1], -v[0]], dtype=float)

def _rotate_ccw(v: np.ndarray) -> np.ndarray:
    return np.array([-v[1], v[0]], dtype=float)

def _line_intersection_2d(p: np.ndarray, d: np.ndarray, q: np.ndarray, e: np.ndarray):
    M = np.array([d, -e], float).T
    det = np.linalg.det(M)
    if abs(det) < 1e-12: return None
    t, s = np.linalg.solve(M, (q - p))
    return p + t * d

def _polyline_lengths(path: List[Tuple[float,float]]):
    P = np.array(path, float)
    segs = P[1:] - P[:-1]
    lens = np.linalg.norm(segs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(lens)])
    return cum, float(cum[-1])

def _sample_polyline_at_s(path: List[Tuple[float,float]], s: float):
    P = np.array(path, float)
    cum, total = _polyline_lengths(path)
    if total == 0: return P[0], np.array([1.0, 0.0])
    s_abs = s * total
    i = np.searchsorted(cum, s_abs, side="right") - 1
    i = int(np.clip(i, 0, len(P)-2))
    seg = P[i+1] - P[i]; L = np.linalg.norm(seg)
    if L == 0:
        t = np.array([1.0, 0.0]); pos = P[i]
    else:
        t = seg / L; u = (s_abs - cum[i]) / L; pos = (1-u)*P[i] + u*P[i+1]
    return pos, t

def _is_path_closed(path: List[Tuple[float,float]], threshold: float = 0.01) -> bool:
    """Detect if a path is explicitly closed (first and last points nearly identical).

    Returns True only if the first and last points are within threshold distance.
    Use a small threshold (default 0.01) to only catch truly closed paths where
    the endpoint is repeated.
    """
    if len(path) < 3:
        return False
    P = np.array(path, float)
    first_last_dist = np.linalg.norm(P[0] - P[-1])
    return first_last_dist < threshold

def _offset_polyline(path: List[Tuple[float,float]], offset: float):
    P = np.array(path, float); n = len(P)
    if n < 2: raise ValueError("Polyline needs at least 2 points")
    tangents = []; normals = []
    for i in range(n-1):
        t = _normalize(P[i+1]-P[i])
        if np.linalg.norm(t) < 1e-12: t = np.array([1.0, 0.0])
        tangents.append(t); normals.append(_rotate_ccw(t))
    left_pts = [P[0] + offset * normals[0]]
    right_pts = [P[0] - offset * normals[0]]
    for k in range(1, n-1):
        t_prev, n_prev = tangents[k-1], normals[k-1]
        t_next, n_next = tangents[k], normals[k]
        L1_p = P[k] + offset * n_prev; L1_d = t_prev
        L2_p = P[k] + offset * n_next; L2_d = t_next
        R1_p = P[k] - offset * n_prev; R1_d = t_prev
        R2_p = P[k] - offset * n_next; R2_d = t_next
        L = _line_intersection_2d(L1_p, L1_d, L2_p, L2_d)
        R = _line_intersection_2d(R1_p, R1_d, R2_p, R2_d)
        if L is None: L = 0.5*(L1_p + L2_p)
        if R is None: R = 0.5*(R1_p + R2_p)
        left_pts.append(L); right_pts.append(R)
    left_pts.append(P[-1] + offset * normals[-1])
    right_pts.append(P[-1] - offset * normals[-1])
    return left_pts, right_pts

def _offset_closed_polyline(path: List[Tuple[float,float]], offset: float) -> List[np.ndarray]:
    """Offset a closed polyline, returning a single closed ring.

    Unlike _offset_polyline which returns left/right sides for open paths,
    this returns a single continuous closed ring for paths where first ≈ last point.
    """
    P = np.array(path, float)
    # Remove duplicate closing point if present (first ≈ last)
    if len(P) > 1 and np.linalg.norm(P[0] - P[-1]) < 0.01:
        P = P[:-1]
    n = len(P)
    if n < 3:
        raise ValueError("Closed polyline needs at least 3 points")

    # Compute tangents treating path as closed loop
    # For CCW-oriented polygons, use CW rotation to get outward-pointing normals
    tangents = []
    normals = []
    for i in range(n):
        t = _normalize(P[(i+1) % n] - P[i])
        if np.linalg.norm(t) < 1e-12:
            t = np.array([1.0, 0.0])
        tangents.append(t)
        normals.append(_rotate_cw(t))  # CW rotation gives outward normal for CCW polygon

    # Compute offset points with proper miter at each vertex
    offset_pts = []
    for k in range(n):
        t_prev, n_prev = tangents[(k-1) % n], normals[(k-1) % n]
        t_next, n_next = tangents[k], normals[k]
        L1_p = P[k] + offset * n_prev
        L1_d = t_prev
        L2_p = P[k] + offset * n_next
        L2_d = t_next
        pt = _line_intersection_2d(L1_p, L1_d, L2_p, L2_d)
        if pt is None:
            pt = 0.5 * (L1_p + L2_p)
        offset_pts.append(pt)

    return offset_pts

def _polygon_area_2d(poly_xy: np.ndarray) -> float:
    x = poly_xy[:,0]; y = poly_xy[:,1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def _ensure_ccw(poly_xy: np.ndarray) -> np.ndarray:
    return poly_xy if _polygon_area_2d(poly_xy) > 0 else poly_xy[::-1].copy()

def _cross2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0]*b[1] - a[1]*b[0])

def _ear_clipping_triangulation(poly_xy: np.ndarray) -> np.ndarray:
    def is_convex(a, b, c): return _cross2d(b - a, c - b) > 0
    def point_in_tri(p, a, b, c):
        v0=c-a; v1=b-a; v2=p-a
        den=v0[0]*v1[1]-v1[0]*v0[1]
        if abs(den)<1e-15: return False
        u=(v2[0]*v1[1]-v1[0]*v2[1])/den
        v=(v0[0]*v2[1]-v2[0]*v0[1])/den
        return (u>=-1e-12) and (v>=-1e-12) and (u+v<=1+1e-12)
    V = list(range(len(poly_xy))); tris=[]; it=0
    while len(V)>3 and it<10000:
        ear=False; m=len(V)
        for vi in range(m):
            i0=V[(vi-1)%m]; i1=V[vi]; i2=V[(vi+1)%m]
            a,b,c = poly_xy[i0], poly_xy[i1], poly_xy[i2]
            if not is_convex(a,b,c): continue
            inside=False
            for j in range(m):
                if j in [(vi-1)%m,vi,(vi+1)%m]: continue
                pj = poly_xy[V[j]]
                if point_in_tri(pj,a,b,c): inside=True; break
            if inside: continue
            tris.append([i0,i1,i2]); del V[vi]; ear=True; break
        if not ear:
            V2=V.copy()
            for k in range(1,len(V2)-1): tris.append([V2[0],V2[k],V2[k+1]])
            V=[V2[0],V2[-1],V2[-2]]
        it+=1
    tris.append([V[0],V[1],V[2]])
    return np.array(tris,int)


def _extract_boundary_polygon(V: np.ndarray, F: np.ndarray) -> Optional[np.ndarray]:
    """Extract ordered boundary polygon from a triangulated mesh.

    Finds boundary edges (edges appearing in only one face) and chains
    them together to form an ordered polygon. This correctly handles
    non-convex shapes like L-shaped or U-shaped boundaries.

    Parameters
    ----------
    V : np.ndarray
        Vertices (n, 3) or (n, 2)
    F : np.ndarray
        Faces (m, 3) - triangle indices

    Returns
    -------
    np.ndarray or None
        Ordered boundary vertices XY coordinates (k, 2), or None if
        no boundary edges found (closed mesh).
    """
    if F.size == 0:
        return None

    # Count edge occurrences - boundary edges appear exactly once
    edge_count: Dict[Tuple[int, int], int] = {}
    for face in F:
        for i in range(3):
            v0, v1 = int(face[i]), int(face[(i + 1) % 3])
            edge = (min(v0, v1), max(v0, v1))
            edge_count[edge] = edge_count.get(edge, 0) + 1

    # Boundary edges appear exactly once
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]

    if not boundary_edges:
        return None  # No boundary (closed mesh)

    # Build adjacency from boundary edges
    adj: Dict[int, List[int]] = {}
    for v0, v1 in boundary_edges:
        adj.setdefault(v0, []).append(v1)
        adj.setdefault(v1, []).append(v0)

    # Chain the boundary edges starting from any vertex
    start = boundary_edges[0][0]
    polygon = [start]
    prev = None
    curr = start

    max_iter = len(boundary_edges) + 1
    for _ in range(max_iter):
        neighbors = adj.get(curr, [])
        # Pick the neighbor that isn't the previous vertex
        next_candidates = [n for n in neighbors if n != prev]
        if not next_candidates:
            break
        next_v = next_candidates[0]
        if next_v == start:
            break  # Completed the loop
        polygon.append(next_v)
        prev = curr
        curr = next_v

    # Extract XY coordinates
    return V[polygon, :2]


# ---------------- Mesh IO & metrics ----------------

def write_obj_with_groups(path: str, groups: Dict[str, Tuple[np.ndarray, np.ndarray]]):
    lines=[]; offset=1
    for g,(V,F) in groups.items():
        lines.append(f"g {g}")
        for v in V: lines.append(f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}")
        for tri in F:
            a,b,c = tri + offset
            lines.append(f"f {a} {b} {c}")
        offset += V.shape[0]
    with open(path,"w") as f: f.write("\n".join(lines))

def parse_obj_groups(path: str):
    verts=[]; faces_by_group={}; current="default"
    with open(path,"r") as f:
        for line in f:
            if not line.strip(): continue
            if line.startswith("v "):
                _,x,y,z = line.strip().split()
                verts.append([float(x),float(y),float(z)])
            elif line.startswith("g "):
                current = line.strip().split(maxsplit=1)[1]
                faces_by_group.setdefault(current, [])
            elif line.startswith("f "):
                parts = line.strip().split()
                idxs = [int(p.split('/')[0])-1 for p in parts[1:4]]
                faces_by_group.setdefault(current, []).append(idxs)
    V=np.array(verts,float)
    faces_by_group = {k:(np.array(v,int) if len(v) else np.zeros((0,3),int)) for k,v in faces_by_group.items()}
    return V, faces_by_group

def triangle_areas(V,F):
    p0=V[F[:,0]]; p1=V[F[:,1]]; p2=V[F[:,2]]
    return 0.5*np.linalg.norm(np.cross(p1-p0,p2-p0),axis=1)

def surface_area(V,F): return float(triangle_areas(V,F).sum())

def surface_area_by_group(obj_path: str):
    V, fbg = parse_obj_groups(obj_path)
    return {g: float(surface_area(V,F)) for g,F in fbg.items()}

def signed_volume_of_closed_surface(V,F):
    p0=V[F[:,0]]; p1=V[F[:,1]]; p2=V[F[:,2]]
    vol = np.einsum('ij,ij->i', p0, np.cross(p1,p2))
    return float(vol.sum()/6.0)

def volume_by_groups_as_closed(obj_path: str, names):
    V, fbg = parse_obj_groups(obj_path)
    Fs=[fbg[n] for n in names if n in fbg]
    if not Fs: return 0.0
    F=np.vstack(Fs)
    return signed_volume_of_closed_surface(V,F)

def flux_volume_from_closed_groups(obj_path: str, names):
    V, fbg = parse_obj_groups(obj_path)
    F_all = np.vstack([fbg[n] for n in names if n in fbg])
    p0=V[F_all[:,0]]; p1=V[F_all[:,1]]; p2=V[F_all[:,2]]
    cent=(p0+p1+p2)/3.0; Fvec=cent/3.0; nvec=np.cross(p1-p0,p2-p0)
    return float(((Fvec*nvec).sum(axis=1)/2.0).sum())


def _combine_groups(groups: Dict[str, Tuple[np.ndarray, np.ndarray]], names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    vertices: List[np.ndarray] = []
    faces: List[np.ndarray] = []
    offset = 0
    for name in names:
        entry = groups.get(name)
        if entry is None:
            continue
        V, F = entry
        if V.size == 0 or F.size == 0:
            continue
        vertices.append(V)
        faces.append(F + offset)
        offset += V.shape[0]
    if not vertices:
        return np.zeros((0, 3), float), np.zeros((0, 3), int)
    return np.vstack(vertices), np.vstack(faces)


def _compute_surface_metrics(
    groups: Dict[str, Tuple[np.ndarray, np.ndarray]],
    extra: Dict[str, Any],
    spec: SceneSpec,
) -> Dict[str, Any]:
    areas = {
        name: float(surface_area(V, F))
        for name, (V, F) in groups.items()
    }
    closed_names = ["trench_walls", "trench_bottom", "trench_cap_for_volume"]
    V_closed, F_closed = _combine_groups(groups, closed_names)
    if F_closed.size == 0:
        vol_surface = 0.0
        vol_flux = 0.0
    else:
        vol_surface = signed_volume_of_closed_surface(V_closed, F_closed)
        p0 = V_closed[F_closed[:, 0]]
        p1 = V_closed[F_closed[:, 1]]
        p2 = V_closed[F_closed[:, 2]]
        cent = (p0 + p1 + p2) / 3.0
        Fvec = cent / 3.0
        nvec = np.cross(p1 - p0, p2 - p0)
        vol_flux = float(((Fvec * nvec).sum(axis=1) / 2.0).sum())

    metrics = {
        "surface_area_by_group": areas,
        "closed_surface_sets": {"trench_closed_groups": closed_names},
        "volumes": {
            "trench_from_surface": vol_surface,
            "trench_flux_integral_div1": vol_flux,
        },
        "footprint_area_top": float(areas.get("trench_cap_for_volume", 0.0)),
        "footprint_area_bottom": float(areas.get("trench_bottom", 0.0)),
        "width_top": float(extra.get("width_top", spec.width)),
        "width_bottom": float(extra.get("width_bottom", spec.width)),
        "noise": asdict(spec.noise) if spec.noise else None,
    }
    return metrics


def _render_surface_previews(
    groups: Dict[str, Tuple[np.ndarray, np.ndarray]],
    exclude_groups: Optional[frozenset] = None,
) -> Dict[str, bytes]:
    if plt is None or not groups:
        return {}
    if exclude_groups:
        groups = {k: v for k, v in groups.items() if k not in exclude_groups}
    all_vertices = [V for (V, F) in groups.values() if V.size > 0]
    if not all_vertices:
        return {}
    stack = np.vstack(all_vertices)
    mins, maxs = stack.min(axis=0), stack.max(axis=0)
    previews: Dict[str, bytes] = {}
    viewset = [("top", (90, 0)), ("side", (0, 0)), ("oblique", (22, -60))]
    for name, (elev, azim) in viewset:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        for group_name, (V, F) in groups.items():
            if F.shape[0] == 0:
                continue
            tris = [V[idx] for idx in F]
            if not tris:
                continue
            pc = Poly3DCollection(tris, linewidths=0.1)
            color = color_for_group(group_name)
            alpha = opacity_for_group(group_name)
            pc.set_facecolor(color)
            pc.set_edgecolor(color)
            pc.set_alpha(alpha)
            ax.add_collection3d(pc)
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=elev, azim=azim)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        previews[name] = buf.getvalue()
    return previews

# ---------------- Scene & primitives ----------------

@dataclass
class PipeSpec:
    radius: float
    length: float
    angle_deg: float
    s_center: float = 0.5
    z: Optional[float] = None
    offset_u: float = 0.0
    n_theta: int = 96
    n_along: int = 48
    clearance_scale: float = 1.0

@dataclass
class BoxSpec:
    along: float
    across: float
    height: float
    s: float = 0.5
    offset_u: float = 0.0
    z: Optional[float] = None

@dataclass
class SphereSpec:
    radius: float
    s: float = 0.7
    offset_u: float = 0.0
    z: Optional[float] = None

@dataclass
class NoiseSpec:
    enable: bool = False
    amplitude: float = 0.02
    corr_length: float = 0.5
    octaves: int = 2
    gain: float = 0.5
    seed: int = 42
    apply_to: Tuple[str,...] = ("trench_walls","trench_bottom")

@dataclass
class GroundSpec:
    z0: float = 0.0
    slope: Tuple[float,float] = (0.0, 0.0)   # (dz/dx, dz/dy)
    size_margin: float = 3.0
    fill_interior: bool = False  # For closed paths: fill the interior with ground surface

@dataclass
class SceneSpec:
    path_xy: List[Tuple[float,float]]
    width: float
    depth: float
    wall_slope: float = 0.0      # m horizontal per m depth (each side)
    ground_margin: float = 0.0   # legacy; used if ground.size_margin==0
    pipes: List[PipeSpec] = field(default_factory=list)
    boxes: List[BoxSpec] = field(default_factory=list)
    spheres: List[SphereSpec] = field(default_factory=list)
    noise: NoiseSpec = field(default_factory=NoiseSpec)
    ground: GroundSpec = field(default_factory=GroundSpec)


@dataclass(frozen=True)
class SurfaceMeshFiles:
    obj_path: Path
    metrics_path: Path
    preview_paths: Tuple[Path, ...]
    sdf_metadata_path: Optional[Path] = None


@dataclass
class SurfaceMeshResult:
    spec: SceneSpec
    groups: Dict[str, Tuple[np.ndarray, np.ndarray]]
    object_counts: Dict[str, int]
    metrics: Dict[str, Any]
    previews: Dict[str, bytes]

    def _build_sdf_metadata(self) -> Dict[str, Any]:
        """Build SDF metadata for downstream consumers.

        This metadata enables generic mesh-to-SDF pipelines to correctly
        interpret the mesh geometry without trenchfoot-specific heuristics.
        """
        # Extract trench opening polygon from the trench cap geometry
        trench_opening_vertices = None
        if "trench_cap_for_volume" in self.groups:
            V_cap, F_cap = self.groups["trench_cap_for_volume"]
            if V_cap.size > 0:
                # Extract boundary polygon by finding boundary edges (edges in only one face)
                # and chaining them together. This correctly handles non-convex shapes
                # like L-shaped or U-shaped trenches.
                boundary_xy = _extract_boundary_polygon(V_cap, F_cap)
                if boundary_xy is not None:
                    trench_opening_vertices = boundary_xy.tolist()
                else:
                    # Fallback: just use all vertices (unordered)
                    trench_opening_vertices = V_cap[:, :2].tolist()

        # Determine geometry type
        is_closed = _is_path_closed(self.spec.path_xy)
        geometry_type = "closed_well" if is_closed else "open_trench"

        # Build surface group info
        surface_groups = {}
        for name in self.groups:
            if name in _INTERNAL_GROUPS:
                continue
            if "bottom" in name:
                surface_groups[name] = {"normal_direction": "up", "surface_type": "floor"}
            elif "wall" in name:
                surface_groups[name] = {"normal_direction": "inward", "surface_type": "wall"}
            elif "ground" in name:
                surface_groups[name] = {"normal_direction": "up", "surface_type": "ground"}
            elif "pipe" in name:
                surface_groups[name] = {"normal_direction": "outward", "surface_type": "embedded_object"}
            elif "box" in name or "sphere" in name:
                surface_groups[name] = {"normal_direction": "outward", "surface_type": "embedded_object"}
            else:
                surface_groups[name] = {"normal_direction": "unknown", "surface_type": "other"}

        return {
            "sdf_metadata": {
                "version": "2.0",
                "normal_convention": "into_void",
                "geometry_type": geometry_type,
                "trench_opening": {
                    "type": "polygon",
                    "vertices_xy": trench_opening_vertices,
                    "z_level": self.spec.ground.z0 if self.spec.ground else 0.0,
                },
                "surface_groups": surface_groups,
                "embedded_objects": {
                    "pipes": self.object_counts.get("pipes", 0),
                    "boxes": self.object_counts.get("boxes", 0),
                    "spheres": self.object_counts.get("spheres", 0),
                },
            }
        }

    def persist(self, out_dir: str | Path, *, include_previews: bool = False, include_sdf_metadata: bool = True) -> SurfaceMeshFiles:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        obj_path = out_path / "trench_scene.obj"
        # Exclude internal groups (like trench_cap_for_volume) from OBJ export
        export_groups = {k: v for k, v in self.groups.items() if k not in _INTERNAL_GROUPS}
        write_obj_with_groups(obj_path.as_posix(), export_groups)
        metrics_path = out_path / "metrics.json"
        with metrics_path.open("w") as fh:
            json.dump(self.metrics, fh, indent=2)

        # Export SDF metadata
        sdf_metadata_path = None
        if include_sdf_metadata:
            sdf_metadata = self._build_sdf_metadata()
            sdf_metadata_path = out_path / "sdf_metadata.json"
            with sdf_metadata_path.open("w") as fh:
                json.dump(sdf_metadata, fh, indent=2)

        preview_paths: List[Path] = []
        if include_previews and self.previews:
            for name, data in self.previews.items():
                target = out_path / f"preview_{name}.png"
                target.write_bytes(data)
                preview_paths.append(target)
        return SurfaceMeshFiles(
            obj_path=obj_path,
            metrics_path=metrics_path,
            preview_paths=tuple(preview_paths),
            sdf_metadata_path=sdf_metadata_path,
        )

def _ground_fn(g: GroundSpec):
    sx, sy = g.slope
    def fn(x, y): return g.z0 + sx*float(x) + sy*float(y)
    return fn

def _frame_from_axis(axis_dir: np.ndarray) -> np.ndarray:
    v=_normalize(axis_dir)
    helper=np.array([0.0,0.0,1.0],float)
    if abs(np.dot(helper,v))>0.99: helper=np.array([1.0,0.0,0.0],float)
    u=_normalize(np.cross(helper,v)); w=np.cross(v,u)
    return np.column_stack([u,v,w])

def make_cylinder(
    center: np.ndarray,
    axis_dir: np.ndarray,
    radius: float,
    length: float,
    n_theta: int = 64,
    n_along: int = 32,
    with_caps: bool = True,
    neg_extent: Optional[float] = None,
    pos_extent: Optional[float] = None,
    cap_plane_neg: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    cap_plane_pos: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Generate a cylinder mesh with optional truncation and angled caps.

    Parameters
    ----------
    center : np.ndarray
        Center point of the cylinder (3D).
    axis_dir : np.ndarray
        Unit vector along the cylinder axis (3D).
    radius : float
        Cylinder radius.
    length : float
        Total cylinder length (used if neg/pos_extent not specified).
    n_theta : int
        Number of angular divisions.
    n_along : int
        Number of divisions along the axis.
    with_caps : bool
        Whether to generate end caps.
    neg_extent : float, optional
        Distance from center to negative end. If None, uses -length/2.
    pos_extent : float, optional
        Distance from center to positive end. If None, uses +length/2.
    cap_plane_neg : tuple, optional
        (normal, point) defining the plane for negative cap. If provided,
        generates an elliptical cap on this plane instead of a flat circular cap.
    cap_plane_pos : tuple, optional
        (normal, point) for positive cap.

    Returns
    -------
    dict
        Dictionary with 'pipe_side' and optionally 'pipe_cap_neg', 'pipe_cap_pos'.
    """
    n_theta = max(8, int(n_theta))
    n_along = max(1, int(n_along))

    # Use extents if provided, otherwise symmetric from length
    y_neg = neg_extent if neg_extent is not None else -length / 2.0
    y_pos = pos_extent if pos_extent is not None else length / 2.0

    # Build coordinate frame
    M = _frame_from_axis(axis_dir)

    def xform(V: np.ndarray) -> np.ndarray:
        return (center + V @ M.T).astype(float)

    # Generate cylinder side surface
    thetas = np.linspace(0, 2 * np.pi, n_theta + 1)
    ys = np.linspace(y_neg, y_pos, n_along + 1)
    Vloc = []
    for j in range(n_along + 1):
        y = ys[j]
        for i in range(n_theta + 1):
            th = thetas[i]
            x = radius * np.cos(th)
            z = radius * np.sin(th)
            Vloc.append([x, y, z])
    Vloc = np.array(Vloc, float)

    def idx(i: int, j: int) -> int:
        return j * (n_theta + 1) + i

    F = []
    for j in range(n_along):
        for i in range(n_theta):
            v00 = idx(i, j)
            v10 = idx(i + 1, j)
            v01 = idx(i, j + 1)
            v11 = idx(i + 1, j + 1)
            F.append([v00, v01, v11])
            F.append([v00, v11, v10])
    F = np.array(F, int)

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {"pipe_side": (xform(Vloc), F)}

    if with_caps:
        # Generate caps (either flat circular or angled elliptical)
        Vn, Fn = _make_cylinder_cap(
            radius, y_neg, thetas[:-1], M, center, axis_dir,
            cap_plane_neg, is_negative=True
        )
        Vp, Fp = _make_cylinder_cap(
            radius, y_pos, thetas[:-1], M, center, axis_dir,
            cap_plane_pos, is_negative=False
        )
        out['pipe_cap_neg'] = (Vn, Fn)
        out['pipe_cap_pos'] = (Vp, Fp)

    return out


def _make_cylinder_cap(
    radius: float,
    y_extent: float,
    thetas: np.ndarray,
    M: np.ndarray,
    center: np.ndarray,
    axis_dir: np.ndarray,
    cap_plane: Optional[Tuple[np.ndarray, np.ndarray]],
    is_negative: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a cylinder end cap, either flat or angled.

    For angled caps, we compute where each point on the cylinder rim
    intersects the cap plane, creating an elliptical cap.
    """
    n_theta = len(thetas)

    if cap_plane is None:
        # Flat circular cap perpendicular to axis
        ring = np.array([
            [radius * np.cos(t), y_extent, radius * np.sin(t)]
            for t in thetas
        ], float)
        cap_center = np.array([[0.0, y_extent, 0.0]], float)
        V = np.vstack([cap_center, ring])

        # Apply transformation to world coordinates
        V_world = center + V @ M.T

        # Generate fan triangles
        if is_negative:
            # Negative cap: winding for outward normal (toward -axis)
            F = np.array([[0, 1 + (i + 1) % n_theta, 1 + i] for i in range(n_theta)], int)
        else:
            # Positive cap: winding for outward normal (toward +axis)
            F = np.array([[0, 1 + i, 1 + (i + 1) % n_theta] for i in range(n_theta)], int)

        return V_world.astype(float), F
    else:
        # Angled cap: intersect cylinder rim with plane
        plane_normal, plane_point = cap_plane
        plane_normal = _normalize(plane_normal)

        # Points on the cylinder rim at y_extent (in local coords before xform)
        ring_local = np.array([
            [radius * np.cos(t), y_extent, radius * np.sin(t)]
            for t in thetas
        ], float)

        # Transform ring to world coordinates
        ring_world = center + ring_local @ M.T

        # For each ring point, project along axis onto the cap plane
        # Line: P = ring_point + t * axis_dir
        # Plane: dot(P - plane_point, plane_normal) = 0
        # => t = dot(plane_point - ring_point, plane_normal) / dot(axis_dir, plane_normal)
        denom = np.dot(axis_dir, plane_normal)
        if abs(denom) < 1e-10:
            # Axis is parallel to plane - fall back to flat cap
            V_world = np.vstack([
                center + np.array([0.0, y_extent, 0.0]) @ M.T,
                ring_world
            ])
            if is_negative:
                F = np.array([[0, 1 + (i + 1) % n_theta, 1 + i] for i in range(n_theta)], int)
            else:
                F = np.array([[0, 1 + i, 1 + (i + 1) % n_theta] for i in range(n_theta)], int)
            return V_world.astype(float), F

        # Project ring points onto plane
        cap_verts = []
        for ring_pt in ring_world:
            t = np.dot(plane_point - ring_pt, plane_normal) / denom
            cap_pt = ring_pt + t * axis_dir
            cap_verts.append(cap_pt)
        cap_verts = np.array(cap_verts, float)

        # Cap center: project axis center onto plane
        axis_pt = center + y_extent * axis_dir
        t_center = np.dot(plane_point - axis_pt, plane_normal) / denom
        cap_center = axis_pt + t_center * axis_dir

        V_world = np.vstack([cap_center.reshape(1, 3), cap_verts])

        # Generate fan triangles
        if is_negative:
            F = np.array([[0, 1 + (i + 1) % n_theta, 1 + i] for i in range(n_theta)], int)
        else:
            F = np.array([[0, 1 + i, 1 + (i + 1) % n_theta] for i in range(n_theta)], int)

        return V_world.astype(float), F

def make_box(center: np.ndarray, frame_cols: np.ndarray, dims: Tuple[float,float,float]):
    a,b,h=dims; u=frame_cols[:,0]; v=frame_cols[:,1]; w=frame_cols[:,2]
    corners=[]
    for sx in [-0.5,0.5]:
      for sy in [-0.5,0.5]:
        for sz in [-0.5,0.5]:
          corners.append(center + sx*a*u + sy*b*v + sz*h*w)
    corners=np.array(corners,float)
    def vid(sx,sy,sz):
        ix=0 if sx<0 else 1; iy=0 if sy<0 else 1; iz=0 if sz<0 else 1
        return ix*4 + iy*2 + iz
    quads=[
      [vid( 0.5,-0.5,-0.5), vid( 0.5, 0.5,-0.5), vid( 0.5, 0.5, 0.5), vid( 0.5,-0.5, 0.5)],
      [vid(-0.5, 0.5,-0.5), vid( 0.5, 0.5,-0.5), vid( 0.5, 0.5, 0.5), vid(-0.5, 0.5, 0.5)],
      [vid(-0.5,-0.5, 0.5), vid( 0.5,-0.5,  0.5), vid( 0.5, 0.5, 0.5), vid(-0.5, 0.5, 0.5)],
      [vid(-0.5,-0.5,-0.5), vid( 0.5,-0.5,-0.5), vid( 0.5, 0.5,-0.5), vid(-0.5, 0.5,-0.5)],
      [vid(-0.5,-0.5,-0.5), vid(-0.5, 0.5,-0.5), vid(-0.5, 0.5, 0.5), vid(-0.5,-0.5, 0.5)],
      [vid(-0.5,-0.5,-0.5), vid( 0.5,-0.5,-0.5), vid( 0.5,-0.5, 0.5), vid(-0.5,-0.5, 0.5)],
    ]
    faces=[]
    for q in quads: faces.append([q[0],q[1],q[2]]); faces.append([q[0],q[2],q[3]])
    return corners, np.array(faces,int)

def make_sphere(center: np.ndarray, radius: float, n_theta: int=48, n_phi: int=24):
    n_theta=max(8,int(n_theta)); n_phi=max(4,int(n_phi))
    thetas=np.linspace(0,2*np.pi,n_theta+1); phis=np.linspace(0,np.pi,n_phi+1)
    V=[]
    for j in range(n_phi+1):
        phi=phis[j]
        for i in range(n_theta+1):
            th=thetas[i]
            x=radius*np.sin(phi)*np.cos(th); y=radius*np.sin(phi)*np.sin(th); z=radius*np.cos(phi)
            V.append([center[0]+x, center[1]+y, center[2]+z])
    V=np.array(V,float)
    def idx(i,j): return j*(n_theta+1)+i
    F=[]
    for j in range(n_phi):
        for i in range(n_theta):
            v00=idx(i,j); v10=idx(i+1,j); v01=idx(i,j+1); v11=idx(i+1,j+1)
            F.append([v00,v01,v11]); F.append([v00,v11,v10])
    return V, np.array(F,int)

# --------------- Sloped trench surfaces with ground ---------------

def _ring_from_LR(L: List[np.ndarray], R: List[np.ndarray]) -> np.ndarray:
    return np.array(L + list(R[::-1]), float)


def _extend_polyline_ends(
    L: List[np.ndarray], R: List[np.ndarray],
    path_xy: List[Tuple[float, float]], extension: float
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Extend L/R offset polylines beyond path endpoints.

    Adds extra points at each end that extend the polylines backward (at start)
    and forward (at end) along the path tangent direction.
    """
    P = np.array(path_xy, float)

    # Tangent directions at endpoints
    t_start = _normalize(P[1] - P[0])
    t_end = _normalize(P[-1] - P[-2])

    # Extend start: add points before L[0] and R[0]
    L_start_ext = np.array(L[0]) - extension * t_start
    R_start_ext = np.array(R[0]) - extension * t_start

    # Extend end: add points after L[-1] and R[-1]
    L_end_ext = np.array(L[-1]) + extension * t_end
    R_end_ext = np.array(R[-1]) + extension * t_end

    # Build extended polylines
    L_ext = [L_start_ext] + list(L) + [L_end_ext]
    R_ext = [R_start_ext] + list(R) + [R_end_ext]

    return L_ext, R_ext


def make_trench_from_path_sloped(path_xy: List[Tuple[float,float]], width_top: float, depth: float, wall_slope: float, ground) -> Tuple[Dict,str,str,dict]:
    # Build top and bottom rings by offsetting centerline
    half_top = width_top/2.0
    shrink = max(0.0, wall_slope * depth)
    half_bot = max(1e-3, half_top - shrink)

    is_closed = _is_path_closed(path_xy)

    if is_closed:
        # For closed paths (like circles), create outer/inner rings
        # Outer ring: centerline offset outward (positive)
        # Inner ring: centerline offset inward (negative)
        outer_top = np.array(_offset_closed_polyline(path_xy, half_top), float)
        inner_top = np.array(_offset_closed_polyline(path_xy, -half_top), float)
        outer_bot = np.array(_offset_closed_polyline(path_xy, half_bot), float)
        inner_bot = np.array(_offset_closed_polyline(path_xy, -half_bot), float)

        # Ensure CCW orientation (outer should be CCW, inner CW for proper normals)
        outer_top = _ensure_ccw(outer_top)
        outer_bot = _ensure_ccw(outer_bot)
        # Inner rings should go opposite direction
        if _polygon_area_2d(inner_top) > 0:
            inner_top = inner_top[::-1].copy()
        if _polygon_area_2d(inner_bot) > 0:
            inner_bot = inner_bot[::-1].copy()

        gfun = _ground_fn(ground)

        # For closed trenches, we need outer wall, inner wall, and bottom (no cap for annular trench)
        # Actually, for annular trench, the "bottom" is an annulus and the "cap" is also an annulus
        z_outer_top = np.array([gfun(x,y) for x,y in outer_top])
        z_inner_top = np.array([gfun(x,y) for x,y in inner_top])
        z_outer_bot = np.array([gfun(x,y) - depth for x,y in outer_bot])
        z_inner_bot = np.array([gfun(x,y) - depth for x,y in inner_bot])

        # Triangulate annular cap and bottom
        cap_verts, cap_faces = _triangulate_annulus(outer_top, inner_top[::-1])  # reverse inner for CCW
        V_cap = np.column_stack([cap_verts, np.concatenate([z_outer_top, z_inner_top[::-1]])])
        F_cap = cap_faces

        bot_verts, bot_faces = _triangulate_annulus(outer_bot, inner_bot[::-1])
        V_bottom = np.column_stack([bot_verts, np.concatenate([z_outer_bot, z_inner_bot[::-1]])])
        # Floor normals point UP (+z) into the trench void for correct SDF sign
        F_bottom = _ensure_upward_normals(V_bottom, bot_faces)

        # Outer wall: connects outer_top to outer_bot (facing outward from trench)
        n_outer = len(outer_top)
        walls_V = []
        walls_F = []
        for i in range(n_outer):
            j = (i + 1) % n_outer
            A_top = np.array([outer_top[i,0], outer_top[i,1], z_outer_top[i]])
            B_top = np.array([outer_top[j,0], outer_top[j,1], z_outer_top[j]])
            A_bot = np.array([outer_bot[i,0], outer_bot[i,1], z_outer_bot[i]])
            B_bot = np.array([outer_bot[j,0], outer_bot[j,1], z_outer_bot[j]])
            base = len(walls_V)
            walls_V.extend([A_top, B_top, B_bot, A_bot])
            # Winding for outward-facing (away from center)
            walls_F.extend([[base, base+1, base+2], [base, base+2, base+3]])

        # Inner wall: connects inner_top to inner_bot (facing inward toward center)
        n_inner = len(inner_top)
        for i in range(n_inner):
            j = (i + 1) % n_inner
            A_top = np.array([inner_top[i,0], inner_top[i,1], z_inner_top[i]])
            B_top = np.array([inner_top[j,0], inner_top[j,1], z_inner_top[j]])
            A_bot = np.array([inner_bot[i,0], inner_bot[i,1], z_inner_bot[i]])
            B_bot = np.array([inner_bot[j,0], inner_bot[j,1], z_inner_bot[j]])
            base = len(walls_V)
            walls_V.extend([A_top, B_top, B_bot, A_bot])
            # Winding for inward-facing (toward center) - reverse winding
            walls_F.extend([[base, base+2, base+1], [base, base+3, base+2]])

        V_walls = np.array(walls_V, float)
        F_walls = np.array(walls_F, int)

        # Inner column lid: cap the top of the inner column at ground level
        # Reverse inner_top to get CCW winding for upward-facing normals
        inner_top_ccw = inner_top[::-1].copy()
        z_inner_top_ccw = z_inner_top[::-1].copy()
        lid_xy, lid_faces = _triangulate_polygon_fan(inner_top_ccw)
        # Assign z-values: polygon vertices use z_inner_top_ccw, centroid uses average
        z_lid = np.concatenate([z_inner_top_ccw, [np.mean(z_inner_top_ccw)]])
        V_lid = np.column_stack([lid_xy, z_lid])
        F_lid = _ensure_upward_normals(V_lid, lid_faces)

        # For closed path, poly_top is the outer ring (used for ground plane hole)
        poly_top = outer_top
        poly_bot = outer_bot

        extra = {
            "width_top": width_top,
            "width_bottom": 2.0*half_bot,
            "area_top": abs(_polygon_area_2d(outer_top)) - abs(_polygon_area_2d(inner_top)),
            "area_bottom": abs(_polygon_area_2d(outer_bot)) - abs(_polygon_area_2d(inner_bot)),
            "is_closed_path": True
        }
    else:
        # Original logic for open paths
        L_top, R_top = _offset_polyline(path_xy, half_top)
        L_bot, R_bot = _offset_polyline(path_xy, half_bot)
        poly_top = _ensure_ccw(_ring_from_LR(L_top, R_top))
        poly_bot = _ensure_ccw(_ring_from_LR(L_bot, R_bot))

        gfun = _ground_fn(ground)
        # Top and bottom rings lie on the ground plane and ground-depth respectively
        z_top = np.array([gfun(x,y) for x,y in poly_top])
        z_bot = np.array([gfun(x,y) - depth for x,y in poly_bot])
        tris_top = _ear_clipping_triangulation(poly_top)
        tris_bot = _ear_clipping_triangulation(poly_bot)
        V_cap = np.column_stack([poly_top, z_top])
        V_bottom = np.column_stack([poly_bot, z_bot])
        F_cap = tris_top
        # Floor normals point UP (+z) into the trench void for correct SDF sign
        F_bottom = _ensure_upward_normals(V_bottom, tris_bot)

        # Walls: connect corresponding indices
        N = len(poly_top)
        assert N == len(poly_bot)
        walls_V = []
        walls_F = []
        for i in range(N):
            j = (i+1) % N
            A_top = np.array([poly_top[i,0], poly_top[i,1], z_top[i]])
            B_top = np.array([poly_top[j,0], poly_top[j,1], z_top[j]])
            A_bot = np.array([poly_bot[i,0], poly_bot[i,1], z_bot[i]])
            B_bot = np.array([poly_bot[j,0], poly_bot[j,1], z_bot[j]])
            base = len(walls_V)
            walls_V.extend([A_top, B_top, B_bot, A_bot])
            walls_F.extend([[base, base+1, base+2], [base, base+2, base+3]])
        V_walls = np.array(walls_V, float)
        F_walls = np.array(walls_F, int)

        extra = {
            "width_top": width_top,
            "width_bottom": 2.0*half_bot,
            "area_top": abs(_polygon_area_2d(poly_top)),
            "area_bottom": abs(_polygon_area_2d(poly_bot)),
            "is_closed_path": False
        }

    groups = {
        "trench_bottom": (V_bottom, F_bottom),
        "trench_cap_for_volume": (V_cap, F_cap),
        "trench_walls": (V_walls, F_walls)
    }

    # Add inner column lid for closed paths
    if is_closed:
        groups["inner_column_lid"] = (V_lid, F_lid)

    return groups, poly_top, poly_bot, extra

def _ensure_upward_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Ensure all faces have upward-pointing normals (+z).

    For horizontal surfaces like trench floors, normals should point UP
    into the void for correct SDF computation. This function flips any
    faces with downward-pointing normals.

    Parameters
    ----------
    V : np.ndarray
        Vertices (n, 3)
    F : np.ndarray
        Faces (m, 3) - indices into V

    Returns
    -------
    np.ndarray
        Faces with consistent upward normals (may have winding flipped)
    """
    F_out = F.copy()
    p0 = V[F[:, 0]]
    p1 = V[F[:, 1]]
    p2 = V[F[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)

    # Flip faces with negative z-component normals
    down_mask = normals[:, 2] < 0
    F_out[down_mask] = F_out[down_mask, ::-1]

    return F_out


def _triangulate_polygon_fan(polygon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Triangulate a simple polygon using fan triangulation from centroid.

    Works well for convex or nearly-convex polygons. Returns (vertices, faces)
    where vertices includes the original polygon points plus the centroid.
    """
    n = len(polygon)
    centroid = polygon.mean(axis=0)

    # Vertices: polygon points first, then centroid at the end
    verts = np.vstack([polygon, centroid.reshape(1, -1)])
    centroid_idx = n

    # Fan triangles from centroid to each edge
    tris = []
    for i in range(n):
        j = (i + 1) % n
        tris.append([centroid_idx, i, j])

    return verts, np.array(tris, dtype=int)


def _triangulate_annulus(outer: np.ndarray, inner: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Triangulate the annular region between outer and inner polygons.

    Creates triangles that fill ONLY the region between the two polygons,
    leaving the inner polygon area as an open hole.

    Both polygons should be CCW oriented. Returns (vertices, faces) where
    vertices is the concatenation of outer and inner, and faces index into it.
    """
    n_outer = len(outer)
    n_inner = len(inner)

    # Vertices: outer first, then inner
    verts = np.vstack([outer, inner])

    # Create triangles by "zipping" around the two polygons
    # This works well when both polygons have similar vertex counts
    # For different counts, we need to handle the ratio

    tris = []

    # Use a marching approach: for each outer edge, connect to nearest inner vertices
    # and vice versa. This creates a proper triangulated annulus.

    # Simple approach: interpolate around both polygons simultaneously
    # treating them as having a common parameter t in [0, 1]

    i_outer = 0  # current outer vertex index
    i_inner = 0  # current inner vertex index
    t_outer = 0.0  # parameter position on outer polygon
    t_inner = 0.0  # parameter position on inner polygon

    outer_step = 1.0 / n_outer
    inner_step = 1.0 / n_inner

    # March around creating triangles
    while i_outer < n_outer or i_inner < n_inner:
        # Current vertices
        o_curr = i_outer % n_outer
        o_next = (i_outer + 1) % n_outer
        i_curr = i_inner % n_inner
        i_next = (i_inner + 1) % n_inner

        # Indices in combined vertex array
        vo_curr = o_curr
        vo_next = o_next
        vi_curr = n_outer + i_curr
        vi_next = n_outer + i_next

        if i_outer >= n_outer:
            # Finished outer, just advance inner
            tris.append([vo_curr, vi_next, vi_curr])
            i_inner += 1
            t_inner += inner_step
        elif i_inner >= n_inner:
            # Finished inner, just advance outer
            tris.append([vo_curr, vo_next, vi_curr])
            i_outer += 1
            t_outer += outer_step
        elif t_outer + outer_step <= t_inner + inner_step:
            # Advance outer - create triangle: o_curr, o_next, i_curr
            tris.append([vo_curr, vo_next, vi_curr])
            i_outer += 1
            t_outer += outer_step
        else:
            # Advance inner - create triangle: o_curr, i_next, i_curr
            tris.append([vo_curr, vi_next, vi_curr])
            i_inner += 1
            t_inner += inner_step

    return verts, np.array(tris, dtype=int)


def make_ground_surface_plane(path_xy: List[Tuple[float,float]], width_top: float, ground) -> Dict[str,Tuple[np.ndarray,np.ndarray]]:
    """Create ground surface as an offset polygon around the trench opening.

    The ground surface forms an annulus (ring) around the trench, leaving
    the trench opening as an open hole. This creates a natural shape that
    hugs L-shaped, U-shaped, and curved trenches.

    For closed paths (like circles), the ground surface is a ring around the
    outer edge of the trench, with the trench opening left completely open.
    """
    half_top = width_top / 2.0
    m = float(max(0.5, ground.size_margin))
    gfun = _ground_fn(ground)
    is_closed = _is_path_closed(path_xy)

    if is_closed:
        # For closed paths, ground is an annulus from outer boundary to trench edge.
        # Optionally, if fill_interior is set, also fill the interior island.

        # Trench outer boundary (edge of trench opening)
        trench_outer = np.array(_offset_closed_polyline(path_xy, half_top), float)

        # Ground outer boundary (edge of ground surface)
        ground_outer = np.array(_offset_closed_polyline(path_xy, half_top + m), float)

        # Ensure proper orientations
        trench_outer = _ensure_ccw(trench_outer)
        ground_outer = _ensure_ccw(ground_outer)

        # Outer ground annulus: from ground_outer to trench_outer
        outer_xy, outer_tris = _triangulate_annulus(ground_outer, trench_outer)
        Vg_outer = np.array([[x, y, gfun(x, y)] for (x, y) in outer_xy], float)
        outer_tris = _ensure_upward_normals(Vg_outer, outer_tris)

        result = {"ground_surface": (Vg_outer, outer_tris)}

        # Optionally fill the interior island (for loop trenches, not wells/pits)
        if getattr(ground, 'fill_interior', False):
            trench_inner = np.array(_offset_closed_polyline(path_xy, -half_top), float)
            trench_inner = _ensure_ccw(trench_inner)
            inner_xy, inner_tris = _triangulate_polygon_fan(trench_inner)
            Vg_inner = np.array([[x, y, gfun(x, y)] for (x, y) in inner_xy], float)
            inner_tris = _ensure_upward_normals(Vg_inner, inner_tris)
            result["ground_island"] = (Vg_inner, inner_tris)

        return result
    else:
        # Open paths: ground forms annulus with extensions past trench endpoints.
        # The outer ring (ground boundary) is extended, but the inner ring (trench
        # opening) stays at its original position. This creates ground fill at the
        # trench ends rather than extending the trench opening itself.

        # Inner boundary: trench opening at original position (not extended)
        L_inner, R_inner = _offset_polyline(path_xy, half_top)
        inner_ring = _ensure_ccw(_ring_from_LR(L_inner, R_inner))

        # Outer boundary: offset by margin, extended at ends
        L_outer, R_outer = _offset_polyline(path_xy, half_top + m)
        L_outer_ext, R_outer_ext = _extend_polyline_ends(L_outer, R_outer, path_xy, m)
        outer_ring = _ensure_ccw(_ring_from_LR(L_outer_ext, R_outer_ext))

        # Triangulate the annular region (leaves hole open)
        combined_xy, tris = _triangulate_annulus(outer_ring, inner_ring)

        # Apply ground elevation to get 3D vertices
        Vg = np.array([[x, y, gfun(x, y)] for (x, y) in combined_xy], float)

        # Ground normals should point UP (+z) into the air
        tris = _ensure_upward_normals(Vg, tris)

        return {"ground_surface": (Vg, tris)}

def _half_width_at_depth(half_top: float, slope: float, top_z: float, z: float) -> float:
    return max(1e-6, half_top - slope * (top_z - z))


# ---------------- Object Truncation Helpers ----------------


@dataclass
class TrenchLocalFrame:
    """Local coordinate frame at a point along the trench."""
    centerline_xy: np.ndarray  # 2D position on centerline
    tangent: np.ndarray        # 2D unit tangent along path
    left_normal: np.ndarray    # 2D unit normal pointing left
    top_z: float               # Ground elevation at this point
    half_width_top: float      # Half-width at ground level
    depth: float               # Trench depth
    wall_slope: float          # Slope of walls


def _find_trench_frame_at_xy(
    x: float, y: float,
    path_xy: List[Tuple[float, float]],
    half_top: float,
    depth: float,
    wall_slope: float,
    ground: GroundSpec,
) -> Tuple[TrenchLocalFrame, float]:
    """Find the trench local frame for a given XY position.

    Returns the local coordinate frame and the local 'u' offset (signed
    distance from centerline in left_normal direction).
    """
    _, total = _polyline_lengths(path_xy)

    # Find closest point on path by sampling
    best_dist_sq = float("inf")
    best_s = 0.0
    n_samples = max(200, int(total * 50))  # ~50 samples per unit length for precision
    for s_val in np.linspace(0, 1, n_samples):
        pos, _ = _sample_polyline_at_s(path_xy, s_val)
        dist_sq = (pos[0] - x) ** 2 + (pos[1] - y) ** 2
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_s = s_val

    pos, tangent = _sample_polyline_at_s(path_xy, best_s)
    left_normal = _rotate_ccw(tangent)
    gfun = _ground_fn(ground)
    top_z = gfun(pos[0], pos[1])

    # Compute local u offset
    offset_vec = np.array([x - pos[0], y - pos[1]])
    local_u = float(np.dot(offset_vec, left_normal))

    frame = TrenchLocalFrame(
        centerline_xy=pos,
        tangent=tangent,
        left_normal=left_normal,
        top_z=top_z,
        half_width_top=half_top,
        depth=depth,
        wall_slope=wall_slope,
    )
    return frame, local_u


def _point_inside_trench(
    x: float, y: float, z: float,
    path_xy: List[Tuple[float, float]],
    half_top: float,
    depth: float,
    wall_slope: float,
    ground: GroundSpec,
) -> bool:
    """Check if a 3D point is inside the trench void."""
    frame, local_u = _find_trench_frame_at_xy(x, y, path_xy, half_top, depth, wall_slope, ground)

    # Check vertical bounds
    if z > frame.top_z:
        return False  # Above ground
    if z < frame.top_z - depth:
        return False  # Below floor

    # Check horizontal bounds (accounting for wall slope)
    half_w = _half_width_at_depth(half_top, wall_slope, frame.top_z, z)
    return abs(local_u) <= half_w


@dataclass
class TruncationResult:
    """Result of computing pipe truncation."""
    neg_extent: float          # Distance from center to negative end
    pos_extent: float          # Distance from center to positive end
    neg_cap_plane: Optional[Tuple[np.ndarray, np.ndarray]]  # (normal, point) or None
    pos_cap_plane: Optional[Tuple[np.ndarray, np.ndarray]]  # (normal, point) or None
    was_truncated: bool        # True if any truncation occurred


def _compute_pipe_truncation(
    center: np.ndarray,
    axis_dir: np.ndarray,
    radius: float,
    half_length: float,
    path_xy: List[Tuple[float, float]],
    half_top: float,
    wall_slope: float,
    ground: GroundSpec,
    depth: float,
) -> TruncationResult:
    """Compute where a pipe axis exits the trench void.

    Samples points along the pipe axis and finds where the pipe surface
    (considering radius) would exit the trench. Returns truncated extents
    and the wall/floor planes at each truncation point.

    The truncation includes a safety margin to account for cap projection.
    When the pipe is truncated at an angle, the cap vertices project beyond
    the truncation point. The cap_margin ensures the entire cap stays inside.
    """
    # Cap safety margin: accounts for cap projection when pipe is at an angle
    # to the trench wall. The margin is proportional to radius with a minimum.
    cap_margin = max(0.02, 0.4 * radius)

    def pipe_surface_inside(t: float) -> bool:
        """Check if pipe surface at axis position t is inside trench.

        Includes cap_margin to ensure angled caps stay inside the boundary.
        """
        point = center + t * axis_dir
        x, y, z = point

        # Get local trench frame
        frame, local_u = _find_trench_frame_at_xy(
            x, y, path_xy, half_top, depth, wall_slope, ground
        )

        # Effective radius includes cap margin for conservative truncation
        effective_radius = radius + cap_margin

        # Check floor clearance: pipe bottom must be above floor
        floor_z = frame.top_z - depth
        if z - effective_radius < floor_z:
            return False

        # Check ceiling clearance: pipe top must be below ground
        if z + effective_radius > frame.top_z:
            return False

        # Check wall clearance: pipe surface must not penetrate walls
        half_w = _half_width_at_depth(half_top, wall_slope, frame.top_z, z)
        if abs(local_u) + effective_radius > half_w:
            return False

        return True

    def binary_search_boundary(t_inside: float, t_outside: float, tol: float = 0.001) -> float:
        """Binary search to find boundary between inside and outside."""
        for _ in range(50):  # Max iterations
            if abs(t_outside - t_inside) < tol:
                break
            t_mid = (t_inside + t_outside) / 2
            if pipe_surface_inside(t_mid):
                t_inside = t_mid
            else:
                t_outside = t_mid
        return t_inside

    # Find negative extent
    neg_extent = -half_length
    was_neg_truncated = False
    if pipe_surface_inside(0):
        # Search from center toward negative end
        step = 0.02  # 2cm steps
        t = 0
        while t > -half_length:
            t -= step
            if not pipe_surface_inside(t):
                # Found exit, binary search for exact boundary
                neg_extent = binary_search_boundary(t + step, t)
                was_neg_truncated = True
                break
        if not was_neg_truncated:
            # Never exited, use full length
            neg_extent = -half_length

    # Find positive extent
    pos_extent = half_length
    was_pos_truncated = False
    if pipe_surface_inside(0):
        step = 0.02
        t = 0
        while t < half_length:
            t += step
            if not pipe_surface_inside(t):
                pos_extent = binary_search_boundary(t - step, t)
                was_pos_truncated = True
                break
        if not was_pos_truncated:
            pos_extent = half_length

    # Compute cap planes at truncation points
    neg_cap_plane = None
    pos_cap_plane = None

    if was_neg_truncated:
        neg_cap_plane = _compute_cap_plane_at_truncation(
            center + neg_extent * axis_dir, radius,
            path_xy, half_top, wall_slope, ground, depth
        )

    if was_pos_truncated:
        pos_cap_plane = _compute_cap_plane_at_truncation(
            center + pos_extent * axis_dir, radius,
            path_xy, half_top, wall_slope, ground, depth
        )

    was_truncated = was_neg_truncated or was_pos_truncated
    return TruncationResult(
        neg_extent=neg_extent,
        pos_extent=pos_extent,
        neg_cap_plane=neg_cap_plane,
        pos_cap_plane=pos_cap_plane,
        was_truncated=was_truncated,
    )


def _compute_cap_plane_at_truncation(
    point: np.ndarray,
    radius: float,
    path_xy: List[Tuple[float, float]],
    half_top: float,
    wall_slope: float,
    ground: GroundSpec,
    depth: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the plane where the pipe intersects the trench boundary.

    Determines which boundary (left wall, right wall, floor) the pipe exits
    through and returns (normal, point) for that plane.
    """
    x, y, z = point
    frame, local_u = _find_trench_frame_at_xy(
        x, y, path_xy, half_top, depth, wall_slope, ground
    )

    floor_z = frame.top_z - depth
    half_w = _half_width_at_depth(half_top, wall_slope, frame.top_z, z)

    # Determine which boundary was hit
    dist_to_floor = z - radius - floor_z
    dist_to_ceiling = frame.top_z - (z + radius)
    dist_to_left_wall = half_w - (local_u + radius)
    dist_to_right_wall = half_w - (-local_u + radius)

    # For sloped walls, the wall normal has a horizontal and vertical component
    # Wall slope means the wall goes inward as we go down
    # Wall normal = (left_normal_2d_x, left_normal_2d_y, slope) normalized
    if wall_slope > 0:
        wall_normal_3d = np.array([
            frame.left_normal[0],
            frame.left_normal[1],
            wall_slope
        ], float)
        wall_normal_3d = wall_normal_3d / np.linalg.norm(wall_normal_3d)
    else:
        wall_normal_3d = np.array([frame.left_normal[0], frame.left_normal[1], 0.0], float)

    # Find which boundary is closest (most violated)
    min_dist = min(dist_to_floor, dist_to_ceiling, dist_to_left_wall, dist_to_right_wall)

    if min_dist == dist_to_floor or dist_to_floor < 0:
        # Hit floor - horizontal plane at floor level
        plane_normal = np.array([0.0, 0.0, 1.0], float)
        plane_point = np.array([x, y, floor_z], float)
    elif min_dist == dist_to_ceiling or dist_to_ceiling < 0:
        # Hit ceiling (ground) - horizontal plane at ground level
        plane_normal = np.array([0.0, 0.0, -1.0], float)
        plane_point = np.array([x, y, frame.top_z], float)
    elif min_dist == dist_to_left_wall or dist_to_left_wall < 0:
        # Hit left wall
        plane_normal = -wall_normal_3d  # Points inward (into trench)
        wall_x = frame.centerline_xy[0] + half_w * frame.left_normal[0]
        wall_y = frame.centerline_xy[1] + half_w * frame.left_normal[1]
        plane_point = np.array([wall_x, wall_y, z], float)
    else:
        # Hit right wall
        plane_normal = wall_normal_3d  # Flip for right wall
        plane_normal[0] = -plane_normal[0]
        plane_normal[1] = -plane_normal[1]
        wall_x = frame.centerline_xy[0] - half_w * frame.left_normal[0]
        wall_y = frame.centerline_xy[1] - half_w * frame.left_normal[1]
        plane_point = np.array([wall_x, wall_y, z], float)

    return plane_normal, plane_point


def _compute_box_fit(
    center: np.ndarray,
    along: float,
    across: float,
    height: float,
    path_xy: List[Tuple[float, float]],
    half_top: float,
    wall_slope: float,
    ground: GroundSpec,
    depth: float,
    clearance: float = 0.02,
) -> Tuple[float, float, float]:
    """Compute shrunk box dimensions to fit within trench.

    Returns (along, across, height) that fit within the trench at the given
    center position.
    """
    x, y, z = center
    frame, local_u = _find_trench_frame_at_xy(
        x, y, path_xy, half_top, depth, wall_slope, ground
    )

    floor_z = frame.top_z - depth

    # Max height: from floor to ground, minus clearance
    max_height = (frame.top_z - floor_z) - 2 * clearance
    fit_height = min(height, max_height)

    # Recompute z to center the box if height was shrunk
    if fit_height < height:
        # Center box vertically in trench
        z_new = floor_z + clearance + fit_height / 2
    else:
        z_new = z

    # At the box center depth, what's the half-width?
    half_w = _half_width_at_depth(half_top, wall_slope, frame.top_z, z_new)

    # Max across: must fit within walls accounting for offset from centerline
    # The box extends ±across/2 from center, so:
    # local_u + across/2 <= half_w - clearance
    # local_u - across/2 >= -(half_w - clearance)
    # => across/2 <= min(half_w - clearance - local_u, half_w - clearance + local_u)
    max_across = 2 * (half_w - clearance - abs(local_u))
    fit_across = max(0.01, min(across, max_across))

    # Along dimension: no shrinking needed for typical cases
    # (pipes along trench axis can be arbitrarily long within path bounds)
    fit_along = along

    return fit_along, fit_across, fit_height


def _compute_sphere_fit(
    center: np.ndarray,
    radius: float,
    path_xy: List[Tuple[float, float]],
    half_top: float,
    wall_slope: float,
    ground: GroundSpec,
    depth: float,
    clearance: float = 0.02,
) -> float:
    """Compute shrunk sphere radius to fit within trench.

    Returns the maximum radius that fits within the trench at the given center.
    """
    x, y, z = center
    frame, local_u = _find_trench_frame_at_xy(
        x, y, path_xy, half_top, depth, wall_slope, ground
    )

    floor_z = frame.top_z - depth

    # Distance to floor
    dist_floor = z - floor_z - clearance

    # Distance to ground
    dist_ground = frame.top_z - z - clearance

    # Distance to nearest wall (accounting for offset)
    half_w = _half_width_at_depth(half_top, wall_slope, frame.top_z, z)
    dist_wall = half_w - abs(local_u) - clearance

    # Maximum radius is minimum of all distances
    max_radius = max(0.01, min(dist_floor, dist_ground, dist_wall))

    return min(radius, max_radius)


def _clip_vertices_to_trench(
    V: np.ndarray,
    path_xy: List[Tuple[float, float]],
    half_top: float,
    wall_slope: float,
    ground: GroundSpec,
    depth: float,
) -> np.ndarray:
    """Clip vertices to stay inside the trench boundary.

    Any vertex outside the trench is projected back to the nearest boundary.
    This handles edge cases where cap projection pushes vertices outside.
    """
    V_clipped = V.copy()

    for i, vert in enumerate(V):
        x, y, z = vert
        frame, local_u = _find_trench_frame_at_xy(
            x, y, path_xy, half_top, depth, wall_slope, ground
        )

        floor_z = frame.top_z - depth
        half_w = _half_width_at_depth(half_top, wall_slope, frame.top_z, z)

        # Clip z to floor/ceiling
        z_clipped = np.clip(z, floor_z, frame.top_z)

        # Clip u to walls (need to update xy)
        if abs(local_u) > half_w:
            # Project point back to wall
            u_clipped = np.sign(local_u) * half_w
            # Adjust xy: move toward centerline
            delta_u = u_clipped - local_u
            x_clipped = x + delta_u * frame.left_normal[0]
            y_clipped = y + delta_u * frame.left_normal[1]
        else:
            x_clipped = x
            y_clipped = y

        V_clipped[i] = [x_clipped, y_clipped, z_clipped]

    return V_clipped


# ---------------- Noise ----------------

def vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    n=np.zeros_like(V)
    p0=V[F[:,0]]; p1=V[F[:,1]]; p2=V[F[:,2]]
    fn=np.cross(p1-p0,p2-p0)
    for i in range(3): np.add.at(n, F[:,i], fn)
    norms=np.linalg.norm(n,axis=1); norms[norms==0]=1.0
    return n / norms[:,None]

def smooth_noise_field(points: np.ndarray, seed: int, corr_length: float, octaves: int=2, gain: float=0.5) -> np.ndarray:
    rng=np.random.default_rng(seed); K=7
    val=np.zeros(points.shape[0],float)
    base_k=2.0*np.pi/max(corr_length,1e-6)
    for o in range(octaves):
        kscale=(2**o)*base_k; amp=(gain**o)
        ks=rng.normal(size=(K,3)); ks=ks/np.linalg.norm(ks,axis=1)[:,None]*kscale
        phase=rng.uniform(0,2*np.pi,size=(K,))
        proj=points@ks.T
        val += amp * np.sum(np.cos(proj + phase), axis=1) / K
    return val

def apply_vertex_noise(groups: Dict[str, Tuple[np.ndarray, np.ndarray]], patterns: List[str],
                       amplitude: float, seed: int, corr_length: float, octaves:int=2, gain:float=0.5):
    import fnmatch
    out={}
    for name,(V,F) in groups.items():
        if any(fnmatch.fnmatch(name, pat) for pat in patterns):
            nrm=vertex_normals(V,F)
            field=smooth_noise_field(V, seed, corr_length, octaves, gain)
            Vn=V + (amplitude*field)[:,None]*nrm
            out[name]=(Vn, F.copy())
        else:
            out[name]=(V.copy(), F.copy())
    return out

# --------------- Scene builder ---------------

def _build_surface_groups(
    spec: SceneSpec,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, int], Dict[str, Any]]:
    groups: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    trench_groups, _, _, extra = make_trench_from_path_sloped(
        spec.path_xy, spec.width, spec.depth, spec.wall_slope, spec.ground
    )
    groups.update(trench_groups)

    if spec.ground and spec.ground.size_margin > 0:
        groups.update(make_ground_surface_plane(spec.path_xy, spec.width, spec.ground))
    else:
        L, R = _offset_polyline(spec.path_xy, spec.width / 2.0)
        gfun = _ground_fn(spec.ground)

        def tri_quad_ccw(v0, v1, v2, v3):
            poly = np.array([v0, v1, v2, v3], float)
            if _polygon_area_2d(poly[:, :2]) < 0:
                poly = poly[::-1]
            return np.array([[0, 1, 2], [0, 2, 3]], int), poly

        V_left: List[List[float]] = []
        F_left: List[List[int]] = []
        for i in range(len(L) - 1):
            v0 = [L[i][0], L[i][1], gfun(*L[i])]
            v1 = [L[i + 1][0], L[i + 1][1], gfun(*L[i + 1])]
            v2 = [spec.path_xy[i + 1][0], spec.path_xy[i + 1][1], gfun(*spec.path_xy[i + 1])]
            v3 = [spec.path_xy[i][0], spec.path_xy[i][1], gfun(*spec.path_xy[i])]
            tris, poly = tri_quad_ccw(v0, v1, v2, v3)
            base = len(V_left)
            V_left += poly.tolist()
            F_left += (tris + base).tolist()
        if V_left:
            groups["ground_left_strip"] = (np.array(V_left, float), np.array(F_left, int))

        V_right: List[List[float]] = []
        F_right: List[List[int]] = []
        for i in range(len(R) - 1):
            v0 = [spec.path_xy[i][0], spec.path_xy[i][1], gfun(*spec.path_xy[i])]
            v1 = [spec.path_xy[i + 1][0], spec.path_xy[i + 1][1], gfun(*spec.path_xy[i + 1])]
            v2 = [R[i + 1][0], R[i + 1][1], gfun(*R[i + 1])]
            v3 = [R[i][0], R[i][1], gfun(*R[i])]
            tris, poly = tri_quad_ccw(v0, v1, v2, v3)
            base = len(V_right)
            V_right += poly.tolist()
            F_right += (tris + base).tolist()
        if V_right:
            groups["ground_right_strip"] = (np.array(V_right, float), np.array(F_right, int))

    half_top = spec.width * 0.5
    gfun = _ground_fn(spec.ground)
    clearance = 0.02

    for idx, p in enumerate(spec.pipes):
        pos_xy, tangent = _sample_polyline_at_s(spec.path_xy, p.s_center)
        angle = math.radians(p.angle_deg)
        t_rot = np.array(
            [
                math.cos(angle) * tangent[0] - math.sin(angle) * tangent[1],
                math.sin(angle) * tangent[0] + math.cos(angle) * tangent[1],
            ],
            float,
        )
        axis_dir = np.array([t_rot[0], t_rot[1], 0.0], float)
        left_normal = _rotate_ccw(tangent)
        top_z = gfun(pos_xy[0], pos_xy[1])
        req_u = float(p.offset_u)
        req_z = float(p.z if p.z is not None else (top_z - spec.depth * 0.5))

        # Cap margin accounts for cap projection when pipe is at angle to wall
        cap_margin = max(0.02, 0.4 * p.radius)
        effective_radius = p.radius + cap_margin

        z_min = top_z - spec.depth + (effective_radius + clearance)
        z_max = top_z - (effective_radius + clearance)
        zc = float(np.clip(req_z, z_min, z_max))
        half_w = _half_width_at_depth(half_top, spec.wall_slope, top_z, zc)
        umax = max(0.0, half_w - (effective_radius + clearance))
        u = float(np.clip(req_u, -umax, umax))
        ctr_xy = pos_xy + u * left_normal
        center = np.array([ctr_xy[0], ctr_xy[1], zc], float)

        # Compute pipe truncation at trench boundaries
        trunc = _compute_pipe_truncation(
            center, axis_dir, p.radius, p.length / 2.0,
            spec.path_xy, half_top, spec.wall_slope, spec.ground, spec.depth
        )

        cyl = make_cylinder(
            center, axis_dir, p.radius, p.length,
            p.n_theta, p.n_along, with_caps=True,
            neg_extent=trunc.neg_extent,
            pos_extent=trunc.pos_extent,
            cap_plane_neg=trunc.neg_cap_plane,
            cap_plane_pos=trunc.pos_cap_plane,
        )
        for key, (V, F) in cyl.items():
            # Clip cap vertices to trench boundary to handle projection overshoot
            if "cap" in key:
                V = _clip_vertices_to_trench(
                    V, spec.path_xy, half_top, spec.wall_slope, spec.ground, spec.depth
                )
            groups[f"pipe{idx}_{key}"] = (V, F)

    for j, b in enumerate(spec.boxes):
        pos_xy, tangent = _sample_polyline_at_s(spec.path_xy, b.s)
        left_normal = _rotate_ccw(tangent)
        top_z = gfun(pos_xy[0], pos_xy[1])
        req_u = float(b.offset_u)
        req_z = float(b.z if b.z is not None else (top_z - spec.depth + b.height * 0.5))
        z_min = top_z - spec.depth + (b.height * 0.5 + clearance)
        z_max = top_z - (b.height * 0.5 + clearance)
        zc = float(np.clip(req_z, z_min, z_max))
        half_w = _half_width_at_depth(half_top, spec.wall_slope, top_z, zc)
        umax = max(0.0, half_w - (b.across * 0.5 + clearance))
        u = float(np.clip(req_u, -umax, umax))
        ctr_xy = pos_xy + u * left_normal
        center = np.array([ctr_xy[0], ctr_xy[1], zc], float)

        # Compute shrunk box dimensions to fit within trench
        fit_along, fit_across, fit_height = _compute_box_fit(
            center, b.along, b.across, b.height,
            spec.path_xy, half_top, spec.wall_slope, spec.ground, spec.depth, clearance
        )

        # Re-center if height was shrunk
        if fit_height < b.height:
            floor_z = top_z - spec.depth
            zc = floor_z + clearance + fit_height / 2
            center = np.array([ctr_xy[0], ctr_xy[1], zc], float)

        frame_cols = np.column_stack(
            [
                np.array([tangent[0], tangent[1], 0.0]),
                np.array([left_normal[0], left_normal[1], 0.0]),
                np.array([0.0, 0.0, 1.0]),
            ]
        )
        Vb, Fb = make_box(center, frame_cols, (fit_along, fit_across, fit_height))
        groups[f"box{j}"] = (Vb, Fb)

    for k, s in enumerate(spec.spheres):
        pos_xy, tangent = _sample_polyline_at_s(spec.path_xy, s.s)
        left_normal = _rotate_ccw(tangent)
        top_z = gfun(pos_xy[0], pos_xy[1])
        req_u = float(s.offset_u)
        req_z = float(s.z if s.z is not None else (top_z - spec.depth + s.radius))
        z_min = top_z - spec.depth + (s.radius + clearance)
        z_max = top_z - (s.radius + clearance)
        zc = float(np.clip(req_z, z_min, z_max))
        half_w = _half_width_at_depth(half_top, spec.wall_slope, top_z, zc)
        umax = max(0.0, half_w - (s.radius + clearance))
        u = float(np.clip(req_u, -umax, umax))
        ctr_xy = pos_xy + u * left_normal
        center = np.array([ctr_xy[0], ctr_xy[1], zc], float)

        # Compute shrunk sphere radius to fit within trench
        fit_radius = _compute_sphere_fit(
            center, s.radius,
            spec.path_xy, half_top, spec.wall_slope, spec.ground, spec.depth, clearance
        )

        # Re-center if radius was shrunk significantly
        if fit_radius < s.radius:
            floor_z = top_z - spec.depth
            # Center the smaller sphere optimally
            zc = floor_z + clearance + fit_radius
            zc = min(zc, top_z - clearance - fit_radius)  # Also respect ceiling
            center = np.array([ctr_xy[0], ctr_xy[1], zc], float)

        Vs, Fs = make_sphere(center, fit_radius, n_theta=64, n_phi=32)
        groups[f"sphere{k}"] = (Vs, Fs)

    if spec.noise and spec.noise.enable:
        groups = apply_vertex_noise(
            groups,
            list(spec.noise.apply_to),
            amplitude=spec.noise.amplitude,
            seed=spec.noise.seed,
            corr_length=spec.noise.corr_length,
            octaves=spec.noise.octaves,
            gain=spec.noise.gain,
        )

    object_counts = {
        "pipes": len(spec.pipes),
        "boxes": len(spec.boxes),
        "spheres": len(spec.spheres),
    }
    return groups, object_counts, extra


def generate_surface_mesh(spec: SceneSpec, *, make_preview: bool = False) -> SurfaceMeshResult:
    groups, object_counts, extra = _build_surface_groups(spec)
    metrics = _compute_surface_metrics(groups, extra, spec)
    # Exclude internal groups (like cap) from previews to show open-topped trenches
    previews = _render_surface_previews(groups, exclude_groups=_INTERNAL_GROUPS) if make_preview else {}
    return SurfaceMeshResult(
        spec=spec,
        groups=groups,
        object_counts=object_counts,
        metrics=metrics,
        previews=previews,
    )


def build_scene(spec: SceneSpec, out_dir: str, make_preview=False):
    result = generate_surface_mesh(spec, make_preview=make_preview)
    files = result.persist(out_dir, include_previews=make_preview)
    return {
        "obj_path": files.obj_path.as_posix(),
        "metrics": result.metrics,
        "previews": [p.as_posix() for p in files.preview_paths],
        "object_counts": result.object_counts,
        "surface_result": result,
    }

# ---------------- CLI ----------------

def scene_spec_from_dict(cfg: Dict[str, Any]) -> SceneSpec:
    pipes=[PipeSpec(**p) for p in cfg.get("pipes", [])]
    boxes=[BoxSpec(**b) for b in cfg.get("boxes", [])]
    spheres=[SphereSpec(**s) for s in cfg.get("spheres", [])]
    noise_cfg = cfg.get("noise", {})
    noise = NoiseSpec(**noise_cfg) if noise_cfg else NoiseSpec(enable=False)
    ground_cfg = cfg.get("ground", {})
    ground = GroundSpec(**ground_cfg) if ground_cfg else GroundSpec()
    return SceneSpec(path_xy=[tuple(map(float, p)) for p in cfg["path_xy"]],
                     width=float(cfg["width"]), depth=float(cfg["depth"]),
                     wall_slope=float(cfg.get("wall_slope", 0.0)),
                     ground_margin=float(cfg.get("ground_margin", 0.0)),
                     pipes=pipes, boxes=boxes, spheres=spheres, noise=noise, ground=ground)

def load_scene_spec_from_json(path: str) -> SceneSpec:
    with open(path,"r") as f: cfg=json.load(f)
    return scene_spec_from_dict(cfg)

def main():
    ap=argparse.ArgumentParser(description="Synthetic trench scene (surface, sloped walls, grounded)")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preview", action="store_true")
    args=ap.parse_args()
    spec=load_scene_spec_from_json(args.spec)
    out=build_scene(spec, args.out, make_preview=args.preview)
    response = {
        "obj_path": out["obj_path"],
        "metrics_path": os.path.join(args.out, "metrics.json"),
        "objects": out["object_counts"],
        "previews": out["previews"],
        "preview_count": len(out["previews"]),
        "footprint_top": out["metrics"]["footprint_area_top"],
        "footprint_bottom": out["metrics"]["footprint_area_bottom"],
        "trench_from_surface": out["metrics"]["volumes"]["trench_from_surface"]
    }
    if args.preview and plt is None:
        response["preview_note"] = "matplotlib_unavailable"
    print(json.dumps(response, indent=2))

if __name__ == "__main__":
    main()
