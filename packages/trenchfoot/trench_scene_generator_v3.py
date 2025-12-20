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
import os, json, math, argparse, re
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
_INTERNAL_GROUPS = frozenset({"trench_cap_for_volume"})

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


@dataclass
class SurfaceMeshResult:
    spec: SceneSpec
    groups: Dict[str, Tuple[np.ndarray, np.ndarray]]
    object_counts: Dict[str, int]
    metrics: Dict[str, Any]
    previews: Dict[str, bytes]

    def persist(self, out_dir: str | Path, *, include_previews: bool = False) -> SurfaceMeshFiles:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        obj_path = out_path / "trench_scene.obj"
        # Exclude internal groups (like trench_cap_for_volume) from OBJ export
        export_groups = {k: v for k, v in self.groups.items() if k not in _INTERNAL_GROUPS}
        write_obj_with_groups(obj_path.as_posix(), export_groups)
        metrics_path = out_path / "metrics.json"
        with metrics_path.open("w") as fh:
            json.dump(self.metrics, fh, indent=2)
        preview_paths: List[Path] = []
        if include_previews and self.previews:
            for name, data in self.previews.items():
                target = out_path / f"preview_{name}.png"
                target.write_bytes(data)
                preview_paths.append(target)
        return SurfaceMeshFiles(obj_path=obj_path, metrics_path=metrics_path, preview_paths=tuple(preview_paths))

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

def make_cylinder(center: np.ndarray, axis_dir: np.ndarray, radius: float, length: float,
                  n_theta: int=64, n_along: int=32, with_caps: bool=True):
    n_theta=max(8,int(n_theta)); n_along=max(1,int(n_along))
    thetas=np.linspace(0,2*np.pi,n_theta+1); ys=np.linspace(-length/2.0,length/2.0,n_along+1)
    Vloc=[]
    for j in range(n_along+1):
        y=ys[j]
        for i in range(n_theta+1):
            th=thetas[i]; x=radius*np.cos(th); z=radius*np.sin(th)
            Vloc.append([x,y,z])
    Vloc=np.array(Vloc,float)
    def idx(i,j): return j*(n_theta+1)+i
    F=[]
    for j in range(n_along):
        for i in range(n_theta):
            v00=idx(i,j); v10=idx(i+1,j); v01=idx(i,j+1); v11=idx(i+1,j+1)
            F.append([v00,v01,v11]); F.append([v00,v11,v10])
    F=np.array(F,int)
    caps={}
    if with_caps:
        ring=np.array([[radius*np.cos(t),-length/2.0,radius*np.sin(t)] for t in thetas[:-1]],float)
        Vn=np.vstack([np.array([[0.0,-length/2.0,0.0]],float), ring])
        Fn=np.array([[0,1+(i+1)%len(ring),1+i] for i in range(len(ring))],int)
        ring=np.array([[radius*np.cos(t),+length/2.0,radius*np.sin(t)] for t in thetas[:-1]],float)
        Vp=np.vstack([np.array([[0.0,+length/2.0,0.0]],float), ring])
        Fp=np.array([[0,1+i,1+(i+1)%len(ring)] for i in range(len(ring))],int)
        caps['pipe_cap_neg']=(Vn,Fn); caps['pipe_cap_pos']=(Vp,Fp)
    M=_frame_from_axis(axis_dir)
    def xform(V): return (center + V @ M.T).astype(float)
    out={"pipe_side": (xform(Vloc), F)}
    if with_caps:
        Vn,Fn=caps['pipe_cap_neg']; Vp,Fp=caps['pipe_cap_pos']
        out['pipe_cap_neg']=(xform(Vn),Fn); out['pipe_cap_pos']=(xform(Vp),Fp)
    return out

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
        F_bottom = bot_faces[:, ::-1]  # flip for outward normals

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
        F_bottom = tris_bot[:, ::-1]  # outward

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

    return groups, poly_top, poly_bot, extra

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
        # For closed paths (like circular wells), the ground is an annulus
        # from the outer ground boundary to the outer edge of the trench opening.
        # The center (inside the trench) is left completely open.

        # Trench outer boundary (edge of trench opening)
        trench_outer = np.array(_offset_closed_polyline(path_xy, half_top), float)

        # Ground outer boundary (edge of ground surface)
        ground_outer = np.array(_offset_closed_polyline(path_xy, half_top + m), float)

        # Ensure proper orientations
        trench_outer = _ensure_ccw(trench_outer)
        ground_outer = _ensure_ccw(ground_outer)

        # Ground annulus: from ground_outer to trench_outer
        combined_xy, tris = _triangulate_annulus(ground_outer, trench_outer)
        Vg = np.array([[x, y, gfun(x, y)] for (x, y) in combined_xy], float)

        return {"ground_surface": (Vg, tris)}
    else:
        # Original logic for open paths
        # Inner boundary: the trench opening (same as trench top ring)
        L_inner, R_inner = _offset_polyline(path_xy, half_top)
        inner_ring = _ensure_ccw(_ring_from_LR(L_inner, R_inner))

        # Outer boundary: offset by additional margin
        L_outer, R_outer = _offset_polyline(path_xy, half_top + m)
        outer_ring = _ensure_ccw(_ring_from_LR(L_outer, R_outer))

        # Triangulate the annular region (leaves hole open)
        combined_xy, tris = _triangulate_annulus(outer_ring, inner_ring)

        # Apply ground elevation to get 3D vertices
        Vg = np.array([[x, y, gfun(x, y)] for (x, y) in combined_xy], float)

        return {"ground_surface": (Vg, tris)}

def _half_width_at_depth(half_top: float, slope: float, top_z: float, z: float) -> float:
    return max(1e-6, half_top - slope * (top_z - z))

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
        z_min = top_z - spec.depth + (p.radius + clearance)
        z_max = top_z - (p.radius + clearance)
        zc = float(np.clip(req_z, z_min, z_max))
        half_w = _half_width_at_depth(half_top, spec.wall_slope, top_z, zc)
        umax = max(0.0, half_w - (p.radius + clearance))
        u = float(np.clip(req_u, -umax, umax))
        ctr_xy = pos_xy + u * left_normal
        center = np.array([ctr_xy[0], ctr_xy[1], zc], float)
        cyl = make_cylinder(center, axis_dir, p.radius, p.length, p.n_theta, p.n_along, with_caps=True)
        for key, (V, F) in cyl.items():
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
        frame_cols = np.column_stack(
            [
                np.array([tangent[0], tangent[1], 0.0]),
                np.array([left_normal[0], left_normal[1], 0.0]),
                np.array([0.0, 0.0, 1.0]),
            ]
        )
        Vb, Fb = make_box(center, frame_cols, (b.along, b.across, b.height))
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
        Vs, Fs = make_sphere(center, s.radius, n_theta=64, n_phi=32)
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
