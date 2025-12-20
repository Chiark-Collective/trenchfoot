#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gmsh_sloped_trench_mesher.py
Ground-aware volumetric mesher with sloped walls and conformal pipes.
Usage:
  python gmsh_sloped_trench_mesher.py --spec scene.json --out ./vol --lc 0.3
"""
import json, math, os, sys, argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import gmsh  # pip install gmsh

PIPE_CLEARANCE_BASE = 0.05  # baseline minimum (metres) between pipe surfaces and trench walls


@dataclass(frozen=True)
class VolumeElementBlock:
    gmsh_type: int
    element_tags: np.ndarray
    node_tags: np.ndarray


@dataclass(frozen=True)
class PhysicalGroupInfo:
    dimension: int
    tag: int
    name: str
    entity_tags: Tuple[int, ...]
    element_tags: Dict[int, np.ndarray]


@dataclass
class VolumeMeshResult:
    node_tags: np.ndarray
    nodes: np.ndarray
    element_blocks: List[VolumeElementBlock]
    physical_groups: List[PhysicalGroupInfo]
    pipe_clearances: List[Dict[str, object]]
    persisted_path: Optional[Path]
    mesh_characteristic_length: Optional[float]

def _normalize(v): 
    n = np.linalg.norm(v); 
    return v if n==0 else v/n

def _rotate_ccw(v): return np.array([-v[1], v[0]], float)

def _is_path_closed(path, threshold=0.01):
    """Detect if path is explicitly closed (first and last points nearly identical)."""
    if len(path) < 3:
        return False
    P = np.array(path, float)
    first_last_dist = np.linalg.norm(P[0] - P[-1])
    return first_last_dist < threshold

def _offset_closed_polyline(path, offset):
    """Offset a closed polyline, returning a single closed ring."""
    P = np.array(path, float)
    n = len(P)
    if n < 3:
        raise ValueError("Closed polyline needs at least 3 points")
    tangents = []
    normals = []
    for i in range(n):
        t = _normalize(P[(i+1) % n] - P[i])
        if np.linalg.norm(t) < 1e-12:
            t = np.array([1.0, 0.0])
        tangents.append(t)
        normals.append(np.array([-t[1], t[0]], float))
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

def _line_intersection_2d(p, d, q, e):
    M = np.array([d, -e], float).T; det = np.linalg.det(M)
    if abs(det) < 1e-12: return None
    t, s = np.linalg.solve(M, (q - p)); return p + t * d

def _offset_polyline(path, offset):
    P = np.array(path, float); n = len(P)
    tangents=[]; normals=[]
    for i in range(n-1):
        t = _normalize(P[i+1]-P[i]); 
        if np.linalg.norm(t) < 1e-12: t = np.array([1.0, 0.0])
        tangents.append(t); normals.append(np.array([-t[1], t[0]], float))
    L=[P[0] + offset*normals[0]]; R=[P[0] - offset*normals[0]]
    for k in range(1,n-1):
        t_prev, n_prev = tangents[k-1], normals[k-1]
        t_next, n_next = tangents[k], normals[k]
        L1_p = P[k] + offset * n_prev; L1_d = t_prev
        L2_p = P[k] + offset * n_next; L2_d = t_next
        R1_p = P[k] - offset * n_prev; R1_d = t_prev
        R2_p = P[k] - offset * n_next; R2_d = t_next
        Lp = _line_intersection_2d(L1_p, L1_d, L2_p, L2_d)
        Rp = _line_intersection_2d(R1_p, R1_d, R2_p, R2_d)
        if Lp is None: Lp = 0.5*(L1_p+L2_p)
        if Rp is None: Rp = 0.5*(R1_p+R2_p)
        L.append(Lp); R.append(Rp)
    L.append(P[-1] + offset*normals[-1])
    R.append(P[-1] - offset*normals[-1])
    return L, R

def _ring_from_LR(L, R): return np.array(L + list(R[::-1]), float)

def _add_closed_wire_xyz(points_xyz):
    pt = [gmsh.model.occ.addPoint(float(x), float(y), float(z)) for x,y,z in points_xyz]
    lines = []
    for i in range(len(pt)):
        a = pt[i]; b = pt[(i+1) % len(pt)]
        lines.append(gmsh.model.occ.addLine(a, b))
    loop = gmsh.model.occ.addCurveLoop(lines)
    return loop

def _sample_polyline_at_s(path, s):
    P = np.array(path, float)
    segs = P[1:] - P[:-1]; lens = np.linalg.norm(segs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(lens)]); total = float(cum[-1])
    if total == 0: return P[0], np.array([1.0, 0.0])
    s_abs = s * total; i = int(np.searchsorted(cum, s_abs, side="right") - 1); i = max(0, min(i, len(P)-2))
    seg = P[i+1] - P[i]; L = np.linalg.norm(seg)
    if L == 0: return P[i], np.array([1.0, 0.0])
    u = (s_abs - cum[i]) / L; pos = (1-u)*P[i] + u*P[i+1]; t = seg / L; 
    return pos, t

def _polyline_lengths(path):
    P = np.array(path, float)
    if len(P) < 2:
        return np.array([0.0]), 0.0
    segs = P[1:] - P[:-1]
    lens = np.linalg.norm(segs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(lens)])
    return cum, float(cum[-1])

def _line_segment_intersection_param(p, direction, a, b):
    seg = np.array(b, float) - np.array(a, float)
    M = np.array([direction, -seg]).T
    det = np.linalg.det(M)
    if abs(det) < 1e-12:
        return None
    t, u = np.linalg.solve(M, np.array(a, float) - np.array(p, float))
    if -1e-9 <= u <= 1 + 1e-9:
        return float(t)
    return None

def _max_extent_in_polygon(center, direction, poly_xy, clearance):
    direction = _normalize(np.array(direction, float))
    pos_intersections = []
    neg_intersections = []
    m = len(poly_xy)
    for i in range(m):
        a = poly_xy[i]
        b = poly_xy[(i + 1) % m]
        t = _line_segment_intersection_param(center, direction, a, b)
        if t is None:
            continue
        if t > 0:
            pos_intersections.append(t)
        elif t < 0:
            neg_intersections.append(t)
    if not pos_intersections or not neg_intersections:
        return math.inf
    positive_min = min(pos_intersections)
    negative_max = max(neg_intersections)
    return max(0.0, min(positive_min, -negative_max) - clearance)

def _pipe_clearance(radius: float, wall_slope: float) -> float:
    """Adaptive clearance: scale guard band with pipe radius and wall slope."""
    radius_term = radius * 0.35
    slope_term = abs(float(wall_slope)) * 0.02 * radius
    return max(PIPE_CLEARANCE_BASE, radius_term + slope_term)

def generate_trench_volume(
    cfg,
    *,
    lc=0.3,
    persist_path: Optional[str | Path] = None,
    finalize: bool = True,
    debug_callback: Optional[Callable[[Dict[str, object]], None]] = None,
    debug_export: Optional[str] = None,
):
    gmsh.initialize()
    persist_path_obj = Path(persist_path) if persist_path is not None else None
    try:
        gmsh.model.add("trench_volume")

        path_xy = [tuple(map(float, p)) for p in cfg["path_xy"]]
        width_top = float(cfg["width"])
        depth = float(cfg["depth"])
        slope = float(cfg.get("wall_slope", 0.0))
        ground_cfg = cfg.get("ground", {})
        z0 = float(ground_cfg.get("z0", 0.0))
        sx, sy = tuple(ground_cfg.get("slope", (0.0, 0.0)))

        def g(x, y): return z0 + sx*float(x) + sy*float(y)

        half_top = width_top / 2.0
        half_bot = max(1e-3, half_top - slope * depth)

        is_closed = _is_path_closed(path_xy)

        if is_closed:
            # For closed paths, create annular (ring-shaped) trench with outer and inner walls
            outer_top = np.array(_offset_closed_polyline(path_xy, half_top), float)
            inner_top = np.array(_offset_closed_polyline(path_xy, -half_top), float)
            outer_bot = np.array(_offset_closed_polyline(path_xy, half_bot), float)
            inner_bot = np.array(_offset_closed_polyline(path_xy, -half_bot), float)

            outer_top_xyz = [(x, y, g(x, y)) for (x, y) in outer_top]
            inner_top_xyz = [(x, y, g(x, y)) for (x, y) in inner_top]
            outer_bot_xyz = [(x, y, g(x, y) - depth) for (x, y) in outer_bot]
            inner_bot_xyz = [(x, y, g(x, y) - depth) for (x, y) in inner_bot]

            # Create outer and inner wire loops at top and bottom
            outer_top_loop = _add_closed_wire_xyz(outer_top_xyz)
            inner_top_loop = _add_closed_wire_xyz(inner_top_xyz)
            outer_bot_loop = _add_closed_wire_xyz(outer_bot_xyz)
            inner_bot_loop = _add_closed_wire_xyz(inner_bot_xyz)

            # Outer wall: loft from outer_top to outer_bot
            outDimTags_outer = []
            try:
                gmsh.model.occ.addThruSections(
                    [outer_top_loop, outer_bot_loop], makeSolid=False, makeRuled=True, outDimTags=outDimTags_outer
                )
            except TypeError:
                outDimTags_outer = gmsh.model.occ.addThruSections(
                    [outer_top_loop, outer_bot_loop], makeSolid=False, makeRuled=True
                ) or []

            # Inner wall: loft from inner_top to inner_bot
            outDimTags_inner = []
            try:
                gmsh.model.occ.addThruSections(
                    [inner_top_loop, inner_bot_loop], makeSolid=False, makeRuled=True, outDimTags=outDimTags_inner
                )
            except TypeError:
                outDimTags_inner = gmsh.model.occ.addThruSections(
                    [inner_top_loop, inner_bot_loop], makeSolid=False, makeRuled=True
                ) or []

            # Create top and bottom surfaces (annular)
            top_surf = gmsh.model.occ.addPlaneSurface([outer_top_loop, inner_top_loop])
            bot_surf = gmsh.model.occ.addPlaneSurface([outer_bot_loop, inner_bot_loop])

            # Collect all surfaces to form a shell
            all_surfs = [top_surf, bot_surf]
            all_surfs.extend([tag for (dim, tag) in outDimTags_outer if dim == 2])
            all_surfs.extend([tag for (dim, tag) in outDimTags_inner if dim == 2])

            # Create surface loop and volume
            gmsh.model.occ.synchronize()
            surf_loop = gmsh.model.occ.addSurfaceLoop(all_surfs)
            trench_vol = gmsh.model.occ.addVolume([surf_loop])

            gmsh.model.occ.healShapes()
            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()

            # Use outer_top for ring_top in clearance calculations
            ring_top = outer_top
        else:
            # Original logic for open paths
            Ltop, Rtop = _offset_polyline(path_xy, half_top)
            Lbot, Rbot = _offset_polyline(path_xy, half_bot)
            ring_top = _ring_from_LR(Ltop, Rtop)
            ring_bot = _ring_from_LR(Lbot, Rbot)

            ring_top_xyz = [(x, y, g(x, y)) for (x, y) in ring_top]
            ring_bot_xyz = [(x, y, g(x, y) - depth) for (x, y) in ring_bot]

            top_loop = _add_closed_wire_xyz(ring_top_xyz)
            bot_loop = _add_closed_wire_xyz(ring_bot_xyz)

            outDimTags = []
            try:
                gmsh.model.occ.addThruSections(
                    [top_loop, bot_loop], makeSolid=True, makeRuled=True, outDimTags=outDimTags
                )
            except TypeError:  # gmsh >= 4.14 removed outDimTags kwarg
                outDimTags = gmsh.model.occ.addThruSections(
                    [top_loop, bot_loop], makeSolid=True, makeRuled=True
                )
                if outDimTags is None:
                    outDimTags = []
            gmsh.model.occ.healShapes()
            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()

            vols = [tag for (dim, tag) in outDimTags if dim == 3]
            assert len(vols) >= 1, "Loft did not create a volume"
            trench_vol = vols[0]

        pipe_cfgs = cfg.get("pipes", [])
        cum_lengths, total_length = _polyline_lengths(path_xy)
        pipe_dimtags = []
        clearance_records: List[Dict[str, object]] = []
        for i, p in enumerate(pipe_cfgs):
            radius = float(p["radius"])
            orig_length = float(p["length"])
            angle = math.radians(float(p["angle_deg"]))
            s_center = float(p.get("s_center", 0.5))
            zc = float(p.get("z", -depth * 0.5))
            offset_u = float(p.get("offset_u", 0.0))
            clearance_scale = float(p.get("clearance_scale", 1.0))
            if not math.isfinite(clearance_scale) or clearance_scale <= 0:
                raise ValueError(f"clearance_scale for pipe[{i}] must be > 0")
            pos_xy, tangent = _sample_polyline_at_s(path_xy, s_center)
            axis_xy = _normalize(np.array(
                [
                    math.cos(angle) * tangent[0] - math.sin(angle) * tangent[1],
                    math.sin(angle) * tangent[0] + math.cos(angle) * tangent[1],
                ],
                float,
            ))
            left_n = _rotate_ccw(tangent)
            # clamp inside
            top_here = g(pos_xy[0], pos_xy[1])
            clearance = _pipe_clearance(radius, slope) * clearance_scale
            warn_threshold = 0.5 * clearance
            z_min = top_here - depth + (radius + clearance)
            z_max = top_here - (radius + clearance)
            zc = max(z_min, min(zc, z_max))
            half_w = max(1e-6, half_top - slope * (top_here - zc))
            umax = max(0.0, half_w - (radius + clearance))
            offset_u = max(-umax, min(offset_u, umax))
            ctr_xy = pos_xy + offset_u * left_n
            # clamp length to stay within trench extents
            t_component = float(np.dot(axis_xy, tangent))
            n_component = float(np.dot(axis_xy, left_n))
            s_abs = s_center * total_length
            dist_start = s_abs
            dist_end = total_length - s_abs
            path_margin = radius + clearance
            allow_path = max(0.0, min(dist_start, dist_end) - path_margin)
            half_len_allow_path = (
                allow_path / max(abs(t_component), 1e-6) if allow_path > 0 else math.inf
            )
            lateral_allow = max(0.0, half_w - (radius + clearance))
            half_len_allow_lat = (
                lateral_allow / max(abs(n_component), 1e-6) if lateral_allow > 0 else math.inf
            )
            poly_extent = _max_extent_in_polygon(ctr_xy, axis_xy, ring_top, radius + clearance)
            half_len_allow_poly = poly_extent
            half_len_cap = min(half_len_allow_path, half_len_allow_lat, half_len_allow_poly)
            length = orig_length
            if math.isfinite(half_len_cap):
                max_length = max(0.0, 2.0 * half_len_cap)
                if max_length > 0.0:
                    length = min(length, max_length)
            length = max(length, radius * 0.5)

            half_length = 0.5 * length
            axis_margin_poly = (
                (half_len_allow_poly - half_length) if math.isfinite(half_len_allow_poly) else math.inf
            )
            axis_margin_lat = (
                (half_len_allow_lat - half_length) if math.isfinite(half_len_allow_lat) else math.inf
            )
            axis_margin_path = (
                (half_len_allow_path - half_length) if math.isfinite(half_len_allow_path) else math.inf
            )
            lateral_gap = half_w - abs(offset_u) - radius
            lateral_margin = lateral_gap - clearance
            min_margin_val = min(axis_margin_poly, axis_margin_lat, axis_margin_path, lateral_margin)

            def _finite(val: float) -> Optional[float]:
                return float(val) if math.isfinite(val) else None

            stored_min_margin = _finite(min_margin_val)

            clearance_records.append(
                {
                    "pipe_index": i,
                    "radius": radius,
                    "length": length,
                    "s_center": s_center,
                    "axis_margin_poly": _finite(axis_margin_poly),
                    "axis_margin_lat": _finite(axis_margin_lat),
                    "axis_margin_path": _finite(axis_margin_path),
                    "lateral_gap": float(lateral_gap),
                    "lateral_margin": float(lateral_margin),
                    "min_margin": stored_min_margin,
                    "clearance": float(clearance),
                    "warn_threshold": float(warn_threshold),
                    "clearance_scale": float(clearance_scale),
                    "center_xy": (float(ctr_xy[0]), float(ctr_xy[1])),
                    "offset_u": float(offset_u),
                }
            )

            # axis in XY, constant Z
            dx, dy, dz = (axis_xy[0] * length, axis_xy[1] * length, 0.0)
            base_x = float(ctr_xy[0] - 0.5 * dx)
            base_y = float(ctr_xy[1] - 0.5 * dy)
            base_z = float(zc - 0.5 * dz)
            cyl = gmsh.model.occ.addCylinder(base_x, base_y, base_z, dx, dy, dz, radius)
            pipe_dimtags.append((3, cyl))

        critical_clearance = [
            rec for rec in clearance_records if rec["min_margin"] is not None and rec["min_margin"] < -1e-6
        ]
        near_violation = [
            rec
            for rec in clearance_records
            if rec["min_margin"] is not None and rec["min_margin"] < rec["warn_threshold"]
        ]
        if critical_clearance or near_violation:
            def _fmt(rec: Dict[str, object]) -> str:
                idx = rec["pipe_index"]
                min_margin_val = rec["min_margin"]
                lat_gap = rec["lateral_gap"]
                axis_poly = rec["axis_margin_poly"]
                axis_path = rec["axis_margin_path"]

                def _fmt_opt(val: Optional[float]) -> str:
                    if val is None:
                        return "inf"
                    return f"{val:.4f}"

                return (
                    f"pipe[{idx}] min_margin={_fmt_opt(min_margin_val)}m "
                    f"(lateral_gap={lat_gap:.4f}m, axis_poly_margin={_fmt_opt(axis_poly)}m, "
                    f"axis_path_margin={_fmt_opt(axis_path)}m, clearance_scale={rec['clearance_scale']:.2f})"
                )

            if critical_clearance:
                details = "\n".join(_fmt(rec) for rec in critical_clearance)
                raise ValueError(
                    "Pipe clearance fell below the required guard band; adjust scenario geometry:\n" + details
                )
            else:
                for rec in near_violation:
                    print(
                        "[trenchfoot] clearance warning: "
                        + _fmt(rec)
                        + f" (< {rec['warn_threshold']:.3f}m headroom)",
                        file=sys.stderr,
                    )

        gmsh.model.occ.synchronize()
        if pipe_dimtags:
            outDT, outMap = gmsh.model.occ.fragment(
                [(3, trench_vol)], pipe_dimtags, removeObject=True, removeTool=True
            )
            gmsh.model.occ.synchronize()
            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()
            available_vols = {tag for (dim, tag) in gmsh.model.getEntities(dim=3)}
            if isinstance(outMap, dict):
                trench_new = outMap.get((3, trench_vol), [])
                pipes_new_lists = [outMap.get(dimtag, []) for dimtag in pipe_dimtags]
            else:  # gmsh >= 4.14 returns list of lists
                trench_new = outMap[0] if len(outMap) >= 1 else []
                pipes_new_lists = outMap[1:1 + len(pipe_dimtags)]
                while len(pipes_new_lists) < len(pipe_dimtags):
                    pipes_new_lists.append([])
            pipe_volume_tags = {t for lst in pipes_new_lists for (d, t) in lst if d == 3}
            trench_tags = [t for (d, t) in trench_new if d == 3 and t not in pipe_volume_tags]
            if not trench_tags:
                trench_tags = [t for (d, t) in outDT if d == 3 and t not in pipe_volume_tags]
            trench_tags = [t for t in trench_tags if t in available_vols]
            if trench_tags:
                gmsh.model.addPhysicalGroup(3, trench_tags, tag=1, name="TrenchAir")
            for i, lst in enumerate(pipes_new_lists):
                vol_tags = [t for (d, t) in lst if d == 3 and t in available_vols]
                if vol_tags:
                    gmsh.model.addPhysicalGroup(3, vol_tags, tag=100 + i, name=f"Pipe{i}")
        else:
            available_vols = {tag for (dim, tag) in gmsh.model.getEntities(dim=3)}
            trench_tags = [t for t in available_vols] or ([trench_vol] if trench_vol in available_vols else [])
            if trench_tags:
                gmsh.model.addPhysicalGroup(3, trench_tags, tag=1, name="TrenchAir")

        if debug_export:
            os.makedirs(debug_export, exist_ok=True)
            gmsh.write(os.path.join(debug_export, "geom_pre_mesh.brep"))

        if debug_callback is not None:
            ctx: Dict[str, object] = {
                "volumes": gmsh.model.getEntities(dim=3),
                "surfaces": gmsh.model.getEntities(dim=2),
                "physical_groups": gmsh.model.getPhysicalGroups(),
                "out_msh": persist_path_obj.as_posix() if persist_path_obj else None,
                "pipe_clearances": clearance_records,
            }
            try:
                debug_callback(ctx)
            except Exception as exc:  # pragma: no cover - debug aid only
                print(f"[trenchfoot] debug_callback failed: {exc}")

        if lc is not None:
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
        gmsh.model.mesh.generate(3)

        persisted = None
        if persist_path_obj is not None:
            persist_path_obj.parent.mkdir(parents=True, exist_ok=True)
            gmsh.write(persist_path_obj.as_posix())
            persisted = persist_path_obj

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_tags_arr = np.array(node_tags, dtype=int)
        node_coords_arr = np.array(node_coords, dtype=float)
        nodes = node_coords_arr.reshape(-1, 3) if node_coords_arr.size else np.zeros((0, 3), float)

        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements()
        element_blocks: List[VolumeElementBlock] = []
        for etype, tags, node_conn in zip(elem_types, elem_tags, elem_node_tags):
            tags_arr = np.array(tags, dtype=int)
            node_conn_arr = np.array(node_conn, dtype=int)
            if tags_arr.size:
                if node_conn_arr.size:
                    nodes_per_elem = node_conn_arr.size // tags_arr.size
                    node_conn_arr = node_conn_arr.reshape(tags_arr.size, nodes_per_elem)
                else:
                    node_conn_arr = np.empty((tags_arr.size, 0), dtype=int)
            else:
                node_conn_arr = np.empty((0, 0), dtype=int)
            element_blocks.append(
                VolumeElementBlock(
                    gmsh_type=int(etype),
                    element_tags=tags_arr,
                    node_tags=node_conn_arr,
                )
            )

        physical_groups: List[PhysicalGroupInfo] = []
        for dim, tag in gmsh.model.getPhysicalGroups():
            name = gmsh.model.getPhysicalName(dim, tag) or f"dim{dim}_tag{tag}"
            entities = tuple(gmsh.model.getEntitiesForPhysicalGroup(dim, tag))
            elem_accum: Dict[int, List[int]] = {}
            for entity_tag in entities:
                et_types, et_elem_tags, _ = gmsh.model.mesh.getElements(dim, entity_tag)
                for etype, tags in zip(et_types, et_elem_tags):
                    bucket = elem_accum.setdefault(int(etype), [])
                    bucket.extend(int(t) for t in tags)
            elements_map: Dict[int, np.ndarray] = {
                etype: np.array(values, dtype=int)
                for etype, values in elem_accum.items()
            }
            physical_groups.append(
                PhysicalGroupInfo(
                    dimension=dim,
                    tag=tag,
                    name=name,
                    entity_tags=entities,
                    element_tags=elements_map,
                )
            )

        return VolumeMeshResult(
            node_tags=node_tags_arr,
            nodes=nodes,
            element_blocks=element_blocks,
            physical_groups=physical_groups,
            pipe_clearances=clearance_records,
            persisted_path=persisted,
            mesh_characteristic_length=float(lc) if lc is not None else None,
        )
    finally:
        if finalize:
            gmsh.finalize()


def build_trench_volume_from_spec(
    cfg,
    lc=0.3,
    out_msh="mesh.msh",
    *,
    finalize: bool = True,
    debug_callback: Optional[Callable[[Dict[str, object]], None]] = None,
    debug_export: Optional[str] = None,
):
    result = generate_trench_volume(
        cfg,
        lc=lc,
        persist_path=out_msh,
        finalize=finalize,
        debug_callback=debug_callback,
        debug_export=debug_export,
    )
    if result.persisted_path is None:
        return out_msh
    return result.persisted_path.as_posix()


def main():
    ap = argparse.ArgumentParser(description="Volumetric sloped-trench mesher (ground-aware)")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lc", type=float, default=0.3, help="Target mesh size")
    args = ap.parse_args()
    with open(args.spec, "r") as f: cfg = json.load(f)
    os.makedirs(args.out, exist_ok=True)
    msh = build_trench_volume_from_spec(cfg, lc=args.lc, out_msh=os.path.join(args.out, "trench_volume.msh"))
    print(json.dumps({"msh": msh}, indent=2))

if __name__ == "__main__":
    main()
