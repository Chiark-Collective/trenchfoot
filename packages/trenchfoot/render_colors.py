import re
from typing import Optional

_TRENCH_COLORS = {
    "trench_walls": "#b87333",  # copper
    "trench_bottom": "#d2a679",
    "trench_cap_for_volume": "#e6cfa4",
    "ground_surface": "#b0b0b0",
}

_PIPE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
_BOX_COLORS = ["#17becf", "#bcbd22"]
_SPHERE_COLORS = ["#e377c2", "#7f7f7f"]

_OBJECT_ALPHA = 0.9
_SURFACE_ALPHA = 0.55
_DEFAULT_COLOR = "#c7c7c7"


def _match_index(name: str, prefix: str) -> Optional[int]:
    match = re.search(rf"{prefix}(\d+)", name, re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def is_object_group(name: str) -> bool:
    lower = name.lower()
    return lower.startswith("pipe") or lower.startswith("box") or lower.startswith("sphere")


def color_for_group(name: str) -> str:
    lower = name.lower()
    for key, color in _TRENCH_COLORS.items():
        if lower.startswith(key):
            return color
    if lower.startswith("ground"):
        return _TRENCH_COLORS["ground_surface"]

    idx = _match_index(name, "pipe")
    if idx is not None:
        return _PIPE_COLORS[idx % len(_PIPE_COLORS)]

    idx = _match_index(name, "box")
    if idx is not None:
        return _BOX_COLORS[idx % len(_BOX_COLORS)]

    idx = _match_index(name, "sphere")
    if idx is not None:
        return _SPHERE_COLORS[idx % len(_SPHERE_COLORS)]

    return _DEFAULT_COLOR


def opacity_for_group(name: str) -> float:
    return _OBJECT_ALPHA if is_object_group(name) else _SURFACE_ALPHA
