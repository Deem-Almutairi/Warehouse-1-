"""Single-block warehouse geometry.

Model: parallel picking aisles joined by a front and a back cross-aisle. A pick point is
(aisle, y) with y = distance from the front cross-aisle along the aisle centre line. Both
rack sides are served from the same centre line. The depot is at the front cross-aisle
in front of ``params.depot_aisle``.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from config import Params


def aisle_x(aisle, p: Params):
    return (np.asarray(aisle, dtype=float) - 1.0) * p.aisle_spacing_m


def pair_dist(a1, y1, a2, y2, p: Params):
    """Shortest walking distance between pick points (vectorised, exact for this geometry)."""
    a1, y1, a2, y2 = (np.asarray(v, dtype=float) for v in (a1, y1, a2, y2))
    L = p.aisle_length_m
    dx = np.abs(a1 - a2) * p.aisle_spacing_m
    via_front = y1 + y2 + dx
    via_back = (L - y1) + (L - y2) + dx
    return np.where(a1 == a2, np.abs(y1 - y2), np.minimum(via_front, via_back))


def depot_point(p: Params):
    return (int(p.depot_aisle), 0.0)


def location_points(locations: pd.DataFrame) -> dict:
    """location_id -> (aisle, y)."""
    return {r.location_id: (int(r.aisle), float(r.position_m)) for r in locations.itertuples()}


def depot_distance(locations: pd.DataFrame, p: Params) -> pd.Series:
    da, dy = depot_point(p)
    d = pair_dist(locations["aisle"].values, locations["position_m"].values, da, dy, p)
    return pd.Series(d, index=locations["location_id"].values)


def path_between(p1, p2, p: Params):
    """Waypoints (x, y) in metres for the shortest path between two pick points."""
    (a1, y1), (a2, y2) = p1, p2
    L = p.aisle_length_m
    x1, x2 = float(aisle_x(a1, p)), float(aisle_x(a2, p))
    if a1 == a2:
        return [(x1, y1), (x2, y2)]
    dx = abs(x1 - x2)
    front = y1 + y2 + dx
    back = (L - y1) + (L - y2) + dx
    yc = 0.0 if front <= back else L
    return [(x1, y1), (x1, yc), (x2, yc), (x2, y2)]
