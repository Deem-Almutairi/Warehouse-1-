"""Picker routing for a single-block warehouse.

Methods
-------
s_shape, return : standard textbook routing POLICIES (heuristics). Used to represent the
                  *assumed current routing practice*.
optimal         : own exact implementation (Held-Karp dynamic programming over the shortest-path
                  metric) for orders with <= ``exact_limit`` distinct pick locations; larger
                  orders fall back to min(S-shape, nearest-neighbour + 2-opt), flagged as NOT optimal.
                  This is NOT the repository's DAP code; see dap_adapter.py for that interface.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import permutations
import numpy as np
from config import Params
from layout import pair_dist, depot_point, aisle_x, path_between


@dataclass
class RouteResult:
    distance: float
    waypoints: list = field(default_factory=list)   # [(x, y), ...] metres
    method: str = ""
    optimal: bool = False
    n_locations: int = 0


def _dedupe(points):
    return sorted(set((int(a), float(y)) for a, y in points))


# ----------------------------------------------------------------------------- policies
def route_s_shape(points, p: Params) -> RouteResult:
    pts = _dedupe(points)
    dep = depot_point(p)
    xd = float(aisle_x(dep[0], p))
    if not pts:
        return RouteResult(0.0, [(xd, 0.0)], "s_shape", False, 0)
    L = p.aisle_length_m
    ymax = {}
    for a, y in pts:
        ymax[a] = max(ymax.get(a, 0.0), y)
    aisles = sorted(ymax)
    dist, cur_x, at_front = 0.0, xd, True
    wp = [(xd, 0.0)]
    for i, a in enumerate(aisles):
        x = float(aisle_x(a, p))
        dist += abs(x - cur_x)
        wp.append((x, 0.0 if at_front else L))
        if i == len(aisles) - 1 and at_front:
            dist += 2 * ymax[a]
            wp += [(x, ymax[a]), (x, 0.0)]
        else:
            dist += L
            wp.append((x, L if at_front else 0.0))
            at_front = not at_front
        cur_x = x
    dist += abs(cur_x - xd)
    wp.append((xd, 0.0))
    return RouteResult(dist, wp, "s_shape", False, len(pts))


def route_return(points, p: Params) -> RouteResult:
    pts = _dedupe(points)
    dep = depot_point(p)
    xd = float(aisle_x(dep[0], p))
    if not pts:
        return RouteResult(0.0, [(xd, 0.0)], "return", False, 0)
    ymax = {}
    for a, y in pts:
        ymax[a] = max(ymax.get(a, 0.0), y)
    left = sorted([a for a in ymax if aisle_x(a, p) < xd], reverse=True)
    right = sorted([a for a in ymax if aisle_x(a, p) >= xd])
    dist, cur_x, wp = 0.0, xd, [(xd, 0.0)]
    for a in left + right:
        x = float(aisle_x(a, p))
        dist += abs(x - cur_x) + 2 * ymax[a]
        wp += [(x, 0.0), (x, ymax[a]), (x, 0.0)]
        cur_x = x
    dist += abs(cur_x - xd)
    wp.append((xd, 0.0))
    return RouteResult(dist, wp, "return", False, len(pts))


# ------------------------------------------------------------------------- exact / heuristic
def dist_matrix(points, p: Params):
    """Node 0 = depot, nodes 1..k = pick points."""
    nodes = [depot_point(p)] + list(points)
    a = np.array([n[0] for n in nodes], dtype=float)
    y = np.array([n[1] for n in nodes], dtype=float)
    return pair_dist(a[:, None], y[:, None], a[None, :], y[None, :], p), nodes


def held_karp(D):
    """Optimal closed tour from node 0 through all other nodes. Returns (length, order of node ids)."""
    n = len(D) - 1
    if n == 0:
        return 0.0, [0]
    if n == 1:
        return float(2 * D[0, 1]), [0, 1]
    C = D[1:, 1:]
    full = 1 << n
    INF = np.inf
    dp = np.full((full, n), INF)
    par = np.full((full, n), -1, dtype=np.int64)
    for j in range(n):
        dp[1 << j, j] = D[0, j + 1]
    for mask in range(1, full):
        row = dp[mask]
        if not np.isfinite(row).any():
            continue
        cand = row[:, None] + C                      # cand[j, k] = dp[mask, j] + C[j, k]
        best_j = np.argmin(cand, axis=0)
        best_v = cand[best_j, np.arange(n)]
        for k in range(n):
            if mask >> k & 1:
                continue
            nm = mask | (1 << k)
            if best_v[k] < dp[nm, k]:
                dp[nm, k] = best_v[k]
                par[nm, k] = best_j[k]
    last = full - 1
    tot = dp[last] + D[1:, 0]
    j = int(np.argmin(tot))
    length = float(tot[j])
    order, mask = [], last
    while j != -1:
        order.append(j + 1)
        pj = int(par[mask, j])
        mask ^= (1 << j)
        j = pj
    return length, [0] + order[::-1]


def brute_force(D):
    n = len(D) - 1
    best = (np.inf, None)
    for perm in permutations(range(1, n + 1)):
        s, cur = 0.0, 0
        for v in perm:
            s += D[cur, v]
            cur = v
        s += D[cur, 0]
        if s < best[0]:
            best = (s, [0] + list(perm))
    return float(best[0]), best[1]


def _two_opt(D, tour):
    improved = True
    n = len(tour)
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[(j + 1) % n]
                if D[a, c] + D[b, d] < D[a, b] + D[c, d] - 1e-9:
                    tour[i:j + 1] = reversed(tour[i:j + 1])
                    improved = True
    return tour


def nn_two_opt(D):
    n = len(D)
    unvisited, tour = set(range(1, n)), [0]
    while unvisited:
        nxt = min(unvisited, key=lambda v: D[tour[-1], v])
        tour.append(nxt)
        unvisited.remove(nxt)
    tour = _two_opt(D, tour)
    length = sum(D[tour[i], tour[(i + 1) % n]] for i in range(n))
    return float(length), tour


def route_optimal(points, p: Params) -> RouteResult:
    pts = _dedupe(points)
    if not pts:
        return RouteResult(0.0, [(float(aisle_x(depot_point(p)[0], p)), 0.0)], "optimal", True, 0)
    D, nodes = dist_matrix(pts, p)
    if len(pts) <= p.exact_limit:
        length, order = held_karp(D)
        exact, label = True, "optimal"
    else:
        length, order = nn_two_opt(D)
        ss = route_s_shape(pts, p)
        if ss.distance <= length:                       # keep the better heuristic
            ss.method, ss.optimal = "heuristic_fallback", False
            return ss
        exact, label = False, "heuristic_fallback"
    seq = [nodes[i] for i in order] + [nodes[0]]
    wp = []
    for u, v in zip(seq[:-1], seq[1:]):
        seg = path_between(u, v, p)
        wp += seg if not wp else seg[1:]
    return RouteResult(length, wp, label, exact, len(pts))


def route(points, method: str, p: Params) -> RouteResult:
    if method == "s_shape":
        return route_s_shape(points, p)
    if method == "return":
        return route_return(points, p)
    if method == "optimal":
        return route_optimal(points, p)
    raise ValueError(f"Unknown routing method: {method}")
