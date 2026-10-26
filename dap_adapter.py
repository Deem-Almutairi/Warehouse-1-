"""Adapter for the DAP routing code of github.com/MyReStudy/single_block (Liu, Shen & Xia, IEEM 2024).

STATUS: IMPLEMENTED BUT NOT EXECUTED IN THIS ENVIRONMENT (the repository code was only READ, not run:
the build sandbox has no network access). The file formats below were inferred from reading
DAP_algorithm.py and PickingInfo.py. Run ``scripts/validate_dap.py`` after cloning the repository to test it.

What DAP expects (inferred from source)
  * distance matrix (xlsx, index_col=0): rows/cols 'a{i}' (top junction of aisle i), 'b{i}' (bottom junction)
    and one row/col per pick vertex; aisles must be numbered 1..m consecutively with no empty aisle
    (the first stage uses aisle 1 explicitly), equal aisle length (aisleLength = d(a1,b1)).
  * warehouse information (xlsx, index_col=0): index = pick vertex, column 'aisle'.
  * returns (state, tour length, backtracked configuration path) via DAP(...).formingPath().
Known limitations: single block, equal-length aisles, no depot, no capacity.
Depot handling here: the depot is added as an extra pick vertex at the front of ``depot_aisle``; since DAP
returns an optimal CLOSED tour, visiting the depot vertex makes the tour equivalent to a depot-based route.
This is an unverified modelling assumption until validate_dap.py confirms it.
"""
from __future__ import annotations
import os
import sys
import importlib
import numpy as np
import pandas as pd
from config import Params
from layout import pair_dist, depot_point

REQUIRED_FILES = ["DAP_algorithm.py", "PickingInfo.py"]


class DapUnavailable(RuntimeError):
    pass


def dap_status(repo_path: str | None) -> dict:
    if not repo_path or not os.path.isdir(repo_path):
        return {"available": False, "reason": "Repository path not set or not found. Clone "
                "https://github.com/MyReStudy/single_block and set the path (see README)."}
    miss = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(repo_path, f))]
    if miss:
        return {"available": False, "reason": f"Missing file(s) in repository path: {miss}"}
    return {"available": True, "reason": "Files found (execution still needs to be validated with scripts/validate_dap.py)"}


def build_dap_inputs(points, p: Params, out_dir: str, tag: str = "order"):
    """Write the two Excel inputs for one order. ``points`` = [(aisle, y), ...] (distinct)."""
    os.makedirs(out_dir, exist_ok=True)
    pts = sorted(set((int(a), float(y)) for a, y in points) | {depot_point(p)})
    aisles = sorted({a for a, _ in pts})
    ren = {a: i + 1 for i, a in enumerate(aisles)}                 # consecutive numbering, empty aisles removed
    gap = {}                                                        # preserve real inter-aisle spacing
    xs = {a: (a - 1) * p.aisle_spacing_m for a in aisles}
    names = [f"p{i}" for i in range(len(pts))]
    # nodes: (name, real aisle, y) incl. junction nodes
    nodes = []
    for a in aisles:
        nodes.append((f"a{ren[a]}", a, 0.0))
    for a in aisles:
        nodes.append((f"b{ren[a]}", a, p.aisle_length_m))
    for n, (a, y) in zip(names, pts):
        nodes.append((n, a, y))
    idx = [n[0] for n in nodes]
    A = np.array([n[1] for n in nodes], dtype=float)
    Y = np.array([n[2] for n in nodes], dtype=float)
    D = pair_dist(A[:, None], Y[:, None], A[None, :], Y[None, :], p)
    dm = pd.DataFrame(D, index=idx, columns=idx)
    wi = pd.DataFrame({"aisle": [ren[a] for a, _ in pts]}, index=names)
    f1, f2 = os.path.join(out_dir, f"{tag}_distance_matrix.xlsx"), os.path.join(out_dir, f"{tag}_warehouse_info.xlsx")
    dm.to_excel(f1)
    wi.to_excel(f2)
    return f1, f2


def run_dap(repo_path: str, points, p: Params, work_dir: str, tag: str = "order") -> dict:
    st = dap_status(repo_path)
    if not st["available"]:
        raise DapUnavailable(st["reason"])
    f1, f2 = build_dap_inputs(points, p, work_dir, tag)
    sys.path.insert(0, repo_path)
    try:
        mod = importlib.import_module("DAP_algorithm")
        state, length, path = mod.DAP(f1, f2).formingPath()
    finally:
        sys.path.remove(repo_path)
    return {"state": state, "distance": float(length), "path": path}
