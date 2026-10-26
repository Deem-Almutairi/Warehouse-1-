"""Storage-location assignment strategies and feasibility checks.

Assumption (documented in the UI): one SKU per storage location.
Implemented here: current, ABC (heuristic), random (reference heuristic), external import.
NOT implemented: a mathematical SLA optimiser (see sla_interface.py).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from config import Params
from layout import depot_distance


def _eligible(locations: pd.DataFrame) -> pd.DataFrame:
    if "eligible" in locations:
        return locations[locations["eligible"].fillna(1).astype(int) == 1]
    return locations


def current_assignment(products: pd.DataFrame) -> dict:
    if "current_location" not in products:
        return {}
    s = products.dropna(subset=["current_location"])
    return dict(zip(s["product_id"], s["current_location"]))


def feasibility_report(assign: dict, products: pd.DataFrame, locations: pd.DataFrame):
    """Return (checks, label). Each check = (name, 'pass'|'fail'|'unverifiable', detail).
    label: 'Feasible' | 'Feasible (unverified: ...)' | 'Infeasible'."""
    checks = []
    locs = locations.set_index("location_id")
    missing = [p for p in products["product_id"] if p not in assign]
    checks.append(("Every product is assigned", "fail" if missing else "pass",
                   f"{len(missing)} product(s) have no location (e.g. {missing[:3]})." if missing else "all products assigned"))
    unknown = [l for l in assign.values() if l not in locs.index]
    checks.append(("All assigned locations exist", "fail" if unknown else "pass",
                   f"{len(unknown)} assignment(s) use unknown locations (e.g. {unknown[:3]})." if unknown else "all valid"))
    used = pd.Series(list(assign.values()), dtype=object)
    dup = used[used.duplicated()].unique().tolist()
    checks.append(("One product per location", "fail" if dup else "pass",
                   f"{len(dup)} location(s) hold more than one SKU (single-SKU assumption), e.g. {dup[:3]}." if dup else "no duplicates"))
    if "eligible" in locs and locs["eligible"].notna().any():
        bad = [l for l in assign.values() if l in locs.index and int(locs.at[l, "eligible"] if pd.notna(locs.at[l, "eligible"]) else 1) != 1]
        checks.append(("Only eligible locations used", "fail" if bad else "pass",
                       f"{len(bad)} product(s) are in ineligible locations, e.g. {bad[:3]}." if bad else "all eligible"))
    else:
        checks.append(("Only eligible locations used", "unverifiable", "no eligibility data supplied"))
    if "capacity_units" in locs and locs["capacity_units"].notna().any() and "quantity" in products and products["quantity"].notna().any():
        q = products.set_index("product_id")["quantity"]
        over = [pid for pid, l in assign.items() if l in locs.index and pid in q.index
                and pd.notna(q[pid]) and pd.notna(locs.at[l, "capacity_units"]) and q[pid] > locs.at[l, "capacity_units"]]
        checks.append(("Location capacity respected", "fail" if over else "pass",
                       f"{len(over)} product(s) exceed location capacity, e.g. {over[:3]}." if over else "all within capacity"))
    else:
        checks.append(("Location capacity respected", "unverifiable", "needs location capacity and product quantity"))
    fails = [c for c in checks if c[1] == "fail"]
    unver = [c[0] for c in checks if c[1] == "unverifiable"]
    label = "Infeasible" if fails else ("Feasible" if not unver else "Feasible (unverified: " + "; ".join(unver) + ")")
    return checks, label


def check_assignment(assign: dict, products: pd.DataFrame, locations: pd.DataFrame) -> list:
    """Human-readable constraint violations (empty list = no violated constraint)."""
    checks, _ = feasibility_report(assign, products, locations)
    return [c[2] for c in checks if c[1] == "fail"]


def _greedy(order_products, products, locations, params, location_order):
    """Assign products (in the given order) to the first free eligible location that fits."""
    q = products.set_index("product_id")["quantity"] if "quantity" in products else None
    cap = locations.set_index("location_id")["capacity_units"] if "capacity_units" in locations else None
    free = list(location_order)
    assign, unassigned = {}, []
    for pid in order_products:
        need = q.get(pid) if q is not None else None
        for l in free:
            c = cap.get(l) if cap is not None else None
            if pd.notna(need) and pd.notna(c) and need > c:
                continue
            assign[pid] = l
            free.remove(l)
            break
        else:
            unassigned.append(pid)
    return assign, unassigned


def abc_assignment(abc_table: pd.DataFrame, products, locations, params: Params):
    """Highest-activity products -> eligible locations closest to the depot (walking distance)."""
    el = _eligible(locations)
    dd = depot_distance(el, params).sort_values(kind="stable")
    prods = abc_table.sort_values(["rank"])["product_id"].tolist()
    prods += [p for p in products["product_id"] if p not in set(prods)]
    return _greedy(prods, products, locations, params, dd.index.tolist())


def random_assignment(products, locations, params: Params, seed: int = 42):
    el = _eligible(locations)
    rng = np.random.default_rng(seed)
    locs = el["location_id"].tolist()
    rng.shuffle(locs)
    pids = products["product_id"].tolist()
    rng.shuffle(pids)
    return _greedy(pids, products, locations, params, locs)


def external_assignment(df: pd.DataFrame) -> dict:
    need = {"product_id", "location_id"}
    if not need.issubset(df.columns):
        raise ValueError("Assignment file needs columns: product_id, location_id")
    return dict(zip(df["product_id"], df["location_id"]))


def count_moves(base: dict, new: dict) -> int:
    return sum(1 for p, l in new.items() if base.get(p) != l)


def utilization(assign: dict, locations: pd.DataFrame) -> float:
    el = set(_eligible(locations)["location_id"])
    occ = len(set(assign.values()) & el)
    return occ / len(el) if el else float("nan")
