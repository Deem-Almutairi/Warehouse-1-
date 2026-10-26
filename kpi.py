"""KPI definitions and order-level evaluation of an assignment under a routing method."""
from __future__ import annotations
from dataclasses import dataclass, field
import time
import numpy as np
import pandas as pd
from config import Params
from layout import location_points
from routing import route
from assignment import utilization

KPI_DEFS = pd.DataFrame([
    ["Total picking distance", "sum over orders of (route distance x order frequency)", "m", "lower",
     "Walking distance; model-generated (routing policy/algorithm applied to the layout)",
     "layout, current/assigned locations, orders, accepted routing assumption"],
    ["Average distance per order", "total distance / sum of order frequencies", "m/order", "lower",
     "Model-generated", "same as above"],
    ["Estimated travel time", "route distance / picker speed", "s", "lower",
     "ESTIMATE: walking only; excludes picking/handling, waiting and congestion", "picker speed"],
    ["Average travel time per order", "total travel time / sum of order frequencies", "s/order", "lower",
     "ESTIMATE (walking only)", "picker speed"],
    ["Time incl. handling (optional)", "travel time + service time per line x order lines", "s", "lower",
     "ESTIMATE using the supplied service-time assumption", "picker speed, service time"],
    ["Storage utilization", "occupied eligible locations / total eligible locations", "%", "context-dependent",
     "Measured from the assignment (location-count definition; not volumetric)", "locations, assignment"],
    ["Storage locations used", "number of distinct locations holding a SKU", "count", "context-dependent",
     "Measured", "assignment"],
    ["Avg. product locations visited per order", "mean distinct pick locations per order (frequency-weighted)",
     "locations/order", "lower", "Measured from orders + assignment", "orders, assignment"],
    ["Orders exceeding distance target", "% of orders (freq-weighted) with route distance > target", "%", "lower",
     "Only when a target is supplied", "user target"],
    ["Orders exceeding time target", "% of orders (freq-weighted) with travel time > target", "%", "lower",
     "Only when a target is supplied", "user target, picker speed"],
    ["Orders above picker capacity", "% of orders whose total units exceed picker capacity", "%", "lower",
     "FLAG ONLY: multi-trip splitting is not modelled", "picker capacity, order quantities"],
], columns=["KPI", "Formula", "Unit", "Better when", "Nature / interpretation", "Data required"])


@dataclass
class EvalResult:
    label: str
    routing: str
    orders: pd.DataFrame
    kpis: dict
    runtime_s: float
    n_fallback: int = 0
    n_exact: int = 0
    notes: list = field(default_factory=list)


def order_table(orders: pd.DataFrame) -> pd.DataFrame:
    o = orders.copy()
    if "order_frequency" not in o:
        o["order_frequency"] = 1
    o["order_frequency"] = o["order_frequency"].fillna(1)
    return o


def evaluate(assign: dict, orders: pd.DataFrame, locations: pd.DataFrame, params: Params,
             routing: str, label: str = "", keep_routes: bool = False) -> EvalResult:
    t0 = time.perf_counter()
    o = order_table(orders)
    pts = location_points(locations)
    cache, rows, n_fb, n_ex = {}, [], 0, 0
    for oid, g in o.groupby("order_id", sort=False):
        locs = frozenset(assign[p] for p in g["product_id"])
        key = (locs, routing)
        if key not in cache:
            cache[key] = route([pts[l] for l in locs], routing, params)
        rr = cache[key]
        if routing == "optimal":
            n_ex += int(rr.optimal)
            n_fb += int(not rr.optimal and rr.n_locations > 0)
        units = g["quantity"].sum() if "quantity" in g and g["quantity"].notna().all() else np.nan
        rows.append({"order_id": oid, "frequency": float(g["order_frequency"].iloc[0]), "lines": len(g),
                     "locations_visited": len(locs), "distance_m": rr.distance, "units": units,
                     "route_method": rr.method, "waypoints": rr.waypoints if keep_routes else None})
    df = pd.DataFrame(rows)
    res = EvalResult(label, routing, df, {}, 0.0, n_fb, n_ex)
    res.kpis = compute_kpis(df, assign, locations, params)
    res.runtime_s = time.perf_counter() - t0
    if routing == "optimal" and n_fb:
        res.notes.append(f"{n_fb} order(s) exceeded the exact-routing limit ({params.exact_limit} locations) and used a "
                         "heuristic fallback; their routes are NOT proven optimal.")
    return res


def compute_kpis(df: pd.DataFrame, assign: dict, locations: pd.DataFrame, params: Params) -> dict:
    w = df["frequency"]
    nW = w.sum()
    k = {}
    tot = float((df["distance_m"] * w).sum())
    k["total_distance_m"] = tot
    k["avg_distance_per_order_m"] = tot / nW
    if params.picker_speed_mps:
        t = df["distance_m"] / params.picker_speed_mps
        k["total_travel_time_s"] = float((t * w).sum())
        k["avg_travel_time_per_order_s"] = k["total_travel_time_s"] / nW
        if params.service_time_s_per_line is not None:
            tt = t + df["lines"] * params.service_time_s_per_line
            k["avg_time_incl_handling_s"] = float((tt * w).sum() / nW)
    else:
        t = None
        k["total_travel_time_s"] = k["avg_travel_time_per_order_s"] = None
    k["utilization_pct"] = utilization(assign, locations) * 100
    k["locations_used"] = len(set(assign.values()))
    k["avg_locations_visited_per_order"] = float((df["locations_visited"] * w).sum() / nW)
    k["pct_orders_over_distance_target"] = (
        float(w[df["distance_m"] > params.target_max_order_distance_m].sum() / nW * 100)
        if params.target_max_order_distance_m is not None else None)
    k["pct_orders_over_time_target"] = (
        float(w[t > params.target_max_order_time_s].sum() / nW * 100)
        if (params.target_max_order_time_s is not None and t is not None) else None)
    k["pct_orders_over_picker_capacity"] = (
        float(w[df["units"] > params.picker_capacity_units].sum() / nW * 100)
        if (params.picker_capacity_units is not None and df["units"].notna().all()) else None)
    return k


def distance_reduction_pct(baseline_total, strategy_total):
    """((baseline - strategy) / baseline) x 100. Positive = improvement, negative = longer distance.
    Baseline 0 -> 0.0 if the strategy is also 0, otherwise None (undefined). Missing inputs -> None."""
    if baseline_total is None or strategy_total is None:
        return None
    if baseline_total == 0:
        return 0.0 if strategy_total == 0 else None
    return (baseline_total - strategy_total) / baseline_total * 100.0


def gap_table(kpis: dict, params: Params) -> pd.DataFrame:
    spec = [("Average distance per order (m)", "avg_distance_per_order_m", params.target_avg_distance_m),
            ("Average travel time per order (s)", "avg_travel_time_per_order_s", params.target_avg_time_s),
            ("% orders over distance target", "pct_orders_over_distance_target", None),
            ("% orders over time target", "pct_orders_over_time_target", None)]
    rows = []
    for name, key, tgt in spec[:2]:
        v = kpis.get(key)
        if v is None:
            rows.append([name, "Unavailable", tgt, None, "Cannot evaluate (data missing)"])
        elif tgt is None:
            rows.append([name, v, None, None, "No target defined - no judgement made"])
        else:
            rows.append([name, v, tgt, v - tgt, "Meets target" if v <= tgt else "Above target"])
    return pd.DataFrame(rows, columns=["KPI", "Value", "Target", "Gap (value - target)", "Status"])
