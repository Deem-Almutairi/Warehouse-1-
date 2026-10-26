"""Interface for a FUTURE mathematical SLA model (GAMS, Pyomo, OR-Tools, ...). STATUS: PENDING.

No SLA optimiser exists in this project or in the referenced repository. This module only
defines files that a solver can read/write, so the DSS can evaluate its result under the same
routing and constraints as every other strategy.

Export (DSS -> solver):  sla_products.csv   product_id, activity, quantity (if known)
                         sla_locations.csv  location_id, aisle, position_m, depot_distance_m,
                                            capacity_units (if known), eligible
                         sla_coactivity.csv product_a, product_b, weight (orders containing both)
Import (solver -> DSS):  sla_assignment.csv product_id, location_id
An imported assignment is evaluated as 'Externally supplied SLA assignment'. The DSS verifies
feasibility and recomputes routes, but it does NOT verify that the assignment is optimal.
"""
from __future__ import annotations
from itertools import combinations
from collections import Counter
import pandas as pd
from layout import depot_distance

STATUS = "PENDING - no mathematical SLA model is integrated"


def export_inputs(act: pd.DataFrame, products, orders, locations, params, out_dir: str) -> list:
    import os
    os.makedirs(out_dir, exist_ok=True)
    p = products[["product_id"] + [c for c in ("quantity",) if c in products]].merge(
        act[["product_id", "activity"]], on="product_id", how="left")
    l = locations.copy()
    l["depot_distance_m"] = l["location_id"].map(depot_distance(locations, params))
    cnt = Counter()
    for _, g in orders.groupby("order_id"):
        for a, b in combinations(sorted(set(g["product_id"])), 2):
            cnt[(a, b)] += 1
    co = pd.DataFrame([(a, b, w) for (a, b), w in cnt.items()], columns=["product_a", "product_b", "weight"])
    paths = []
    for name, df in (("sla_products.csv", p), ("sla_locations.csv", l), ("sla_coactivity.csv", co)):
        path = os.path.join(out_dir, name)
        df.to_csv(path, index=False)
        paths.append(path)
    return paths
