"""Data loading, sample data, templates and validation."""
from __future__ import annotations
from dataclasses import dataclass, field
import io
import numpy as np
import pandas as pd
from config import Params

REQUIRED = {
    "products": ["product_id"],
    "orders": ["order_id", "product_id"],
    "locations": ["location_id", "aisle", "position_m"],
}
OPTIONAL = {
    "products": ["description", "demand", "quantity", "current_location"],
    "orders": ["quantity", "order_frequency"],
    "locations": ["side", "capacity_units", "eligible"],
}
SAMPLE_LABEL = ("SAMPLE DATA (synthetic, fixed seed). Its current layout is randomly generated, so improvements over it "
                "demonstrate the method only - they are not evidence about any real warehouse or about which model is better.")


def read_table(file, name: str = "") -> pd.DataFrame:
    fname = getattr(file, "name", name) or name
    if str(fname).lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def generate_sample(seed: int = 7, n_aisles: int = 6, slots_per_side: int = 10, n_products: int = 60,
                    n_orders: int = 200):
    """Synthetic data. The current assignment is deliberately NOT activity-sorted (random)."""
    rng = np.random.default_rng(seed)
    params = Params(aisle_length_m=20.0, aisle_spacing_m=3.0, depot_aisle=1,
                    picker_speed_mps=1.2, service_time_s_per_line=15.0, picker_capacity_units=40.0,
                    reassign_cost_per_move=5.0, current_routing_policy="s_shape")
    slot = params.aisle_length_m / slots_per_side
    rows = []
    for a in range(1, n_aisles + 1):
        for side in ("L", "R"):
            for k in range(slots_per_side):
                rows.append((f"A{a}-{side}{k + 1:02d}", a, side, round((k + 0.5) * slot, 2), 100.0, 1))
    locations = pd.DataFrame(rows, columns=["location_id", "aisle", "side", "position_m", "capacity_units", "eligible"])
    pids = [f"P{i:03d}" for i in range(1, n_products + 1)]
    demand = np.maximum(1, np.round(1500 / np.arange(1, n_products + 1) ** 1.1)).astype(int)
    qty = rng.integers(10, 80, n_products)
    cur = rng.permutation(locations["location_id"].values)[:n_products]
    products = pd.DataFrame({"product_id": pids, "description": [f"Item {i}" for i in range(1, n_products + 1)],
                             "demand": demand, "quantity": qty, "current_location": cur})
    w = demand / demand.sum()
    lines = []
    for o in range(1, n_orders + 1):
        k = int(rng.integers(2, 9))
        chosen = rng.choice(n_products, size=k, replace=False, p=w)
        for c in chosen:
            lines.append((f"O{o:04d}", pids[c], int(rng.integers(1, 6)), 1))
    orders = pd.DataFrame(lines, columns=["order_id", "product_id", "quantity", "order_frequency"])
    return products, orders, locations, params


def templates() -> dict:
    return {
        "products_template.csv": pd.DataFrame([{"product_id": "P001", "description": "example", "demand": 100,
                                                "quantity": 40, "current_location": "A1-L01"}]),
        "orders_template.csv": pd.DataFrame([{"order_id": "O0001", "product_id": "P001", "quantity": 2,
                                              "order_frequency": 1}]),
        "locations_template.csv": pd.DataFrame([{"location_id": "A1-L01", "aisle": 1, "side": "L",
                                                 "position_m": 1.0, "capacity_units": 100, "eligible": 1}]),
        "parameters_template.csv": Params().to_frame(),
    }


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


@dataclass
class ValidationReport:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    availability: dict = field(default_factory=dict)   # analysis -> (bool, reason)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate(products, orders, locations, params: Params, is_sample: bool = False) -> ValidationReport:
    r = ValidationReport()
    tables = {"products": products, "orders": orders, "locations": locations}
    for t, df in tables.items():
        if df is None or len(df) == 0:
            r.errors.append(f"The {t} table is missing or empty.")
            continue
        miss = [c for c in REQUIRED[t] if c not in df.columns]
        if miss:
            r.errors.append(f"{t}: required column(s) missing: {miss}.")
    if r.errors:
        return r
    if products["product_id"].duplicated().any():
        r.errors.append("products: duplicate product_id values.")
    if locations["location_id"].duplicated().any():
        r.errors.append("locations: duplicate location_id values.")
    for c in ("aisle", "position_m"):
        if pd.to_numeric(locations[c], errors="coerce").isna().any():
            r.errors.append(f"locations: non-numeric or empty values in '{c}'.")
    if not r.errors:
        if (locations["position_m"] < 0).any() or (locations["position_m"] > params.aisle_length_m).any():
            r.errors.append(f"locations: position_m must lie within 0..{params.aisle_length_m} m (aisle length).")
        if (locations["aisle"] < 1).any() or (locations["aisle"] != locations["aisle"].round()).any():
            r.errors.append("locations: aisle must be a positive integer.")
        if int(params.depot_aisle) < 1:
            r.errors.append("Parameters: depot_aisle must be >= 1.")
    unk = set(orders["product_id"]) - set(products["product_id"])
    if unk:
        r.errors.append(f"orders: {len(unk)} product_id(s) not found in products, e.g. {sorted(unk)[:3]}.")
    if "current_location" in products:
        cur = products.dropna(subset=["current_location"])
        bad = set(cur["current_location"]) - set(locations["location_id"])
        if bad:
            r.errors.append(f"products: {len(bad)} current_location value(s) not in locations, e.g. {sorted(bad)[:3]}.")
        if cur["current_location"].duplicated().any():
            r.errors.append("products: several products share one current_location (single-SKU-per-location assumption).")
        nm = products["current_location"].isna().sum()
        if nm:
            r.warnings.append(f"products: {nm} product(s) have no current_location; the baseline cannot be evaluated until fixed.")
    else:
        r.warnings.append("products: no 'current_location' column - the Baseline strategy and distance reduction (%) "
                          "cannot be calculated; ABC/mathematical strategies can still be compared with each other.")
    if len(products) > len(locations):
        r.warnings.append("More products than storage locations: no feasible one-SKU-per-location assignment exists.")
    # availability map
    has = lambda t, c: c in tables[t].columns and tables[t][c].notna().any()
    ok_geo = not r.errors
    r.availability = {
        "Baseline strategy": (ok_geo and has("products", "current_location"), "requires current product-to-location assignments"),
        "Intervention analysis (SLA vs PRO)": (ok_geo and params.routing_assumption_accepted,
                                               "requires baseline + an ACCEPTED current-routing-policy assumption"),
        "Estimated travel time": (params.picker_speed_mps is not None, "requires picker speed"),
        "Time incl. handling": (params.picker_speed_mps is not None and params.service_time_s_per_line is not None,
                                "requires picker speed and service time per line"),
        "ABC (demand)": (has("products", "demand"), "requires products.demand"),
        "ABC (pick frequency)": (True, "derived from orders"),
        "ABC (units ordered)": (has("orders", "quantity"), "requires orders.quantity"),
        "Capacity checks": (has("locations", "capacity_units") and has("products", "quantity"),
                            "requires location capacity and product quantity"),
        "Picker-capacity flag": (params.picker_capacity_units is not None and has("orders", "quantity"),
                                 "requires picker capacity and order quantities"),
        "Reassignment cost": (params.reassign_cost_per_move is not None, "requires cost per move"),
        "Target gaps": (any(getattr(params, k) is not None for k in
                            ("target_avg_distance_m", "target_avg_time_s", "target_max_order_distance_m",
                             "target_max_order_time_s")), "requires at least one user-defined target"),
        "Mathematical SLA": (False, "PENDING: no model integrated (an external assignment file can be imported)"),
    }
    if is_sample:
        r.warnings.append(SAMPLE_LABEL)
    return r
