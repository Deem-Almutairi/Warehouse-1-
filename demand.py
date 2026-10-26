"""Demand indicators and system-level descriptive KPIs (historical data only; no forecasting is implemented)."""
from __future__ import annotations
import pandas as pd


def product_demand(products: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Per product: picking frequency (number of orders containing it) and total demand quantity.
    If order_frequency is supplied, each order is counted that many times."""
    o = orders.copy()
    w = o["order_frequency"].fillna(1) if "order_frequency" in o else 1
    o["_w"] = w
    freq = o.groupby("product_id")["_w"].sum()
    out = products[["product_id"] + (["description"] if "description" in products else [])].copy()
    out["picking_frequency"] = out["product_id"].map(freq).fillna(0).astype(float)
    if "quantity" in o and o["quantity"].notna().all():
        qty = (o["quantity"] * o["_w"]).groupby(o["product_id"]).sum()
        out["total_demand_quantity"] = out["product_id"].map(qty).fillna(0).astype(float)
    else:
        out["total_demand_quantity"] = float("nan")     # unavailable - never filled with invented values
    return out.sort_values(["picking_frequency", "product_id"], ascending=[False, True]).reset_index(drop=True)


def system_kpis(products: pd.DataFrame, orders: pd.DataFrame, abc_tbl: pd.DataFrame | None = None) -> dict:
    d = product_demand(products, orders)
    w = orders["order_frequency"].fillna(1) if "order_frequency" in orders else pd.Series(1, index=orders.index)
    n_orders = float(orders.groupby("order_id").size().index.size)
    k = {
        "total_products": int(len(products)),
        "total_orders": int(n_orders),
        "total_demand_quantity": (float(d["total_demand_quantity"].sum()) if d["total_demand_quantity"].notna().all() else None),
        "products_never_ordered": int((d["picking_frequency"] == 0).sum()),
        "avg_lines_per_order": float(len(orders) / n_orders) if n_orders else None,
        "demand_frequency_min": float(d["picking_frequency"].min()),
        "demand_frequency_median": float(d["picking_frequency"].median()),
        "demand_frequency_max": float(d["picking_frequency"].max()),
    }
    if abc_tbl is not None:
        dist = abc_tbl["abc_class"].value_counts().reindex(["A", "B", "C"]).fillna(0).astype(int)
        k["abc_distribution"] = {c: {"products": int(dist[c]),
                                     "share_of_products_pct": float(dist[c] / len(abc_tbl) * 100),
                                     "share_of_activity_pct": float(abc_tbl.loc[abc_tbl.abc_class == c, "share_pct"].sum())}
                                 for c in ("A", "B", "C")}
    return k


ORDER_PROFILE = {
    "Order-by-Order": "SUPPORTED - each order is routed separately.",
    "Batch Orders": "NOT SUPPORTED - the implemented routing handles individual orders only; batching is not evaluated.",
}


def order_profile(orders: pd.DataFrame) -> dict:
    sizes = orders.groupby("order_id").size()
    return {"mode": "Order-by-Order", "orders": int(len(sizes)), "min_lines": int(sizes.min()),
            "median_lines": float(sizes.median()), "max_lines": int(sizes.max()), "support": ORDER_PROFILE}
