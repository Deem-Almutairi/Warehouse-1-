"""ABC classification (a prioritisation heuristic, not an optimisation algorithm)."""
from __future__ import annotations
import pandas as pd

MEASURES = {
    "demand": "Demand (products table)",
    "pick_frequency": "Pick frequency (order lines x order frequency)",
    "units_ordered": "Units ordered (quantity x order frequency)",
}


def activity_table(products: pd.DataFrame, orders: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Return product_id, description, activity for the chosen measure; raises if unavailable."""
    base = products[["product_id"] + (["description"] if "description" in products else [])].copy()
    o = orders.copy()
    o["_f"] = o["order_frequency"].fillna(1) if "order_frequency" in o else 1
    if measure == "demand":
        if "demand" not in products or products["demand"].isna().all():
            raise ValueError("Column 'demand' is not available in the products table.")
        act = products.set_index("product_id")["demand"]
    elif measure == "pick_frequency":
        act = o.groupby("product_id")["_f"].sum()
    elif measure == "units_ordered":
        if "quantity" not in o or o["quantity"].isna().all():
            raise ValueError("Order quantities are not available.")
        act = (o["quantity"] * o["_f"]).groupby(o["product_id"]).sum()
    else:
        raise ValueError(f"Unknown activity measure {measure}")
    base["activity"] = base["product_id"].map(act).fillna(0.0).astype(float)
    return base


def abc_classify(act: pd.DataFrame, a_pct: float = 80.0, b_pct: float = 15.0) -> pd.DataFrame:
    """A product is class A while the cumulative share *before* it is < a_pct, etc.
    (so the product that crosses the threshold stays in the higher class)."""
    if not (0 < a_pct < 100 and 0 <= b_pct < 100 and a_pct + b_pct <= 100):
        raise ValueError("Thresholds must satisfy 0 < A < 100, B >= 0 and A + B <= 100.")
    df = act.sort_values(["activity", "product_id"], ascending=[False, True]).reset_index(drop=True)
    total = df["activity"].sum()
    if total <= 0:
        raise ValueError("Total activity is zero; ABC classification is not possible.")
    df["rank"] = df.index + 1
    df["share_pct"] = df["activity"] / total * 100
    df["cum_pct"] = df["share_pct"].cumsum()
    prev = df["cum_pct"] - df["share_pct"]
    df["abc_class"] = ["A" if v < a_pct - 1e-9 else ("B" if v < a_pct + b_pct - 1e-9 else "C") for v in prev]
    return df
