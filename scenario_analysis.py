"""OPTIONAL robustness analysis on HYPOTHETICAL inputs. Results are never mixed with the historical-data results.

Kinds
  demand_multiplier   : every order's frequency is multiplied by k (uniform demand growth). Distances scale linearly, so
                        rankings are not expected to change - this is shown, not assumed.
  distribution_shift  : order lines are re-drawn (fixed seed) with product probabilities proportional to
                        historical_activity ** gamma (gamma < 1 flatter demand, > 1 more concentrated), keeping each
                        order's size and quantities. ABC is re-planned on the scenario orders; baseline layout and any
                        externally supplied assignment stay fixed (the DSS cannot re-optimise them).
Not supported: warehouse-capacity scenarios, forecasting, batching.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from abc_analysis import activity_table, abc_classify
from strategies import evaluate_strategies, recommend

KINDS = {"demand_multiplier": "Increased demand (x k)", "distribution_shift": "Changed product demand distribution (exponent gamma)"}
LABEL = "HYPOTHETICAL SCENARIO - not historical data and not a forecast"


def make_scenario_orders(orders: pd.DataFrame, kind: str, value: float, seed: int = 123) -> pd.DataFrame:
    o = orders.copy()
    if "order_frequency" not in o:
        o["order_frequency"] = 1.0
    o["order_frequency"] = o["order_frequency"].fillna(1.0)
    if kind == "demand_multiplier":
        if value <= 0:
            raise ValueError("Multiplier must be positive.")
        o["order_frequency"] = o["order_frequency"] * value
        return o
    if kind == "distribution_shift":
        act = o.groupby("product_id").size()
        w = act.values.astype(float) ** value
        w = w / w.sum()
        rng = np.random.default_rng(seed)
        rows = []
        for oid, g in o.groupby("order_id", sort=False):
            k = min(len(g), len(act))
            ch = rng.choice(len(act), size=k, replace=False, p=w)
            for c, (_, line) in zip(ch, g.head(k).iterrows()):
                d = line.to_dict()
                d["product_id"] = act.index[c]
                rows.append(d)
        return pd.DataFrame(rows)
    raise ValueError(f"Unknown scenario kind: {kind}")


def run_scenario(products, orders, locations, params, kind, value, ext_assign=None, abc_a=80, abc_b=15,
                 measure="pick_frequency"):
    so = make_scenario_orders(orders, kind, value)
    tbl = abc_classify(activity_table(products, so, measure), abc_a, abc_b)
    res = evaluate_strategies(products, so, locations, params, tbl, ext_assign)
    dec = recommend(res, params)
    assumptions = [LABEL, f"Kind: {KINDS[kind]}, parameter = {value}",
                   "Baseline and imported assignments stay fixed; ABC is re-planned on the scenario orders.",
                   "Same layout, routing method and constraints as the historical evaluation."]
    return {"label": LABEL, "assumptions": assumptions, "results": res, "decision": dec, "orders": so}
