"""Operational parameters. Every field is optional unless noted; None = 'not provided'."""
from __future__ import annotations
from dataclasses import dataclass, asdict, fields
from typing import Optional
import pandas as pd

ROUTING_POLICIES = {
    "s_shape": "S-shape (traversal) policy",
    "return": "Return policy (enter and leave each aisle from the front)",
}


@dataclass
class Params:
    # --- layout geometry (single-block warehouse) ---
    aisle_length_m: float = 20.0          # required
    aisle_spacing_m: float = 3.0          # required, centre-to-centre
    depot_aisle: int = 1                  # depot sits at the front cross-aisle, in front of this aisle
    # --- operational data ---
    picker_speed_mps: Optional[float] = None
    service_time_s_per_line: Optional[float] = None
    picker_capacity_units: Optional[float] = None
    reassign_cost_per_move: Optional[float] = None
    routing_change_cost: Optional[float] = None     # one-off cost of adopting an improved routing method
    max_moves_allowed: Optional[int] = None
    reassign_budget: Optional[float] = None
    # --- user-defined operational targets ---
    target_avg_distance_m: Optional[float] = None
    target_avg_time_s: Optional[float] = None
    target_max_order_distance_m: Optional[float] = None
    target_max_order_time_s: Optional[float] = None
    # --- analysis settings ---
    current_routing_policy: str = "s_shape"
    routing_assumption_accepted: bool = False
    exact_limit: int = 12                 # max distinct pick locations for exact routing
    evaluation_routing: str = "optimal"   # routing used for the primary strategy comparison (same for all strategies)
    similarity_tolerance_pct: float = 1.0 # REPORTING threshold only: gaps below this are called 'similar'; never changes the winner
    priority: str = "least_effort"        # or "max_improvement"
    min_material_improvement_pct: float = 0.0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"parameter": [f.name for f in fields(self)],
                             "value": [getattr(self, f.name) for f in fields(self)]})

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "Params":
        p = cls()
        kinds = {f.name: str(f.type) for f in fields(cls)}
        for _, r in df.iterrows():
            name, val = str(r["parameter"]).strip(), r["value"]
            if name not in kinds or pd.isna(val) or str(val).strip() == "":
                continue
            t = kinds[name]
            if "bool" in t:
                setattr(p, name, str(val).strip().lower() in ("1", "true", "yes"))
            elif "str" in t:
                setattr(p, name, str(val).strip())
            elif "int" in t:
                setattr(p, name, int(float(val)))
            else:
                setattr(p, name, float(val))
        return p

    def as_dict(self):
        return asdict(self)
