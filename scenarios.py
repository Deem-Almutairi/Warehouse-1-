"""Scenario construction. Routes are RE-CALCULATED for every assignment (never reused)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
from config import Params, ROUTING_POLICIES
from kpi import evaluate, EvalResult
from assignment import (current_assignment, check_assignment, abc_assignment, random_assignment,
                         count_moves)


@dataclass
class Scenario:
    sid: str
    name: str
    component: str                    # baseline | SLA | PRO | SLA+PRO | reference
    assignment_label: str
    routing: str                      # s_shape | return | optimal
    status: str = "Evaluated"         # Evaluated | Model not integrated | Not evaluated | Infeasible
    result: Optional[EvalResult] = None
    assign: dict = field(default_factory=dict)
    moves: int = 0
    reloc_cost: Optional[float] = None
    violations: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    changes_assignment: bool = False
    changes_routing: bool = False

    @property
    def evaluated(self) -> bool:
        return self.status == "Evaluated" and self.result is not None


def _mk(sid, name, comp, label, routing, assign, base, products, orders, locations, params,
        unassigned=None, base_routing=None, keep_routes=False, notes=None):
    s = Scenario(sid, name, comp, label, routing, assign=assign, notes=list(notes or []))
    s.violations = check_assignment(assign, products, locations)
    if unassigned:
        s.violations.append(f"{len(unassigned)} product(s) could not be placed (capacity/eligibility/space).")
    s.moves = count_moves(base, assign)
    s.changes_assignment = s.moves > 0
    s.changes_routing = routing != base_routing
    if params.reassign_cost_per_move is not None:
        s.reloc_cost = s.moves * params.reassign_cost_per_move
    if s.violations:
        s.status = "Infeasible"
        return s
    s.result = evaluate(assign, orders, locations, params, routing, label=name, keep_routes=keep_routes)
    s.notes += s.result.notes
    return s


def build_scenarios(products, orders, locations, params: Params, abc_table: pd.DataFrame,
                    ext_assign: Optional[dict] = None, random_seed: int = 42) -> list:
    if not params.routing_assumption_accepted:
        raise ValueError("Accept the current-routing-policy assumption before evaluating baseline distances.")
    cur = current_assignment(products)
    if not cur:
        raise ValueError("Intervention analysis needs a baseline (current product-to-location assignment).")
    pol = params.current_routing_policy
    pol_name = ROUTING_POLICIES[pol]
    abc_a, abc_un = abc_assignment(abc_table, products, locations, params)
    rnd_a, rnd_un = random_assignment(products, locations, params, random_seed)
    a = dict(products=products, orders=orders, locations=locations, params=params, base=cur, base_routing=pol)
    sc = [
        _mk("S0", "Current arrangement + current routing", "baseline", "Current", pol, cur, **a,
            notes=[f"Current routing is ASSUMED to follow the {pol_name}."]),
        _mk("S1", "ABC storage + current routing", "SLA", "ABC (heuristic)", pol, abc_a, unassigned=abc_un, **a),
        _mk("S2", "Current storage + optimal routing", "PRO", "Current", "optimal", cur, **a),
        _mk("S3", "ABC storage + optimal routing (routes recalculated)", "SLA+PRO", "ABC (heuristic)", "optimal",
            abc_a, unassigned=abc_un, **a),
    ]
    if ext_assign:
        sc.append(_mk("S4", "External SLA assignment + optimal routing", "SLA+PRO",
                      "Externally supplied SLA assignment (optimality NOT verified)", "optimal", ext_assign, **a))
        sc.append(_mk("S4a", "External SLA assignment + current routing", "SLA",
                      "Externally supplied SLA assignment (optimality NOT verified)", pol, ext_assign, **a))
    else:
        sc.append(Scenario("S4", "Mathematical SLA + optimal routing", "SLA+PRO", "Mathematical SLA model", "optimal",
                           status="Model not integrated",
                           notes=["No mathematical SLA model is integrated; see dss/sla_interface.py."]))
    sc.append(_mk("R", "Random storage + current routing (reference only)", "reference", "Random (heuristic)", pol,
                  rnd_a, unassigned=rnd_un, **a))
    # the optimal-routing scenarios are only comparable if the exact limit was respected
    return sc


def comparison_table(scenarios: list, params: Params) -> pd.DataFrame:
    base = next(s for s in scenarios if s.sid == "S0")
    rows = []
    for s in scenarios:
        row = {"Scenario": s.sid, "Description": s.name, "Storage": s.assignment_label, "Routing": s.routing,
               "Status": s.status}
        if s.evaluated:
            k, kb = s.result.kpis, base.result.kpis
            row.update({
                "Total distance (m)": k["total_distance_m"],
                "Avg distance/order (m)": k["avg_distance_per_order_m"],
                "Avg travel time/order (s, est.)": k["avg_travel_time_per_order_s"],
                "Storage utilization (%)": k["utilization_pct"],
                "Avg locations/order": k["avg_locations_visited_per_order"],
                "Distance change vs S0 (%)": (kb["avg_distance_per_order_m"] - k["avg_distance_per_order_m"])
                / kb["avg_distance_per_order_m"] * 100,
                "Relocations": s.moves, "Relocation cost": s.reloc_cost,
                "Compute time (s)": s.result.runtime_s})
        else:
            for c in ["Total distance (m)", "Avg distance/order (m)", "Avg travel time/order (s, est.)",
                      "Storage utilization (%)", "Avg locations/order", "Distance change vs S0 (%)"]:
                row[c] = s.status if s.status == "Model not integrated" else "Not evaluated"
            row["Relocations"], row["Relocation cost"] = s.moves, s.reloc_cost
        row["Feasibility"] = "Feasible" if s.evaluated else ("; ".join(s.violations) or s.status)
        rows.append(row)
    return pd.DataFrame(rows)
