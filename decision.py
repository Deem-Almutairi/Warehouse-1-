"""Transparent decision engine (rule-based; NO weighted score, NO arbitrary KPI weights).

Order of reasoning
  1. Feasibility & mandatory constraints (assignment violations, max relocations, budget).
  2. Is there a gap against the USER-DEFINED targets? (no target -> no judgement of 'poor')
  3. Which component explains improvement? SLA (S1) vs PRO (S2) vs combined (S3); interaction = S3 - S1 - S2
     (effects are NOT assumed additive).
  4. SLA strategy: keep ABC if it meets targets; use a mathematical model only if one is integrated.
  5. Choose among feasible alternatives by the stated priority:
       least_effort    -> among alternatives meeting the targets, fewest relocations/lowest cost, then fewer
                          changed components, then larger improvement
       max_improvement -> largest measured improvement among feasible alternatives
  6. If the evidence is insufficient, no recommendation is made.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd
from config import Params

INFLUENCE = pd.DataFrame([
    ["Improve SLA (ABC / math model)", "Distance, time, locations visited per order; changes utilization pattern",
     "Does not change routing logic; requires physical relocations"],
    ["Improve PRO (better routing)", "Distance and time per order",
     "No relocations; cannot change storage utilization or how scattered the products are"],
    ["Improve both", "All of the above", "Effects are not additive; evaluated on the new locations with re-computed routes"],
], columns=["Intervention", "KPIs it can reasonably influence", "Notes"])

NAMES = {"S1": "ABC-based storage assignment", "S2": "Picking route optimization",
         "S3": "Combined storage (ABC) and routing improvement",
         "S4": "Mathematical/external storage optimization + routing", "S4a": "External storage assignment (current routing)"}


@dataclass
class Decision:
    recommendation: str
    status: str                  # recommended | no_intervention | no_recommendation | incomplete
    reasons: list = field(default_factory=list)
    limitations: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    evidence: pd.DataFrame = None
    gap_nature: str = ""
    attribution: dict = field(default_factory=dict)
    sla_decision: str = ""
    chosen: str = ""


def _metrics(s, params):
    k = s.result.kpis
    return k["avg_distance_per_order_m"], k.get("avg_travel_time_per_order_s")


def _targets(params):
    t = []
    if params.target_avg_distance_m is not None:
        t.append(("avg distance/order", "avg_distance_per_order_m", params.target_avg_distance_m))
    if params.target_avg_time_s is not None and params.picker_speed_mps:
        t.append(("avg travel time/order", "avg_travel_time_per_order_s", params.target_avg_time_s))
    return t


def _meets(s, targets):
    if not targets:
        return None
    return all(s.result.kpis.get(key) is not None and s.result.kpis[key] <= tgt for _, key, tgt in targets)


def _effort(s, params):
    cost = None
    if params.reassign_cost_per_move is not None and (params.routing_change_cost is not None or not s.changes_routing):
        cost = s.moves * params.reassign_cost_per_move + (params.routing_change_cost or 0) * int(s.changes_routing)
    comps = int(s.changes_assignment) + int(s.changes_routing)
    return (s.moves, comps), cost


def decide(scenarios: list, params: Params) -> Decision:
    by = {s.sid: s for s in scenarios}
    d = Decision("No recommendation yet", "no_recommendation")
    base = by.get("S0")
    if base is None or not base.evaluated:
        d.reasons.append("The baseline (current arrangement) could not be evaluated, so no intervention can be judged.")
        d.missing.append("Valid current storage assignment and layout; accepted routing-policy assumption.")
        return d
    targets = _targets(params)
    b_dist = base.result.kpis["avg_distance_per_order_m"]
    d.limitations += [
        "Distances are model-generated from single-block geometry; the current routing policy is an assumption.",
        "Travel time is an estimate (distance / speed) and excludes handling, waiting and congestion unless stated.",
        "Orders are routed one at a time; picker capacity is only flagged (no multi-trip splitting).",
        "ABC is a heuristic; no claim of optimal storage is made.",
        "The recommendation is only as valid as the supplied data; the sample dataset is synthetic."]
    math_ok = by.get("S4") is not None and by["S4"].evaluated
    if not math_ok:
        d.limitations.append("Mathematical SLA model: NOT integrated - the SLA comparison is incomplete.")
    # --- step 1: candidates, feasibility ---
    cands, rows = [], []
    for sid in ("S1", "S2", "S3", "S4a", "S4"):
        s = by.get(sid)
        if s is None:
            continue
        row = {"Alternative": f"{sid}: {NAMES[sid]}", "Status": s.status}
        if not s.evaluated:
            row.update({"Feasible": "No" if s.status == "Infeasible" else "Not evaluated",
                        "Why": "; ".join(s.violations) or s.status})
            rows.append(row)
            continue
        dist, tm = _metrics(s, params)
        imp = (b_dist - dist) / b_dist * 100
        reasons = []
        if params.max_moves_allowed is not None and s.moves > params.max_moves_allowed:
            reasons.append(f"{s.moves} relocations exceed the allowed {params.max_moves_allowed}")
        if params.reassign_budget is not None and s.reloc_cost is not None and s.reloc_cost > params.reassign_budget:
            reasons.append(f"relocation cost {s.reloc_cost:.0f} exceeds budget {params.reassign_budget:.0f}")
        feas = not reasons
        meets = _meets(s, targets)
        eff, cost = _effort(s, params)
        row.update({"Feasible": "Yes" if feas else "No", "Avg distance/order (m)": dist,
                    "Improvement vs S0 (%)": imp, "Meets targets": ("n/a (no target)" if meets is None else ("Yes" if meets else "No")),
                    "Relocations": s.moves, "Relocation/adoption cost": cost,
                    "Compute time (s)": s.result.runtime_s, "Why": "; ".join(reasons)})
        rows.append(row)
        if feas:
            cands.append(dict(sid=sid, s=s, imp=imp, meets=meets, eff=eff, cost=cost, dist=dist))
    d.evidence = pd.DataFrame(rows)
    # effort key: money only if it is known for EVERY feasible candidate, otherwise relocations (then components)
    if cands and all(c["cost"] is not None for c in cands):
        for c in cands:
            c["eff"] = (c["cost"], c["eff"][1])
        effort_basis = "cost"
    else:
        effort_basis = "relocations"
    # --- step 2: gap vs targets ---
    base_meets = _meets(base, targets)
    if not targets:
        d.missing.append("User-defined target(s) (average distance or time per order) - needed to decide whether "
                         "performance is insufficient and which alternative is 'enough'.")
    # --- step 3: attribution ---
    if all(k in by and by[k].evaluated for k in ("S1", "S2", "S3")):
        dS, dP, dC = (b_dist - by["S1"].result.kpis["avg_distance_per_order_m"],
                      b_dist - by["S2"].result.kpis["avg_distance_per_order_m"],
                      b_dist - by["S3"].result.kpis["avg_distance_per_order_m"])
        d.attribution = {"SLA only (S1) reduction, m/order": dS, "PRO only (S2) reduction, m/order": dP,
                         "Combined (S3) reduction, m/order": dC, "Interaction = S3 - S1 - S2, m/order": dC - dS - dP}
        mat = params.min_material_improvement_pct / 100 * b_dist
        sla_m, pro_m = dS > mat and dS > 0, dP > mat and dP > 0
        d.gap_nature = ("Both storage assignment and picking routes contribute" if sla_m and pro_m else
                        "Mainly storage assignment" if sla_m else "Mainly picking routes" if pro_m else
                        "No evaluated intervention gives a material improvement - cause cannot be determined from these alternatives")
        d.reasons.append(f"Cause analysis: {d.gap_nature}. Largest single-factor reduction is not assumed to be the "
                         f"only cause; the combined effect ({dC:.1f} m/order) differs from the sum of the parts "
                         f"({dS + dP:.1f} m/order) by {dC - dS - dP:+.1f} m/order.")
    if base_meets:
        d.recommendation, d.status, d.chosen = "No intervention required (baseline already meets the targets)", "no_intervention", "S0"
        d.reasons.append("The baseline meets all user-defined targets, so there is no supported performance gap.")
        return d
    # --- step 4: SLA strategy (ABC vs math) ---
    abc_ok = [c for c in cands if c["sid"] in ("S1", "S3") and c["meets"]]
    if targets and abc_ok:
        d.sla_decision = ("If a storage change is pursued, ABC already meets the targets, so a mathematical SLA model "
                          "is not needed on the available evidence.")
    elif targets and not math_ok:
        d.sla_decision = "ABC does not meet the targets and no mathematical model is integrated: SLA comparison is INCOMPLETE."
    elif targets and math_ok:
        d.sla_decision = "ABC does not meet the targets; the externally supplied SLA assignment was evaluated (see table)."
    else:
        d.sla_decision = "Cannot judge ABC vs mathematical SLA without a target."
    d.reasons.append(d.sla_decision)
    if not cands:
        d.reasons.append("No feasible alternative could be evaluated.")
        return d
    # --- step 5: choose ---
    if params.priority == "max_improvement":
        pool = [c for c in cands if c["imp"] > max(params.min_material_improvement_pct, 0)]
        if not pool:
            d.reasons.append("No feasible alternative exceeds the minimum material improvement.")
            d.status = "no_recommendation"
            return d
        best = sorted(pool, key=lambda c: (-c["imp"], c["eff"]))[0]
        d.reasons.append("Priority = maximise measured distance reduction among feasible alternatives.")
    else:
        if not targets:
            d.reasons.append("Priority = 'least effort that meets the targets', but no target is defined: no recommendation is forced.")
            d.status = "no_recommendation"
            return d
        meeting = [c for c in cands if c["meets"]]
        if not meeting:
            best_partial = sorted(cands, key=lambda c: -c["imp"])[0]
            d.status = "incomplete"
            d.recommendation = "No evaluated alternative meets the targets"
            d.reasons.append(f"Closest evaluated alternative: {NAMES[best_partial['sid']]} "
                             f"({best_partial['dist']:.1f} m/order, {best_partial['imp']:.1f}% better than S0) but it "
                             "does not reach the target. Additional options (e.g. a validated mathematical SLA model, "
                             "layout change, batching) or a revised target are required.")
            d.chosen = best_partial["sid"]
            return d
        best = sorted(meeting, key=lambda c: (c["eff"], -c["imp"]))[0]
        d.reasons.append("Priority = least implementation effort among feasible alternatives that meet the targets "
                         f"(effort basis used here: {effort_basis}; ties -> fewer changed components, then larger improvement).")
    s = best["s"]
    d.chosen, d.status, d.recommendation = best["sid"], "recommended", NAMES[best["sid"]]
    d.reasons.append(f"{NAMES[best['sid']]}: {best['dist']:.1f} m/order vs {b_dist:.1f} baseline "
                     f"({best['imp']:.1f}% reduction, {s.moves} relocations).")
    others = [c for c in cands if c["sid"] != best["sid"]]
    for c in others:
        d.reasons.append(f"Compared with {NAMES[c['sid']]}: {c['dist']:.1f} m/order, {c['s'].moves} relocations, "
                         f"meets targets: {'n/a' if c['meets'] is None else ('yes' if c['meets'] else 'no')}.")
    if best["sid"] in ("S2", "S3", "S4") and params.routing_change_cost is None:
        d.limitations.append("Effort of adopting the improved routing method is not quantified (routing_change_cost not supplied).")
    if params.reassign_cost_per_move is None:
        d.missing.append("Reassignment cost per move - effort was compared by number of relocations only.")
    return d
