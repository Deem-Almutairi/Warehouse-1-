"""Primary DSS workflow: compare storage STRATEGIES (Baseline / ABC / Mathematical model) on identical data,
layout and routing method, check feasibility, and recommend by OPI.

OPI (Overall Performance Index) = Total Picking Distance. Lower is better. No weights, no other scores.
Storage assignment and routing remain separate components: the strategy produces an assignment, then the
routing method (same for all strategies) is applied to every order on the NEW locations.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
from config import Params
from kpi import evaluate, distance_reduction_pct, EvalResult
from assignment import current_assignment, abc_assignment, feasibility_report, count_moves

KEYS = ("baseline", "abc", "math")
LABELS = {"baseline": "Baseline (current assignment)", "abc": "ABC-based storage (heuristic)",
          "math": "Mathematical model"}


@dataclass
class StrategyResult:
    key: str
    name: str
    status: str                       # Evaluated | Not available | Not implemented | Not evaluable
    routing: str = ""
    assign: dict = field(default_factory=dict)
    result: Optional[EvalResult] = None
    checks: list = field(default_factory=list)
    feasibility: str = "Not checked"
    moves: Optional[int] = None
    reloc_cost: Optional[float] = None
    reduction_pct: Optional[float] = None
    notes: list = field(default_factory=list)

    @property
    def evaluated(self) -> bool:
        return self.status == "Evaluated" and self.result is not None

    @property
    def total_distance(self):
        return self.result.kpis["total_distance_m"] if self.evaluated else None

    @property
    def opi(self):                    # OPI := Total Picking Distance
        return self.total_distance

    @property
    def avg_distance(self):
        return self.result.kpis["avg_distance_per_order_m"] if self.evaluated else None

    @property
    def is_feasible(self) -> bool:
        return self.feasibility.startswith("Feasible")


def _evaluable(assign: dict, orders: pd.DataFrame, locations: pd.DataFrame):
    locs = set(locations["location_id"])
    bad = sorted({p for p in orders["product_id"] if p not in assign or assign[p] not in locs})
    return (not bad), bad


def _one(key, name, assign, base_assign, products, orders, locations, params, routing, notes=None):
    r = StrategyResult(key, name, "Evaluated", routing, assign=assign, notes=list(notes or []))
    r.checks, r.feasibility = feasibility_report(assign, products, locations)
    if base_assign:
        r.moves = count_moves(base_assign, assign)
        if params.reassign_cost_per_move is not None:
            r.reloc_cost = r.moves * params.reassign_cost_per_move
    ok, bad = _evaluable(assign, orders, locations)
    if not ok:
        r.status = "Not evaluable"
        r.notes.append(f"Routes cannot be calculated: {len(bad)} ordered product(s) have no valid location (e.g. {bad[:3]}).")
        return r
    r.result = evaluate(assign, orders, locations, params, routing, label=name)
    r.notes += r.result.notes
    return r


def evaluate_strategies(products, orders, locations, params: Params, abc_tbl: pd.DataFrame,
                        ext_assign: Optional[dict] = None, routing: Optional[str] = None) -> list:
    routing = routing or params.evaluation_routing
    base = current_assignment(products)
    out = []
    if base:
        out.append(_one("baseline", LABELS["baseline"], base, base, products, orders, locations, params, routing))
    else:
        out.append(StrategyResult("baseline", LABELS["baseline"], "Not available",
                                  notes=["No current product-to-location assignment was supplied."]))
    abc_a, abc_un = abc_assignment(abc_tbl, products, locations, params)
    out.append(_one("abc", LABELS["abc"], abc_a, base, products, orders, locations, params, routing))
    if ext_assign:
        out.append(_one("math", "Mathematical model (externally supplied assignment; optimality NOT verified)",
                        ext_assign, base, products, orders, locations, params, routing,
                        notes=["The assignment was imported from outside this application; it was not produced or verified here."]))
    else:
        out.append(StrategyResult("math", LABELS["math"], "Not implemented",
                                  notes=["No mathematical storage-assignment model is implemented or imported "
                                         "(the reference repository only contains picker-routing code)."]))
    ref = out[0].total_distance if out[0].evaluated else None
    for r in out:
        if r.evaluated:
            r.reduction_pct = 0.0 if r.key == "baseline" and ref is not None else distance_reduction_pct(ref, r.total_distance)
    return out


def comparison_table(results: list) -> pd.DataFrame:
    """Rows = metrics, columns = strategies (as in the specification). Only calculated values are shown."""
    cols = {}
    for r in results:
        label = {"baseline": "Baseline", "abc": "ABC", "math": "Mathematical Model"}[r.key]
        if r.evaluated:
            if r.key == "baseline":
                red = "Reference"
            elif r.reduction_pct is not None:
                red = f"{r.reduction_pct:.2f}"
            else:
                red = "Undefined (baseline distance is 0)" if results[0].evaluated else "Baseline unavailable"
            cols[label] = {
                "Total Picking Distance (m)": f"{r.total_distance:,.2f}",
                "Average Picking Distance per Order (m)": f"{r.avg_distance:,.2f}",
                "Picking Distance Reduction (%)": red,
                "Storage Assignment Feasibility": r.feasibility,
                "OPI (= Total Picking Distance, m)": f"{r.opi:,.2f}",
            }
        else:
            na = "Not implemented" if r.status == "Not implemented" else "Unavailable"
            cols[label] = {m: na for m in ["Total Picking Distance (m)", "Average Picking Distance per Order (m)",
                                            "Picking Distance Reduction (%)", "Storage Assignment Feasibility",
                                            "OPI (= Total Picking Distance, m)"]}
            if r.status == "Not evaluable":
                cols[label]["Storage Assignment Feasibility"] = r.feasibility
    return pd.DataFrame(cols)


@dataclass
class StrategyDecision:
    status: str                       # recommended | no_recommendation
    selected: Optional[str] = None
    explanation: list = field(default_factory=list)
    limitations: list = field(default_factory=list)
    ranking: pd.DataFrame = None
    tie: bool = False
    similar: bool = False
    routing: str = ""


def recommend(results: list, params: Params) -> StrategyDecision:
    """Select the FEASIBLE evaluated strategy with the lowest OPI (total picking distance).
    Ties: fewer relocations, then fixed order baseline < ABC < math (reported, never hidden)."""
    d = StrategyDecision("no_recommendation", routing=results[0].routing if results and results[0].routing else
                         next((r.routing for r in results if r.routing), ""))
    rows, cands = [], []
    for r in results:
        row = {"Strategy": r.name, "Status": r.status, "OPI / total distance (m)": r.opi, "Feasibility": r.feasibility,
               "Relocations": r.moves, "Considered": ""}
        if not r.evaluated:
            row["Considered"] = "No - " + (r.notes[0] if r.notes else r.status)
        elif not r.is_feasible:
            row["Considered"] = "No - infeasible: " + "; ".join(c[2] for c in r.checks if c[1] == "fail")
        elif (params.max_moves_allowed is not None and r.key != "baseline" and r.moves is not None
              and r.moves > params.max_moves_allowed):
            row["Considered"] = f"No - {r.moves} relocations exceed the user limit of {params.max_moves_allowed}"
        else:
            row["Considered"] = "Yes"
            cands.append(r)
        rows.append(row)
    d.ranking = pd.DataFrame(rows)
    inf_shorter = [r for r in results if r.evaluated and not r.is_feasible and cands
                   and r.opi < min(c.opi for c in cands)]
    if not any(r.key == "math" and r.evaluated for r in results):
        d.limitations.append("Mathematical model: NOT implemented/imported - the strategy comparison is incomplete.")
    d.limitations += [
        f"Distances come from single-block geometry with routing method '{d.routing}' applied identically to all strategies.",
        "Orders are routed individually (no batching).",
        "ABC is a heuristic; its result is not claimed to be optimal.",
        "Historical orders only; no forecast is used."]
    unver = sorted({c[0] for r in cands for c in r.checks if c[1] == "unverifiable"})
    if unver:
        d.limitations.append("Constraints that could not be verified (no data): " + "; ".join(unver) + ".")
    if len(cands) < 2:
        d.explanation.append("A valid winner cannot be established: fewer than two feasible, evaluated strategies are available.")
        for r in inf_shorter:
            d.explanation.append(f"{r.name} has a shorter distance ({r.opi:,.1f} m) but is infeasible and is not recommended.")
        return d
    best = min(c.opi for c in cands)
    tied = [c for c in cands if abs(c.opi - best) <= 1e-9 * max(1.0, best)]
    order = {k: i for i, k in enumerate(KEYS)}
    if len(tied) > 1:
        d.tie = True
        tied.sort(key=lambda c: ((c.moves if c.moves is not None else 0), order[c.key]))
    sel = tied[0]
    d.status, d.selected = "recommended", sel.key
    d.explanation.append(f"{sel.name} was selected because it is feasible and achieved the lowest total picking distance "
                         f"(OPI = {sel.opi:,.2f} m) among the {len(cands)} feasible strategies evaluated under the same data and routing method.")
    if d.tie:
        d.explanation.append("TIE: " + ", ".join(c.name for c in tied) + f" have identical OPI ({best:,.2f} m). "
                             "Predefined tie-break: fewer relocations first, then the order Baseline < ABC < Mathematical model.")
    rest = sorted([c for c in cands if c is not sel], key=lambda c: c.opi)
    if rest:
        gap = (rest[0].opi - sel.opi) / rest[0].opi * 100 if rest[0].opi else 0.0   # distance saved vs runner-up
        if gap <= params.similarity_tolerance_pct and not d.tie:
            d.similar = True
            d.explanation.append(f"The result is close: {sel.name} is only {gap:.2f}% shorter than the runner-up ({rest[0].name}), "
                                 f"below the {params.similarity_tolerance_pct:g}% similarity threshold, so this should not be read as a significant improvement.")
        elif not d.tie:
            d.explanation.append(f"Runner-up: {rest[0].name} with {rest[0].opi:,.2f} m; the selected strategy is {gap:.1f}% shorter.")
    for r in inf_shorter:
        d.explanation.append(f"{r.name} produced a shorter distance ({r.opi:,.2f} m) but violates constraints and was NOT eligible.")
    if sel.key == "baseline":
        d.explanation.append("No evaluated alternative beats the current assignment, so keeping it is supported by the results.")
    return d
