"""Exports and methodology text."""
from __future__ import annotations
import io
import pandas as pd
from kpi import KPI_DEFS, gap_table
from scenarios import comparison_table as intervention_table
from strategies import comparison_table as strategy_table
from decision import INFLUENCE

COMPONENT_STATUS = pd.DataFrame([
    ["Data input, templates, validation", "Implemented", "Unit-tested (validation rules)"],
    ["Demand indicators, system-level KPIs, ABC distribution", "Implemented", "Hand-computed tests"],
    ["Forecasting", "NOT implemented", "Historical data only; no forecast values are produced"],
    ["Baseline / ABC strategies, distance KPIs, OPI, distance reduction (%)", "Implemented", "Hand-computed tests incl. zero-baseline guard"],
    ["Feasibility checks (Feasible / Infeasible / unverified)", "Implemented", "Unit-tested"],
    ["Strategy recommendation (lowest OPI among feasible)", "Implemented", "Unit-tested incl. ties, similar results, infeasible-but-shorter"],
    ["ABC classification + ABC assignment", "Implemented (HEURISTIC)", "Unit-tested"],
    ["Routing: S-shape / return policies", "Implemented (HEURISTICS, own code)", "Hand-computed cases tested"],
    ["Routing: exact (Held-Karp), own code", "Implemented", "Verified against brute force; never worse than policies"],
    ["Routing: DAP from github.com/MyReStudy/single_block", "Adapter written - NOT EXECUTED", "Only input-file generation tested; run scripts/validate_dap.py"],
    ["Mathematical SLA optimisation (GAMS etc.)", "PENDING - no model exists", "File interface documented (dss/sla_interface.py); imported assignments accepted"],
    ["Batch order picking", "NOT supported", "Routing handles individual orders only"],
    ["Scenario analysis (optional, hypothetical)", "Implemented", "Unit-tested; clearly labelled hypothetical"],
    ["SLA/PRO intervention analysis (additional)", "Implemented", "Unit-tested"],
    ["Streamlit UI", "Implemented", "Not run in the build environment (Streamlit not installable offline); stub smoke test only"],
    ["Hosting", "Streamlit needs a long-running server: NOT deployable on Vercel", "Docker/Render/Streamlit Cloud files provided; deployment itself unverified"],
], columns=["Component", "Status", "Verification"])

METHODOLOGY = """# Methodology

**Primary workflow.** Validate data -> demand indicators and ABC -> generate the assignment of each implemented strategy
(Baseline, ABC, Mathematical model if available/imported) -> calculate routes on the NEW locations with ONE routing method
used for every strategy -> check feasibility -> compare by OPI -> recommend the feasible strategy with the lowest OPI.

**OPI = Total Picking Distance** (sum over evaluated orders; order frequencies, if supplied, act as repetition counts). Lower is
better. No weights and no other composite score. **Picking Distance Reduction (%) = (Baseline - Strategy) / Baseline x 100**
(positive = shorter; undefined if the baseline distance is 0 and the strategy is not 0).

**Feasibility.** Every product assigned; assigned locations exist; one product per location; eligibility and capacity checked
when data exist, otherwise reported as *unverified*. Infeasible strategies are never recommended even if shorter.

**Ties / similar results.** Exact ties are reported and broken by fewer relocations, then Baseline < ABC < Mathematical model.
A runner-up within the similarity threshold (reporting only) is described as similar, not as a significant improvement.
If fewer than two feasible, evaluated strategies exist, no recommendation is made.

**Routing.** Single-block geometry, equal aisle length, depot at the front of a chosen aisle, individual orders (no batching).
Default evaluation method: exact optimal route (Held-Karp) for <= exact_limit pick locations; larger orders use a flagged
heuristic fallback. S-shape / return policies are selectable and shown as a sensitivity table.

**Scenario analysis (optional).** Hypothetical demand growth or demand-distribution changes, labelled as such and never mixed with
historical results. No forecasting is implemented.

**Additional intervention analysis.** Compares SLA-only, PRO-only and combined improvements (S0-S4) against user-defined targets;
it is separate from the strategy recommendation and requires an explicitly accepted current-routing assumption.
"""


def _sys_frame(k: dict) -> pd.DataFrame:
    rows = [(n, v) for n, v in k.items() if not isinstance(v, dict)]
    for c, v in (k.get("abc_distribution") or {}).items():
        rows.append((f"ABC class {c}", f"{v['products']} products ({v['share_of_products_pct']:.1f}% of products, "
                                       f"{v['share_of_activity_pct']:.1f}% of activity)"))
    return pd.DataFrame(rows, columns=["KPI", "Value"]).astype(str)


def markdown_report(strat_results, strat_decision, params, sample: bool, sys_kpis: dict) -> str:
    out = ["# Decision report - Integrated DSS for Warehouse Performance Optimization", ""]
    if sample:
        out += ["> **SAMPLE DATA - synthetic; results illustrate the method and are not real warehouse results.**", ""]
    sel = next((r for r in strat_results if r.key == strat_decision.selected), None)
    out.append("## Recommended storage strategy")
    if sel is None:
        out.append("**No recommendation** - " + " ".join(strat_decision.explanation))
    else:
        red = "n/a" if sel.reduction_pct is None else f"{sel.reduction_pct:.2f}%"
        out += [f"**{sel.name}**", f"- Feasibility: {sel.feasibility}",
                f"- OPI / total picking distance: {sel.opi:,.2f} m", f"- Average distance per order: {sel.avg_distance:,.2f} m",
                f"- Reduction vs baseline: {red}", f"- Routing method (same for all strategies): {strat_decision.routing}"]
    out += ["", "## Explanation"] + [f"- {e}" for e in strat_decision.explanation]
    out += ["", "## Limitations / missing information"] + [f"- {l}" for l in strat_decision.limitations]
    out += ["", "## Strategy comparison", strategy_table(strat_results).to_string(),
            "", "## System-level KPIs", _sys_frame(sys_kpis).to_string(index=False),
            "", "## Component status", COMPONENT_STATUS.to_string(index=False), "", METHODOLOGY]
    return "\n".join(out)


def excel_report(strat_results, strat_decision, params, abc_table, sys_kpis, intervention=None, int_decision=None) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        strategy_table(strat_results).to_excel(xw, sheet_name="Strategy comparison")
        strat_decision.ranking.to_excel(xw, sheet_name="Eligibility", index=False)
        pd.DataFrame({"Selected": [strat_decision.selected or "none"], "Status": [strat_decision.status],
                      "Explanation": [" | ".join(strat_decision.explanation)],
                      "Limitations": [" | ".join(strat_decision.limitations)]}).to_excel(xw, sheet_name="Recommendation", index=False)
        _sys_frame(sys_kpis).to_excel(xw, sheet_name="System KPIs", index=False)
        abc_table.to_excel(xw, sheet_name="ABC", index=False)
        rows = [{"strategy": r.key, "product_id": p, "location_id": l} for r in strat_results for p, l in r.assign.items()]
        pd.DataFrame(rows).to_excel(xw, sheet_name="Assignments", index=False)
        pd.DataFrame([{"strategy": r.key, "check": c[0], "result": c[1], "detail": c[2]} for r in strat_results for c in r.checks]
                     ).to_excel(xw, sheet_name="Feasibility checks", index=False)
        KPI_DEFS.to_excel(xw, sheet_name="KPI definitions", index=False)
        if intervention:
            intervention_table(intervention, params).to_excel(xw, sheet_name="Intervention analysis", index=False)
            if int_decision is not None and int_decision.evidence is not None:
                int_decision.evidence.to_excel(xw, sheet_name="Intervention evidence", index=False)
            INFLUENCE.to_excel(xw, sheet_name="Intervention influence", index=False)
        params.to_frame().to_excel(xw, sheet_name="Parameters", index=False)
        COMPONENT_STATUS.to_excel(xw, sheet_name="Component status", index=False)
    return buf.getvalue()
