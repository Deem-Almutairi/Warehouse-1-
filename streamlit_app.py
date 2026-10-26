"""Streamlit app - Integrated Decision Support System for Warehouse Performance Optimization.

Run locally:  streamlit run streamlit_app.py
NOTE: this is a Streamlit script, not an ASGI/WSGI app. It needs a long-running server with WebSocket support,
so it cannot be hosted on Vercel's serverless Python runtime (see DEPLOYMENT.md). The file is deliberately NOT
called app.py so that Vercel's Python auto-detection does not mistake it for a Flask/FastAPI entrypoint.
"""
import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import Params, ROUTING_POLICIES
from data import (generate_sample, read_table, templates, to_csv_bytes, validate, SAMPLE_LABEL)
from abc_analysis import MEASURES, activity_table, abc_classify
from assignment import (abc_assignment, current_assignment, external_assignment, utilization)
from layout import location_points
from routing import route
from kpi import KPI_DEFS, gap_table
from scenarios import build_scenarios, comparison_table
from decision import decide, INFLUENCE
from dap_adapter import dap_status
from sla_interface import export_inputs, STATUS as SLA_STATUS
from report import COMPONENT_STATUS, METHODOLOGY, excel_report, markdown_report
from demand import product_demand, system_kpis, order_profile
from strategies import evaluate_strategies, comparison_table as strategy_table, recommend, KEYS
from scenario_analysis import KINDS as SCEN_KINDS, run_scenario

st.set_page_config(page_title="Warehouse Performance DSS", page_icon="📦", layout="wide")
PAGES = ["1 Executive Overview", "2 Data Input", "3 ABC Analysis", "4 Storage Strategies",
         "5 Order Picking Routes", "6 KPI Analysis", "7 Decision Engine", "8 Reports"]
S = st.session_state


# ----------------------------------------------------------------------------- helpers
ROUTING_LABELS = {"optimal": "Optimal route (exact Held-Karp; heuristic fallback is flagged)", **ROUTING_POLICIES}


def have_data():
    return all(k in S for k in ("products", "orders", "locations", "params"))


def need_data():
    if not have_data():
        st.info("Load the sample dataset or upload your data on **2 Data Input** first.")
        st.stop()


def need_eval():
    need_data()
    if "strat" not in S:
        st.info("Run the evaluation first (button on **4 Storage Strategies**).")
        st.stop()


def abc_table():
    a, b = S.get("abc_thr", (80, 15))
    act = activity_table(S.products, S.orders, S.get("abc_measure", "pick_frequency"))
    return abc_classify(act, a, b)


def run_eval():
    p = S.params
    rep = validate(S.products, S.orders, S.locations, p, S.get("is_sample", False))
    if not rep.ok:
        st.error("Fix the data validation errors on **2 Data Input** first.")
        return
    try:
        with st.spinner("Evaluating strategies and re-computing routes for each assignment..."):
            tbl = abc_table()
            S.strat = evaluate_strategies(S.products, S.orders, S.locations, p, tbl, S.get("ext_assign"))
            S.strat_dec = recommend(S.strat, p)
            sens = []
            for m in ("optimal", "s_shape", "return"):          # sensitivity: does the winner depend on the routing method?
                rs = evaluate_strategies(S.products, S.orders, S.locations, p, tbl, S.get("ext_assign"), routing=m)
                dd = recommend(rs, p)
                row = {"Routing method": m}
                for r in rs:
                    row[r.key] = r.opi if r.evaluated else "n/a"
                row["Recommended"] = dd.selected or "none"
                sens.append(row)
            S.sens = pd.DataFrame(sens)
            S.pop("scenarios", None); S.pop("decision", None)
            if "current_location" in S.products and p.routing_assumption_accepted:
                S.scenarios = build_scenarios(S.products, S.orders, S.locations, p, tbl, S.get("ext_assign"))
                S.decision = decide(S.scenarios, p)
    except Exception as e:  # surfaced to the user, never silent
        st.error(f"Evaluation failed: {e}")


def fmt(v, nd=1):
    if v is None:
        return "Unavailable"
    return f"{v:,.{nd}f}" if isinstance(v, (int, float)) else str(v)


def layout_fig(assign, tbl, title):
    loc = S.locations.copy()
    inv = {l: p for p, l in assign.items()}
    cls = tbl.set_index("product_id")["abc_class"].to_dict()
    loc["product"] = loc["location_id"].map(inv)
    loc["class"] = loc["product"].map(cls).fillna("empty")
    side = loc["side"] if "side" in loc else pd.Series("L", index=loc.index)
    loc["x"] = (loc["aisle"] - 1) * S.params.aisle_spacing_m + side.map({"L": -0.5, "R": 0.5}).fillna(0)
    fig = px.scatter(loc, x="x", y="position_m", color="class", hover_data=["location_id", "product"],
                     color_discrete_map={"A": "#d62728", "B": "#ff9f1c", "C": "#2a9d8f", "empty": "#cccccc"}, title=title)
    fig.add_trace(go.Scatter(x=[(S.params.depot_aisle - 1) * S.params.aisle_spacing_m], y=[0], mode="markers+text",
                             text=["Depot"], textposition="bottom center", marker=dict(symbol="star", size=14, color="black"),
                             name="Depot"))
    fig.update_yaxes(title="Distance from front cross-aisle (m)", autorange="reversed")
    fig.update_xaxes(title="Aisle position (m)")
    return fig


def recommendation_panel():
    d, res = S.strat_dec, S.strat
    sel = next((r for r in res if r.key == d.selected), None)
    if sel is None:
        st.warning("**Recommended storage strategy: none** - a valid winner cannot be established.")
    else:
        st.success(f"**Recommended storage strategy: {sel.name}**")
        c = st.columns(4)
        c[0].metric("OPI = total picking distance (m)", fmt(sel.opi, 2))
        c[1].metric("Avg distance / order (m)", fmt(sel.avg_distance, 2))
        c[2].metric("Reduction vs baseline (%)", "Reference" if sel.key == "baseline" else fmt(sel.reduction_pct, 2))
        c[3].metric("Feasibility", sel.feasibility)
    for e in d.explanation:
        st.markdown(f"- {e}")
    with st.expander("Limitations and missing information", expanded=True):
        for l in d.limitations:
            st.markdown(f"- {l}")


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("📦 Warehouse DSS")
page = st.sidebar.radio("Navigate", PAGES)
if S.get("is_sample"):
    st.sidebar.warning(SAMPLE_LABEL)
st.sidebar.caption("Workflow: Data → Demand & ABC → Strategies → Routes → Feasibility → OPI comparison → Recommendation")
st.title("Integrated Decision Support System for Warehouse Performance Optimization")

# ----------------------------------------------------------------------------- 2 Data input
if page == PAGES[1]:
    st.header("Warehouse data input")
    c1, c2 = st.columns(2)
    if c1.button("Load SAMPLE dataset", type="primary"):
        pr, o, l, p = generate_sample()
        S.update(products=pr, orders=o, locations=l, params=p, is_sample=True)
        for k in ("strat", "strat_dec", "sens", "scenarios", "decision", "ext_assign", "scen_out"):
            S.pop(k, None)
        st.rerun()
    with c2:
        for name, df in templates().items():
            st.download_button(f"Download {name}", to_csv_bytes(df), name, key=name)
    with st.expander("Upload your own files (CSV or Excel)", expanded=not have_data()):
        up = {k: st.file_uploader(f"{k} file", type=["csv", "xlsx", "xls"], key=f"up_{k}")
              for k in ("products", "orders", "locations", "parameters (optional)")}
        if st.button("Use uploaded files"):
            try:
                tabs = {k: read_table(up[k]) for k in ("products", "orders", "locations") if up[k]}
                if len(tabs) < 3:
                    st.error("Upload products, orders and locations files.")
                else:
                    prm = up["parameters (optional)"]
                    S.params = Params.from_frame(read_table(prm)) if prm else Params()
                    S.products, S.orders, S.locations = tabs["products"], tabs["orders"], tabs["locations"]
                    S.is_sample = False
                    for k in ("strat", "strat_dec", "sens", "scenarios", "decision", "scen_out"):
                        S.pop(k, None)
                    st.rerun()
            except Exception as e:
                st.error(f"Could not read the files: {e}")
    if have_data():
        p = S.params
        st.subheader("Evaluation settings")
        e1, e2 = st.columns(2)
        p.evaluation_routing = e1.selectbox("Routing method used for ALL strategies in the comparison", list(ROUTING_LABELS),
                                            index=list(ROUTING_LABELS).index(p.evaluation_routing), format_func=ROUTING_LABELS.get)
        p.similarity_tolerance_pct = e2.number_input("Similarity threshold (%) - reporting only, never changes the winner",
                                                     value=float(p.similarity_tolerance_pct), min_value=0.0)
        p.exact_limit = int(st.slider("Exact-routing limit (distinct pick locations per order)", 6, 14, int(p.exact_limit)))
        st.subheader("Layout, operational data and targets")
        st.caption("Leave a field empty if it is not available - it will be reported as unavailable, never invented.")
        a, b, c = st.columns(3)
        p.aisle_length_m = a.number_input("Aisle length (m)", value=float(p.aisle_length_m), min_value=1.0)
        p.aisle_spacing_m = a.number_input("Aisle spacing (m)", value=float(p.aisle_spacing_m), min_value=0.1)
        p.depot_aisle = int(a.number_input("Depot in front of aisle #", value=int(p.depot_aisle), min_value=1, step=1))
        p.picker_speed_mps = b.number_input("Picker speed (m/s)", value=p.picker_speed_mps, min_value=0.01)
        p.service_time_s_per_line = b.number_input("Service/handling time per order line (s)", value=p.service_time_s_per_line, min_value=0.0)
        p.picker_capacity_units = b.number_input("Picker capacity (units)", value=p.picker_capacity_units, min_value=0.0)
        p.reassign_cost_per_move = c.number_input("Cost per product relocation", value=p.reassign_cost_per_move, min_value=0.0)
        mm = c.number_input("Max relocations allowed", value=p.max_moves_allowed, min_value=0, step=1)
        p.max_moves_allowed = None if mm is None else int(mm)
        p.reassign_budget = c.number_input("Relocation budget", value=p.reassign_budget, min_value=0.0)
        p.routing_change_cost = c.number_input("One-off cost of adopting improved routing", value=p.routing_change_cost, min_value=0.0)
        with st.expander("Targets and assumption for the additional SLA-vs-PRO intervention analysis (optional)"):
            t1, t2, t3, t4 = st.columns(4)
            p.target_avg_distance_m = t1.number_input("Avg distance/order target (m)", value=p.target_avg_distance_m, min_value=0.0)
            p.target_avg_time_s = t2.number_input("Avg travel time/order target (s)", value=p.target_avg_time_s, min_value=0.0)
            p.target_max_order_distance_m = t3.number_input("Max distance per order (m)", value=p.target_max_order_distance_m, min_value=0.0)
            p.target_max_order_time_s = t4.number_input("Max time per order (s)", value=p.target_max_order_time_s, min_value=0.0)
            p.current_routing_policy = st.selectbox("Current routing practice (ASSUMED - real routes are not in the data)",
                                                    list(ROUTING_POLICIES), index=list(ROUTING_POLICIES).index(p.current_routing_policy),
                                                    format_func=ROUTING_POLICIES.get)
            p.routing_assumption_accepted = st.checkbox("I accept this routing-policy assumption for the intervention analysis",
                                                        value=p.routing_assumption_accepted)
        S.params = p
        rep = validate(S.products, S.orders, S.locations, p, S.get("is_sample", False))
        st.subheader("Validation")
        for e in rep.errors:
            st.error(e)
        for w in rep.warnings:
            st.warning(w)
        if rep.ok:
            st.success("Data passed validation.")
        st.dataframe(pd.DataFrame([{"Analysis": k, "Possible": "Yes" if v[0] else "No", "Condition": v[1]}
                                   for k, v in rep.availability.items()]), hide_index=True, use_container_width=True)
        t = st.tabs(["Demand summary", "Products", "Orders", "Locations"])
        if rep.ok:
            k = system_kpis(S.products, S.orders)
            t[0].write({"Total products": k["total_products"], "Total orders": k["total_orders"],
                        "Total demand quantity": fmt(k["total_demand_quantity"], 0)})
            t[0].dataframe(product_demand(S.products, S.orders), use_container_width=True, hide_index=True)
        t[1].dataframe(S.products, use_container_width=True); t[2].dataframe(S.orders, use_container_width=True)
        t[3].dataframe(S.locations, use_container_width=True)

# ----------------------------------------------------------------------------- 3 ABC
elif page == PAGES[2]:
    need_data()
    st.header("ABC classification")
    st.caption("ABC is a prioritisation HEURISTIC. It is not an optimisation algorithm and does not prove an optimal layout.")
    c1, c2, c3 = st.columns(3)
    S.abc_measure = c1.selectbox("Activity measure", list(MEASURES), format_func=MEASURES.get,
                                 index=list(MEASURES).index(S.get("abc_measure", "pick_frequency")))
    a = c2.number_input("Class A threshold (cumulative %)", 1.0, 99.0, float(S.get("abc_thr", (80, 15))[0]))
    b = c3.number_input("Class B share (%)", 0.0, 99.0, float(S.get("abc_thr", (80, 15))[1]))
    S.abc_thr = (a, b)
    try:
        t = abc_table()
    except ValueError as e:
        st.error(str(e)); st.stop()
    st.caption(f"Classes: A = first {a:.0f}%, B = next {b:.0f}%, C = remaining {100 - a - b:.0f}% of cumulative activity (editable defaults, not universal rules).")
    k = system_kpis(S.products, S.orders, t)
    st.dataframe(pd.DataFrame(k["abc_distribution"]).T.rename(columns={"products": "Products", "share_of_products_pct": "% of products",
                                                                      "share_of_activity_pct": "% of activity"}), use_container_width=True)
    st.dataframe(t, use_container_width=True, hide_index=True)
    fig = go.Figure()
    fig.add_bar(x=t["product_id"], y=t["activity"], marker_color=t["abc_class"].map({"A": "#d62728", "B": "#ff9f1c", "C": "#2a9d8f"}), name="Activity")
    fig.add_scatter(x=t["product_id"], y=t["cum_pct"], yaxis="y2", name="Cumulative %", mode="lines+markers")
    fig.update_layout(title="Pareto chart", yaxis2=dict(overlaying="y", side="right", range=[0, 100], title="Cumulative %"),
                      xaxis=dict(tickangle=-60))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Placement rule:** products are ranked by activity and assigned, in rank order, to the eligible locations with the "
                "shortest walking distance to the depot (A first, then B, then C), skipping locations whose capacity is too small. "
                "**Limitations:** ignores which products are ordered together and the routing method; assumes activity is a good proxy for travel burden.")
    asg, un = abc_assignment(t, S.products, S.locations, S.params)
    st.plotly_chart(layout_fig(asg, t, "ABC-based storage assignment (heuristic)"), use_container_width=True)
    if un:
        st.error(f"{len(un)} product(s) could not be placed.")

# ----------------------------------------------------------------------------- 4 Storage strategies
elif page == PAGES[3]:
    need_data()
    st.header("Storage strategies")
    up = st.file_uploader("Optional: import an assignment produced by an external mathematical model (CSV: product_id, location_id)", type=["csv"])
    if up is not None:
        try:
            S.ext_assign = external_assignment(read_table(up))
            st.success("Imported - it will be evaluated as an EXTERNALLY SUPPLIED assignment (feasibility is checked; optimality is not verified).")
        except Exception as e:
            st.error(str(e))
    if st.button("Run full evaluation", type="primary"):
        run_eval()
    st.info(f"Mathematical SLA: {SLA_STATUS}. Baseline and ABC are evaluated on identical data, layout and routing method "
            f"(**{ROUTING_LABELS[S.params.evaluation_routing]}**).")
    if "strat" in S:
        st.subheader("Strategy comparison (OPI = total picking distance; lower is better)")
        st.dataframe(strategy_table(S.strat), use_container_width=True)
        st.subheader("Is the result sensitive to the routing method?")
        st.dataframe(S.sens, hide_index=True, use_container_width=True)
        st.caption("OPI (m) per strategy under each routing method, and which strategy would be recommended. If the winner changes, "
                   "the conclusion depends on the routing assumption.")
        tbl = abc_table()
        sc = {r.key: r for r in S.strat if r.assign}
        pick = st.selectbox("Show layout of", list(sc), format_func=lambda k: sc[k].name)
        st.plotly_chart(layout_fig(sc[pick].assign, tbl, sc[pick].name), use_container_width=True)
        st.subheader("Feasibility checks")
        st.dataframe(pd.DataFrame([{"Strategy": r.name, "Check": c[0], "Result": c[1], "Detail": c[2]} for r in S.strat for c in r.checks]),
                     hide_index=True, use_container_width=True)
        for r in S.strat:
            for n in r.notes:
                st.caption(f"{r.name}: {n}")
        with st.expander("Export inputs for a future mathematical SLA model"):
            if st.button("Write files to ./sla_export"):
                act = activity_table(S.products, S.orders, S.get("abc_measure", "pick_frequency"))
                st.write(export_inputs(act, S.products, S.orders, S.locations, S.params, "sla_export"))

# ----------------------------------------------------------------------------- 5 Routes
elif page == PAGES[4]:
    need_eval()
    st.header("Order picking routes")
    st.caption("Order Profile: Order-by-Order (each order routed separately). Batch picking is NOT supported.")
    sc = {r.key: r for r in S.strat if r.evaluated}
    c1, c2 = st.columns(2)
    sid = c1.selectbox("Storage strategy", list(sc), format_func=lambda k: sc[k].name)
    oid = c2.selectbox("Order", S.orders["order_id"].unique())
    pts_map, o = location_points(S.locations), S.orders[S.orders["order_id"] == oid]
    pts = list({pts_map[sc[sid].assign[p]] for p in o["product_id"]})
    rows, figs = [], {}
    for m in ("s_shape", "return", "optimal"):
        r = route(pts, m, S.params)
        t = r.distance / S.params.picker_speed_mps if S.params.picker_speed_mps else None
        rows.append({"Method": m + ("" if m != "optimal" else (" (exact)" if r.optimal else " (heuristic fallback)")),
                     "Distance (m)": r.distance, "Est. travel time (s)": t, "Type": "own exact algorithm" if r.optimal else "policy/heuristic"})
        figs[m] = r
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    m = st.radio("Draw route", list(figs), horizontal=True)
    r = figs[m]
    fig = go.Figure()
    fig.add_scatter(x=[w[0] for w in r.waypoints], y=[w[1] for w in r.waypoints], mode="lines", name="Route")
    fig.add_scatter(x=[(a - 1) * S.params.aisle_spacing_m for a, _ in pts], y=[y for _, y in pts], mode="markers", name="Picks",
                    marker=dict(size=11, color="#d62728"))
    fig.add_scatter(x=[(S.params.depot_aisle - 1) * S.params.aisle_spacing_m], y=[0], mode="markers", name="Depot",
                    marker=dict(symbol="star", size=15, color="black"))
    fig.update_yaxes(autorange="reversed", title="Distance from front (m)"); fig.update_xaxes(title="m")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("DAP (github.com/MyReStudy/single_block)")
    path = st.text_input("Local path to the cloned repository", S.get("dap_path", ""))
    S.dap_path = path
    st.write(dap_status(path))
    if os.path.exists("validation_log.json"):
        log = json.load(open("validation_log.json")); st.write({k: log[k] for k in ("passed", "failed", "errors")})
    else:
        st.warning("DAP integration status: NOT VALIDATED (scripts/validate_dap.py has not been run). Its limits: single block, equal aisles, no depot/capacity.")

# ----------------------------------------------------------------------------- 6 KPI
elif page == PAGES[5]:
    need_eval()
    st.header("KPI analysis")
    st.subheader("Demand indicators")
    st.dataframe(product_demand(S.products, S.orders), hide_index=True, use_container_width=True)
    st.subheader("Distance indicators by strategy")
    st.dataframe(strategy_table(S.strat), use_container_width=True)
    ev = [r for r in S.strat if r.evaluated]
    st.plotly_chart(px.bar(pd.DataFrame({"Strategy": [r.name for r in ev], "OPI (total picking distance, m)": [r.opi for r in ev]}),
                           x="Strategy", y="OPI (total picking distance, m)"), use_container_width=True)
    base = next((r for r in S.strat if r.key == "baseline" and r.evaluated), None)
    if base:
        k = base.result.kpis
        st.subheader("Baseline KPIs (calculated with the evaluation routing method)")
        cols = st.columns(4)
        cols[0].metric("Avg distance / order (m)", fmt(k["avg_distance_per_order_m"]))
        cols[1].metric("Est. travel time / order (s)", fmt(k["avg_travel_time_per_order_s"]))
        cols[2].metric("Storage utilization (%)", fmt(k["utilization_pct"]))
        cols[3].metric("Locations visited / order", fmt(k["avg_locations_visited_per_order"], 2))
        st.caption("Distances are MODEL-GENERATED; time is an ESTIMATE (distance / speed, walking only).")
        st.dataframe(gap_table(k, S.params), hide_index=True, use_container_width=True)
        for lbl, key in (("% orders over distance target", "pct_orders_over_distance_target"),
                         ("% orders over time target", "pct_orders_over_time_target"),
                         ("% orders above picker capacity (flag only)", "pct_orders_over_picker_capacity")):
            st.write(f"{lbl}: {fmt(k[key])}")
    st.subheader("KPI definitions")
    st.dataframe(KPI_DEFS, hide_index=True, use_container_width=True)

# ----------------------------------------------------------------------------- 7 Decision
elif page == PAGES[6]:
    need_eval()
    st.header("Decision engine")
    st.caption("Primary decision: the FEASIBLE strategy with the lowest OPI (= total picking distance), evaluated on identical data and routing. "
               "No weights, no scores, no scenario designed to favour a model.")
    recommendation_panel()
    st.dataframe(S.strat_dec.ranking, hide_index=True, use_container_width=True)

    st.subheader("Optional scenario analysis (HYPOTHETICAL inputs)")
    st.caption("Tests whether the conclusion holds under justified what-if conditions. Results are not historical data and not forecasts.")
    k1, k2 = st.columns(2)
    kind = k1.selectbox("Scenario", list(SCEN_KINDS), format_func=SCEN_KINDS.get)
    val = k2.number_input("Parameter (k for demand multiplier, gamma for distribution shift)", value=1.5 if kind == "demand_multiplier" else 0.5, min_value=0.01)
    if st.button("Run scenario"):
        try:
            a, b = S.get("abc_thr", (80, 15))
            S.scen_out = run_scenario(S.products, S.orders, S.locations, S.params, kind, val, S.get("ext_assign"), a, b,
                                      S.get("abc_measure", "pick_frequency"))
        except Exception as e:
            st.error(f"Scenario failed: {e}")
    if "scen_out" in S:
        out = S.scen_out
        st.warning(out["label"])
        for a_ in out["assumptions"][1:]:
            st.markdown(f"- {a_}")
        st.dataframe(strategy_table(out["results"]), use_container_width=True)
        for e in out["decision"].explanation:
            st.markdown(f"- {e}")

    st.subheader("Additional analysis: SLA vs PRO vs combined interventions")
    if "decision" not in S:
        st.info("Needs a baseline assignment and the routing-policy assumption accepted on **2 Data Input** (optional section), then re-run the evaluation.")
    else:
        st.caption("Separate from the strategy recommendation above. Depends on your targets and stated priority.")
        c1, c2 = st.columns(2)
        S.params.priority = c1.radio("Stated priority", ["least_effort", "max_improvement"], horizontal=True,
                                     index=["least_effort", "max_improvement"].index(S.params.priority),
                                     format_func={"least_effort": "Least effort that meets the targets", "max_improvement": "Largest measured improvement"}.get)
        S.params.min_material_improvement_pct = c2.number_input("Minimum improvement (%) treated as material (your threshold)", 0.0, 100.0,
                                                               float(S.params.min_material_improvement_pct))
        S.decision = decide(S.scenarios, S.params)
        d = S.decision
        box = {"recommended": st.success, "no_intervention": st.success, "incomplete": st.warning, "no_recommendation": st.warning}[d.status]
        box(f"**Intervention:** {d.recommendation}")
        for r in d.reasons:
            st.markdown(f"- {r}")
        if d.attribution:
            st.dataframe(pd.DataFrame(list(d.attribution.items()), columns=["Component", "m per order"]), hide_index=True)
        st.dataframe(d.evidence, hide_index=True, use_container_width=True)
        st.dataframe(INFLUENCE, hide_index=True, use_container_width=True)
        for m_ in d.missing:
            st.markdown(f"- Missing: {m_}")

# ----------------------------------------------------------------------------- 8 Reports
elif page == PAGES[7]:
    need_eval()
    st.header("Reports")
    k = system_kpis(S.products, S.orders, abc_table())
    st.download_button("Download Excel report", excel_report(S.strat, S.strat_dec, S.params, abc_table(), k, S.get("scenarios"), S.get("decision")),
                       "warehouse_dss_report.xlsx")
    st.download_button("Download decision report (Markdown)", markdown_report(S.strat, S.strat_dec, S.params, S.get("is_sample", False), k),
                       "decision_report.md")
    st.download_button("Download strategy comparison (CSV)", strategy_table(S.strat).to_csv().encode("utf-8"), "strategy_comparison.csv")
    st.subheader("Component status")
    st.dataframe(COMPONENT_STATUS, hide_index=True, use_container_width=True)
    st.subheader("Methodology")
    st.markdown(METHODOLOGY)

# ----------------------------------------------------------------------------- 1 Overview
else:
    st.header("Executive overview")
    st.markdown("Analyses historical order data, evaluates storage strategies on identical conditions, checks feasibility and "
                "**recommends the feasible strategy with the lowest total picking distance (OPI)**.")
    if not have_data():
        st.info("Start on **2 Data Input** (a sample dataset is available).")
    else:
        tbl = None
        try:
            tbl = abc_table()
        except ValueError as e:
            st.error(str(e))
        k = system_kpis(S.products, S.orders, tbl)
        c = st.columns(4)
        c[0].metric("Total products", k["total_products"])
        c[1].metric("Total orders", k["total_orders"])
        c[2].metric("Total demand quantity", fmt(k["total_demand_quantity"], 0))
        c[3].metric("Product demand frequency (min / median / max)",
                    f"{k['demand_frequency_min']:.0f} / {k['demand_frequency_median']:.0f} / {k['demand_frequency_max']:.0f}")
        if tbl is not None:
            st.markdown("**ABC classification distribution** (products / share of activity): " + "; ".join(
                f"{cl}: {v['products']} / {v['share_of_activity_pct']:.1f}%" for cl, v in k["abc_distribution"].items()))
        op = order_profile(S.orders)
        st.markdown(f"**Order profile:** {op['mode']} - {op['orders']} orders, {op['min_lines']}-{op['max_lines']} lines "
                    f"(median {op['median_lines']:.0f}). Batch orders: not supported (routing handles individual orders only).")
        if "strat" not in S:
            st.info("Run the evaluation on **4 Storage Strategies** to see the recommended strategy.")
        else:
            sel = next((r for r in S.strat if r.key == S.strat_dec.selected), None)
            if sel:
                inv = len(set(sel.assign.values()))
                st.markdown(f"**Storage location assignment (recommended strategy):** {inv} of {len(S.locations)} locations occupied "
                            f"({fmt(utilization(sel.assign, S.locations) * 100)}% of eligible locations).")
            st.subheader("Recommended storage strategy")
            recommendation_panel()
    st.subheader("Component status")
    st.dataframe(COMPONENT_STATUS, hide_index=True, use_container_width=True)
