# app.py — Streamlit frontend for the annual business planning agent
# run: streamlit run app.py

import os
import json
from pathlib import Path
import streamlit as st

# --- *** enter your openai api key here *** ---
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.error("API key not found. Add OPENAI_API_KEY to your .env file.")
    st.stop()

st.set_page_config(
    page_title="Annual Business Planning Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from planning_agent import (
    build_planning_workflow, get_initial_planning_state,
    load_df, load_assumptions, generate_word_document
)

# --- styles ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+3:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family:'Source Sans 3',sans-serif; background:#080f18; color:#cde0ef; }
.main, .block-container { background:#080f18; }
.plan-header {
    background:linear-gradient(135deg,#0a1520 0%,#1a2a3d 50%,#0a2010 100%);
    padding:1.8rem 2.2rem; border-radius:14px; margin-bottom:1.2rem;
    border-left:5px solid #2e7d32;
}
.plan-header h1 { font-family:'Playfair Display',serif; font-size:1.9rem; color:#fff; margin:0; }
.plan-header p  { color:#8ab4c9; font-size:0.82rem; margin:0.3rem 0 0 0; }
.agent-card {
    background:#0d1b2a; border:1px solid #1e3a52; border-radius:10px;
    padding:1rem 1.2rem; margin-bottom:0.8rem;
}
.agent-card h4 { color:#2e7d32; margin:0 0 0.4rem 0; font-size:0.9rem; }
.agent-complete { border-left:3px solid #4caf7d; }
.agent-running  { border-left:3px solid #f5a623; }
.agent-waiting  { border-left:3px solid #5ba8d9; }
.metric-card {
    background:#0d1b2a; border:1px solid #1e3a52; border-radius:10px;
    padding:1.2rem; text-align:center;
}
.metric-card .value { font-size:1.6rem; font-weight:700; color:#4caf7d; }
.metric-card .label { font-size:0.75rem; color:#5a80a0; margin-top:0.2rem; }
.autogen-box {
    background:#0d1b0d; border:1px solid #1a4a1a; border-radius:10px;
    padding:1rem 1.2rem; margin-bottom:0.8rem;
}
.review-card {
    background:#0f1f2d; border:1px solid #2d4a6a; border-radius:8px;
    padding:1rem 1.2rem; margin-bottom:0.7rem;
}
section[data-testid="stSidebar"] { background:#060c14; border-right:1px solid #1e3a52; }
.stButton>button { background:#2e7d32 !important; color:#fff !important;
    font-weight:600 !important; border:none !important; border-radius:8px !important; }
.stButton>button:hover { background:#1b5e20 !important; }
hr { border-color:#1e3a52 !important; }
</style>
""", unsafe_allow_html=True)

# --- session state ---
for k, v in {
    "stage": "setup",  # setup → running → reviewing → complete
    "workflow": None,
    "agent_outputs": {},
    "target_data": {},
    "loyalty_data": {},
    "baseline_data": {},
    "challenger_feedback": "",
    "final_plan_text": "",
    "docx_path": "",
    "run_log": [],
    "company_growth": 15.0,
    "north_growth": 15.0,
    "south_growth": 18.0,
    "east_growth": 12.0,
    "west_growth": 16.0,
    "loyalty_target": 45.0,
    "loyalty_contrib": 75.0,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --- sidebar — planning parameters ---
with st.sidebar:
    st.markdown("### 📋 Planning Parameters")
    st.markdown("<div style='color:#5a80a0;font-size:0.75rem;'>All inputs editable — change and re-run</div>",
                unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**Revenue Growth Targets**")
    st.session_state.company_growth = st.slider(
        "Company Growth %", 5.0, 40.0, st.session_state.company_growth, 0.5)

    with st.expander("Regional Growth %"):
        st.session_state.north_growth = st.slider("North %", 5.0, 35.0, st.session_state.north_growth, 0.5)
        st.session_state.south_growth = st.slider("South %", 5.0, 35.0, st.session_state.south_growth, 0.5)
        st.session_state.east_growth  = st.slider("East %",  5.0, 35.0, st.session_state.east_growth,  0.5)
        st.session_state.west_growth  = st.slider("West %",  5.0, 35.0, st.session_state.west_growth,  0.5)

    st.markdown("---")
    st.markdown("**Loyalty Targets**")
    st.session_state.loyalty_target = st.slider(
        "Chain Penetration Target %", 30.0, 70.0, st.session_state.loyalty_target, 1.0)
    st.session_state.loyalty_contrib = st.slider(
        "Loyalty Revenue Contribution %", 50.0, 90.0, st.session_state.loyalty_contrib, 1.0)

    st.markdown("---")
    if st.button("🚀 Generate FY26 Plan"):
        st.session_state.update({
            "stage": "running", "agent_outputs": {}, "target_data": {},
            "loyalty_data": {}, "baseline_data": {}, "challenger_feedback": "",
            "final_plan_text": "", "docx_path": "", "run_log": [], "workflow": None
        })
        st.rerun()

    # reset clears workflow state and all agent outputs
    if st.button("🔄 Reset"):
        st.session_state.update({
            "stage": "setup", "agent_outputs": {}, "target_data": {},
            "loyalty_data": {}, "baseline_data": {}, "challenger_feedback": "",
            "final_plan_text": "", "docx_path": "", "run_log": [], "workflow": None
        })
        st.rerun()

    st.markdown("---")
    colors = {"setup":"#5ba8d9","running":"#f5a623","reviewing":"#9c27b0","complete":"#4caf7d"}
    c = colors.get(st.session_state.stage, "#5ba8d9")
    st.markdown(f"<span style='color:{c}'>● {st.session_state.stage.title()}</span>",
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""<div style='font-size:0.72rem;color:#3a5a7a;line-height:1.6;'>
<b style='color:#5a80a0;'>Stack</b><br>
LangGraph 4-Node Workflow<br>
AutoGen Proposer/Challenger<br>
Human-in-the-Loop Review<br>
python-docx Plan Document<br>
52-Week Seasonal Data<br>
OTB Cascade Engine
</div>""", unsafe_allow_html=True)

# --- header ---
st.markdown(f"""
<div class="plan-header">
    <h1>📊 Annual Business Planning Agent</h1>
    <p>FY26 Target Cascade · Store / Category / Brand OTB · Loyalty Plan · AutoGen Challenge Pattern
    &nbsp;|&nbsp; Growth Target: <b>{st.session_state.company_growth:.1f}%</b>
    &nbsp;|&nbsp; Loyalty Target: <b>{st.session_state.loyalty_target:.1f}%</b></p>
</div>""", unsafe_allow_html=True)

tab_workflow, tab_targets, tab_loyalty, tab_autogen, tab_review, tab_plan, tab_data, tab_about = st.tabs([
    "🤖 Workflow", "🎯 Store Targets", "💳 Loyalty Plan",
    "⚡ AutoGen Challenge", "✅ Review & Approve",
    "📄 Business Plan", "📂 Data", "ℹ️ About"
])

# --- workflow tab ---
with tab_workflow:
    # Key metrics row
    if st.session_state.target_data:
        import pandas as pd
        td = st.session_state.target_data
        ld = st.session_state.loyalty_data
        chain_fy25 = sum(d["fy25_revenue"] for d in td.values())
        chain_fy26 = sum(d["fy26_target"] for d in td.values())
        chain_loy  = sum(d.get("fy26_loyalty_revenue_target_L", 0) for d in ld.values())
        chain_enroll = sum(d.get("new_enrollment_target", 0) for d in ld.values())

        c1, c2, c3, c4 = st.columns(4)
        for col, val, label in [
            (c1, f"INR {chain_fy25:.0f}L", "FY25 Actual Revenue"),
            (c2, f"INR {chain_fy26:.0f}L", "FY26 Target Revenue"),
            (c3, f"INR {chain_loy:.0f}L",  "Loyalty Revenue Target"),
            (c4, f"{chain_enroll:,}",        "New Enrollments Target"),
        ]:
            col.markdown(f"""<div class="metric-card">
<div class="value">{val}</div>
<div class="label">{label}</div>
</div>""", unsafe_allow_html=True)
        st.markdown("")

    # Agent cards
    agent_defs = [
        ("1", "Performance Baseline Analyst",
         "52-week FY25 actuals · seasonal analysis · store/category/brand baseline",
         "baseline_analysis"),
        ("2", "Target Cascade Engine",
         "Company → Region → Store → Category → Brand OTB waterfall",
         "target_plan"),
        ("3", "Loyalty Planning Agent",
         "FY26 penetration targets · enrollment by segment · loyalty revenue",
         "loyalty_plan"),
        ("4", "Business Plan Writer",
         "Synthesises all agents → formatted Word document with all plan tables",
         "final_plan_text"),
    ]

    for num, role, desc, key in agent_defs:
        out = st.session_state.agent_outputs.get(key, "")
        css = "agent-complete" if out else ("agent-running" if st.session_state.stage == "running" else "agent-waiting")
        badge = "✅ Complete" if out else ("⏳ Running..." if st.session_state.stage == "running" else "⏸️ Waiting")
        st.markdown(f"""<div class="agent-card {css}">
<h4>Agent {num} — {role} &nbsp;<small style='color:#5a80a0;font-weight:400;'>{badge}</small></h4>
<div style='color:#7a9bb5;font-size:0.78rem;'>{desc}</div>
</div>""", unsafe_allow_html=True)
        if out:
            with st.expander(f"View Agent {num} Full Output"):
                st.markdown(
                    f"<div style='color:#cde0ef;font-size:0.83rem;line-height:1.7;'>"
                    f"{out.replace(chr(10),'<br>')}</div>",
                    unsafe_allow_html=True
                )

    # RUN WORKFLOW
    if st.session_state.stage == "running":
        with st.spinner("Running planning agents — Agents 1-3 take 2-3 minutes..."):
            try:
                import time
                wf = build_planning_workflow()
                st.session_state.workflow = wf
                cfg = {"configurable": {"thread_id": f"cs3_{int(time.time())}"}}
                st.session_state.graph_config = cfg

                init_state = get_initial_planning_state(
                    company_growth_pct=st.session_state.company_growth,
                    regional_growth={
                        "North": st.session_state.north_growth,
                        "South": st.session_state.south_growth,
                        "East":  st.session_state.east_growth,
                        "West":  st.session_state.west_growth,
                    },
                    loyalty_target_pct=st.session_state.loyalty_target,
                    loyalty_revenue_contrib_pct=st.session_state.loyalty_contrib,
                    api_key=OPENAI_API_KEY,
                )

                # Run until interrupt before business_plan_writer
                result = wf.invoke(init_state, config=cfg)

                st.session_state.agent_outputs = {
                    "baseline_analysis": result.get("baseline_analysis", ""),
                    "target_plan":       result.get("target_plan", ""),
                    "loyalty_plan":      result.get("loyalty_plan", ""),
                    "final_plan_text":   result.get("final_plan_text", ""),
                }
                st.session_state.target_data        = result.get("target_data", {})
                st.session_state.loyalty_data       = result.get("loyalty_data", {})
                st.session_state.baseline_data      = result.get("baseline_data", {})
                st.session_state.challenger_feedback = result.get("challenger_feedback", "")
                st.session_state.final_plan_text    = result.get("final_plan_text", "")
                st.session_state.docx_path          = result.get("docx_path", "")
                st.session_state.run_log            = result.get("session_log", [])
                st.session_state.stage = "reviewing"
                st.rerun()

            except Exception as e:
                st.error(f"Workflow error: {e}")
                import traceback; st.code(traceback.format_exc())
                st.session_state.stage = "setup"

    if st.session_state.run_log:
        with st.expander("Run Log"):
            for entry in st.session_state.run_log:
                st.markdown(f"<div style='color:#5a80a0;font-size:0.75rem;'>✓ {entry}</div>",
                            unsafe_allow_html=True)

# --- store targets tab ---
with tab_targets:
    if not st.session_state.target_data:
        st.info("Run the planning workflow to generate store targets.")
    else:
        import pandas as pd
        td = st.session_state.target_data
        ld = st.session_state.loyalty_data

        # Store targets table
        rows = []
        for store, data in td.items():
            loy = ld.get(store, {})
            rows.append({
                "Store": store,
                "Region": data["region"],
                "Maturity": data["maturity"],
                "FY25 Revenue (L)": f"INR {data['fy25_revenue']:.1f}L",
                "FY26 Target (L)": f"INR {data['fy26_target']:.1f}L",
                "Growth %": f"+{data['growth_rate_pct']:.1f}%",
                "Loyalty Penetration Target": f"{loy.get('fy26_penetration_target_pct',45):.1f}%",
                "New Enrollments": f"{loy.get('new_enrollment_target',0):,}",
            })

        df = pd.DataFrame(rows)
        st.markdown("### FY26 Store Revenue Targets")
        st.dataframe(df, use_container_width=True, height=380)

        # Category targets per store
        st.markdown("### Category Targets by Store")
        store_sel = st.selectbox("Select store:", list(td.keys()))
        if store_sel:
            cat_rows = []
            for cat, rev in td[store_sel]["category_targets"].items():
                fy25 = st.session_state.baseline_data.get(store_sel, {}).get(
                    "categories", {}).get(cat, {}).get("revenue", 0)
                growth = ((rev - fy25) / fy25 * 100) if fy25 > 0 else 0
                cat_rows.append({
                    "Category": cat,
                    "FY25 Revenue (L)": f"INR {fy25:.1f}L",
                    "FY26 Target (L)": f"INR {rev:.1f}L",
                    "Growth %": f"+{growth:.1f}%",
                })
            st.dataframe(pd.DataFrame(cat_rows), use_container_width=True)

        # Brand OTB table
        st.markdown("### Brand OTB — Top 20 by Value")
        otb_rows = []
        for store, data in td.items():
            for key, otb in data.get("brand_otb", {}).items():
                otb_rows.append({
                    "Store": store,
                    "Category": otb["category"],
                    "Brand": otb["brand"],
                    "Revenue Target (L)": f"INR {otb['revenue_target_L']:.2f}L",
                    "OTB Units": f"{otb['otb_units']:,}",
                    "OTB Value (L)": f"INR {otb['otb_value_L']:.2f}L",
                    "Space": "Expand" if otb["space_multiplier"] > 1 else ("Reduce" if otb["space_multiplier"] < 1 else "Stable"),
                })

        otb_df = pd.DataFrame(otb_rows)
        if not otb_df.empty:
            # Chain-level OTB summary
            chain_otb = pd.DataFrame([{
                "Category": r["Category"], "Brand": r["Brand"],
                "OTB Value (L)": float(r["OTB Value (L)"].replace("INR ","").replace("L",""))
            } for r in otb_rows])
            chain_agg = chain_otb.groupby(["Category","Brand"])["OTB Value (L)"].sum().reset_index()
            chain_agg = chain_agg.sort_values("OTB Value (L)", ascending=False).head(20)
            chain_agg["OTB Value (L)"] = chain_agg["OTB Value (L)"].apply(lambda x: f"INR {x:.2f}L")
            st.dataframe(chain_agg, use_container_width=True, height=420)

# --- loyalty tab ---
with tab_loyalty:
    if not st.session_state.loyalty_data:
        st.info("Run the planning workflow to generate loyalty targets.")
    else:
        import pandas as pd
        ld = st.session_state.loyalty_data

        # Loyalty summary table
        loy_rows = []
        for store, data in ld.items():
            loy_rows.append({
                "Store": store,
                "FY25 Penetration": f"{data['fy25_penetration_pct']:.1f}%",
                "FY26 Target": f"{data['fy26_penetration_target_pct']:.1f}%",
                "Stretch (pp)": f"+{data['fy26_penetration_target_pct'] - data['fy25_penetration_pct']:.1f}pp",
                "Loyalty Revenue Target": f"INR {data['fy26_loyalty_revenue_target_L']:.1f}L",
                "Loyalty Contribution": f"{data['loyalty_contribution_pct']:.0f}%",
                "New Enrollments": f"{data['new_enrollment_target']:,}",
                "Current Members": f"{data['current_active_members']:,}",
                "Basket Premium (INR)": f"+{data['avg_basket_premium_INR']:,}",
            })

        st.markdown("### FY26 Loyalty Targets by Store")
        st.dataframe(pd.DataFrame(loy_rows), use_container_width=True, height=380)

        # Loyalty plan narrative
        with st.expander("📋 Full Loyalty Plan", expanded=False):
            plan = st.session_state.agent_outputs.get("loyalty_plan", "")
            st.markdown(
                f"<div style='color:#cde0ef;font-size:0.85rem;line-height:1.7;'>"
                f"{plan.replace(chr(10),'<br>')}</div>",
                unsafe_allow_html=True
            )

# --- autogen challenge tab ---
with tab_autogen:
    st.markdown("""<div style='background:#0d1b0d;border:1px solid #1a4a1a;border-radius:10px;
padding:1rem 1.5rem;margin-bottom:1rem;'>
<div style='color:#4caf7d;font-weight:600;margin-bottom:0.4rem;'>
⚡ AutoGen Proposer / Challenger Pattern
</div>
<div style='color:#8ab4c9;font-size:0.85rem;'>
Agent 2 runs two GPT-4o calls in sequence.
The <b>Proposer</b> sets targets using the cascade logic.
The <b>Challenger</b> stress-tests each target for realism — flagging aggressive,
conservative, or at-risk targets. This debate pattern produces more robust plans
than a single-pass generation.
</div>
</div>""", unsafe_allow_html=True)

    if not st.session_state.challenger_feedback:
        st.info("Run the workflow to see the AutoGen challenge output.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📈 Proposer Output (Target Plan)")
            plan = st.session_state.agent_outputs.get("target_plan", "")
            st.markdown(
                f"<div style='background:#0d1b2a;border:1px solid #1e3a52;border-radius:8px;"
                f"padding:1rem;color:#cde0ef;font-size:0.82rem;line-height:1.7;"
                f"max-height:500px;overflow-y:auto;'>"
                f"{plan.replace(chr(10),'<br>')}</div>",
                unsafe_allow_html=True
            )
        with col2:
            st.markdown("### 🔍 Challenger Output (Stress Test)")
            st.markdown(
                f"<div style='background:#0d1b0d;border:1px solid #1a4a1a;border-radius:8px;"
                f"padding:1rem;color:#cde0ef;font-size:0.82rem;line-height:1.7;"
                f"max-height:500px;overflow-y:auto;'>"
                f"{st.session_state.challenger_feedback.replace(chr(10),'<br>')}</div>",
                unsafe_allow_html=True
            )

# --- review & approve tab ---
with tab_review:
    if st.session_state.stage not in ["reviewing", "complete"]:
        st.info("Run the workflow first. Agents 1-3 must complete before you can review.")
    else:
        st.markdown("""<div style='background:#0d1b2a;border:1px solid #2e7d32;border-radius:10px;
padding:1rem 1.5rem;margin-bottom:1rem;'>
<div style='color:#4caf7d;font-weight:600;margin-bottom:0.4rem;'>
✅ Human-in-the-Loop — Planning Review
</div>
<div style='color:#8ab4c9;font-size:0.85rem;'>
Agents 1-3 have completed. Review the proposed targets.
Adjust any store growth rates if needed, then approve to generate the final plan document.
</div>
</div>""", unsafe_allow_html=True)

        td = st.session_state.target_data
        if not td:
            st.warning("No target data found. Please re-run the workflow.")
        else:
            st.markdown("### Adjust Store Growth Targets (Optional)")
            adjustments = {}
            for store, data in td.items():
                col1, col2, col3 = st.columns([3, 2, 2])
                with col1:
                    st.markdown(
                        f"<div style='color:#cde0ef;font-size:0.85rem;padding-top:0.5rem;'>"
                        f"<b>{store}</b> ({data['region']}, {data['maturity']})</div>",
                        unsafe_allow_html=True
                    )
                with col2:
                    st.markdown(
                        f"<div style='color:#5ba8d9;font-size:0.82rem;padding-top:0.5rem;'>"
                        f"Proposed: +{data['growth_rate_pct']:.1f}%</div>",
                        unsafe_allow_html=True
                    )
                with col3:
                    adj = st.number_input(
                        "Override %", min_value=0.0, max_value=50.0,
                        value=data["growth_rate_pct"],
                        step=0.5, key=f"adj_{store}",
                        label_visibility="collapsed"
                    )
                    adjustments[store] = adj

            st.markdown("---")
            if st.session_state.stage == "reviewing":
                if st.button("✅ Approve & Generate Business Plan Document", type="primary"):
                    with st.spinner("Generating final business plan and Word document..."):
                        try:
                            # Apply any adjustments to target data
                            for store, new_growth in adjustments.items():
                                if store in st.session_state.target_data:
                                    old_rev = st.session_state.target_data[store]["fy25_revenue"]
                                    st.session_state.target_data[store]["fy26_target"] = round(
                                        old_rev * (1 + new_growth/100), 2)
                                    st.session_state.target_data[store]["growth_rate_pct"] = new_growth

                            # Generate final plan text and document
                            from langchain_openai import ChatOpenAI
                            from langchain_core.messages import HumanMessage, SystemMessage
                            import pandas as pd

                            llm = ChatOpenAI(
                                model="gpt-4o", temperature=0.1,
                                openai_api_key=OPENAI_API_KEY
                            )

                            td = st.session_state.target_data
                            ld = st.session_state.loyalty_data
                            chain_fy25 = sum(d["fy25_revenue"] for d in td.values())
                            chain_fy26 = sum(d["fy26_target"] for d in td.values())
                            chain_loy  = sum(d.get("fy26_loyalty_revenue_target_L", 0) for d in ld.values())

                            store_lines = [
                                f"{s}: INR {d['fy25_revenue']:.1f}L → INR {d['fy26_target']:.1f}L (+{d['growth_rate_pct']:.1f}%)"
                                for s, d in td.items()
                            ]

                            resp = llm.invoke([
                                SystemMessage(content="You are a Chief Strategy Officer writing an annual retail business plan for the Board."),
                                HumanMessage(content=f"""Write the FY26 Annual Business Plan.

CHAIN: FY25 INR {chain_fy25:.1f}L → FY26 INR {chain_fy26:.1f}L (+{st.session_state.company_growth:.1f}%)
LOYALTY: Chain Revenue Target INR {chain_loy:.1f}L | Penetration {st.session_state.loyalty_target:.1f}%

STORE TARGETS:
{chr(10).join(store_lines)}

BASELINE CONTEXT: {st.session_state.agent_outputs.get('baseline_analysis','')[:400]}
LOYALTY CONTEXT: {st.session_state.agent_outputs.get('loyalty_plan','')[:400]}
CHALLENGER RISKS: {st.session_state.challenger_feedback[:300]}

Write a complete plan:
**EXECUTIVE SUMMARY**
**FY26 TARGETS AT A GLANCE**
**REGIONAL STRATEGY** (North/South/East/West)
**STORE PERFORMANCE TARGETS** (all 10 stores with rationale)
**CATEGORY STRATEGY** — write one paragraph each for ONLY these 7 categories: Ladies Western, Ladies Ethnic, Men Formal, Men Casual, Kids, Sportswear, Footwear. Do NOT invent other categories.
**BRAND INVESTMENT PRIORITIES** — use ONLY these brands: AND, W, Biba, Fabindia, Van Heusen, Arrow, Nike, Adidas, Puma, Levis, UCB, Bata, Woodland. Do NOT invent brand names.
**LOYALTY PROGRAMME PLAN**
**KEY RISKS AND MITIGATIONS**
**PLANNING ASSUMPTIONS**""")
                            ])

                            plan_text = resp.content
                            st.session_state.final_plan_text = plan_text
                            st.session_state.agent_outputs["final_plan_text"] = plan_text

                            # Build mock state for docx generator
                            mock = {
                                "target_data": st.session_state.target_data,
                                "loyalty_data": st.session_state.loyalty_data,
                                "company_growth_pct": st.session_state.company_growth,
                                "loyalty_target_pct": st.session_state.loyalty_target,
                                "loyalty_revenue_contrib_pct": st.session_state.loyalty_contrib,
                                "baseline_analysis": st.session_state.agent_outputs.get("baseline_analysis",""),
                                "target_plan": st.session_state.agent_outputs.get("target_plan",""),
                                "challenger_feedback": st.session_state.challenger_feedback,
                                "loyalty_plan": st.session_state.agent_outputs.get("loyalty_plan",""),
                            }

                            docx_path = generate_word_document(mock, plan_text)
                            st.session_state.docx_path = docx_path
                            st.session_state.stage = "complete"
                            st.session_state.run_log.append("Plan approved and document generated")
                            st.rerun()

                        except Exception as e:
                            st.error(f"Error: {e}")
                            import traceback; st.code(traceback.format_exc())
            else:
                st.success("✅ Plan approved and document generated. See Business Plan tab.")

# --- business plan tab ---
with tab_plan:
    if not st.session_state.final_plan_text:
        st.info("Approve the plan in the Review tab to generate the final document.")
    else:
        st.markdown("""<div style='background:#0d3320;border:1px solid #1a6640;border-radius:8px;
padding:0.7rem 1rem;margin-bottom:1rem;'>
<span style='color:#4caf7d;font-weight:600;'>✅ FY26 Business Plan Generated</span>
</div>""", unsafe_allow_html=True)

        # Download button
        docx_path = st.session_state.docx_path
        if docx_path and Path(docx_path).exists():
            with open(docx_path, "rb") as f:
                file_bytes = f.read()
            ext = "docx" if docx_path.endswith(".docx") else "txt"
            st.download_button(
                label="⬇️ Download FY26 Business Plan (Word)",
                data=file_bytes,
                file_name=f"FY26_Annual_Business_Plan.{ext}",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

        # Display plan
        st.markdown(
            f"<div style='background:#0d1b2a;border:1px solid #1e3a52;border-radius:10px;"
            f"padding:1.5rem;color:#cde0ef;font-size:0.88rem;line-height:1.7;'>"
            f"{st.session_state.final_plan_text.replace(chr(10),'<br>')}"
            f"</div>",
            unsafe_allow_html=True
        )

# --- data preview tab ---
with tab_data:
    import pandas as pd
    fmap = {
        "FY25 Weekly Sales (52 weeks)": "./data/01_FY25_Weekly_Sales_Actuals.xlsx",
        "FY25 Brand Performance":       "./data/02_FY25_Brand_Performance.xlsx",
        "FY25 Loyalty Actuals":         "./data/03_FY25_Loyalty_Actuals.xlsx",
        "Planning Assumptions — Stores":"./data/04_Planning_Assumptions.xlsx",
    }
    chosen = st.selectbox("Select file:", list(fmap.keys()))
    fp = fmap[chosen]

    if Path(fp).exists():
        if "Planning Assumptions" in chosen:
            sheet = st.selectbox("Sheet:", ["Store Assumptions", "Brand Assumptions", "Planning Parameters"])
            df = pd.read_excel(fp, sheet_name=sheet)
        else:
            df = pd.read_excel(fp)
            if 'Store' in df.columns:
                stores = ["All"] + sorted(df['Store'].unique().tolist())
                sel_store = st.selectbox("Filter by store:", stores, key="data_store")
                if sel_store != "All":
                    df = df[df['Store'] == sel_store]

        st.caption(f"{len(df):,} rows · {len(df.columns)} columns")
        st.dataframe(df.head(100), use_container_width=True, height=420)
    else:
        st.warning(f"File not found: {fp}. Run generate_data.py first.")

# --- about tab ---
with tab_about:
    st.markdown("""<div style='color:#cde0ef;line-height:1.75;font-size:0.88rem;'>
<h3 style='font-family:Playfair Display,serif;color:#2e7d32;margin-top:0;'>
Annual Business Planning Agent</h3>

<h4 style='color:#5ba8d9;'>The Business Problem</h4>
<p>Annual planning in large format retail takes weeks. Category managers submit brand plans.
Store managers submit store plans. Finance consolidates. The CEO reviews. Multiple rounds of
revision with Excel files flying around. This agent compresses that into one session — takes
52 weeks of FY25 actuals, applies planning logic, and produces a complete FY26 plan with
targets at every level and a downloadable Word document.</p>

<h4 style='color:#5ba8d9;'>The AutoGen Pattern</h4>
<p>Agent 2 uses a two-LLM-call debate pattern. The <b>Proposer</b> generates targets using
the cascade waterfall. The <b>Challenger</b> (acting as CFO) stress-tests each store target —
flagging aggressive, conservative, or execution-risk targets. This produces more robust plans
than single-pass generation. This is the AutoGen Proposer/Challenger pattern implemented
natively in LangGraph without the AutoGen framework.</p>

<h4 style='color:#5ba8d9;'>Target Cascade Logic</h4>
<p>Company growth % → Regional breakdown → Store targets (capped by maturity: New 40%,
Mid 20%, Mature 13%) → Category mix from FY25 actuals → Brand OTB using the formula:
OTB Units = Revenue Target / (MRP × Sell-Through Target). Floor plate changes,
food courts, and SIS additions adjust store targets accordingly.</p>

<h4 style='color:#5ba8d9;'>Tech Stack</h4>
<p>LangGraph · GPT-4o · AutoGen Proposer/Challenger Pattern · Human-in-the-Loop · python-docx · pandas · Streamlit</p>
</div>""", unsafe_allow_html=True)
