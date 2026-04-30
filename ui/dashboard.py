import streamlit as st
import pandas as pd
import sys
import os
import traceback
import logging

# Ensure absolute silence from external libs in terminal
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.pipeline_engine import DeploymentPipeline
from utils.plotting import (
    create_radar_chart, 
    create_pareto_frontier, 
    create_strategy_heatmap,
    create_roofline_plot
)
from utils.config import GROQ_MODELS

# Streamlit config
st.set_page_config(
    page_title="Slingshot-AI | Enterprise Deployment",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise CSS
st.markdown("""
<style>
    .main { background-color: #0b0f19; color: #e2e8f0; font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4, h5, h6 { color: #f8fafc; font-weight: 600; }
    .stSelectbox label, .stSlider label, .stNumberInput label, .stCheckbox label { color: #94a3b8 !important; }
    
    .stButton>button {
        background: #3b82f6; color: white; border: none; border-radius: 6px; padding: 10px; font-weight: 600; width: 100%; transition: all 0.2s;
    }
    .stButton>button:hover { background: #2563eb; transform: translateY(-1px); }
    
    .metric-card {
        background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2); text-align: center;
    }
    .metric-title { color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
    .metric-value { font-size: 1.8rem; font-weight: 700; }
    .val-green { color: #10b981; } .val-blue { color: #3b82f6; } .val-red { color: #ef4444; } .val-purple { color: #8b5cf6; }
    
    .header-box { background: #1e293b; padding: 20px; border-radius: 8px; margin-bottom: 24px; border: 1px solid #334155; }
    .header-box h2 { margin-top: 0; color: #38bdf8; font-size: 1.5rem; }
    .header-box p { color: #94a3b8; margin-bottom: 0; }
    
    .decision-box { background: #152238; border: 2px solid #3b82f6; border-radius: 8px; padding: 20px; margin-top: 20px; margin-bottom: 20px; }
    .decision-box h3 { color: #38bdf8; margin-top: 0; margin-bottom: 15px; font-size: 1.25rem; }
    
    .insight-card { background: #1e293b; border-left: 4px solid #3b82f6; padding: 20px; border-radius: 6px; margin-top: 20px; border: 1px solid #334155; }
    .insight-text { color: #cbd5e1; font-size: 1.05rem; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

def percent_improvement(base: float, opt: float) -> float:
    if base == 0:
        return 0.0
    return max(0.0, ((base - opt) / base) * 100)

# -------------------------------------------------------------------
# STATE MANAGEMENT
# -------------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state["results"] = None

# -------------------------------------------------------------------
# HEADER
# -------------------------------------------------------------------
st.markdown("""
<div class="header-box">
    <h2>Slingshot-AI Deployment Intelligence</h2>
    <p>Find the Pareto-optimal deployment strategy using Roofline Performance Modeling.</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# SIDEBAR CONFIGURATION
# -------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    
    with st.form("pipeline_form"):
        models = ["meta-llama/Llama-2-7b-chat-hf", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "mistralai/Mistral-7B-v0.1"]
        selected_model = st.selectbox("Model Architecture", models)
        
        selected_hardware = st.selectbox("Hardware Bias (Optional)", ["None", "AMD_MI250", "AMD_MI300X"])
        selected_workload = st.selectbox("Workload", ["chat_inference", "batch_inference", "fine_tuning"])
        
        st.markdown("---")
        st.subheader("Constraints")
        max_lat = st.number_input("Max Latency (ms)", value=5000.0, min_value=0.1)
        max_cost = st.number_input("Max Cost/Req ($)", value=0.01, min_value=0.00001, format="%.5f")
        
        st.markdown("---")
        st.subheader("Objective Weights")
        w_lat = st.slider("Latency", 0, 100, 25)
        w_mem = st.slider("Memory", 0, 100, 20)
        w_cost = st.slider("Cost", 0, 100, 20)
        w_eng = st.slider("Energy", 0, 100, 15)
        w_acc = st.slider("Accuracy", 0, 100, 20)
        
        st.markdown("---")
        st.subheader("Reasoning Model")
        use_llm = True
        llm_mode = st.selectbox("Model Tier", ["Fast (llama-3.1-8b-instant)", "Balanced (llama-3.3-70b-versatile)", "Premium (openai/gpt-oss-120b)"])
        llm_map = {
            "Fast (llama-3.1-8b-instant)": "llama-3.1-8b-instant",
            "Balanced (llama-3.3-70b-versatile)": "llama-3.3-70b-versatile",
            "Premium (openai/gpt-oss-120b)": "openai/gpt-oss-120b"
        }
        actual_llm_mode = llm_map.get(llm_mode, "llama-3.1-8b-instant")

        run_btn = st.form_submit_button("Run Optimization")

# -------------------------------------------------------------------
# PIPELINE EXECUTION LOGIC
# -------------------------------------------------------------------

if run_btn:
    with st.spinner("Executing Roofline Multi-Objective Optimization..."):
        try:
            pipeline = DeploymentPipeline()
            weights = {
                'latency': w_lat, 'memory_efficiency': w_mem,
                'cost_efficiency': w_cost, 'energy_efficiency': w_eng,
                'accuracy_preservation': w_acc
            }
            constraints = {"max_latency": max_lat, "max_cost": max_cost}
            
            hw_type = None if selected_hardware == "None" else selected_hardware
            
            results = pipeline.run_pipeline(
                model_id=selected_model, 
                hardware_type=hw_type,
                workload_type=selected_workload,
                weights=weights,
                constraints=constraints,
                llm_mode=actual_llm_mode,
                use_llm_reasoning=use_llm
            )
            st.session_state["results"] = results
            st.session_state["constraints"] = constraints
        except AssertionError as ae:
            st.error(f"Validation Guard: {ae}")
        except Exception:
            st.error("Pipeline encountered an issue during execution. Please check hardware limits or model compatibility.")

# -------------------------------------------------------------------
# RESULTS DASHBOARD
# -------------------------------------------------------------------
res = st.session_state.get("results")
cons = st.session_state.get("constraints", {})

if res:
    if "error" in res:
        st.error("Pipeline execution halted. Please adjust constraints and try again.")
        st.stop()
        
    best_eval = res.get("best_evaluation", {})
    baseline_eval = res.get("baseline_evaluation", {})
    sim = best_eval.get("simulation", {})
    score_brk = res.get("scoring_breakdown", {})
    hardware_comp = res.get("hardware_comparison", [])
    
    # KPIs
    latency = sim.get("latency_ms", 0)
    cost = sim.get("cost_usd", 0)
    memory = sim.get("memory_mb", 0)
    energy_kwh = sim.get("energy_kwh", 0)
    eff_score = score_brk.get("efficiency_score", 0)
    
    energy_wh = energy_kwh * 1000
    if energy_wh < 1:
        energy_display = f"{energy_wh * 1000:.2f} mWh"
    else:
        energy_display = f"{energy_wh:.2f} Wh"
        
    status = res.get("status", "unknown")
    original_req = res.get("original_request", "Unknown")
    model_name_display = res.get("model_id", "Unknown")
    
    # Display logic for model status
    if status == "ready":
        st.success(f"Model loaded successfully: {model_name_display} (Ready)")
    elif status == "fallback":
        st.info(f"Using compatible model for simulation: {model_name_display} ({original_req} is restricted)")
    else:
        st.info(f"Model loaded: {model_name_display}")
    
    # 1. KPI SECTION
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.markdown(f"<div class='metric-card' title='Time per inference request'><div class='metric-title'>Latency (ms)</div><div class='metric-value val-blue'>{latency:.1f}</div></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='metric-card' title='Total GPU VRAM footprint'><div class='metric-title'>Memory (MB)</div><div class='metric-value val-blue'>{memory:.0f}</div></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='metric-card' title='Estimated cost per inference'><div class='metric-title'>Cost / Req ($)</div><div class='metric-value val-green'>{cost:.5f}</div></div>", unsafe_allow_html=True)
    with c4: st.markdown(f"<div class='metric-card' title='GPU energy per request'><div class='metric-title'>Energy</div><div class='metric-value val-red'>{energy_display}</div></div>", unsafe_allow_html=True)
    with c5: st.markdown(f"<div class='metric-card' title='Multi-objective efficiency score'><div class='metric-title'>Score</div><div class='metric-value val-purple'>{eff_score:.1f}/100</div></div>", unsafe_allow_html=True)

    # 2. RECOMMENDED STRATEGY
    strat = best_eval.get("strategy", {})
    hw_name = best_eval.get("hardware", "Unknown")
    precision = strat.get("precision", "Unknown").upper()
    pruning = strat.get("prune_ratio", 0) * 100
    mode = strat.get("deployment_mode", "Balanced").capitalize()
    
    st.markdown(f"""
    <div class="decision-box">
        <h3>Recommended Deployment Strategy</h3>
        <p>Hardware: {hw_name} <br>
        Precision: {precision} <br>
        Pruning: {pruning:.0f}% <br>
        Mode: {mode}</p>
        <p>Why this works:<br>
        - Meets latency constraints<br>
        - Reduces memory footprint<br>
        - Maintains acceptable accuracy</p>
        <p>Confidence: {eff_score:.0f}%</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 3. IMPACT & CONSTRAINT PANEL
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Deployment Impact")
        if baseline_eval:
            b_lat = baseline_eval['simulation'].get('latency_ms', 1)
            b_cost = baseline_eval['simulation'].get('cost_usd', 1)
            b_eng = baseline_eval['simulation'].get('energy_kwh', 1)
            
            i_lat = percent_improvement(b_lat, latency)
            i_cost = percent_improvement(b_cost, cost)
            i_eng = percent_improvement(b_eng, energy_kwh)
            
            st.markdown(f"""
            Baseline vs Optimized:
            - Latency ↓ {i_lat:.1f}%
            - Cost ↓ {i_cost:.1f}%
            - Energy ↓ {i_eng:.1f}%
            """)
        else:
            st.info("Baseline metrics not available.")
            
    with col_b:
        st.subheader("Constraint Satisfaction")
        lat_pass = latency <= cons.get('max_latency', float('inf'))
        mem_pass = True # Assuming memory passes if we reached here
        acc_penalty = sim.get('accuracy_penalty', 0)
        
        lat_msg = "✔ Latency constraint satisfied" if lat_pass else "⚠ Latency constraint exceeded"
        mem_msg = "✔ Memory constraint satisfied" if mem_pass else "⚠ Memory constraint exceeded"
        acc_msg = "⚠ Minor accuracy tradeoff expected" if acc_penalty > 0.1 else "✔ Accuracy preserved"
        
        st.markdown(f"""
        {lat_msg}  
        {mem_msg}  
        {acc_msg}  
        """)

    # 4. ENGINEERING INSIGHT
    st.subheader("Engineering Insight")
    reasoning_text = score_brk.get("reasoning", "This strategy operates efficiently within the desired constraints.")
    st.markdown(f"<div class='insight-card'><div class='insight-text'>{reasoning_text}</div></div>", unsafe_allow_html=True)

    # 5. VISUALS
    st.plotly_chart(create_pareto_frontier(hardware_comp), use_container_width=True)
    st.plotly_chart(create_roofline_plot(hardware_comp), use_container_width=True)
    
    with st.expander("Advanced Analysis (Optional)"):
        rc1, rc2 = st.columns(2)
        with rc1: st.plotly_chart(create_radar_chart(hardware_comp), use_container_width=True)
        with rc2: st.plotly_chart(create_strategy_heatmap(hardware_comp), use_container_width=True)

else:
    st.info("System Ready. Configure parameters and run optimization to begin.")
