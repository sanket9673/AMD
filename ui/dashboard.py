import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import sys
import traceback

# Add parent directory to sys.path to allow importing from 'core'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.pipeline_engine import DeploymentPipeline
from core.hardware_profiles import HARDWARE_DATABASE

# Streamlit config
st.set_page_config(
    page_title="Enterprise AI Deployment Intelligence",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise CSS
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #38bdf8;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }
    .stSelectbox label, .stSlider label {
        color: #94a3b8 !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #0284c7 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2563eb 0%, #0284c7 100%);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        transform: translateY(-1px);
        color: white;
    }
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .insight-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-left: 4px solid #3b82f6;
        padding: 24px;
        border-radius: 8px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    .insight-text {
        color: #e2e8f0;
        font-size: 1.1rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌌 Enterprise AI Deployment Intelligence")
st.markdown("Advanced visualization and multi-objective optimization for AMD compute ecosystems.")

if "results" not in st.session_state:
    st.session_state.results = None

with st.sidebar:
    st.header("⚙️ Deployment Configuration")
    
    models = ["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "mistralai/Mistral-7B-v0.1"]
    selected_model = st.selectbox("Model Architecture", models)
    
    # 1. Hardware Selector
    st.subheader("🖥️ Hardware Target")
    hardware_choices = ["AMD_MI250", "AMD_MI300X"]
    selected_hardware = st.selectbox("Target Hardware Suite", hardware_choices)
    
    # 2. Workload Selector
    st.subheader("⚡ Workload Profile")
    workloads = ["chat_inference", "batch_inference", "fine_tuning"]
    selected_workload = st.selectbox("Execution Mode", workloads)
    
    st.markdown("---")
    # 3. Multi-Objective Weight Sliders
    st.subheader("⚖️ Objective Weights")
    w_lat_raw = st.slider("Latency Weight (α)", 0, 100, 25)
    w_mem_raw = st.slider("Memory Weight (β)", 0, 100, 25)
    w_cost_raw = st.slider("Cost Weight (γ)", 0, 100, 25)
    w_eng_raw = st.slider("Energy Weight (δ)", 0, 100, 25)
    
    total_w = w_lat_raw + w_mem_raw + w_cost_raw + w_eng_raw
    if total_w == 0:
        total_w = 1 # prevent div by zero
    
    weights = {
        'latency': w_lat_raw / total_w,
        'memory_efficiency': w_mem_raw / total_w,
        'cost_efficiency': w_cost_raw / total_w,
        'energy_efficiency': w_eng_raw / total_w
    }
    
    st.caption(f"Normalized: α={weights['latency']:.2f}, β={weights['memory_efficiency']:.2f}, γ={weights['cost_efficiency']:.2f}, δ={weights['energy_efficiency']:.2f}")

    st.markdown("---")
    if st.button("🚀 Run Enterprise Pipeline"):
        with st.spinner("Executing Multi-Objective Optimization..."):
            try:
                pipeline = DeploymentPipeline()
                results = pipeline.run_pipeline(
                    model_id=selected_model, 
                    hardware_type=selected_hardware,
                    workload_type=selected_workload,
                    weights=weights
                )
                st.session_state.results = results
            except Exception as e:
                st.error(f"Pipeline Error: {e}\n{traceback.format_exc()}")

if st.session_state.results:
    res = st.session_state.results
    best_eval = res.get("best_evaluation", {})
    sim = best_eval.get("simulation", {})
    hardware_comp = res.get("hardware_comparison", [])
    
    # 6. Enterprise Metrics Panel
    st.subheader("📈 Enterprise Metrics Overview")
    
    # Calculate some enterprise metrics based on sim
    latency = sim.get("latency_estimate", 0.1)
    throughput = sim.get("throughput_estimate", 10.0)
    energy = sim.get("energy_consumption", 250)
    cost = sim.get("cost_estimate", 0.05)
    
    # Assume 1 month continuous running
    req_per_month = 1_000_000
    monthly_cost = req_per_month * cost
    nodes_required = max(1, int((req_per_month / (30*24*3600)) / max(1e-9, throughput)) + 1)
    energy_hr = energy * nodes_required
    eff_score = best_eval.get("score", 0) if isinstance(best_eval, dict) else res.get("scoring_breakdown", {}).get("score", 0)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-card'><div style='color:#94a3b8;font-size:0.9rem'>Est. Monthly Cost</div><div style='font-size:1.8rem;color:#10b981;font-weight:bold'>${monthly_cost:,.2f}</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><div style='color:#94a3b8;font-size:0.9rem'>Nodes Required</div><div style='font-size:1.8rem;color:#3b82f6;font-weight:bold'>{nodes_required}</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><div style='color:#94a3b8;font-size:0.9rem'>Energy / Hour</div><div style='font-size:1.8rem;color:#f59e0b;font-weight:bold'>{energy_hr:,.1f} W</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-card'><div style='color:#94a3b8;font-size:0.9rem'>Efficiency Score</div><div style='font-size:1.8rem;color:#8b5cf6;font-weight:bold'>{eff_score:,.1f} / 100</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Visualization Row
    col_rad, col_bar = st.columns(2)
    
    # 4. Radar Chart
    with col_rad:
        st.subheader("🕸️ Top Strategies Topography")
        if hardware_comp:
            fig_rad = go.Figure()
            # Plot top 3
            top_3 = hardware_comp[:3]
            
            # Find bounds for normalization
            max_lat = max([c['latency_estimate'] for c in hardware_comp]) or 1.0
            max_cost = max([c['cost_estimate'] for c in hardware_comp]) or 1.0
            max_energy = max([c['energy_consumption'] for c in hardware_comp]) or 1.0
            
            categories = ['Latency Efficiency', 'Memory Efficiency', 'Cost Efficiency', 'Energy Efficiency']
            
            for i, c in enumerate(top_3):
                # normalize so larger is better for the radar chart
                # Latency efficiency
                lat_scr = max(0, 1 - (c['latency_estimate'] / max_lat)) * 100
                # Memory efficiency (using hardware_fit_score which acts as memory efficiency)
                mem_scr = c.get('hardware_fit_score', 100.0) 
                # Cost efficiency
                cost_scr = max(0, 1 - (c['cost_estimate'] / max_cost)) * 100
                # Energy efficiency
                en_scr = max(0, 1 - (c['energy_consumption'] / max_energy)) * 100
                
                prec = c['strategy'].get('precision', 'Unknown')
                mode = c['strategy'].get('deployment_mode', 'Unknown')
                
                fig_rad.add_trace(go.Scatterpolar(
                    r=[lat_scr, mem_scr, cost_scr, en_scr],
                    theta=categories,
                    fill='toself',
                    name=f"Rank {i+1} ({c['hardware']} - {prec})"
                ))
            
            fig_rad.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100]),
                    bgcolor='#0b0f19'
                ),
                showlegend=True,
                paper_bgcolor='#0b0f19',
                font=dict(color='#e2e8f0'),
                margin=dict(l=40, r=40, t=40, b=40)
            )
            st.plotly_chart(fig_rad, use_container_width=True)
            
    # 5. Hardware Comparison Graph
    with col_bar:
        st.subheader("📊 Hardware Architecture Comparison")
        # Find best strategy for MI250 and MI300X
        mi250_best = next((c for c in hardware_comp if "MI250" in c['hardware']), None)
        mi300x_best = next((c for c in hardware_comp if "MI300X" in c['hardware']), None)
        
        comp_data = []
        if mi250_best:
            comp_data.append({"Hardware": "AMD_MI250", "Cost": mi250_best['cost_estimate'], "Energy": mi250_best['energy_consumption'], "Latency": mi250_best['latency_estimate']})
        if mi300x_best:
            comp_data.append({"Hardware": "AMD_MI300X", "Cost": mi300x_best['cost_estimate'], "Energy": mi300x_best['energy_consumption'], "Latency": mi300x_best['latency_estimate']})
            
        if comp_data:
            df_comp = pd.DataFrame(comp_data)
            
            # Melt data for grouped bar chart
            df_melt = df_comp.melt(id_vars=["Hardware"], var_name="Metric", value_name="Value")
            
            fig_bar = px.bar(
                df_melt, 
                x="Metric", 
                y="Value", 
                color="Hardware", 
                barmode="group",
                color_discrete_sequence=["#f59e0b", "#3b82f6"]
            )
            fig_bar.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig_bar.update_yaxes(type='log', title="Value (Log Scale)")
            st.plotly_chart(fig_bar, use_container_width=True)

    # 7. AI Insight Section
    st.markdown("---")
    st.subheader("🧠 Reasoning Engine Intelligence")
    reasoning_text = res.get("scoring_breakdown", {}).get("reasoning", "No insights available.")
    st.markdown(f"<div class='insight-card'><div class='insight-text'>{reasoning_text}</div></div>", unsafe_allow_html=True)

else:
    st.info("👈 System Ready. Configure parameters and run the Enterprise Pipeline.")
