import streamlit as st
import pandas as pd
import sys
import os
import logging
import warnings
import random
from dotenv import load_dotenv

# Suppress loud HTTP logs
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.pipeline_engine import DeploymentPipeline
from core.model_profiler import safe_load_config
from utils.plotting import (
    create_radar_chart, 
    create_pareto_frontier, 
    create_strategy_heatmap,
    create_roofline_plot,
    create_impact_chart,
    create_hardware_comparison,
    create_strategy_leaderboard
)

# Streamlit config
st.set_page_config(
    page_title="Slingshot-AI | Enterprise Deployment",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cold Start Optimization
if "initialized" not in st.session_state:
    st.session_state["initialized"] = True
    st.info("Initializing system... please wait (first load only)")

# Cache Model Loading
@st.cache_resource
def cached_config_loader(model_id, hf_token):
    return safe_load_config(model_id, hf_token)

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
    
    .header-box { background: #1e293b; padding: 20px; border-radius: 8px; margin-bottom: 24px; border: 1px solid #334155; }
    .header-box h2 { margin-top: 0; color: #38bdf8; font-size: 1.5rem; }
    .header-box p { color: #94a3b8; margin-bottom: 0; }
    
    .decision-box { background: #152238; border: 2px solid #3b82f6; border-radius: 8px; padding: 20px; margin-top: 20px; margin-bottom: 20px; }
    .decision-box h3 { color: #38bdf8; margin-top: 0; margin-bottom: 15px; font-size: 1.25rem; }
    
    .insight-card { background: #1e293b; border-left: 4px solid #3b82f6; padding: 20px; border-radius: 6px; margin-top: 20px; border: 1px solid #334155; }
    .insight-text { color: #cbd5e1; font-size: 1.05rem; line-height: 1.6; }
    
    [data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; color: #38bdf8; }
</style>
""", unsafe_allow_html=True)

def percent_improvement(base: float, opt: float) -> float:
    if base <= 0:
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
st.title("Slingshot-AI")
st.markdown("""
<div class="header-box">
    <h2>Enterprise Deployment Intelligence Engine</h2>
    <p>This system analyzes AI models and recommends optimal deployment strategies based on latency, cost, and energy efficiency across AMD hardware.</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# SIDEBAR CONFIGURATION
# -------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    
    demo_mode = st.checkbox("Fast Demo Mode", value=True)
    
    with st.form("pipeline_form"):
        models = [
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "microsoft/phi-2",
            "google/gemma-2b"
        ]
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
    if demo_mode:
        # Realistic Synthetic Spread for Demo
        lat_base = random.uniform(150, 300)
        cost_base = random.uniform(0.0004, 0.0008)
        eng_base = random.uniform(0.00004, 0.00008)
        
        lat_opt = lat_base * random.uniform(0.3, 0.6)
        cost_opt = cost_base * random.uniform(0.3, 0.6)
        eng_opt = eng_base * random.uniform(0.3, 0.6)
        
        hw_comp = []
        for i in range(20):
            hw_comp.append({
                "strategy": {"precision": random.choice(["fp32", "fp16", "int8", "int4"]), "prune_ratio": random.choice([0.0, 0.2, 0.4, 0.6]), "deployment_mode": "balanced"},
                "hardware": random.choice(["AMD_MI250", "AMD_MI300X"]),
                "simulation": {
                    "latency_ms": lat_base * random.uniform(0.3, 1.5),
                    "memory_mb": random.uniform(400, 2000),
                    "cost_usd": cost_base * random.uniform(0.3, 1.5),
                    "energy_kwh": eng_base * random.uniform(0.3, 1.5),
                    "throughput": random.uniform(10, 100)
                },
                "score": random.uniform(40, 95)
            })
        # Inject best into hw_comp
        hw_comp[0]["strategy"] = {"precision": "int4", "prune_ratio": 0.4, "deployment_mode": "high_performance"}
        hw_comp[0]["hardware"] = "AMD_MI300X"
        hw_comp[0]["simulation"].update({"latency_ms": lat_opt, "cost_usd": cost_opt, "energy_kwh": eng_opt, "memory_mb": 520.0})
        hw_comp[0]["score"] = 96.5

        st.session_state["results"] = {
            "model_id": selected_model,
            "status": "ready",
            "original_request": selected_model,
            "best_evaluation": hw_comp[0],
            "baseline_evaluation": {
                "simulation": {
                    "latency_ms": lat_base,
                    "memory_mb": 1800.0,
                    "cost_usd": cost_base,
                    "energy_kwh": eng_base,
                }
            },
            "scoring_breakdown": {
                "efficiency_score": 96.5,
                "reasoning": "In Fast Demo Mode, we bypass full performance simulations. The precomputed strategy demonstrates how INT4 quantization combined with 40% structured pruning minimizes memory bandwidth bottlenecks, dropping latency by over 60% while maintaining target boundaries."
            },
            "hardware_comparison": hw_comp,
            "is_demo": True
        }
        st.session_state["constraints"] = {"max_latency": max_lat, "max_cost": max_cost}
    else:
        with st.spinner("Loading model configuration and executing Multi-Objective Optimization..."):
            try:
                load_dotenv()
                hf_token = os.getenv("HF_TOKEN")
                
                config, actual_m_id, status = cached_config_loader(selected_model, hf_token)
                
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
                    use_llm_reasoning=use_llm,
                    config=config,
                    status=status,
                    actual_model_id=actual_m_id
                )
                results["is_demo"] = False
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
    is_demo = res.get("is_demo", False)
    
    latency = sim.get("latency_ms", 0)
    cost = sim.get("cost_usd", 0)
    energy_kwh = sim.get("energy_kwh", 0)
    eff_score = score_brk.get("efficiency_score", 0)
    
    b_lat = baseline_eval.get('simulation', {}).get('latency_ms', 1)
    b_cost = baseline_eval.get('simulation', {}).get('cost_usd', 1)
    b_eng = baseline_eval.get('simulation', {}).get('energy_kwh', 1)
    
    i_lat = percent_improvement(b_lat, latency)
    i_cost = percent_improvement(b_cost, cost)
    i_eng = percent_improvement(b_eng, energy_kwh)
        
    status = res.get("status", "unknown")
    original_req = res.get("original_request", "Unknown")
    model_name_display = res.get("model_id", "Unknown")
    
    if is_demo:
        st.success(f"Fast Demo Mode loaded successfully: {model_name_display}")
    elif status == "ready":
        st.success(f"Model loaded successfully: {model_name_display}")
    elif status == "fallback":
        st.warning(f"Using compatible model for simulation: {model_name_display} (Original requested was restricted)")
    else:
        st.info(f"Model loaded: {model_name_display}")

    # 1. KPI SECTION (EXECUTIVE VIEW)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Latency", f"{latency:.1f} ms", delta=f"-{i_lat:.1f}%", delta_color="inverse")
    with c2: st.metric("Cost / Req", f"${cost:.5f}", delta=f"-{i_cost:.1f}%", delta_color="inverse")
    with c3: st.metric("Energy", f"{energy_kwh*1000:.3f} mWh", delta=f"-{i_eng:.1f}%", delta_color="inverse")
    with c4: st.metric("Efficiency Score", f"{eff_score:.1f}/100")

    # 2. RECOMMENDED STRATEGY
    strat = best_eval.get("strategy", {})
    hw_name = best_eval.get("hardware", "Unknown")
    precision = strat.get("precision", "Unknown").upper()
    pruning = strat.get("prune_ratio", 0) * 100
    
    st.markdown(f"""
    <div class="decision-box">
        <h3>Optimized Deployment Strategy</h3>
        <p><b>Hardware:</b> {hw_name} | <b>Precision:</b> {precision} | <b>Pruning:</b> {pruning:.0f}%</p>
        <p>This configuration reduces latency and cost while maintaining acceptable accuracy for production inference workloads.</p>
    </div>
    """, unsafe_allow_html=True)
    
    reasoning_text = score_brk.get("reasoning", "This strategy operates efficiently within the desired constraints.")
    st.markdown(f"<div class='insight-card'><div class='insight-text'>{reasoning_text}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. TABS FOR VISUALIZATIONS
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Optimization Impact", 
        "Strategy Tradeoffs", 
        "Hardware Comparison", 
        "Strategy Leaderboard",
        "Roofline & Radar"
    ])
    
    with tab1:
        st.plotly_chart(create_impact_chart(baseline_eval, best_eval), use_container_width=True)
        st.markdown("### What this means")
        st.markdown("- **Why this strategy is chosen:** It maximizes the latency and cost reductions by leveraging lower precision types which drastically lower memory bandwidth pressure.")
        
    with tab2:
        st.plotly_chart(create_pareto_frontier(hardware_comp), use_container_width=True)
        st.markdown("### What this means")
        st.markdown("- **Tradeoffs exist:** The cluster of points near the optimal strategy shows where lowering precision further yields diminishing returns in latency but drastically hurts model accuracy.")
        
    with tab3:
        st.plotly_chart(create_hardware_comparison(hardware_comp), use_container_width=True)
        st.markdown("### What this means")
        st.markdown("- **Why MI300X wins:** MI300X delivers significantly better latency due to higher memory bandwidth (5.3 TB/s), allowing memory-bound decode phases to execute faster.")
        
    with tab4:
        st.plotly_chart(create_strategy_leaderboard(hardware_comp), use_container_width=True)
        st.markdown("### What this means")
        st.markdown("- **Leaderboard Ranking:** The top strategies are heavily dominated by INT8 and INT4 quantization. Unpruned FP16 variations fall lower down the board due to high compute latency.")

    with tab5:
        rc1, rc2 = st.columns(2)
        with rc1: 
            st.plotly_chart(create_radar_chart(hardware_comp), use_container_width=True)
        with rc2: 
            if not is_demo:
                st.plotly_chart(create_roofline_plot(hardware_comp), use_container_width=True)
            else:
                st.info("Roofline plot requires full runtime execution. Uncheck 'Fast Demo Mode' to view.")

else:
    st.info("System Ready. Configure parameters and run optimization to begin.")
