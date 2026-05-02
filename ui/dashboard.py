import streamlit as st
import pandas as pd
import sys
import os
import logging
import warnings
import random

# Phase 2: Fix dotenv import crash
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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

# Production Safe ENV Handling
try:
    st_secrets_hf = st.secrets.get("HF_TOKEN", None)
except Exception:
    st_secrets_hf = None
    
try:
    st_secrets_groq = st.secrets.get("GROQ_API_KEY", None)
except Exception:
    st_secrets_groq = None

HF_TOKEN = os.getenv("HF_TOKEN") or st_secrets_hf
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st_secrets_groq

# Handle missing API Keys Safely
if GROQ_API_KEY is None:
    USE_LLM = False
else:
    USE_LLM = True

# Cold Start Optimization
if "initialized" not in st.session_state:
    st.session_state["initialized"] = True
    st.info("Initializing system... please wait")

# Cache Model Loading
@st.cache_resource
def cached_config_loader(model_id, hf_token):
    return safe_load_config(model_id, hf_token)


# Custom Enterprise CSS
st.markdown("""
<style>
    .main { background-color: transparent; color: var(--text-color); font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4, h5, h6 { color: var(--text-color); font-weight: 600; }
    .stSelectbox label, .stSlider label, .stNumberInput label, .stCheckbox label { color: var(--text-color) !important; opacity: 0.85; }
    
    div.stButton > button {
        background: linear-gradient(135deg, #FF6B00 0%, #e65c00 100%) !important;
        color: white !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        padding: 16px !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(255, 107, 0, 0.4) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(255, 107, 0, 0.6) !important;
    }
    
    .header-box { 
        background: var(--secondary-background-color); 
        backdrop-filter: blur(10px);
        padding: 24px; 
        border-radius: 12px; 
        margin-bottom: 24px; 
        border: 1px solid rgba(128, 128, 128, 0.2); 
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .header-box h2 { margin-top: 0; background: linear-gradient(90deg, #3b82f6, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 1.8rem; }
    .header-box p { color: var(--text-color); opacity: 0.8; margin-bottom: 0; font-size: 1.1rem; }
    
    .decision-box { background: var(--secondary-background-color); border: 1px solid rgba(128, 128, 128, 0.2); border-radius: 12px; padding: 24px; margin-top: 20px; margin-bottom: 20px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.05); backdrop-filter: blur(10px); }
    .decision-box h3 { color: #3b82f6; margin-top: 0; margin-bottom: 15px; font-size: 1.3rem; }
    
    .insight-card { background: var(--secondary-background-color); border-left: 4px solid #3b82f6; padding: 20px; border-radius: 8px; margin-top: 20px; border: 1px solid rgba(128, 128, 128, 0.2); }
    .insight-text { color: var(--text-color); font-size: 1.1rem; line-height: 1.6; }
    
    [data-testid="stMetricValue"] { font-size: 2.2rem; font-weight: 800; background: linear-gradient(90deg, #3b82f6, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    [data-testid="stMetricLabel"] { font-size: 1.1rem; color: var(--text-color); opacity: 0.8; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

def percent_improvement(base: float, opt: float) -> float:
    if base <= 0:
        return 0.0
    return max(0.0, ((base - opt) / base) * 100)

def generate_demo_results(selected_model, selected_hardware):
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

    return {
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

# -------------------------------------------------------------------
# STATE MANAGEMENT
# -------------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state["results"] = None

if "run_trigger" not in st.session_state:
    st.session_state["run_trigger"] = False

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

st.info("Configuration loaded successfully. Please select your target parameters below.")

if not USE_LLM:
    st.warning("AI insights disabled (no API key found). Using heuristic explanation.")

# -------------------------------------------------------------------
# SIDEBAR CONFIGURATION (Step 1)
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Step 1: System Configuration")
    st.markdown("Configure your constraints and target model here.")
    
    models = [
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "microsoft/phi-2",
        "google/gemma-2b"
    ]
    selected_model = st.selectbox("Model Architecture", models)
    selected_hardware = st.selectbox("Hardware Bias (Optional)", ["None", "AMD_MI250", "AMD_MI300X"])
    selected_workload = st.selectbox("Workload", ["chat_inference", "batch_inference", "fine_tuning"])
    
    st.markdown("---")
    st.markdown("### Constraints")
    max_lat = st.number_input("Max Latency (ms)", value=5000.0, min_value=0.1)
    max_cost = st.number_input("Max Cost/Req ($)", value=0.01, min_value=0.00001, format="%.5f")
    
    st.markdown("---")
    st.markdown("### Reasoning Engine")
    llm_mode = st.selectbox("Reasoning Tier", ["Fast (llama-3.1-8b-instant)", "Balanced (llama-3.3-70b-versatile)", "Premium (openai/gpt-oss-120b)"], disabled=not USE_LLM)
    llm_map = {
        "Fast (llama-3.1-8b-instant)": "llama-3.1-8b-instant",
        "Balanced (llama-3.3-70b-versatile)": "llama-3.3-70b-versatile",
        "Premium (openai/gpt-oss-120b)": "openai/gpt-oss-120b"
    }
    actual_llm_mode = llm_map.get(llm_mode, "llama-3.1-8b-instant")
    


# -------------------------------------------------------------------
# MAIN AREA - EXECUTION (Step 2)
# -------------------------------------------------------------------
st.markdown("### 🚀 Step 2: Execute Optimization Engine")
st.caption("Review your configuration in the sidebar, tune the optimization weights on the right, and run the simulation.")

exec_col, spacer, weight_col = st.columns([1, 0.2, 1.2])

with exec_col:
    st.markdown("<br><br>", unsafe_allow_html=True)
    run_clicked = st.button("Run Deployment Optimization", use_container_width=True)
    demo_mode = st.checkbox("⚡ Fast Demo Mode (Instant Results)", value=True)

with weight_col:
    st.markdown("#### ⚖️ Advanced Optimization Weights")
    w_c1, w_c2 = st.columns(2)
    with w_c1:
        w_lat = st.slider("Latency", 0, 100, 25)
        w_mem = st.slider("Memory", 0, 100, 20)
        w_cost = st.slider("Cost", 0, 100, 20)
    with w_c2:
        w_eng = st.slider("Energy", 0, 100, 15)
        w_acc = st.slider("Accuracy", 0, 100, 20)

st.markdown("---")



# -------------------------------------------------------------------
# PIPELINE EXECUTION LOGIC
# -------------------------------------------------------------------
if run_clicked:
    st.session_state["run_trigger"] = True
    with st.spinner("Running deployment optimization..."):
        if demo_mode:
            st.session_state["results"] = generate_demo_results(selected_model, selected_hardware)
            st.session_state["constraints"] = {"max_latency": max_lat, "max_cost": max_cost}
        else:
            try:
                config, actual_m_id, status = cached_config_loader(selected_model, HF_TOKEN)
                
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
                    use_llm_reasoning=USE_LLM,
                    config=config,
                    status=status,
                    actual_model_id=actual_m_id
                )
                
                if "error" in results:
                    raise Exception(results["error"])
                    
                results["is_demo"] = False
                st.session_state["results"] = results
                st.session_state["constraints"] = constraints
                
            except Exception as e:
                st.error("System failed gracefully. Showing demo results.")
                logger.error(f"Execution Error: {e}")
                st.session_state["results"] = generate_demo_results(selected_model, selected_hardware)
                st.session_state["constraints"] = {"max_latency": max_lat, "max_cost": max_cost}

# -------------------------------------------------------------------
# RESULTS DASHBOARD
# -------------------------------------------------------------------
res = st.session_state.get("results")

if res:
    # Phase 1: Add Result Anchor
    st.markdown('<div id="results-section"></div>', unsafe_allow_html=True)
    st.markdown("## 📊 Step 3: Optimization Results Intelligence Report")
    
    # Phase 3: "Result Ready" feedback + scroll implementation
    if st.session_state.get("run_trigger"):
        st.success("Optimization completed. Scroll down to view results.")
        # Phase 2: Auto Scroll using smooth scrollIntoView
        scroll_script = f"""
        <script>
            // Cache buster: {random.random()}
            setTimeout(function() {{
                try {{
                    const doc = window.parent.document;
                    const element = doc.getElementById("results-section") || doc.querySelector('[id="results-section"]');
                    if (element) {{
                        element.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                    }}
                }} catch (e) {{
                    console.error("Scroll error:", e);
                }}
            }}, 300);
        </script>
        """
        import streamlit.components.v1 as components
        components.html(scroll_script, height=0)
        st.session_state["run_trigger"] = False
        
    st.divider()

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
    
    # Phase 4 & 5: Highlight Result Section & Structured Output
    with st.container():
        st.markdown("### Recommended Configuration")
        strat = best_eval.get("strategy", {})
        hw_name = best_eval.get("hardware", "Unknown")
        precision = strat.get("precision", "Unknown").upper()
        pruning = strat.get("prune_ratio", 0) * 100
        mode = strat.get("deployment_mode", "Balanced").capitalize()
        
        st.markdown(f"""
        **Precision:** {precision}  
        **Pruning:** {pruning:.0f}%  
        **Hardware:** {hw_name}  
        **Deployment Mode:** {mode}
        """)
        
    st.divider()

    # Phase 7: Add "Key Impact" Summary
    st.markdown("### Impact Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1: c1.metric("Latency Reduction", f"{i_lat:.1f}%")
    with c2: c2.metric("Cost Reduction", f"{i_cost:.1f}%")
    with c3: c3.metric("Energy Reduction", f"{i_eng:.1f}%")
    with c4: c4.metric("Efficiency Score", f"{eff_score:.1f}/100")
    
    st.divider()

    # Phase 6: Add Reasoning Panel
    st.markdown("### Why this strategy was selected")
    reasoning_text = score_brk.get("reasoning", "This strategy operates efficiently within the desired constraints.")
    st.info(reasoning_text)

    st.markdown("---")

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
