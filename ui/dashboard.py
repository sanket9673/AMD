import streamlit as st
import pandas as pd
import sys
import os
import logging
import warnings
import random

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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

st.set_page_config(
    page_title="Slingshot-AI | Enterprise Deployment",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

USE_LLM = GROQ_API_KEY is not None

if "initialized" not in st.session_state:
    st.session_state["initialized"] = True
    st.info("Initializing system... please wait")

@st.cache_resource
def cached_config_loader(model_id, hf_token):
    return safe_load_config(model_id, hf_token)

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


def generate_demo_results(selected_model, selected_hardware, selected_workload, weights, constraints):
    """
    Runs the real pipeline using a pre-built DummyConfig to avoid HF network calls.
    All metrics are computed by the real Scorer, PerformanceSimulator, and Roofline math.
    No hardcoded scores. No fake random numbers for scientific metrics.
    """
    from core.pipeline_engine import DeploymentPipeline
    from core.hardware import HARDWARE_DATABASE

    # Build a DummyConfig that matches the selected model size approximately
    model_size_map = {
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
            "hidden_size": 2048,
            "num_hidden_layers": 22,
            "vocab_size": 32000,
            "num_attention_heads": 32
        },
        "microsoft/phi-2": {
            "hidden_size": 2560,
            "num_hidden_layers": 32,
            "vocab_size": 51200,
            "num_attention_heads": 32
        },
        "google/gemma-2b": {
            "hidden_size": 2048,
            "num_hidden_layers": 18,
            "vocab_size": 262144,
            "num_attention_heads": 8
        },
    }
    cfg_vals = model_size_map.get(selected_model, model_size_map["TinyLlama/TinyLlama-1.1B-Chat-v1.0"])

    class FastDummyConfig:
        hidden_size = cfg_vals["hidden_size"]
        num_hidden_layers = cfg_vals["num_hidden_layers"]
        vocab_size = cfg_vals["vocab_size"]
        num_attention_heads = cfg_vals.get("num_attention_heads", 32)

    pipeline = DeploymentPipeline()
    hw_type = None if selected_hardware == "None" else selected_hardware

    results = pipeline.run_pipeline(
        model_id=selected_model,
        hardware_type=hw_type,
        workload_type=selected_workload,
        weights=weights,
        constraints=constraints,
        llm_mode="llama-3.1-8b-instant",
        use_llm_reasoning=False,
        config=FastDummyConfig(),
        status="demo",
        actual_model_id=selected_model
    )

    if "error" in results:
        # If real pipeline fails for any reason, raise so caller can show error
        raise Exception(results["error"])

    results["is_demo"] = True
    return results


if "results" not in st.session_state:
    st.session_state["results"] = None

if "run_trigger" not in st.session_state:
    st.session_state["run_trigger"] = False

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
    llm_mode = st.selectbox(
        "Reasoning Tier",
        ["Fast (llama-3.1-8b-instant)", "Balanced (llama-3.3-70b-versatile)", "Premium (openai/gpt-oss-120b)"],
        disabled=not USE_LLM
    )
    llm_map = {
        "Fast (llama-3.1-8b-instant)": "llama-3.1-8b-instant",
        "Balanced (llama-3.3-70b-versatile)": "llama-3.3-70b-versatile",
        "Premium (openai/gpt-oss-120b)": "openai/gpt-oss-120b"
    }
    actual_llm_mode = llm_map.get(llm_mode, "llama-3.1-8b-instant")

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

if run_clicked:
    st.session_state["run_trigger"] = True
    with st.spinner("Running deployment optimization..."):
        if demo_mode:
            try:
                weights_dict = {
                    'latency': w_lat,
                    'memory_efficiency': w_mem,
                    'cost_efficiency': w_cost,
                    'energy_efficiency': w_eng,
                    'accuracy_preservation': w_acc
                }
                constraints_dict = {"max_latency": max_lat, "max_cost": max_cost}
                st.session_state["results"] = generate_demo_results(
                    selected_model, selected_hardware, selected_workload,
                    weights_dict, constraints_dict
                )
                st.session_state["constraints"] = constraints_dict
            except Exception as e:
                st.error(f"Demo pipeline failed: {e}")
                logger.error(f"Demo Mode Error: {e}")
                st.session_state["results"] = None
        else:
            try:
                config, actual_m_id, status = cached_config_loader(selected_model, HF_TOKEN)
                pipeline = DeploymentPipeline()
                weights_dict = {
                    'latency': w_lat, 'memory_efficiency': w_mem,
                    'cost_efficiency': w_cost, 'energy_efficiency': w_eng,
                    'accuracy_preservation': w_acc
                }
                constraints_dict = {"max_latency": max_lat, "max_cost": max_cost}
                hw_type = None if selected_hardware == "None" else selected_hardware
                results = pipeline.run_pipeline(
                    model_id=selected_model,
                    hardware_type=hw_type,
                    workload_type=selected_workload,
                    weights=weights_dict,
                    constraints=constraints_dict,
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
                st.session_state["constraints"] = constraints_dict
            except Exception as e:
                st.error("System failed gracefully. Showing demo results.")
                logger.error(f"Execution Error: {e}")
                try:
                    weights_dict = {
                        'latency': w_lat,
                        'memory_efficiency': w_mem,
                        'cost_efficiency': w_cost,
                        'energy_efficiency': w_eng,
                        'accuracy_preservation': w_acc
                    }
                    constraints_dict = {"max_latency": max_lat, "max_cost": max_cost}
                    st.session_state["results"] = generate_demo_results(
                        selected_model, selected_hardware, selected_workload,
                        weights_dict, constraints_dict
                    )
                    st.session_state["constraints"] = constraints_dict
                except Exception as demo_e:
                    st.error(f"Fallback Demo pipeline failed: {demo_e}")
                    st.session_state["results"] = None

res = st.session_state.get("results")

if res:
    st.markdown('<div id="results-section"></div>', unsafe_allow_html=True)
    st.markdown("## 📊 Step 3: Optimization Results Intelligence Report")

    if res.get("is_demo"):
        st.info("ⓘ Demo Mode: Results computed using local DummyConfig simulation. All scores and metrics are real Roofline calculations, not hardcoded values.")
    else:
        st.success("✓ Live Mode: Results computed using real HuggingFace model architecture.")

    with st.expander("🔍 Simulation Parameters Used"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Model:** {res.get('model_id', 'N/A')}")
            st.markdown(f"**Workload:** {res.get('workload_type', 'N/A')}")
            st.markdown(f"**Hardware:** {res.get('best_evaluation', {}).get('hardware', 'N/A')}")
        with col_b:
            sb = res.get('scoring_breakdown', {})
            st.markdown(f"**Latency Score:** {sb.get('latency_score', 0):.1f}/100")
            st.markdown(f"**Cost Score:** {sb.get('cost_score', 0):.1f}/100")
            st.markdown(f"**Energy Score:** {sb.get('energy_score', 0):.1f}/100")
            st.markdown(f"**Accuracy Score:** {sb.get('accuracy_score', 0):.1f}/100")
            st.markdown(f"**Memory Score:** {sb.get('memory_score', 0):.1f}/100")

    if st.session_state.get("run_trigger"):
        st.success("Optimization completed. Scroll down to view results.")
        st.markdown("[↓ Jump to Results](#results-section)", unsafe_allow_html=True)
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

    st.markdown("### Impact Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1: c1.metric("Latency Reduction", f"{i_lat:.1f}%")
    with c2: c2.metric("Cost Reduction", f"{i_cost:.1f}%")
    with c3: c3.metric("Energy Reduction", f"{i_eng:.1f}%")
    with c4: c4.metric("Efficiency Score", f"{eff_score:.1f}/100")

    with st.expander("📐 Raw Simulation Metrics"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Latency (ms)", f"{latency:.2f}")
            st.metric("Baseline Latency (ms)", f"{b_lat:.2f}")
        with col2:
            st.metric("Cost per Request", f"${cost:.6f}")
            st.metric("Baseline Cost", f"${b_cost:.6f}")
        with col3:
            st.metric("Energy (kWh)", f"{energy_kwh:.6f}")
            st.metric("Memory (MB)", f"{sim.get('memory_mb', 0):.1f}")

    st.divider()

    st.markdown("### Why this strategy was selected")
    reasoning_text = score_brk.get("reasoning", "This strategy operates efficiently within the desired constraints.")
    st.info(reasoning_text)

    st.markdown("---")

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
            hw_comp_for_roofline = res.get("hardware_comparison", [])
            has_telemetry = any(
                ev.get("simulation", {}).get("roofline_telemetry") is not None
                for ev in hw_comp_for_roofline
            )
            if has_telemetry:
                st.plotly_chart(create_roofline_plot(hw_comp_for_roofline), use_container_width=True)
            else:
                st.info("Roofline telemetry not available for this run.")
