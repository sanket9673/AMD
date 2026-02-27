import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os

from core.pipeline_engine import DeploymentPipeline
from core.hardware_profiles import HARDWARE_DATABASE

# Streamlit config
st.set_page_config(
    page_title="AMD AI Deployment Intelligence",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for competition-grade UI
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    h1, h2, h3 {
        color: #fca311;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        width: 100%;
        background-color: #14213d;
        color: #ffffff;
        border: 2px solid #fca311;
        border-radius: 8px;
        padding: 10px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #fca311;
        color: #000000;
        border: 2px solid #ffffff;
    }
    .best-strategy-box {
        background-color: #1a1a2e;
        border-left: 5px solid #0f3460;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚀 AMD AI Deployment Intelligence Engine")
st.markdown("Optimize AI models for AMD hardware with real-time profiling, simulation, and strategic reasoning.")

# Ensure we have state for pipeline results
if "results" not in st.session_state:
    st.session_state.results = None
if "model_profile" not in st.session_state:
    st.session_state.model_profile = None

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    models = ["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "meta-llama/Llama-2-7b-chat-hf", "mistralai/Mistral-7B-v0.1"]
    selected_model = st.selectbox("Select Model", models)
    
    hardware_keys = list(HARDWARE_DATABASE.keys())
    selected_hardware = st.selectbox("Select Target Hardware", hardware_keys)
    
    st.markdown("---")
    
    if st.button("▶ Run Analysis"):
        with st.spinner("Running AI Deployment Intelligence Pipeline... Please wait."):
            try:
                pipeline = DeploymentPipeline()
                
                results = pipeline.run_pipeline(selected_model, selected_hardware)
                st.session_state.results = results
                
                # Load the model profile JSON
                model_safe_name = selected_model.replace("/", "_")
                profile_path = os.path.join("data", f"{model_safe_name}_profile.json")
                if os.path.exists(profile_path):
                    with open(profile_path, "r") as f:
                        st.session_state.model_profile = json.load(f)
                        
            except Exception as e:
                st.error(f"Error during analysis: {e}")

# Main body
if st.session_state.results:
    res = st.session_state.results
    best_eval = res.get("best_evaluation", {})
    results_path = res.get("results_path", "")
    
    # Load all ranked strategies for comparison
    ranked_strats = []
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            all_data = json.load(f)
            ranked_strats = all_data.get("ranked_strategies", [])
            
    # Section 1: Model Profile Metrics
    if st.session_state.model_profile:
        st.subheader("📊 Model Profile Summary")
        m_prof = st.session_state.model_profile
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Params (Billion)", f"{m_prof.get('total_parameters', 0) / 1e9:.2f} B")
        col2.metric("Size (FP32 MB)", f"{m_prof.get('estimated_memory_mb', 0):.2f}")
        col3.metric("Layers", m_prof.get("layer_count", "N/A"))
        col4.metric("Hidden Size", m_prof.get("hidden_size", "N/A"))
        
    st.markdown("---")
    
    # Section 2: Best Strategy Highlight
    st.subheader("🏆 Optimal Deployment Strategy")
    if best_eval:
        st.markdown("<div class='best-strategy-box'>", unsafe_allow_html=True)
        colA, colB, colC = st.columns(3)
        strat_info = best_eval.get("strategy", {})
        sim_info = best_eval.get("simulation", {})
        
        with colA:
            st.markdown(f"**Precision:** `{strat_info.get('precision', 'N/A')}`")
            st.markdown(f"**Prune Ratio:** `{strat_info.get('prune_ratio', 0.0)}`")
            st.markdown(f"**Mode:** `{strat_info.get('deployment_mode', 'N/A').replace('_', ' ').title()}`")
        with colB:
            st.markdown(f"**Latency:** `{sim_info.get('latency_estimate', 0):.4f} s`")
            st.markdown(f"**Throughput:** `{sim_info.get('throughput_estimate', 0):.2f} req/s`")
            st.markdown(f"**Memory:** `{sim_info.get('adjusted_memory', 0):.2f} MB`")
        with colC:
            st.markdown(f"**Score:** `{best_eval.get('score', 0):.2f}`")
            st.markdown(f"**Energy:** `{sim_info.get('energy_consumption', 0):.2f} W`")
        
        # LLM explanation
        st.markdown("#### 🤔 Reasoning Engine Explanation")
        if "reasoning" in best_eval:
            st.success(best_eval["reasoning"])
        else:
            st.warning("No reasoning explanation provided.")
            
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Section 3: Strategy Comparison Table
    if ranked_strats:
        st.subheader("📋 Strategy Comparison")
        
        # Build dataframe
        df_rows = []
        for s in ranked_strats:
            df_rows.append({
                "Rank": s["rank"],
                "Precision": s["strategy"]["precision"],
                "Prune Ratio": s["strategy"]["prune_ratio"],
                "Mode": s["strategy"]["deployment_mode"].replace("_", " ").title(),
                "Score": round(s["score"], 3),
                "Latency (s)": round(s["simulation"]["latency_estimate"], 4),
                "Throughput (req/s)": round(s["simulation"]["throughput_estimate"], 2),
                "Memory (MB)": round(s["simulation"]["adjusted_memory"], 2),
                "Energy (W)": round(s["simulation"]["energy_consumption"], 2),
                "Fit Score": round(s["simulation"]["hardware_fit_score"], 3)
            })
        
        df = pd.DataFrame(df_rows)
        # Display as sortable UI element
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("---")
        
        # Section 4: Professional Plotly Visuals
        st.subheader("📈 Performance Visualization")
        
        tab1, tab2, tab3 = st.tabs(["Performance Score vs Rank", "Latency vs Memory", "Energy Consumption"])
        
        with tab1:
            fig1 = px.bar(
                df, 
                x="Rank", 
                y="Score", 
                color="Mode", 
                title="Overall Strategy Score by Rank",
                template="plotly_dark",
                hover_data=["Precision", "Prune Ratio", "Score"]
            )
            fig1.update_layout(xaxis_title="Strategy Rank", yaxis_title="Calculated Score")
            st.plotly_chart(fig1, use_container_width=True)
            
        with tab2:
            fig2 = px.scatter(
                df, 
                x="Memory (MB)", 
                y="Latency (s)", 
                size="Score", 
                color="Precision",
                hover_name="Mode", 
                title="Latency vs Memory Footprint",
                template="plotly_dark"
            )
            st.plotly_chart(fig2, use_container_width=True)
            
        with tab3:
            fig3 = px.bar(
                df, 
                x="Rank", 
                y="Energy (W)", 
                color="Precision", 
                title="Estimated Energy Consumption per Strategy",
                template="plotly_dark"
            )
            fig3.update_layout(xaxis_title="Strategy Rank", yaxis_title="Energy (Watts)")
            st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    
    # Section 5: Download Results
    st.subheader("💾 Export Intelligence")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            json_str = f.read()
            st.download_button(
                label="📥 Download Full Results (JSON)",
                data=json_str,
                file_name=f"deployment_results_{st.session_state.results['model_id'].replace('/', '_')}.json",
                mime="application/json",
                use_container_width=True
            )
else:
    st.info("👈 Please select a model and hardware from the sidebar and click **Run Analysis** to generate insights.")
