import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import List, Dict, Any

def create_impact_chart(baseline: Dict[str, Any], optimized: Dict[str, Any]) -> go.Figure:
    """Phase 2: Baseline vs Optimized Impact"""
    b_sim = baseline['simulation']
    o_sim = optimized['simulation']
    
    lat_drop = max(0, ((b_sim['latency_ms'] - o_sim['latency_ms']) / b_sim['latency_ms']) * 100)
    cost_drop = max(0, ((b_sim['cost_usd'] - o_sim['cost_usd']) / b_sim['cost_usd']) * 100)
    eng_drop = max(0, ((b_sim['energy_kwh'] - o_sim['energy_kwh']) / b_sim['energy_kwh']) * 100)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Latency', 'Cost', 'Energy'],
        y=[lat_drop, cost_drop, eng_drop],
        marker_color=['#3b82f6', '#10b981', '#f59e0b'],
        text=[f"{v:.1f}%" for v in [lat_drop, cost_drop, eng_drop]],
        textposition='auto'
    ))
    fig.update_layout(
        title="Optimization Impact: Reduction (%) vs Baseline",
        yaxis_title="Reduction (%)",
        plot_bgcolor='rgba(0,0,0,0)', 
        paper_bgcolor='rgba(0,0,0,0)', 
        font=dict(color='#e2e8f0'),
        yaxis=dict(range=[0, max(100, max([lat_drop, cost_drop, eng_drop])*1.2)])
    )
    return fig

def create_hardware_comparison(evaluations: List[Dict[str, Any]]) -> go.Figure:
    """Phase 5: Hardware Comparison Chart"""
    # Find best MI250 and MI300X
    best_mi250 = next((e for e in evaluations if e['hardware'] == 'AMD_MI250'), None)
    best_mi300x = next((e for e in evaluations if e['hardware'] == 'AMD_MI300X'), None)
    
    data = []
    if best_mi250:
        data.append({
            "Hardware": "MI250",
            "Latency (ms)": best_mi250['simulation']['latency_ms'],
            "Cost (cents)": best_mi250['simulation']['cost_usd'] * 100,
            "Energy (mWh)": best_mi250['simulation']['energy_kwh'] * 1000
        })
    if best_mi300x:
        data.append({
            "Hardware": "MI300X",
            "Latency (ms)": best_mi300x['simulation']['latency_ms'],
            "Cost (cents)": best_mi300x['simulation']['cost_usd'] * 100,
            "Energy (mWh)": best_mi300x['simulation']['energy_kwh'] * 1000
        })
        
    df = pd.DataFrame(data)
    if df.empty:
        return go.Figure()
        
    df_melt = df.melt(id_vars='Hardware', var_name='Metric', value_name='Value')
    
    fig = px.bar(df_melt, x='Metric', y='Value', color='Hardware', barmode='group',
                 title="Hardware Advantage: MI300X vs MI250", log_y=True)
                 
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e2e8f0'))
    return fig

def create_radar_chart(hardware_comp: List[Dict[str, Any]]) -> go.Figure:
    """Phase 6: Radar Chart Strong Normalized"""
    fig_rad = go.Figure()
    top_3 = hardware_comp[:3]
    categories = ['Latency Efficiency', 'Cost Efficiency', 'Energy Efficiency', 'Throughput', 'Memory Efficiency']
    
    max_tp = max([c['simulation'].get('throughput', 1) for c in hardware_comp]) or 1
    max_mem = max([c['simulation'].get('memory_mb', 1) for c in hardware_comp]) or 1
    
    for i, c in enumerate(top_3):
        lat = c['simulation'].get('latency_ms', 500)
        cost = c['simulation'].get('cost_usd', 0.001)
        eng = c['simulation'].get('energy_kwh', 0.01)
        tp = c['simulation'].get('throughput', 1)
        mem = c['simulation'].get('memory_mb', max_mem)
        
        # Normalize to 0-1
        lat_norm = max(0, 1 - (lat / 500))
        cost_norm = max(0, 1 - (cost / 0.001))
        eng_norm = max(0, 1 - (eng / 0.01))
        tp_norm = tp / max_tp
        mem_norm = max(0, 1 - (mem / max_mem))
        
        prec = c['strategy'].get('precision', 'Unknown')
        
        fig_rad.add_trace(go.Scatterpolar(
            r=[lat_norm*100, cost_norm*100, eng_norm*100, tp_norm*100, mem_norm*100],
            theta=categories,
            fill='toself',
            opacity=0.6,
            name=f"Rank {i+1} ({c['hardware']} - {prec})"
        ))
    
    fig_rad.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100]), bgcolor='#0b0f19'),
        showlegend=True, paper_bgcolor='#0b0f19', font=dict(color='#e2e8f0'), margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig_rad

def create_pareto_frontier(evaluations: List[Dict[str, Any]]) -> go.Figure:
    """Phase 4: Pareto Frontier"""
    data = []
    for ev in evaluations:
        strat_str = f"{ev['hardware']} - {ev['strategy'].get('precision', '')} p={ev['strategy'].get('prune_ratio', 0)} ({ev['strategy'].get('deployment_mode', '')})"
        data.append({
            "Latency (ms)": ev['simulation'].get('latency_ms', 0),
            "Cost ($)": ev['simulation'].get('cost_usd', 0),
            "Memory (MB)": ev['simulation'].get('memory_mb', 0),
            "Strategy": strat_str,
            "Score": ev.get('score', 0)
        })
    df = pd.DataFrame(data)
    if df.empty:
        return go.Figure()
        
    best_strategy = df.loc[df['Score'].idxmax()]
        
    fig = px.scatter(df, x="Latency (ms)", y="Cost ($)", color="Score", size="Memory (MB)", hover_name="Strategy", 
                     title="Strategy Tradeoff (Latency vs Cost vs Memory)", color_continuous_scale="Viridis")
                     
    # Highlight best
    fig.add_trace(go.Scatter(
        x=[best_strategy["Latency (ms)"]],
        y=[best_strategy["Cost ($)"]],
        mode='markers',
        marker=dict(size=18, symbol='star', color='red', line=dict(width=2, color='white')),
        name="Selected Strategy",
        hoverinfo='text',
        hovertext=f"Selected: {best_strategy['Strategy']}"
    ))
    
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e2e8f0'))
    return fig

def create_strategy_leaderboard(evaluations: List[Dict[str, Any]]) -> go.Figure:
    """Phase 7: Strategy Leaderboard"""
    sorted_evals = sorted(evaluations, key=lambda x: x.get('score', 0), reverse=True)[:10]
    
    labels = []
    scores = []
    colors = []
    for i, ev in enumerate(sorted_evals):
        labels.append(f"#{i+1} {ev['hardware']} {ev['strategy']['precision']}")
        scores.append(ev.get('score', 0))
        colors.append('#ef4444' if i == 0 else ('#f59e0b' if i < 3 else '#3b82f6'))
        
    fig = go.Figure(data=[go.Bar(
        x=labels, y=scores,
        marker_color=colors,
        text=[f"{s:.1f}" for s in scores],
        textposition='auto'
    )])
    
    fig.update_layout(
        title="Top 10 Strategy Leaderboard",
        xaxis_title="Strategy",
        yaxis_title="Efficiency Score",
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e2e8f0')
    )
    return fig

def create_strategy_heatmap(evaluations: List[Dict[str, Any]]) -> go.Figure:
    data = []
    for ev in evaluations:
        data.append({
            "Precision": ev['strategy'].get('precision', 'N/A'),
            "Prune Ratio": ev['strategy'].get('prune_ratio', 0.0),
            "Score": ev.get('score', 0)
        })
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.groupby(['Precision', 'Prune Ratio'])['Score'].max().reset_index()
        pivot_df = df.pivot(index='Precision', columns='Prune Ratio', values='Score')
        fig = px.imshow(pivot_df, text_auto=True, color_continuous_scale='Blues', title="Strategy Heatmap (Score)")
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e2e8f0'))
        return fig
    return go.Figure()

def create_roofline_plot(evaluations: List[Dict[str, Any]]) -> go.Figure:
    if not evaluations:
        return go.Figure()
        
    best_eval = evaluations[0]
    telem = best_eval['simulation'].get('roofline_telemetry')
    if not telem:
        return go.Figure()
        
    bandwidth = telem['bandwidth']
    compute = telem['compute']
    ridge = telem['ridge']
    
    fig = go.Figure()

    x_mem = np.linspace(max(ridge / 100, 1e-3), ridge, 100)
    y_mem = bandwidth * x_mem
    fig.add_trace(go.Scatter(
        x=x_mem, y=y_mem,
        mode='lines',
        name="Memory-bound region",
        line=dict(color='#f59e0b', width=3)
    ))
    
    x_comp = np.linspace(ridge, ridge * 100, 100)
    y_comp = [compute] * 100
    fig.add_trace(go.Scatter(
        x=x_comp, y=y_comp,
        mode='lines',
        name="Compute-bound region",
        line=dict(color='#3b82f6', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=[telem['prefill_ai'], telem['decode_ai']],
        y=[telem['prefill_perf'], telem['decode_perf']],
        mode='markers+text',
        text=["Prefill", "Decode"],
        textposition="top center",
        marker=dict(size=18, symbol='star', color='red', line=dict(width=2, color='white')),
        name="Selected Strategy",
        hoverinfo='text',
        hovertext=[f"Prefill AI: {telem['prefill_ai']:.2f}", f"Decode AI: {telem['decode_ai']:.2f}"]
    ))
    
    other_x, other_y = [], []
    for ev in evaluations[1:20]:
        if ev['hardware'] != best_eval['hardware']: continue
        t = ev['simulation'].get('roofline_telemetry')
        if not t: continue
        jx = np.random.uniform(-0.05, 0.05) * t['prefill_ai']
        jy = np.random.uniform(-0.05, 0.05) * t['prefill_perf']
        other_x.extend([t['prefill_ai']+jx, t['decode_ai']*(1+np.random.uniform(-0.05, 0.05))])
        other_y.extend([t['prefill_perf']+jy, t['decode_perf']*(1+np.random.uniform(-0.05, 0.05))])
        
    if other_x:
        fig.add_trace(go.Scatter(
            x=other_x, y=other_y,
            mode='markers',
            marker=dict(size=6, color='gray', opacity=0.5),
            name="Other Strategies",
            hoverinfo='none'
        ))

    fig.update_layout(
        title="Roofline Model: Performance vs Arithmetic Intensity",
        xaxis_title="Arithmetic Intensity (FLOPs / Byte)",
        yaxis_title="Achieved Performance (FLOPs / Sec)",
        xaxis_type="log",
        yaxis_type="log",
        plot_bgcolor='#0b0f19', 
        paper_bgcolor='#0b0f19', 
        font=dict(color='#e2e8f0'),
        showlegend=True,
        legend=dict(x=0.01, y=0.99)
    )

    return fig
