import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import List, Dict, Any

def create_radar_chart(hardware_comp: List[Dict[str, Any]]) -> go.Figure:
    fig_rad = go.Figure()
    top_3 = hardware_comp[:3]
    categories = ['Latency Efficiency', 'Cost Efficiency', 'Energy Efficiency', 'Accuracy', 'Memory Fit']
    
    for i, c in enumerate(top_3):
        lat_scr = c.get('scoring_breakdown', {}).get('latency_score', 0)
        cost_scr = c.get('scoring_breakdown', {}).get('cost_score', 0)
        eng_scr = c.get('scoring_breakdown', {}).get('energy_score', 0)
        acc_scr = c.get('scoring_breakdown', {}).get('accuracy_score', 0)
        mem_scr = c.get('scoring_breakdown', {}).get('memory_score', 0)
        
        prec = c['strategy'].get('precision', 'Unknown')
        
        fig_rad.add_trace(go.Scatterpolar(
            r=[lat_scr, cost_scr, eng_scr, acc_scr, mem_scr],
            theta=categories,
            fill='toself',
            name=f"Rank {i+1} ({c['hardware']} - {prec})"
        ))
    
    fig_rad.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100]), bgcolor='#0b0f19'),
        showlegend=True, paper_bgcolor='#0b0f19', font=dict(color='#e2e8f0'), margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig_rad

def create_pareto_frontier(evaluations: List[Dict[str, Any]]) -> go.Figure:
    data = []
    for ev in evaluations:
        data.append({
            "Latency (ms)": ev['simulation'].get('latency_ms', 0),
            "Cost ($)": ev['simulation'].get('cost_usd', 0),
            "Strategy": f"{ev['hardware']} - {ev['strategy'].get('precision', '')}",
            "Score": ev.get('score', 0)
        })
    df = pd.DataFrame(data)
    if df.empty:
        return go.Figure()
    fig = px.scatter(df, x="Latency (ms)", y="Cost ($)", color="Score", hover_name="Strategy", 
                     title="Pareto Frontier (Latency vs Cost)", color_continuous_scale="Viridis")
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e2e8f0'))
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

    # Memory-bound line: y = bandwidth * x
    x_mem = np.linspace(max(ridge / 100, 1e-3), ridge, 100)
    y_mem = bandwidth * x_mem
    fig.add_trace(go.Scatter(
        x=x_mem, y=y_mem,
        mode='lines',
        name="Memory-bound region",
        line=dict(color='#f59e0b', width=3)
    ))
    
    # Compute-bound line: y = compute
    x_comp = np.linspace(ridge, ridge * 100, 100)
    y_comp = [compute] * 100
    fig.add_trace(go.Scatter(
        x=x_comp, y=y_comp,
        mode='lines',
        name="Compute-bound region",
        line=dict(color='#3b82f6', width=3)
    ))
    
    # Selected Strategy (Best)
    fig.add_trace(go.Scatter(
        x=[telem['prefill_ai'], telem['decode_ai']],
        y=[telem['prefill_perf'], telem['decode_perf']],
        mode='markers',
        marker=dict(size=14, symbol='star', color='red', line=dict(width=2, color='white')),
        name="Selected Strategy",
        hoverinfo='text',
        hovertext=[f"Prefill<br>AI: {telem['prefill_ai']:.2f}<br>Perf: {telem['prefill_perf']:.2e}",
                   f"Decode<br>AI: {telem['decode_ai']:.2f}<br>Perf: {telem['decode_perf']:.2e}"]
    ))
    
    # Other strategies
    other_x = []
    other_y = []
    for ev in evaluations[1:20]:
        if ev['hardware'] != best_eval['hardware']: continue
        t = ev['simulation'].get('roofline_telemetry')
        if not t: continue
        other_x.extend([t['prefill_ai'], t['decode_ai']])
        other_y.extend([t['prefill_perf'], t['decode_perf']])
        
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
