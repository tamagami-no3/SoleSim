import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import numpy as np

# --- GLOBAL PLOT STYLING FOR DARK THEME ---
plt.rcParams.update({
    "figure.facecolor": "#0E1117",
    "axes.facecolor": "#0E1117",
    "axes.edgecolor": "#303030",
    "axes.labelcolor": "white",
    "text.color": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "#303030",
    "figure.autolayout": True
})

def get_custom_cmap():
    # Blue -> Cyan -> Yellow -> Red
    return LinearSegmentedColormap.from_list("p", [(0,0,0.5), (0,1,1), (1,1,0), (1,0,0)])

def plot_static_heatmap(pressure_map, peak_val):
    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(pressure_map, cmap=get_custom_cmap(), aspect='auto')
    cbar = plt.colorbar(im, label="Pressure (N/cm²)")
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.outline.set_edgecolor('#303030')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    ax.set_title(f"Peak Pressure: {peak_val:.2f} N/cm²", color="#00FFFF", fontweight='bold')
    ax.axis('off')
    return fig

def plot_dynamic_heatmap(pressure_map, max_scale):
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.imshow(pressure_map, cmap=get_custom_cmap(), vmin=0, vmax=max_scale, aspect='auto')
    ax.set_title(f"Dynamic Pressure Distribution", color="#00FFFF")
    ax.axis('off')
    return fig

def plot_live_chart(peak_history, phase, max_limit):
    fig, ax = plt.subplots(figsize=(6, 2))
    
    # --- ERROR FIX: REMOVED 'shadow=True' argument ---
    ax.plot(np.linspace(0, phase*100, len(peak_history)), peak_history, color='#00FFFF', linewidth=2)
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, max_limit)
    ax.set_xlabel("Gait Cycle (%)")
    ax.set_ylabel("Peak N/cm²")
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Fill under line for "Cyber" look
    ax.fill_between(np.linspace(0, phase*100, len(peak_history)), peak_history, color='#00FFFF', alpha=0.1)
    
    return fig

def create_3d_topology(pressure_map):
    fig_3d = go.Figure(data=[go.Surface(z=pressure_map, colorscale='Viridis')])
    fig_3d.update_layout(
        title='',
        scene=dict(
            xaxis=dict(title='Width', color='white', gridcolor='#444'),
            yaxis=dict(title='Length', color='white', gridcolor='#444'),
            zaxis=dict(title='Pressure', color='white', gridcolor='#444'),
            bgcolor='#0E1117'
        ),
        paper_bgcolor='#0E1117',
        margin=dict(l=0, r=0, b=0, t=0), 
        height=300
    )
    return fig_3d