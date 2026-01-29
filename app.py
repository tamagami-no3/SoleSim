import streamlit as st
import numpy as np
import time
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# --- 1. ROBUST IMPORT SYSTEM ---
try:
    from modules.physics import SoleSimulation
    from modules.visualization import (
        plot_static_heatmap, plot_dynamic_heatmap, plot_live_chart, create_3d_topology
    )
    from modules.ai_optimizer import run_inverse_design
except ImportError as e:
    st.error(f"⚠️ SYSTEM BOOT FAILURE: Missing Modules.\n\nError Details: {e}")
    st.stop()
except Exception as e:
    st.error(f"⚠️ CRITICAL ERROR: {e}")
    st.stop()

# --- 2. PAGE CONFIG & HACKER CSS ---
st.set_page_config(page_title="SoleSim: Advanced Prototyping", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #E0E0E0; font-family: 'Segoe UI', sans-serif; }
    h1, h2, h3, .stMetricLabel, .metric-value, .stButton button { font-family: 'Courier New', monospace !important; letter-spacing: -0.5px; }
    
    @keyframes pulse-glow {
        0% { text-shadow: 0 0 10px rgba(0, 255, 255, 0.5); }
        50% { text-shadow: 0 0 30px rgba(0, 255, 255, 0.9); }
        100% { text-shadow: 0 0 10px rgba(0, 255, 255, 0.5); }
    }
    .hacker-title {
        font-family: 'Courier New', monospace; 
        font-size: 5rem; 
        font-weight: 900; 
        text-align: center;
        background: linear-gradient(90deg, #00FFFF 0%, #0080FF 50%, #8A2BE2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        animation: pulse-glow 3s infinite ease-in-out;
        margin-bottom: 0px;
    }
    .hacker-subtitle {
        text-align: center; color: #888; font-family: monospace; font-size: 1.2rem; margin-bottom: 50px; letter-spacing: 1px;
    }
    .feature-card { background: rgba(255, 255, 255, 0.05); border: 1px solid #333; padding: 20px; border-radius: 10px; text-align: center; transition: 0.3s; }
    .feature-card:hover { transform: translateY(-5px); border-color: #00FFFF; }
    .feature-icon { font-size: 3rem; margin-bottom: 10px; }
    .feature-title { color: #00FFFF; font-weight: bold; }
    
    div.stButton > button {
        width: 100%; background: rgba(0, 255, 255, 0.05); border: 2px solid #00FFFF; color: #00FFFF;
        padding: 20px 40px; font-size: 1.5rem; font-weight: bold; border-radius: 0px; transition: 0.3s;
        text-transform: uppercase; letter-spacing: 2px;
    }
    div.stButton > button:hover { background: #00FFFF; color: #000; box-shadow: 0 0 30px #00FFFF; transform: scale(1.02); }
    [data-testid="stSidebar"] { background-color: #0b0d10; border-right: 1px solid #303030; }
    .stAlert { background-color: #220000; border: 1px solid #FF0000; color: #FFAAAA; }
</style>
""", unsafe_allow_html=True)

if 'page' not in st.session_state: st.session_state.page = 'landing'
def go_to_dashboard(): st.session_state.page = 'dashboard'

# --- LANDING PAGE ---
def show_landing_page():
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="hacker-title">SOLESIM // PROTO_V2</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="hacker-subtitle">
            MISSION: SIMULATED COMFORT & WEAR PREDICTION FOR FOOTWEAR SOLE DESIGNS<br>
            <span style='font-size: 0.9rem; color: #555;'>SYSTEM STATUS: <span style='color:#00FF00'>ONLINE</span></span>
        </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown('<div class="feature-card"><div class="feature-icon">🧬</div><div class="feature-title">WINKLER PHYSICS</div><div style="color:#aaa">Spring-damper grid arrays.</div></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="feature-card"><div class="feature-icon">🧠</div><div class="feature-title">NEURAL DESIGN</div><div style="color:#aaa">Inverse-design AI algorithms.</div></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="feature-card"><div class="feature-icon">📉</div><div class="feature-title">WEAR PREDICTION</div><div style="color:#aaa">Archard\'s Law forecasting.</div></div>', unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    col_centered = st.columns([1, 2, 1])
    with col_centered[1]: 
        st.button("INITIALIZE_SYSTEM [ >> ]", on_click=go_to_dashboard)

# --- DASHBOARD ---
def show_dashboard():
    MATERIALS = {
        "EVA Foam (Budget)":   {"mod": 25, "rho": 0.25, "wear": 0.020, "cost": 0.40},
        "PU Foam (Standard)":  {"mod": 55, "rho": 0.45, "wear": 0.010, "cost": 0.90},
        "Rubber (Heavy Duty)": {"mod": 90, "rho": 1.10, "wear": 0.005, "cost": 0.70},
        "TPU (Premium)":       {"mod": 70, "rho": 1.20, "wear": 0.008, "cost": 2.10},
        "Custom / AI Mode":    {"mod": 30, "rho": 0.30, "wear": 0.015, "cost": 0.50} 
    }

    with st.sidebar:
        st.markdown("### ⚙️ CONTROLS")
        st.info("System Parameters")
        mat_choice = st.selectbox("Material", list(MATERIALS.keys()))
        mat_props = MATERIALS[mat_choice]
        
        if mat_choice == "Custom / AI Mode":
            p_modulus = st.slider("Stiffness", 10, 100, 30)
            p_density, p_wear_factor = 0.3, 0.015
        else:
            p_modulus, p_density, p_wear_factor = mat_props['mod'], mat_props['rho'], mat_props['wear']
        
        st.divider()
        p_heel = st.slider("Heel Stack (mm)", 5, 50, 25)
        p_fore = st.slider("Forefoot Stack (mm)", 5, 50, 15)
        p_arch = st.slider("Arch Factor", 0.5, 5.0, 1.5)
        p_groove = st.selectbox("Tread Pattern", ["None", "Horizontal Sipes", "Grid Pattern", "Honeycomb"])
        
        st.divider()
        p_gait = st.selectbox("Gait Profile", ["Neutral", "Overpronator (Flat Foot)", "Supinator (High Arch)"])
        p_weight = st.slider("Weight (kg)", 40, 120, 75)
        
        st.divider()
        run_walk = st.button("▶ EXECUTE_WALK_CYCLE", type="primary")

        try:
            sim_temp = SoleSimulation()
            sim_temp.update_design(p_heel, p_fore, p_arch, p_modulus, p_groove, p_wear_factor)
            vol = np.sum(sim_temp.thickness_map / 10) * (sim_temp.dx ** 2)
            mass = vol * p_density
            cost = vol * mat_props.get("cost", 0.5)
            st.divider()
            st.caption(f"MASS: {mass:.1f}g | COST: ₹{cost:.2f}")
        except Exception as e:
            st.error(f"Spec Calc Error: {e}")

    st.markdown('<div class="hacker-title" style="font-size: 2.5rem;">SOLESIM // DASHBOARD</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    try:
        sim = SoleSimulation()
        sim.update_gait(p_gait)
        sim.update_design(p_heel, p_fore, p_arch, p_modulus, p_groove, p_wear_factor)
        pressure = sim.solve_static(p_weight)
        peak_p = np.max(pressure)
    except Exception as e:
        st.error(f"PHYSICS ENGINE FAILURE: {e}")
        st.stop()

    with col1:
        st.subheader("1. STATIC LOAD ANALYSIS")
        if peak_p > 250:
             st.warning("⚠️ WARNING: CRITICAL PRESSURE. Material Yield Limit Reached.")
        
        st.pyplot(plot_static_heatmap(pressure, peak_p))
        st.metric("COMFORT INDEX", f"{max(100 - (peak_p * 2.5), 0):.0f}/100")
        st.markdown("### 3D TOPOLOGY")
        st.plotly_chart(create_3d_topology(pressure), use_container_width=True)

    with col2:
        st.subheader("2. DYNAMIC GAIT SURROGATE")
        st.markdown("""<div style="background:#000; border:1px solid #333; padding:10px; border-radius:5px; text-align:center;"><span style="color:#00FFFF; font-family:monospace;">LIVE SENSOR DATA FEED</span></div>""", unsafe_allow_html=True)
        
        # --- FIXED: ADDED TWO PLACEHOLDERS (Heatmap AND Line Graph) ---
        chart_spot = st.empty()
        graph_spot = st.empty()
        
        if run_walk:
            try:
                sim.wear_map[:] = 0
                peak_hist = []
                phases = np.linspace(0, 1, 25)
                bar = st.progress(0)
                for i, phase in enumerate(phases):
                    w_p, load = sim.solve_walking_step(p_weight, phase)
                    peak_hist.append(np.max(w_p))
                    
                    # UPDATE BOTH GRAPHS
                    chart_spot.pyplot(plot_dynamic_heatmap(w_p, peak_p*1.5))
                    graph_spot.pyplot(plot_live_chart(peak_hist, phase, max(peak_p*1.5, max(peak_hist) + 1)))
                    
                    bar.progress((i+1)/25)
                    time.sleep(0.05)
                bar.empty()
                
                st.success("SEQUENCE COMPLETE")
                with st.expander("📉 DURABILITY REPORT", expanded=True):
                    fig_w, ax_w = plt.subplots(figsize=(6, 2))
                    ax_w.imshow(sim.wear_map, cmap='inferno', aspect='auto')
                    ax_w.axis('off')
                    ax_w.set_title("PREDICTED WEAR ZONES", color="white")
                    st.pyplot(fig_w)
            except Exception as e:
                st.error(f"SIMULATION ABORTED: {e}")
        else:
            st.info("AWAITING INPUT: Press 'EXECUTE' to run dynamics.")

    st.markdown("---")
    with st.expander("🤖 AI_INVERSE_DESIGN_MODULE"):
        c_ai1, c_ai2, c_ai3 = st.columns([1, 1, 1])
        with c_ai1: target_score = st.number_input("TARGET SCORE", 0, 100, 90)
        with c_ai2: target_weight = st.number_input("USER WEIGHT (kg)", 40, 150, 75)
        with c_ai3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("GENERATE OPTIMIZATION"):
                try:
                    with st.spinner("NEURAL NETWORK COMPUTING..."):
                        best, pred = run_inverse_design(sim, target_weight, target_score)
                    st.code(f"OPTIMIZATION RESULT\nHEEL: {best[0]}mm | FORE: {best[1]}mm | ARCH: {best[2]}x | MOD: {best[3]} | SCORE: {pred:.1f}")
                    st.success("AI SOLUTION FOUND")
                except Exception as e:
                    st.error(f"AI MODULE FAILURE: {e}")

if st.session_state.page == 'landing': show_landing_page()
elif st.session_state.page == 'dashboard': show_dashboard()