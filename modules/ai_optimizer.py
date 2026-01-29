import numpy as np
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

def run_inverse_design(sim, weight_kg, target_score):
    progress_text = f"Training AI for {weight_kg}kg body weight..."
    my_bar = st.progress(0, text=progress_text)
    
    X_train, y_train = [], []
    for i in range(50):
        r_heel = int(np.random.uniform(5, 50))
        r_fore = int(np.random.uniform(5, 50))
        r_arch = round(np.random.uniform(0.5, 5.0), 1)
        r_mod = int(np.random.uniform(10, 100))
        r_weight = int(np.random.uniform(40, 120))
        
        sim.update_design(r_heel, r_fore, r_arch, r_mod, "None")
        p_map = sim.solve_static(r_weight)
        score = max(100 - (np.max(p_map) * 2.5), 0)
        
        X_train.append([score, r_weight])
        y_train.append([r_heel, r_fore, r_arch, r_mod])
        my_bar.progress((i+1)/50)
        
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train, y_train)
    
    pred = rf.predict([[target_score, weight_kg]])[0]
    best_design = [
        int(np.clip(pred[0], 5, 50)),
        int(np.clip(pred[1], 5, 50)),
        round(np.clip(pred[2], 0.5, 5.0), 1),
        int(np.clip(pred[3], 10, 100))
    ]
    
    sim.update_design(*best_design, "None")
    p_map_v = sim.solve_static(weight_kg)
    real_score = max(100 - (np.max(p_map_v) * 2.5), 0)
    
    my_bar.empty()
    return best_design, real_score