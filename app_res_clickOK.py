
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from model import calculate_forward

st.set_page_config(page_title="1D Resistivity Forward Modeling", layout="wide")

st.title("1D Resistivity Forward Modeling (Schlumberger Array)")
st.markdown("""
This app simulates the **Apparent Resistivity** curve for a 1D layered earth model.
**Click** on the plot to adjust the resistivity of the layer at that depth.
""")

# --- Sidebar Configuration ---
st.sidebar.header("Survey Parameters")
min_ab2 = st.sidebar.number_input("Min AB/2 (m)", value=1.0, min_value=0.1)
max_ab2 = st.sidebar.number_input("Max AB/2 (m)", value=1000.0, min_value=10.0)
num_points = st.sidebar.slider("Number of Points", 10, 100, 30)

# Generate AB/2 values (logarithmic spacing)
ab2_values = np.logspace(np.log10(min_ab2), np.log10(max_ab2), num_points)

# --- Main Layout ---
col1, col2 = st.columns([1, 2])

# Initialize model variables
model_thick = []
model_res = []

# Initialize keys for data editor reset
if "editor_key" not in st.session_state:
    st.session_state.editor_key = 0

with col1:
    st.subheader("Layer Model")
    st.markdown("Edit the layers below. **Top layer** is Row 0.")
    
    # Initial Data (6 Layers)
    default_data = {
        "Thickness (m)": [5.0, 5.0, 10.0, 20.0, 50.0, 0.0],  # 0.0 for last layer (Infinity)
        "Resistivity (Ohm-m)": [200.0, 100.0, 50.0, 20.0, 10.0, 500.0]
    }
    
    if "layer_df" not in st.session_state:
        st.session_state.layer_df = pd.DataFrame(default_data)

    # Use dynamic key to force reset when updated programmatically
    key_name = f"editor_{st.session_state.editor_key}"
    
    edited_df = st.data_editor(
        st.session_state.layer_df,
        num_rows="dynamic",
        key=key_name
    )
    
    st.info("Note: The last layer's thickness is ignored (Infinite Half-space).")
    
    # Update Session State from Editor (if user edited manually)
    # If the user edits the table, edited_df changes. We should sync this back to session_state
    # so that if we later click the plot, we start from the user's latest manual edits.
    # However, we must be careful not to create a loop.
    # Actually, st.data_editor initializes from the dataframe but returns the EDITED one.
    # We should use the returned `edited_df` as the source of truth for the *current run calculation*.
    # And we should update `st.session_state.layer_df` to match `edited_df` so that if we rerun without
    # programmatic update, it stays consistent.
    
    # IMPORTANT: Only update session state if we are NOT in the middle of a programmatic update flow.
    # But here we just want to persist user edits.
    if not edited_df.equals(st.session_state.layer_df):
        st.session_state.layer_df = edited_df

    # Data Processing
    valid_df = edited_df.dropna()
    res_list = valid_df["Resistivity (Ohm-m)"].astype(float).tolist()
    thick_list = valid_df["Thickness (m)"].astype(float).tolist()
    
    if len(res_list) > 0:
        model_thick = thick_list[:-1] # Drop last
        model_res = res_list

with col2:
    st.subheader("Combined Analysis")
    
    if len(model_res) > 0:
        with st.spinner("Calculating forward model..."):
            try:
                # --- 1. Forward Modeling ---
                rho_a = calculate_forward(model_thick, model_res, ab2_values)
                
                # --- 2. Prepare Model for Plotting (Step Plot) ---
                plot_depths = [0]
                current_d = 0
                for t in model_thick:
                    current_d += t
                    plot_depths.append(current_d)
                # Dummy deep point for visual
                max_plot_depth = plot_depths[-1] * 2 if plot_depths[-1] > 0 else 100
                plot_depths.append(max_plot_depth)
                
                step_depths = []
                step_res = []
                for i in range(len(model_res)):
                    step_res.append(model_res[i])
                    # Start of layer i
                    step_depths.append(plot_depths[i]) 
                    step_res.append(model_res[i])
                    # End of layer i (Start of i+1)
                    step_depths.append(plot_depths[i+1]) 
                
                # --- 3. Combined Plotting ---
                fig = go.Figure()
                
                # Ghost Grid for Interaction
                min_x_grid = min(min_ab2, plot_depths[1] if len(plot_depths)>1 else 1) * 0.5
                max_x_grid = max_ab2 * 1.5
                min_y_grid = min(min(model_res), min(rho_a)) * 0.1
                max_y_grid = max(max(model_res), max(rho_a)) * 10
                
                grid_x = np.logspace(np.log10(min_x_grid), np.log10(max_x_grid), 50)
                grid_y = np.logspace(np.log10(min_y_grid), np.log10(max_y_grid), 50)
                
                gx, gy = np.meshgrid(grid_x, grid_y)
                
                fig.add_trace(go.Scatter(
                    x=gx.flatten(),
                    y=gy.flatten(),
                    mode='markers',
                    marker=dict(size=10, opacity=0.001, color='rgba(0,0,0,0)'), # Invisible
                    name='Interaction Grid',
                    hoverinfo='none',
                    showlegend=False
                ))

                # Add Model Trace (Blue)
                fig.add_trace(go.Scatter(
                    x=step_depths, 
                    y=step_res,
                    mode='lines',
                    name='True Resistivity (Model)',
                    line=dict(color='blue', width=2),
                    hoverinfo='skip'
                ))

                # Add Sounding Curve Trace (Red)
                fig.add_trace(go.Scatter(
                    x=ab2_values, 
                    y=rho_a,
                    mode='lines+markers',
                    name='Apparent Resistivity (V.E.S)',
                    line=dict(color='red', width=2, dash='dot'),
                    hoverinfo='skip'
                ))
                
                fig.update_layout(
                    title="Model & Sounding Curve",
                    xaxis_title="Spacing (AB/2) / Depth (m)",
                    yaxis_title="Resistivity (Ohm-m)",
                    xaxis=dict(type="log", range=[np.log10(min_x_grid), np.log10(max_x_grid)]),
                    yaxis=dict(type="log", range=[np.log10(min_y_grid), np.log10(max_y_grid)]),
                    height=600,
                    showlegend=True,
                    template="plotly_white",
                    dragmode='pan'
                )
                
                # Use on_select to capture clicks
                selection = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points")
                
                # --- Interaction Logic ---
                if selection and selection["selection"]["points"]:
                    point = selection["selection"]["points"][0]
                    clicked_x = point["x"] # Depth / Spacing
                    clicked_y = point["y"] # Resistivity
                    
                    # Logic: Find which layer this falls into (by Depth)
                    target_layer_idx = -1
                    interfaces = plot_depths[:-1]
                    
                    for i in range(len(interfaces) - 1):
                        d_top = interfaces[i]
                        d_bottom = interfaces[i+1]
                        if d_top <= clicked_x < d_bottom:
                            target_layer_idx = i
                            break
                    
                    if target_layer_idx == -1:
                        if clicked_x >= interfaces[-1]:
                            target_layer_idx = len(interfaces) - 1
                    
                    if target_layer_idx != -1:
                        current_df = st.session_state.layer_df
                        
                        if 0 <= target_layer_idx < len(current_df):
                            # Update Resistivity
                            current_df.at[target_layer_idx, "Resistivity (Ohm-m)"] = clicked_y
                            st.session_state.layer_df = current_df
                            
                            # Increment key to force data_editor reset
                            st.session_state.editor_key += 1
                            
                            st.rerun()

            except Exception as e:
                st.error(f"Error calculating model: {e}")
                st.warning("Please ensure resistivities are positive and layers are defined correctly.")
    else:
        st.warning("Please add at least one layer.")

st.markdown("---")
st.markdown("Powered by **Streamlit**, **Plotly**, and **empymod**.")
