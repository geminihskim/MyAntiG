
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from model import calculate_forward

st.set_page_config(page_title="1D Resistivity Forward Modeling", layout="wide")

st.title("1D Resistivity Forward Modeling (Schlumberger Array)")
st.markdown("""
This app simulates the **Apparent Resistivity** curve for a 1D layered earth model.
Use the **Edit Mode** toggle below to choose which parameter to adjust by clicking on the plot.
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

# --- Helper Functions for Data Sync ---
def recalculate_depths(df):
    """Update 'Bottom Depth (m)' based on 'Thickness (m)'."""
    depths = []
    current_d = 0.0
    for t in df["Thickness (m)"]:
        current_d += t
        depths.append(current_d)
    # Last layer depth is infinite/irrelevant, but let's just keep cumulative or set to None?
    # Numeric column needs numbers. Let's keep cumulative but ignore it in logic.
    df["Bottom Depth (m)"] = depths
    return df

def recalculate_thicknesses(df, old_df):
    """Update 'Thickness (m)' based on changed 'Bottom Depth (m)'."""
    # Find which depth changed
    # Logic: t[i] = d[i] - d[i-1].
    # But this resets ALL thicknesses based on depths.
    # This is safer than finding diffs.
    
    # Validation: Depths must be strictly increasing.
    depths = df["Bottom Depth (m)"].tolist()
    
    # Check monotonicity (excluding last layer if we ignore it)
    # Actually, simplistic approach:
    # t[0] = d[0]
    # t[i] = d[i] - d[i-1]
    
    new_thicknesses = []
    last_d = 0.0
    
    # We only care about up to N-1 layers for thickness. 
    # Last layer thickness is 0 (Inf) in our model logic.
    # But in the table it might be 0.
    
    valid_update = True
    
    for i in range(len(depths)):
        d = depths[i]
        if i < len(depths) - 1: # Normal layers
            if d <= last_d:
                valid_update = False # Depth must increase
                break
            th = d - last_d
            new_thicknesses.append(th)
            last_d = d
        else:
             # Last layer
             new_thicknesses.append(0.0)

    if valid_update:
        df["Thickness (m)"] = new_thicknesses
        return df
    else:
        st.toast("Invalid Depth! Depth must be greater than previous layer.", icon="⚠️")
        return old_df

with col1:
    st.subheader("Layer Model")
    st.markdown("Edit **Thickness**, **Resistivity**, or **Bottom Depth**.")
    
    # Initial Data (6 Layers)
    if "layer_df" not in st.session_state:
        default_data = {
            "Thickness (m)": [5.0, 5.0, 10.0, 20.0, 50.0, 0.0],  # 0.0 for last layer (Infinity)
            "Resistivity (Ohm-m)": [200.0, 100.0, 50.0, 20.0, 10.0, 500.0]
        }
        df = pd.DataFrame(default_data)
        st.session_state.layer_df = recalculate_depths(df)

    # Use dynamic key
    key_name = f"editor_{st.session_state.editor_key}"
    
    # Order columns
    column_order = ["Thickness (m)", "Bottom Depth (m)", "Resistivity (Ohm-m)"]
    
    # Display Editor
    edited_df = st.data_editor(
        st.session_state.layer_df,
        column_order=column_order,
        num_rows="dynamic",
        key=key_name,
        disabled=["Bottom Depth (m)"] if False else [] # Enable editing
    )
    
    # --- Synchronization Logic ---
    # We need to determine if User edited Thickness or Depth.
    # We compare edited_df with st.session_state.layer_df
    
    previous_df = st.session_state.layer_df
    
    if not edited_df.equals(previous_df):
        # Something changed. Check what.
        
        # Check Thickness first
        thick_changed = not edited_df["Thickness (m)"].equals(previous_df["Thickness (m)"])
        depth_changed = not edited_df["Bottom Depth (m)"].equals(previous_df["Bottom Depth (m)"])
        
        final_df = edited_df.copy()
        
        if thick_changed and not depth_changed:
            # Thickness edited -> Update Depths
            final_df = recalculate_depths(final_df)
            
        elif depth_changed and not thick_changed:
            # Depth edited -> Update Thicknesses
            final_df = recalculate_thicknesses(final_df, previous_df)
            
            # Recalculate depths again just to ensure precision consistency?
            # Or if invalid update returned old_df, we revert.
            if final_df.equals(previous_df): 
                # Meaning invalid depth was rejected
                st.session_state.editor_key += 1 # Force reset UI
                st.rerun()
            
        elif thick_changed and depth_changed:
             # Both changed? Rare. Prioritize Thickness?
             final_df = recalculate_depths(final_df)
        
        # Update Session State
        st.session_state.layer_df = final_df
        
        # If we modified the dataframe (computed columns), we might need to rerun to show updated values in editor immediately?
        # Streamlit data_editor might not show the computed update until next rerun if we don't force it.
        # But setting session_state.layer_df updates the source.
        # If we want the editor to REFLECT the calculated values immediately, we need a rerun.
        if not final_df.equals(edited_df):
             st.session_state.editor_key += 1
             st.rerun()

    st.info("Note: Last layer is Infinite (Thickness 0 used as placeholder).")
    
    # Data Processing for Model
    valid_df = st.session_state.layer_df.dropna() # Use verified session state
    res_list = valid_df["Resistivity (Ohm-m)"].astype(float).tolist()
    thick_list = valid_df["Thickness (m)"].astype(float).tolist()
    
    if len(res_list) > 0:
        model_thick = thick_list[:-1] # Drop last
        model_res = res_list

with col2:
    st.subheader("Combined Analysis")
    
    st.info("💡 **Interaction Tip**: Select a mode below, then click on the plot.")
    edit_mode = st.radio(
        "**Edit Mode**",
        ["Modify Resistivity", "Modify Layer Depth"],
        horizontal=True,
        captions=["Adjust the Y-value (Resistivity) of a layer.", "Move the nearest layer boundary (X-axis)."]
    )
    
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
                
                movable_interfaces = plot_depths[1:] 
                
                # Dummy deep point for visual
                max_plot_depth = plot_depths[-1] * 2 if plot_depths[-1] > 0 else 100
                plot_depths_vis = plot_depths + [max_plot_depth]
                
                step_depths = []
                step_res = []
                for i in range(len(model_res)):
                    step_res.append(model_res[i])
                    step_depths.append(plot_depths_vis[i]) 
                    step_res.append(model_res[i])
                    step_depths.append(plot_depths_vis[i+1]) 
                
                # --- 3. Combined Plotting ---
                fig = go.Figure()
                
                # Ghost Grid
                min_x_grid = min(min_ab2, plot_depths[1] if len(plot_depths)>1 else 1) * 0.5
                max_x_grid = max_ab2 * 1.5
                min_y_grid = min(min(model_res), min(rho_a)) * 0.1
                max_y_grid = max(max(model_res), max(rho_a)) * 10
                
                grid_x = np.logspace(np.log10(min_x_grid), np.log10(max_x_grid), 100)
                grid_y = np.logspace(np.log10(min_y_grid), np.log10(max_y_grid), 100)
                gx, gy = np.meshgrid(grid_x, grid_y)
                
                fig.add_trace(go.Scatter(
                    x=gx.flatten(), y=gy.flatten(),
                    mode='markers',
                    marker=dict(size=8, opacity=0.001, color='rgba(0,0,0,0)'), 
                    name='Interaction Grid', hoverinfo='none', showlegend=False
                ))

                # Model Trace
                fig.add_trace(go.Scatter(
                    x=step_depths, y=step_res,
                    mode='lines',
                    name='True Resistivity (Model)',
                    line=dict(color='blue', width=2),
                    hoverinfo='skip'
                ))
                
                # Interfaces
                if "Depth" in edit_mode:
                    fig.add_trace(go.Scatter(
                        x=movable_interfaces, 
                        y=[model_res[i] for i in range(len(movable_interfaces))],
                        mode='markers',
                        marker=dict(symbol='line-ns', size=30, color='green', line_width=3),
                        name='Adjustable Interfaces',
                        hoverinfo='x'
                    ))

                # Sounding Curve
                fig.add_trace(go.Scatter(
                    x=ab2_values, y=rho_a,
                    mode='lines+markers',
                    name='Apparent Resistivity (V.E.S)',
                    line=dict(color='red', width=2, dash='dot'),
                    hoverinfo='skip'
                ))
                
                title_suffix = " (Click to Set Resistivity)" if "Resistivity" in edit_mode else " (Click to Move Interface)"
                
                fig.update_layout(
                    title="Model & Sounding Curve" + title_suffix,
                    xaxis_title="Spacing (AB/2) / Depth (m)",
                    yaxis_title="Resistivity (Ohm-m)",
                    xaxis=dict(type="log", range=[np.log10(min_x_grid), np.log10(max_x_grid)]),
                    yaxis=dict(type="log", range=[np.log10(min_y_grid), np.log10(max_y_grid)]),
                    height=600, showlegend=True, template="plotly_white", dragmode='pan'
                )
                
                selection = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points")
                
                # --- Interaction Logic ---
                if selection and selection["selection"]["points"]:
                    point = selection["selection"]["points"][0]
                    clicked_x = point["x"]
                    clicked_y = point["y"]
                    
                    current_df = st.session_state.layer_df.copy() # Work on copy
                    updated = False
                    
                    if "Resistivity" in edit_mode:
                         target_layer_idx = -1
                         for i in range(len(plot_depths) - 1):
                            if i == len(plot_depths) - 1:
                                if clicked_x >= plot_depths[i]: target_layer_idx = i; break
                            else:
                                if plot_depths[i] <= clicked_x < plot_depths[i+1]: target_layer_idx = i; break
                         if target_layer_idx == -1 and clicked_x >= plot_depths[-1]: target_layer_idx = len(model_res) - 1
                         
                         if 0 <= target_layer_idx < len(current_df):
                            current_df.at[target_layer_idx, "Resistivity (Ohm-m)"] = clicked_y
                            updated = True
                            
                    elif "Depth" in edit_mode:
                         if len(movable_interfaces) > 0:
                            diffs = [abs(d - clicked_x) for d in movable_interfaces]
                            nearest_idx = np.argmin(diffs)
                            
                            prev_bound = 0.0 if nearest_idx == 0 else movable_interfaces[nearest_idx - 1]
                            next_bound = movable_interfaces[nearest_idx + 1] if nearest_idx < len(movable_interfaces) - 1 else float('inf')
                            
                            if prev_bound < clicked_x < next_bound:
                                # Update Thicknesses based on new interface depth
                                new_interfaces = list(movable_interfaces)
                                new_interfaces[nearest_idx] = clicked_x
                                
                                new_thicknesses = []
                                last_d = 0
                                for d in new_interfaces:
                                    new_thicknesses.append(d - last_d)
                                    last_d = d
                                new_thicknesses.append(0.0) # Last layer
                                
                                current_df["Thickness (m)"] = new_thicknesses
                                # Recalculate depths to be safe (sync)
                                current_df = recalculate_depths(current_df)
                                updated = True
                            else:
                                pass

                    if updated:
                        st.session_state.layer_df = current_df
                        st.session_state.editor_key += 1
                        st.rerun()

            except Exception as e:
                st.error(f"Error calculating model: {e}")
                st.warning("Please ensure resistivities are positive and layers are defined correctly.")
    else:
        st.warning("Please add at least one layer.")

st.markdown("---")
st.markdown("Powered by **Streamlit**, **Plotly**, and **empymod**.")
