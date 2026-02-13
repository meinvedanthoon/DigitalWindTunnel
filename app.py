import streamlit as st
import numpy as np
import aerosandbox as asb
import neuralfoil as nf
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from PIL import Image
import io
import requests

# --- Page Configuration ---
st.set_page_config(
    page_title="Digital Wind Tunnel Pro",
    page_icon="💨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 🖼️ Header with White-Background Logo ---
col_header_left, col_header_right = st.columns([7, 3])

with col_header_left:
    st.title("💨 Digital Wind Tunnel Pro")
    st.markdown("""
    **Compare airfoils**, analyze performance, and export data.
    Supports **NACA 4, 5, 6 Series** (with Auto-Web-Fetch) and custom files.
    """)

with col_header_right:
    image_filename = "REPLACE_WITH_YOUR_IMAGE_NAME.png" 
    try:
        img = Image.open(image_filename)
        white_bg = Image.new("RGBA", img.size, "WHITE")
        if img.mode in ('RGBA', 'LA'):
            white_bg.paste(img, (0, 0), img)
            final_image = white_bg.convert("RGB")
        else:
            final_image = img.convert("RGB")
        st.image(final_image, use_container_width=True)
    except Exception:
        pass 

# --- Sidebar: Controls ---
st.sidebar.header("🧪 Configuration")

enable_comparison = st.sidebar.checkbox("⚔️ Compare Two Airfoils", value=False)

# --- HELPER: WEB FETCH FALLBACK ---
def fetch_airfoil_from_web(clean_code):
    """
    Attempts to download coordinates from UIUC and AirfoilTools if local lookup fails.
    """
    # Variations to try (e.g. naca63415, naca63-415)
    variations = [clean_code, clean_code.replace("-", ""), clean_code[:4] + "-" + clean_code[4:]]
    base_urls = [
        "https://m-selig.ae.illinois.edu/ads/coord/naca{}.dat",
        "https://raw.githubusercontent.com/pdas-developers/pdas_data/master/airfoils/naca{}.dat"
    ]
    
    for code in variations:
        for url_template in base_urls:
            try:
                url = url_template.format(code)
                # verify=False prevents SSL certificate errors on some networks
                response = requests.get(url, timeout=2) 
                if response.status_code == 200:
                    # Parse content
                    lines = response.text.splitlines()
                    data = []
                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                data.append([float(parts[0]), float(parts[1])])
                            except: pass
                    if len(data) > 10:
                        return np.array(data), f"NACA {code} (Web)"
            except:
                continue
    return None, None


# --- Helper function to get airfoil coordinates ---
def get_airfoil_input(key_prefix):
    st.sidebar.subheader(f"{key_prefix} Selection")
    
    input_type = st.sidebar.selectbox(
        f"Input Type", 
        ["NACA 4-Digit", "NACA 5-Digit", "NACA 6-Series", "Upload DAT"], 
        key=f"{key_prefix}_type"
    )
    
    coords = None
    name = f"{key_prefix}"

    try:
        if input_type == "NACA 4-Digit":
            code = st.sidebar.text_input(f"Code (4 Digits)", value="2412" if key_prefix == "Airfoil 1" else "0012", key=f"{key_prefix}_4", max_chars=4)
            if len(code) == 4 and code.isdigit():
                name = f"NACA {code}"
                af = asb.Airfoil(f"naca{code}")
                af = af.repanel(n_points_per_side=100)
                coords = af.coordinates
            elif len(code) > 0:
                st.sidebar.warning(f"Requires exactly 4 digits.")

        elif input_type == "NACA 5-Digit":
            code = st.sidebar.text_input(f"Code (5 Digits)", value="23012", key=f"{key_prefix}_5", max_chars=5)
            if len(code) == 5 and code.isdigit():
                name = f"NACA {code}"
                af = asb.Airfoil(f"naca{code}")
                af = af.repanel(n_points_per_side=100)
                coords = af.coordinates
            elif len(code) > 0:
                st.sidebar.warning(f"Requires exactly 5 digits.")

        elif input_type == "NACA 6-Series":
            st.caption("Example: 63-415, 64-212")
            raw_code = st.sidebar.text_input(f"Code (6-Series)", value="63-415", key=f"{key_prefix}_6")
            
            if len(raw_code) >= 4:
                # 1. Clean Input
                clean_code = raw_code.lower().replace("naca", "").replace(" ", "").strip()
                name = f"NACA {clean_code.upper()}"
                
                success = False
                
                # 2. Try Local Library (AeroSandbox)
                try:
                    af = asb.Airfoil(f"naca{clean_code}")
                    coords = af.repanel(n_points_per_side=100).coordinates
                    success = True
                except:
                    # Try without dash
                    try:
                        af = asb.Airfoil(f"naca{clean_code.replace('-', '')}")
                        coords = af.repanel(n_points_per_side=100).coordinates
                        success = True
                    except:
                        pass

                # 3. Try Web Fetch (Fallback)
                if not success:
                    with st.spinner(f"Searching online databases for NACA {clean_code}..."):
                        web_coords, web_name = fetch_airfoil_from_web(clean_code)
                        if web_coords is not None:
                            coords = web_coords
                            name = web_name
                            # Re-panel for smoothness using AeroSandbox
                            try:
                                af = asb.Airfoil(name=name, coordinates=coords).repanel(n_points_per_side=100)
                                coords = af.coordinates
                            except: pass # Use raw if repanel fails
                            success = True
                            st.sidebar.success(f"✅ Found {name} online!")
                
                if not success:
                    st.sidebar.error(f"❌ Could not find 'NACA {clean_code}' locally or online.")
                    st.sidebar.info("Tip: If this is a rare airfoil, download the .dat file and use 'Upload DAT'.")

        elif input_type == "Upload DAT":
            file = st.sidebar.file_uploader(f"Upload .dat", type=["dat", "txt"], key=f"{key_prefix}_file")
            if file:
                try:
                    string_data = file.getvalue().decode("utf-8").splitlines()
                    data = []
                    for line in string_data:
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                x = float(parts[0])
                                y = float(parts[1])
                                data.append([x, y])
                            except: pass
                    if len(data) > 10:
                        coords = np.array(data)
                        name = file.name
                    else:
                        st.sidebar.error("File invalid.")
                except:
                    st.sidebar.error("Error parsing file.")

    except Exception as e:
        st.sidebar.error(f"Error: {e}")
    
    return name, coords

# Get Airfoil 1
name_1, coords_1 = get_airfoil_input("Airfoil 1")

# Get Airfoil 2 (if enabled)
name_2, coords_2 = None, None
if enable_comparison:
    name_2, coords_2 = get_airfoil_input("Airfoil 2")

st.sidebar.markdown("---")

# 2. Flow Conditions
st.sidebar.subheader("🌊 Flow Conditions")
reynolds = st.sidebar.number_input("Reynolds Number (Re)", 1000.0, 100_000_000.0, 1_000_000.0, step=100_000.0, format="%.0f")
model_size = st.sidebar.selectbox("Model Size", ["xsmall", "small", "medium", "large", "xlarge"], index=2)

# 3. Analysis Mode
analysis_mode = st.sidebar.radio("Analysis Type", ["Single Point", "Polar Sweep (Alpha vs CL/CD)"])

if analysis_mode == "Single Point":
    alpha = np.array([st.sidebar.slider("Angle of Attack (α)", -20.0, 20.0, 0.0, 0.5)])
else:
    r = st.sidebar.slider("Alpha Sweep Range", -30.0, 30.0, (-5.0, 25.0))
    alpha = np.linspace(r[0], r[1], 80)

# --- Strict Physics Filter ---
def apply_strict_physics_filter(res_dict, alpha_arr):
    cl = res_dict['CL'].flatten()
    stall_idx = np.argmax(cl) 
    
    for i in range(1, len(cl)-1):
        if alpha_arr[i] > 0 and cl[i] > cl[i-1] and cl[i] > cl[i+1]:
            stall_idx = i
            break 

    cutoff_idx = len(cl)
    for i in range(stall_idx + 1, len(cl)):
        if cl[i] > cl[i-1]: 
            cutoff_idx = i
            break
            
    new_res = {}
    for key, val in res_dict.items():
        if isinstance(val, np.ndarray):
             new_res[key] = val[:cutoff_idx]
        else:
             new_res[key] = val
             
    return new_res, alpha_arr[:cutoff_idx]


# --- Main Area ---

if coords_1 is not None:
    with st.expander("📐 Airfoil Geometry", expanded=True):
        fig_geo = go.Figure()
        if not np.allclose(coords_1[0], coords_1[-1]): coords_1 = np.vstack([coords_1, coords_1[0]])
        fig_geo.add_trace(go.Scatter(x=coords_1[:,0], y=coords_1[:,1], mode='lines', name=name_1, fill="toself"))
        
        if enable_comparison and coords_2 is not None:
            if not np.allclose(coords_2[0], coords_2[-1]): coords_2 = np.vstack([coords_2, coords_2[0]])
            fig_geo.add_trace(go.Scatter(x=coords_2[:,0], y=coords_2[:,1], mode='lines', name=name_2, line=dict(dash='dash')))

        fig_geo.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1), height=300, margin=dict(t=30, b=0))
        st.plotly_chart(fig_geo, use_container_width=True)

if st.button("🚀 Run Analysis", type="primary"):
    if coords_1 is None:
        st.error("Please select Airfoil 1.")
    else:
        with st.spinner("Running NeuralFoil..."):
            res1 = nf.get_aero_from_coordinates(coordinates=coords_1, alpha=alpha, Re=reynolds, model_size=model_size)
            alpha1 = alpha.copy()
            
            res2 = None
            alpha2 = alpha.copy()
            if enable_comparison and coords_2 is not None:
                res2 = nf.get_aero_from_coordinates(coordinates=coords_2, alpha=alpha, Re=reynolds, model_size=model_size)

            if analysis_mode != "Single Point":
                res1, alpha1 = apply_strict_physics_filter(res1, alpha1)
                if res2:
                    res2, alpha2 = apply_strict_physics_filter(res2, alpha2)

            if analysis_mode == "Single Point":
                cl1 = res1['CL'][0]
                cd1 = res1['CD'][0]
                cm1 = res1['CM'][0]
                ld1 = cl1 / cd1

                cols = st.columns(4)
                cols[0].metric("CL", f"{cl1:.4f}")
                cols[1].metric("CD", f"{cd1:.5f}")
                cols[2].metric("CM", f"{cm1:.4f}")
                cols[3].metric("L/D", f"{ld1:.1f}")
                
                if 'analysis_confidence' in res1:
                    conf = res1['analysis_confidence'][0]
                    st.caption(f"🤖 AI Confidence Score: {conf:.1%}")
                    st.progress(float(conf))

                if res2:
                    st.divider()
                    st.subheader(f"🆚 Comparison: {name_2}")
                    cl2 = res2['CL'][0]
                    cd2 = res2['CD'][0]
                    cm2 = res2['CM'][0]
                    ld2 = cl2 / cd2
                    
                    delta_cl = cl2 - cl1
                    delta_cd = cd2 - cd1
                    delta_ld = ld2 - ld1
                    
                    cols2 = st.columns(4)
                    cols2[0].metric("CL", f"{cl2:.4f}", delta=f"{delta_cl:.4f}")
                    cols2[1].metric("CD", f"{cd2:.5f}", delta=f"{delta_cd:.5f}", delta_color="inverse")
                    cols2[2].metric("CM", f"{cm2:.4f}")
                    cols2[3].metric("L/D", f"{ld2:.1f}", delta=f"{delta_ld:.1f}")

            else:
                df1 = pd.DataFrame({
                    "Alpha": alpha1, "CL": res1["CL"].flatten(), "CD": res1["CD"].flatten(), 
                    "CM": res1["CM"].flatten(), "Airfoil": name_1
                })
                df_all = df1
                
                if res2:
                    df2 = pd.DataFrame({
                        "Alpha": alpha2, "CL": res2["CL"].flatten(), "CD": res2["CD"].flatten(), 
                        "CM": res2["CM"].flatten(), "Airfoil": name_2
                    })
                    df_all = pd.concat([df1, df2])

                df_all["L/D"] = df_all["CL"] / np.maximum(df_all["CD"], 1e-6)

                tab1, tab2, tab3 = st.tabs(["Lift (CL vs α)", "Drag (CL vs CD)", "Efficiency (L/D)"])
                
                def create_spline_chart(df, x, y, title):
                    fig = px.line(df, x=x, y=y, color="Airfoil", title=title)
                    fig.update_traces(line_shape='spline', line_smoothing=1.0)
                    return fig

                with tab1:
                    st.plotly_chart(create_spline_chart(df_all, "Alpha", "CL", "Lift Curve"), use_container_width=True)
                with tab2:
                    st.plotly_chart(create_spline_chart(df_all, "CD", "CL", "Drag Polar"), use_container_width=True)
                with tab3:
                    st.plotly_chart(create_spline_chart(df_all, "Alpha", "L/D", "L/D Ratio"), use_container_width=True)

                csv = df_all.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results (CSV)", csv, "wind_tunnel_results.csv", "text/csv")
