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
import re

# Page Configuration
st.set_page_config(
    page_title="Digital Wind Tunnel Pro",
    page_icon="💨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
col_header_left, col_header_right = st.columns([7, 3])

with col_header_left:
    st.title("💨 Digital Wind Tunnel Pro")
    st.markdown("""
    **Compare airfoils**, analyze performance, and export data.
    Supports **NACA 4, 5, 6 Series** (with Auto-Web-Fetch) and custom files.
    """)

with col_header_right:
    image_filename = "kcg_logo.png" 
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

# Sidebar: Controls
st.sidebar.header("🧪 Configuration")

enable_comparison = st.sidebar.checkbox("⚔️ Compare Two Airfoils", value=False)


# WEB FETCH WITH NACA 6-SERIES LOGIC
def fetch_airfoil_from_web(clean_code, is_6_series=False):
    """
    Enhanced fetcher with NACA 6-series specific logic.
    
    For 6-series, we try multiple naming conventions:
    - 63-415, 63415, 634-15, 63-4-15, naca63-415, etc.
    """
    
    # Generate variations based on airfoil type
    variations = []
    
    if is_6_series:
        # Extract components: 
        # Try to parse the format
        base = clean_code.replace("naca", "").replace(" ", "").lower()
        
        # Common patterns for 6-series
        variations.extend([
            base,                                    # 63-415
            base.replace("-", ""),                   # 63415
            base[:2] + base[3:] if "-" in base else base,  # 63415 from 63-415
            base[:2] + "-" + base[2:] if "-" not in base else base,  # 63-415 from 63415
        ])
        
        # Try adding/removing dashes in different positions
        if "-" in base:
            parts = base.split("-")
            if len(parts) == 2:
                variations.extend([
                    parts[0] + parts[1],              # Remove all dashes
                    parts[0] + "-" + parts[1][:1] + "-" + parts[1][1:],  # 63-4-15
                ])
        else:
            # If no dash, try adding them
            if len(base) >= 5:
                variations.extend([
                    base[:2] + "-" + base[2:],        # 63-415
                    base[:2] + "-" + base[2:3] + "-" + base[3:],  # 63-4-15
                ])
        
        # Add "a" suffix variations (some databases use this)
        for v in variations.copy():
            variations.append(v + "a")
            
    else:
        # Standard 4/5 digit variations
        variations = [
            clean_code, 
            clean_code.replace("-", ""), 
            clean_code[:4] + "-" + clean_code[4:] if len(clean_code) > 4 else clean_code
        ]
    
    # Remove duplicates while preserving order
    variations = list(dict.fromkeys(variations))
    
    # Database URLs to try
    base_urls = [
        "https://m-selig.ae.illinois.edu/ads/coord/{}.dat",
        "https://m-selig.ae.illinois.edu/ads/coord/naca{}.dat",
        "http://airfoiltools.com/airfoil/seligdatfile?airfoil={}",
        "http://airfoiltools.com/airfoil/seligdatfile?airfoil=naca{}",
    ]
    
    st.sidebar.info(f"🔍 Trying {len(variations)} naming variations across {len(base_urls)} databases...")
    
    for code in variations:
        for url_template in base_urls:
            try:
                # Try with and without 'naca' prefix
                for prefix in ['', 'naca']:
                    test_code = prefix + code if prefix else code
                    url = url_template.format(test_code)
                    
                    response = requests.get(url, timeout=3, verify=False)
                    
                    if response.status_code == 200 and len(response.text) > 100:
                        # Parse content
                        lines = response.text.splitlines()
                        data = []
                        header_skipped = False
                        
                        for line in lines:
                            line = line.strip()
                            
                            # Skip header lines
                            if not header_skipped and (not line or 
                                                      'name' in line.lower() or 
                                                      'naca' in line.lower() and '.' not in line):
                                continue
                            header_skipped = True
                            
                            parts = line.split()
                            if len(parts) >= 2:
                                try:
                                    x = float(parts[0])
                                    y = float(parts[1])
                                    # Sanity check
                                    if -2 <= x <= 2 and -2 <= y <= 2:
                                        data.append([x, y])
                                except ValueError:
                                    continue
                        
                        # Valid airfoil should have at least 20 points
                        if len(data) >= 20:
                            coords = np.array(data)
                            
                            # Normalize if needed (some files have scaled coordinates)
                            x_range = coords[:, 0].max() - coords[:, 0].min()
                            if x_range > 1.5:  # Likely scaled to 100
                                coords = coords / 100.0
                            
                            return coords, f"NACA {clean_code.upper()} (Web)"
            except Exception as e:
                continue
    
    return None, None


# Helper function to get airfoil coordinates
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
            code = st.sidebar.text_input(
                f"Code (4 Digits)", 
                value="2412" if key_prefix == "Airfoil 1" else "0012", 
                key=f"{key_prefix}_4", 
                max_chars=4
            )
            if len(code) == 4 and code.isdigit():
                name = f"NACA {code}"
                af = asb.Airfoil(f"naca{code}")
                af = af.repanel(n_points_per_side=100)
                coords = af.coordinates
            elif len(code) > 0:
                st.sidebar.warning(f"Requires exactly 4 digits.")

        elif input_type == "NACA 5-Digit":
            code = st.sidebar.text_input(
                f"Code (5 Digits)", 
                value="23012", 
                key=f"{key_prefix}_5", 
                max_chars=5
            )
            if len(code) == 5 and code.isdigit():
                name = f"NACA {code}"
                af = asb.Airfoil(f"naca{code}")
                af = af.repanel(n_points_per_side=100)
                coords = af.coordinates
            elif len(code) > 0:
                st.sidebar.warning(f"Requires exactly 5 digits.")

        elif input_type == "NACA 6-Series":
            st.sidebar.caption("✅ Examples: 63-415, 64-212, 65-018, 66-209")
            st.sidebar.caption("💡 Format: 6X-YZZ (X=pressure location, Y=CL×10, ZZ=thickness%)")
            
            raw_code = st.sidebar.text_input(
                f"Code (6-Series)", 
                value="63-415" if key_prefix == "Airfoil 1" else "64-212", 
                key=f"{key_prefix}_6"
            )
            
            if len(raw_code) >= 4:
                # Clean input
                clean_code = raw_code.lower().replace("naca", "").replace(" ", "").strip()
                
                # Validate format 
                digits_only = re.sub(r'[^0-9]', '', clean_code)
                if not (len(digits_only) >= 4 and digits_only[0] == '6'):
                    st.sidebar.error("❌ Invalid format. Must start with '6' (e.g., 63-415)")
                else:
                    name = f"NACA {clean_code.upper()}"
                    success = False
                    
                    # Strategy 1: Try Local AeroSandbox Library
                    st.sidebar.info("🔄 Step 1/3: Checking local database...")
                    for variant in [clean_code, clean_code.replace('-', ''), f"naca{clean_code}"]:
                        try:
                            af = asb.Airfoil(variant)
                            coords = af.repanel(n_points_per_side=100).coordinates
                            success = True
                            st.sidebar.success(f"✅ Found locally as '{variant}'")
                            break
                        except:
                            continue
                    
                    # Strategy 2: Web Fetch with Enhanced Logic
                    if not success:
                        st.sidebar.info("🔄 Step 2/3: Searching online databases...")
                        with st.spinner(f"Fetching NACA {clean_code} from web..."):
                            web_coords, web_name = fetch_airfoil_from_web(clean_code, is_6_series=True)
                            
                            if web_coords is not None:
                                coords = web_coords
                                name = web_name
                                
                                # Re-panel for smoothness
                                try:
                                    af = asb.Airfoil(name=name, coordinates=coords)
                                    af = af.repanel(n_points_per_side=100)
                                    coords = af.coordinates
                                except:
                                    pass  # Use raw coords if repanel fails
                                
                                success = True
                                st.sidebar.success(f"✅ {web_name} downloaded!")
                    
                    # Strategy 3: Generate Analytically 
                    if not success:
                        st.sidebar.info("🔄 Step 3/3: Attempting analytical generation...")
                        try:
                            # Try to generate using NACA 6-series math
                            # This is a simplified approximation
                            coords = generate_naca_6_series_approximation(clean_code)
                            if coords is not None:
                                success = True
                                st.sidebar.warning(f"⚠️ Using analytical approximation (may be less accurate)")
                        except:
                            pass
                    
                    if not success:
                        st.sidebar.error(f"❌ Could not find or generate NACA {clean_code}")
                        st.sidebar.markdown("""
                        **Troubleshooting:**
                        - Verify the code format (e.g., 63-415, 64-212)
                        - Try alternative formats: 63415 or 63-4-15
                        - Download `.dat` file manually and use 'Upload DAT'
                        - Check [UIUC Database](https://m-selig.ae.illinois.edu/ads/coord_database.html)
                        """)

        elif input_type == "Upload DAT":
            file = st.sidebar.file_uploader(
                f"Upload .dat", 
                type=["dat", "txt"], 
                key=f"{key_prefix}_file"
            )
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
                            except:
                                pass
                    
                    if len(data) > 10:
                        coords = np.array(data)
                        name = file.name.replace('.dat', '').replace('.txt', '')
                    else:
                        st.sidebar.error("File contains too few valid points.")
                except Exception as e:
                    st.sidebar.error(f"Error parsing file: {e}")

    except Exception as e:
        st.sidebar.error(f"Error: {e}")
    
    return name, coords


# HELPER: Analytical NACA 6-Series Approximation
def generate_naca_6_series_approximation(code):
    """
    Simplified NACA 6-series generator using thick airfoil theory.
    This is an APPROXIMATION and may not match exact profiles.
    """
    try:
        # Parse code
        digits = re.sub(r'[^0-9]', '', code)
        
        if len(digits) < 5:
            return None
        
        series = int(digits[0])  # Should be 6
        pressure_loc = int(digits[1]) / 10  # Position of min pressure
        design_cl_thickness = digits[2:]
        
        # Extract design CL and thickness
        if len(design_cl_thickness) == 3:
            design_cl = int(design_cl_thickness[0]) / 10
            thickness = int(design_cl_thickness[1:]) / 100
        else:
            # Fallback parsing
            thickness = int(design_cl_thickness[-2:]) / 100
            design_cl = 0.2  # Default
        
        # Generate x coordinates with cosine spacing
        n_points = 100
        beta = np.linspace(0, np.pi, n_points)
        x = (1 - np.cos(beta)) / 2
        
        # Thickness distribution (using NACA 4-digit formula as base)
        yt = 5 * thickness * (
            0.2969 * np.sqrt(x) - 
            0.1260 * x - 
            0.3516 * x**2 + 
            0.2843 * x**3 - 
            0.1015 * x**4
        )
        
        # Simplified camber 
        # Real 6-series uses complex mean line equations
        camber_scale = design_cl * 0.3
        yc = camber_scale * x * (1 - x)
        
        # Upper and lower surfaces
        x_upper = x
        y_upper = yc + yt
        x_lower = x
        y_lower = yc - yt
        
        # Combine coordinates 
        coords = np.vstack([
            np.column_stack([x_upper[::-1], y_upper[::-1]]),
            np.column_stack([x_lower[1:], y_lower[1:]])
        ])
        
        return coords
        
    except Exception as e:
        st.sidebar.error(f"Generation failed: {e}")
        return None


# Get Airfoil 1
name_1, coords_1 = get_airfoil_input("Airfoil 1")

# Get Airfoil 2
name_2, coords_2 = None, None
if enable_comparison:
    name_2, coords_2 = get_airfoil_input("Airfoil 2")

st.sidebar.markdown("---")

# 2. Flow Conditions
st.sidebar.subheader("🌊 Flow Conditions")
reynolds = st.sidebar.number_input(
    "Reynolds Number (Re)", 
    1000.0, 100_000_000.0, 1_000_000.0, 
    step=100_000.0, 
    format="%.0f"
)
model_size = st.sidebar.selectbox(
    "Model Size", 
    ["xsmall", "small", "medium", "large", "xlarge"], 
    index=2
)

# 3. Analysis Mode
analysis_mode = st.sidebar.radio("Analysis Type", ["Single Point", "Polar Sweep (Alpha vs CL/CD)"])

if analysis_mode == "Single Point":
    alpha = np.array([st.sidebar.slider("Angle of Attack (α)", -20.0, 20.0, 0.0, 0.5)])
else:
    r = st.sidebar.slider("Alpha Sweep Range", -30.0, 30.0, (-5.0, 25.0))
    alpha = np.linspace(r[0], r[1], 80)


# Physics Filter
def apply_strict_physics_filter(res_dict, alpha_arr):
    """Remove post-stall unphysical regions"""
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


# Main Area

if coords_1 is not None:
    with st.expander("📐 Airfoil Geometry", expanded=True):
        fig_geo = go.Figure()
        
        # Close the loop if needed
        if not np.allclose(coords_1[0], coords_1[-1]): 
            coords_1 = np.vstack([coords_1, coords_1[0]])
        
        fig_geo.add_trace(go.Scatter(
            x=coords_1[:,0], y=coords_1[:,1], 
            mode='lines', name=name_1, 
            fill="toself", fillcolor="rgba(0,100,200,0.2)"
        ))
        
        if enable_comparison and coords_2 is not None:
            if not np.allclose(coords_2[0], coords_2[-1]): 
                coords_2 = np.vstack([coords_2, coords_2[0]])
            
            fig_geo.add_trace(go.Scatter(
                x=coords_2[:,0], y=coords_2[:,1], 
                mode='lines', name=name_2, 
                line=dict(dash='dash', color='red'),
                fill="toself", fillcolor="rgba(200,0,0,0.1)"
            ))

        fig_geo.update_layout(
            yaxis=dict(scaleanchor="x", scaleratio=1), 
            height=300, 
            margin=dict(t=30, b=0),
            hovermode='closest'
        )
        st.plotly_chart(fig_geo, use_container_width=True)

if st.button("🚀 Run Analysis", type="primary"):
    if coords_1 is None:
        st.error("❌ Please select Airfoil 1 first.")
    else:
        with st.spinner("Running NeuralFoil simulation..."):
            try:
                res1 = nf.get_aero_from_coordinates(
                    coordinates=coords_1, 
                    alpha=alpha, 
                    Re=reynolds, 
                    model_size=model_size
                )
                alpha1 = alpha.copy()
                
                res2 = None
                alpha2 = alpha.copy()
                if enable_comparison and coords_2 is not None:
                    res2 = nf.get_aero_from_coordinates(
                        coordinates=coords_2, 
                        alpha=alpha, 
                        Re=reynolds, 
                        model_size=model_size
                    )

                if analysis_mode != "Single Point":
                    res1, alpha1 = apply_strict_physics_filter(res1, alpha1)
                    if res2:
                        res2, alpha2 = apply_strict_physics_filter(res2, alpha2)

                if analysis_mode == "Single Point":
                    cl1 = res1['CL'][0]
                    cd1 = res1['CD'][0]
                    cm1 = res1['CM'][0]
                    ld1 = cl1 / cd1 if cd1 > 0 else 0

                    st.success(f"✅ Analysis complete for {name_1}")
                    cols = st.columns(4)
                    cols[0].metric("CL (Lift)", f"{cl1:.4f}")
                    cols[1].metric("CD (Drag)", f"{cd1:.5f}")
                    cols[2].metric("CM (Moment)", f"{cm1:.4f}")
                    cols[3].metric("L/D Ratio", f"{ld1:.1f}")
                    
                    if 'analysis_confidence' in res1:
                        conf = res1['analysis_confidence'][0]
                        st.caption(f"🤖 Neural Network Confidence: {conf:.1%}")
                        st.progress(float(conf))

                    if res2:
                        st.divider()
                        st.subheader(f"🆚 Comparison: {name_2}")
                        cl2 = res2['CL'][0]
                        cd2 = res2['CD'][0]
                        cm2 = res2['CM'][0]
                        ld2 = cl2 / cd2 if cd2 > 0 else 0
                        
                        delta_cl = cl2 - cl1
                        delta_cd = cd2 - cd1
                        delta_ld = ld2 - ld1
                        
                        cols2 = st.columns(4)
                        cols2[0].metric("CL", f"{cl2:.4f}", delta=f"{delta_cl:+.4f}")
                        cols2[1].metric("CD", f"{cd2:.5f}", delta=f"{delta_cd:+.5f}", delta_color="inverse")
                        cols2[2].metric("CM", f"{cm2:.4f}", delta=f"{cm2-cm1:+.4f}")
                        cols2[3].metric("L/D", f"{ld2:.1f}", delta=f"{delta_ld:+.1f}")

                else:  # Polar Sweep
                    df1 = pd.DataFrame({
                        "Alpha": alpha1, 
                        "CL": res1["CL"].flatten(), 
                        "CD": res1["CD"].flatten(), 
                        "CM": res1["CM"].flatten(), 
                        "Airfoil": name_1
                    })
                    df_all = df1
                    
                    if res2:
                        df2 = pd.DataFrame({
                            "Alpha": alpha2, 
                            "CL": res2["CL"].flatten(), 
                            "CD": res2["CD"].flatten(), 
                            "CM": res2["CM"].flatten(), 
                            "Airfoil": name_2
                        })
                        df_all = pd.concat([df1, df2])

                    df_all["L/D"] = df_all["CL"] / np.maximum(df_all["CD"], 1e-6)

                    tab1, tab2, tab3 = st.tabs(["📈 Lift (CL vs α)", "📉 Drag (CL vs CD)", "⚡ Efficiency (L/D)"])
                    
                    def create_spline_chart(df, x, y, title, x_label, y_label):
                        fig = px.line(df, x=x, y=y, color="Airfoil", title=title)
                        fig.update_traces(line_shape='spline', line_smoothing=1.0)
                        fig.update_layout(
                            xaxis_title=x_label,
                            yaxis_title=y_label,
                            hovermode='x unified'
                        )
                        return fig

                    with tab1:
                        st.plotly_chart(
                            create_spline_chart(df_all, "Alpha", "CL", "Lift Curve", "Angle of Attack (°)", "CL"), 
                            use_container_width=True
                        )
                    
                    with tab2:
                        st.plotly_chart(
                            create_spline_chart(df_all, "CD", "CL", "Drag Polar", "CD", "CL"), 
                            use_container_width=True
                        )
                    
                    with tab3:
                        st.plotly_chart(
                            create_spline_chart(df_all, "Alpha", "L/D", "Lift-to-Drag Ratio", "Angle of Attack (°)", "L/D"), 
                            use_container_width=True
                        )

                    # Export Data
                    st.divider()
                    csv = df_all.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Results (CSV)", 
                        csv, 
                        "wind_tunnel_results.csv", 
                        "text/csv",
                        help="Download analysis data for further processing"
                    )
                    
            except Exception as e:
                st.error(f"❌ Simulation failed: {e}")
                st.exception(e)

