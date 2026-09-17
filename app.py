import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st

# ---------------------------------------------------------
# PAGE CONFIGURATION & HEADER
# ---------------------------------------------------------
st.set_page_config(
    page_title="ShipDesign Synth - Parametric Sizer", layout="wide"
)
st.title("🚢 Preliminary Ship Design Synthesis & Sizing Tool")
st.markdown(
    "Automated preliminary ship sizing powered by **Clarkson World Fleet Data**"
    " and classical naval architectural design spiral formulations."
)


# ---------------------------------------------------------
# STEP 1: DATA INGESTION & ROBUST CLEANING
# ---------------------------------------------------------
@st.cache_data
def load_and_clean_clarkson(file_buffer):
  xls = pd.ExcelFile(file_buffer)
  target_sheet = (
      "Listing" if "Listing" in xls.sheet_names else xls.sheet_names[0]
  )
  raw_df = pd.read_excel(xls, sheet_name=target_sheet, header=None)

  # Locate header row dynamically
  header_row_idx = None
  for idx, row in raw_df.iterrows():
    row_str = " ".join([str(val).lower() for val in row.dropna()])
    if "name" in row_str and "dwt" in row_str:
      header_row_idx = idx
      break

  if header_row_idx is None:
    return None

  df = raw_df.iloc[header_row_idx + 1 :].copy()
  df.columns = [str(col).strip() for col in raw_df.iloc[header_row_idx]]
  df.reset_index(drop=True, inplace=True)

  column_map = {
      "Name": "name",
      "Type": "ship_type",
      "Built": "year_built",
      "Dwt": "dwt",
      "LOA (m)": "loa",
      "Beam Mld (m)": "beam",
      "Draft (m)": "draft",
      "Depth Moulded (m)": "depth",
      "Status": "status",
      "Company": "company",
      "GT": "gt",
  }
  available_cols = {k: v for k, v in column_map.items() if k in df.columns}
  clean_df = df[list(available_cols.keys())].rename(columns=available_cols)

  for col in ["dwt", "loa", "beam", "draft", "depth", "year_built", "gt"]:
    if col in clean_df.columns:
      clean_df[col] = pd.to_numeric(
          clean_df[col].replace(["-", "N/A", "NA", ""], np.nan),
          errors="coerce",
      )

  # Intelligent speed consolidation hierarchy
  speed_candidates = [
      "Service Speed (knots)",
      "Operational Speed (knots)",
      "Speed (knots)",
      "Trial Speed (knots)",
  ]
  consolidated_speed = pd.Series(np.nan, index=df.index, dtype=float)
  for speed_col in speed_candidates:
    if speed_col in df.columns:
      parsed_col = pd.to_numeric(
          df[speed_col].replace(["-", "N/A", "NA", ""], np.nan),
          errors="coerce",
      )
      consolidated_speed = consolidated_speed.combine_first(parsed_col)

  clean_df["speed_knots"] = consolidated_speed
  valid = (
      (clean_df["dwt"] > 0)
      & (clean_df["loa"] > 0)
      & (clean_df["beam"] > 0)
      & (clean_df["draft"] > 0)
  )
  clean_df = clean_df[valid].reset_index(drop=True)

  clean_df["l_b"] = clean_df["loa"] / clean_df["beam"]
  clean_df["b_t"] = clean_df["beam"] / clean_df["draft"]
  if "depth" in clean_df.columns:
    clean_df["l_d"] = clean_df["loa"] / clean_df["depth"]
  return clean_df


# ---------------------------------------------------------
# STEP 2: NEAREST-NEIGHBOR PARENT SELECTION
# ---------------------------------------------------------
def select_parent_fleet(
    df,
    target_dwt,
    target_speed=None,
    ship_type=None,
    max_draft=None,
    max_beam=None,
    max_loa=None,
    top_n=12,
):
  subset = df.copy()
  if ship_type and ship_type != "All":
    subset = subset[
        subset["ship_type"].str.contains(ship_type, case=False, na=False)
    ]
  if max_draft:
    subset = subset[subset["draft"] <= max_draft]
  if max_beam:
    subset = subset[subset["beam"] <= max_beam]
  if max_loa:
    subset = subset[subset["loa"] <= max_loa]

  if subset.empty:
    return pd.DataFrame(), None

  dwt_std = subset["dwt"].std() if subset["dwt"].std() > 0 else 1.0
  dwt_term = ((subset["dwt"] - target_dwt) / dwt_std) ** 2

  if target_speed and subset["speed_knots"].notna().sum() > 1:
    speed_std = (
        subset["speed_knots"].std() if subset["speed_knots"].std() > 0 else 1.0
    )
    speed_series = subset["speed_knots"].fillna(subset["speed_knots"].mean())
    speed_term = ((speed_series - target_speed) / speed_std) ** 2
    distance = np.sqrt(0.70 * dwt_term + 0.30 * speed_term)
  else:
    distance = np.sqrt(dwt_term)

  subset["similarity_distance"] = distance
  parents = subset.sort_values("similarity_distance").head(top_n).copy()
  stats_df = (
      parents[["loa", "beam", "draft", "l_b", "b_t", "speed_knots"]]
      .describe()
      .loc[["mean", "std", "min", "max"]]
      .round(2)
  )
  return parents, stats_df


# ---------------------------------------------------------
# STEP 3: EMPIRICAL SIZING CONVERGENCE LOOP
# ---------------------------------------------------------
def synthesize_particulars(
    target_dwt,
    speed_knots,
    range_nm=5000.0,
    parent_stats=None,
    max_draft=None,
    max_beam=None,
    max_loa=None,
    seawater_density=1.025,
):
  target_bt = (
      parent_stats.loc["mean", "b_t"] if parent_stats is not None else 2.85
  )
  c_dwt = 0.82
  delta_est = target_dwt / c_dwt
  tolerance = 1.0
  converged = False

  for iteration in range(50):
    disp_vol = delta_est / seawater_density
    c_schneekluth = 3.2
    l_bp = c_schneekluth * (delta_est**0.3) * (speed_knots**0.3)
    loa = 1.04 * l_bp

    if max_loa and loa > max_loa:
      loa = max_loa
      l_bp = loa / 1.04

    l_ft = l_bp * 3.28084
    fn = (speed_knots * 0.514444) / np.sqrt(9.81 * l_bp)
    cb = float(np.clip(1.05 - 0.5 * (speed_knots / np.sqrt(l_ft)), 0.70, 0.86))

    draft_calc = np.sqrt(disp_vol / (l_bp * target_bt * cb))
    beam_calc = draft_calc * target_bt

    if max_draft and draft_calc > max_draft:
      draft = max_draft
      beam = disp_vol / (l_bp * draft * cb)
    else:
      draft = draft_calc
      beam = beam_calc

    if max_beam and beam > max_beam:
      beam = max_beam
      draft = disp_vol / (l_bp * beam * cb)

    depth = l_bp / 11.8
    freeboard = depth - draft

    c_adm = 580.0
    pb_kw = (delta_est ** (2 / 3) * speed_knots**3) / c_adm * 0.7355
    sfc = 175.0 / 1e6
    voyage_hours = range_nm / speed_knots
    fuel_weight = pb_kw * sfc * voyage_hours * 1.15

    w_st = 0.072 * (l_bp * beam * depth)
    w_ot = 0.28 * (l_bp * beam)
    w_m = 0.70 * (pb_kw**0.7)
    lightship = w_st + w_ot + w_m

    delta_new = target_dwt + lightship + fuel_weight
    if abs(delta_new - delta_est) < tolerance:
      converged = True
      delta_est = delta_new
      break
    delta_est = 0.6 * delta_est + 0.4 * delta_new

  return {
      "converged": converged,
      "iterations": iteration + 1,
      "dwt": target_dwt,
      "displacement": round(delta_est, 1),
      "lightship": round(lightship, 1),
      "w_steel": round(w_st, 1),
      "w_outfit": round(w_ot, 1),
      "w_machinery": round(w_m, 1),
      "w_fuel": round(fuel_weight, 1),
      "loa": round(loa, 2),
      "lbp": round(l_bp, 2),
      "beam": round(beam, 2),
      "draft": round(draft, 2),
      "depth": round(depth, 2),
      "freeboard": round(freeboard, 2),
      "cb": round(cb, 3),
      "fn": round(fn, 3),
      "power_kw": round(pb_kw, 0),
      "l_b": round(loa / beam, 2),
      "b_t": round(beam / draft, 2),
      "l_d": round(l_bp / depth, 2),
  }


# ---------------------------------------------------------
# STEP 4: STABILITY & REGULATORY COMPLIANCE CHECKS
# ---------------------------------------------------------
def evaluate_compliance(p):
  c_wp = min(p["cb"] + 0.10, 0.92)
  kb = p["draft"] * (0.833 - 0.333 * (p["cb"] / c_wp))
  bm = (0.075 * c_wp + 0.005) * (p["beam"] ** 2) / (p["draft"] * p["cb"])
  km = kb + bm
  kg = 0.62 * p["depth"]
  gm = km - kg

  ld_ratio = p["lbp"] / p["depth"]
  f_min = 0.022 * p["lbp"]
  fb = p["depth"] - p["draft"]

  return {
      "gm": round(gm, 2),
      "gm_pass": gm >= 0.15,
      "ld_ratio": round(ld_ratio, 2),
      "ld_pass": ld_ratio <= 14.5,
      "fb_actual": round(fb, 2),
      "fb_min": round(f_min, 2),
      "fb_pass": fb >= f_min,
  }


# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.header("⚙️ Project Vessel Requirements")
uploaded_file = st.sidebar.file_uploader(
    "Upload Clarkson Dataset (.xlsx)", type=["xlsx", "xlsb"]
)

target_dwt = st.sidebar.number_input(
    "Target Deadweight (Dwt)",
    min_value=1000,
    max_value=400000,
    value=48000,
    step=1000,
)
target_speed = st.sidebar.slider(
    "Service Speed (knots)", min_value=8.0, max_value=25.0, value=14.0, step=0.5
)
target_range = st.sidebar.number_input(
    "Endurance Range (nm)",
    min_value=1000,
    max_value=25000,
    value=6000,
    step=500,
)

st.sidebar.subheader("🚫 Navigational & Port Limits")
apply_draft = st.sidebar.checkbox("Limit Draft")
max_draft = (
    st.sidebar.number_input("Max Draft (m)", value=12.0, step=0.1)
    if apply_draft
    else None
)

apply_beam = st.sidebar.checkbox("Limit Beam (e.g., Panamax)")
max_beam = (
    st.sidebar.number_input("Max Beam (m)", value=32.26, step=0.1)
    if apply_beam
    else None
)

apply_loa = st.sidebar.checkbox("Limit LOA")
max_loa = (
    st.sidebar.number_input("Max LOA (m)", value=225.0, step=1.0)
    if apply_loa
    else None
)

# ---------------------------------------------------------
# MAIN LOGIC & TABBED DASHBOARD
# ---------------------------------------------------------
if uploaded_file is not None:
  df_clean = load_and_clean_clarkson(uploaded_file)
  if df_clean is not None and not df_clean.empty:
    ship_types = ["All"] + sorted(list(df_clean["ship_type"].dropna().unique()))
    selected_type = st.sidebar.selectbox("Filter Vessel Type", ship_types)

    parents, stats_df = select_parent_fleet(
        df_clean,
        target_dwt,
        target_speed,
        selected_type,
        max_draft,
        max_beam,
        max_loa,
        top_n=12,
    )

    if not parents.empty:
      synth = synthesize_particulars(
          target_dwt,
          target_speed,
          target_range,
          stats_df,
          max_draft,
          max_beam,
          max_loa,
      )
      comp = evaluate_compliance(synth)

      # METRIC KPI CARDS
      st.subheader("📋 Synthesized Principal Particulars")
      col1, col2, col3, col4, col5 = st.columns(5)
      col1.metric("LOA (m)", f"{synth['loa']} m", f"LBP: {synth['lbp']} m")
      col2.metric("Beam (m)", f"{synth['beam']} m", f"B/T: {synth['b_t']}")
      col3.metric(
          "Draft (m)", f"{synth['draft']} m", f"Fb: {synth['freeboard']} m"
      )
      col4.metric(
          "Displacement",
          f"{synth['displacement']:,} t",
          f"DWT: {synth['dwt']:,} t",
      )
      col5.metric(
          "Power (kW)", f"{synth['power_kw']:,} kW", f"Speed: {target_speed} kn"
      )

      # TABBED INTERFACE
      tab1, tab2, tab3 = st.tabs([
          "📊 Fleet Benchmark Plots & 90% Lanes",
          "🔍 Top Parent Vessels",
          "⚖️ Weight & Compliance",
      ])

      with tab1:
        col_plot1, col_plot2 = st.columns(2)

        # Plot 1: DWT vs LOA
        fig1, ax1 = plt.subplots(figsize=(6.5, 4.5), dpi=100)
        x_vals = df_clean["dwt"].values
        y_vals = df_clean["loa"].values
        poly = np.polyfit(x_vals, y_vals, 1)
        res = y_vals - np.polyval(poly, x_vals)
        s_yx = np.sqrt(np.sum(res**2) / (len(x_vals) - 2))
        delta = stats.t.ppf(0.95, len(x_vals) - 2) * s_yx
        x_grid = np.linspace(min(x_vals), max(x_vals), 100)

        ax1.scatter(
            x_vals,
            y_vals,
            alpha=0.4,
            color="#1f77b4",
            label="Clarkson Parent Fleet",
        )
        ax1.plot(
            x_grid,
            np.polyval(poly, x_grid),
            color="#003366",
            label="Linear Trendline",
        )
        ax1.plot(
            x_grid,
            np.polyval(poly, x_grid) + delta,
            "--r",
            label="90% Upper Lane",
        )
        ax1.plot(
            x_grid,
            np.polyval(poly, x_grid) - delta,
            "--r",
            label="90% Lower Lane",
        )
        ax1.scatter(
            synth["dwt"],
            synth["loa"],
            color="#ff7f0e",
            marker="D",
            s=110,
            zorder=5,
            edgecolor="black",
            label="Synthesized Vessel",
        )
        ax1.set_title("DWT vs LOA (m)", fontweight="bold")
        ax1.set_xlabel("Deadweight (DWT)")
        ax1.set_ylabel("Length Overall, LOA (m)")
        ax1.legend(fontsize=8)
        ax1.grid(True, linestyle=":", alpha=0.6)
        col_plot1.pyplot(fig1)

        # Plot 2: DWT vs Beam
        fig2, ax2 = plt.subplots(figsize=(6.5, 4.5), dpi=100)
        y_b = df_clean["beam"].values
        poly_b = np.polyfit(x_vals, y_b, 1)
        res_b = y_b - np.polyval(poly_b, x_vals)
        s_yx_b = np.sqrt(np.sum(res_b**2) / (len(x_vals) - 2))
        delta_b = stats.t.ppf(0.95, len(x_vals) - 2) * s_yx_b

        ax2.scatter(
            x_vals,
            y_b,
            alpha=0.4,
            color="#2ca02c",
            label="Clarkson Parent Fleet",
        )
        ax2.plot(
            x_grid,
            np.polyval(poly_b, x_grid),
            color="#003366",
            label="Linear Trendline",
        )
        ax2.plot(
            x_grid,
            np.polyval(poly_b, x_grid) + delta_b,
            "--r",
            label="90% Upper Lane",
        )
        ax2.plot(
            x_grid,
            np.polyval(poly_b, x_grid) - delta_b,
            "--r",
            label="90% Lower Lane",
        )
        ax2.scatter(
            synth["dwt"],
            synth["beam"],
            color="#ff7f0e",
            marker="D",
            s=110,
            zorder=5,
            edgecolor="black",
            label="Synthesized Vessel",
        )
        ax2.set_title("DWT vs Beam (m)", fontweight="bold")
        ax2.set_xlabel("Deadweight (DWT)")
        ax2.set_ylabel("Beam Moulded (m)")
        ax2.legend(fontsize=8)
        ax2.grid(True, linestyle=":", alpha=0.6)
        col_plot2.pyplot(fig2)

      with tab2:
        st.write("#### 🏆 Extracted Nearest-Neighbor Parent Fleet")
        st.dataframe(
            parents[[
                "name",
                "ship_type",
                "dwt",
                "loa",
                "beam",
                "draft",
                "speed_knots",
                "l_b",
                "b_t",
                "similarity_distance",
            ]],
            use_container_width=True,
        )
        st.write("#### 📊 Statistical Envelopes of Parent Cluster")
        st.dataframe(stats_df, use_container_width=True)

      with tab3:
        c_w1, c_w2 = st.columns(2)
        with c_w1:
          st.write("#### ⚖️ Weight Breakdown (tonnes)")
          w_df = pd.DataFrame({
              "Component": [
                  "Hull Steel (W_ST)",
                  "Outfitting (W_OT)",
                  "Machinery (W_M)",
                  "Fuel & Margins",
                  "Deadweight (DWT)",
                  "Total Displacement",
              ],
              "Weight [t]": [
                  synth["w_steel"],
                  synth["w_outfit"],
                  synth["w_machinery"],
                  synth["w_fuel"],
                  synth["dwt"],
                  synth["displacement"],
              ],
              "Fraction [%]": [
                  round(synth["w_steel"] / synth["displacement"] * 100, 1),
                  round(synth["w_outfit"] / synth["displacement"] * 100, 1),
                  round(synth["w_machinery"] / synth["displacement"] * 100, 1),
                  round(synth["w_fuel"] / synth["displacement"] * 100, 1),
                  round(synth["dwt"] / synth["displacement"] * 100, 1),
                  100.0,
              ],
          })
          st.dataframe(w_df, hide_index=True, use_container_width=True)

        with c_w2:
          st.write("#### 🛡️ Compliance & Safety Sanity Checks")
          st.write(
              f"- **Metacentric Height (GM):** `{comp['gm']} m` "
              + (
                  "✅ PASS (GM >= 0.15 m)"
                  if comp["gm_pass"]
                  else "❌ FAIL (Deficient GM)"
              )
          )
          st.write(
              f"- **L/D Structural Slenderness:** `{comp['ld_ratio']}` "
              + (
                  "✅ PASS (L/D <= 14.5)"
                  if comp["ld_pass"]
                  else "❌ High bending risk"
              )
          )
          st.write(
              f"- **Geometric Freeboard:** `{comp['fb_actual']} m` vs Approx"
              f" `{comp['fb_min']} m` "
              + ("✅ PASS" if comp["fb_pass"] else "⚠️ Check ICLL corrections")
          )
          st.write(
              f"- **Design Froude Number:** `Fn = {synth['fn']}` (Favorable"
              " regime avoiding wave tuning)"
          )
    else:
      st.error("No vessels meet the selected constraints.")
else:
  st.info(
      "👈 Upload your Clarkson database export (.xlsx) in the sidebar to begin"
      " synthesis."
  )