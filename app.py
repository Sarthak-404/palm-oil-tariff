# app.py
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Viz
import plotly.express as px
import plotly.graph_objects as go

# Optional: skops loader for version-agnostic models
try:
    from skops.io import load as skops_load
except Exception:
    skops_load = None


# ---- Human-friendly labels, units & tooltips ----
FIELD_META = {
    # Tariff & taxes
    "effective_tariff_rate_pct": {
        "label": "Effective tariff rate (%)",
        "help": "Overall effective rate on CPO imports after exemptions/credits.",
    },
    "tariff_advalorem_pct": {
        "label": "Ad valorem duty (%)",
        "help": "Percentage duty applied on the customs/CIF base.",
    },
    "tariff_specific_inr_per_t": {
        "label": "Specific duty (₹/ton)",
        "help": "Fixed rupee amount per metric ton.",
    },

    # Global prices & FX
    "world_cpo_price_usd_per_t": {
        "label": "World CPO price (USD/ton)",
        "help": "Reference global crude palm oil price per ton (USD).",
    },
    "usd_inr": {
        "label": "USD/INR exchange rate",
        "help": "Rupees per US dollar.",
    },
    "brent_usd": {
        "label": "Brent crude (USD)",
        "help": "Benchmark crude oil price (indicator for freight/energy).",
    },

    # Export levies & freight
    "indo_export_levy_usd_per_t": {
        "label": "Indonesia export levy (USD/ton)",
        "help": "Levies/export taxes from Indonesia on palm oil exports.",
    },
    "mys_export_levy_usd_per_t": {
        "label": "Malaysia export levy (USD/ton)",
        "help": "Levies/export taxes from Malaysia on palm oil exports.",
    },
    "freight_usd_per_t": {
        "label": "Freight (USD/ton)",
        "help": "Ocean freight and related logistics per ton.",
    },
    "freight_index": {
        "label": "Freight index",
        "help": "Indexed freight proxy; higher = more expensive shipping.",
    },

    # Domestic market balance
    "domestic_supply_ton": {
        "label": "Domestic supply (tons)",
        "help": "Monthly domestic CPO availability.",
    },
    "consumption_ton": {
        "label": "Consumption (tons)",
        "help": "Monthly domestic CPO consumption.",
    },
    "imports_ton": {
        "label": "Imports (tons)",
        "help": "Monthly CPO imports.",
    },

    # State/NMEO-OP (if any of these appear in expected inputs)
    "mature_ha": {
        "label": "Mature oil palm area (ha)",
        "help": "Mature planted area under bearing.",
    },
    "immature_ha": {
        "label": "Immature oil palm area (ha)",
        "help": "Young area not yet bearing.",
    },
    "yield_ffb_ton_per_ha_month": {
        "label": "FFB yield (t/ha/month)",
        "help": "Fresh fruit bunches yield per hectare per month.",
    },
    "ffb_prod_ton": {
        "label": "FFB production (tons)",
        "help": "Fresh fruit bunches produced.",
    },
    "cpo_oil_prod_ton": {
        "label": "CPO production (tons)",
        "help": "Domestic crude palm oil output.",
    },
    "subsidy_intensity": {
        "label": "Subsidy intensity (index)",
        "help": "Relative strength of subsidy support in the state.",
    },
    "rain_anom": {
        "label": "Rainfall anomaly (σ)",
        "help": "Standardized rainfall deviation; + = wetter than normal.",
    },
    "heat_anom": {
        "label": "Heat anomaly (σ)",
        "help": "Standardized temperature deviation; + = hotter than normal.",
    },
}

def pretty_label(name: str) -> str:
    """Return a clean, human label for any feature name."""
    if name in FIELD_META:
        return FIELD_META[name]["label"]
    # Generic fallback: turn snake_case into Title Case and expand common suffixes
    label = name.replace("_per_t", " (per ton)").replace("_usd", " (USD)")
    label = label.replace("_usd_per_t", " (USD/ton)").replace("_inr", " (₹)")
    label = label.replace("_pct", " (%)").replace("_ha", " (ha)")
    label = label.replace("_ton", " (tons)")
    label = label.replace("_", " ").strip().title()
    return label

def field_help(name: str) -> str:
    return FIELD_META.get(name, {}).get("help", "")



# -------------------------
# Helpers (unchanged + new)
# -------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path: Path):
    """Load a model from .pkl (joblib) or .skops (skops)."""
    suffix = model_path.suffix.lower()
    if suffix == ".skops":
        if skops_load is None:
            st.error("Model is .skops but 'skops' is not installed. pip install skops")
            st.stop()
        return skops_load(str(model_path), trusted=True)
    else:
        import joblib
        return joblib.load(model_path)

@st.cache_resource(show_spinner=False)
def load_metadata(meta_path: Path) -> Dict[str, Any]:
    with open(meta_path, "r") as f:
        return json.load(f)

def parse_float_or_none(txt: str) -> Optional[float]:
    txt = (txt or "").strip()
    if txt == "":
        return None
    try:
        return float(txt)
    except ValueError:
        return None

def make_one_row_df(inputs_dict: Dict[str, Any], expected_cols):
    row = {col: inputs_dict.get(col, None) for col in expected_cols}
    return pd.DataFrame([row])

def is_number(x) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False

def nan_to_none(x):
    return None if pd.isna(x) else x

# Derived economics ---------------------------------
def landed_cost_inr_per_t(row: pd.Series) -> Optional[float]:
    """
    Stylized landed cost builder for decomposition & policy deltas.
    Uses columns if present; otherwise returns None.
    """
    req = ["world_cpo_price_usd_per_t", "usd_inr"]
    if not all(c in row.index for c in req):
        return None
    world = row.get("world_cpo_price_usd_per_t", np.nan)
    usd_inr = row.get("usd_inr", np.nan)
    indo_levy = row.get("indo_export_levy_usd_per_t", 0.0) or 0.0
    mys_levy = row.get("mys_export_levy_usd_per_t", 0.0) or 0.0
    freight_usd = row.get("freight_usd_per_t", 0.0) or 0.0

    adval = (row.get("tariff_advalorem_pct", 0.0) or 0.0) / 100.0
    spec_inr = row.get("tariff_specific_inr_per_t", 0.0) or 0.0

    base_usd = (world or 0.0) + (indo_levy or 0.0) + (mys_levy or 0.0) + (freight_usd or 0.0)
    base_inr = base_usd * (usd_inr or 0.0)
    adval_inr = base_inr * adval
    landed = base_inr + adval_inr + spec_inr
    return float(landed)

def tariff_revenue_inr_per_t(row: pd.Series) -> Optional[float]:
    # On a per-ton basis (stylized)
    if "usd_inr" not in row.index or "world_cpo_price_usd_per_t" not in row.index:
        return None
    world = row.get("world_cpo_price_usd_per_t", 0.0) or 0.0
    usd_inr = row.get("usd_inr", 0.0) or 0.0
    indo_levy = row.get("indo_export_levy_usd_per_t", 0.0) or 0.0
    mys_levy  = row.get("mys_export_levy_usd_per_t", 0.0) or 0.0
    freight   = row.get("freight_usd_per_t", 0.0) or 0.0
    adval = (row.get("tariff_advalorem_pct", 0.0) or 0.0) / 100.0
    spec_inr = row.get("tariff_specific_inr_per_t", 0.0) or 0.0
    base_inr = (world + indo_levy + mys_levy + freight) * usd_inr
    return float(base_inr * adval + spec_inr)

def import_dependency(consumption_ton: float, domestic_supply_ton: float) -> Optional[float]:
    if consumption_ton is None or domestic_supply_ton is None:
        return None
    if consumption_ton <= 0:
        return None
    gap = max(0.0, consumption_ton - domestic_supply_ton)
    return 100.0 * gap / consumption_ton

def kpi_card_grid(kpis: List[Tuple[str, Optional[float], str]]):
    c1, c2, c3, c4, c5 = st.columns(5)
    cols = [c1, c2, c3, c4, c5]
    for i, (label, val, suffix) in enumerate(kpis[:5]):
        with cols[i]:
            if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                st.metric(label, value="—")
            else:
                if suffix:
                    st.metric(label, value=f"{val:,.2f} {suffix}")
                else:
                    st.metric(label, value=f"{val:,.2f}")

# Some commonly entered fields to highlight first
PRIORITY_FIELDS = [
    "effective_tariff_rate_pct",
    "tariff_advalorem_pct",
    "tariff_specific_inr_per_t",
    "usd_inr",
    "world_cpo_price_usd_per_t",
]

# -------------------------
# Page
# -------------------------
st.set_page_config(page_title="🌴 CPO Tariff Policy Simulator", layout="wide")
st.title("🌴 CPO Tariff Policy Simulator")

with st.sidebar:
    st.header("⚙️ Model & Data")
    default_dir = Path("model_artifacts")
    model_path = st.text_input(
        "Model path (.pkl or .skops)",
        value=str(default_dir / "retail_inr_per_t_gbr_pipeline.pkl"),
    )
    meta_path = st.text_input(
        "Metadata path (.json)",
        value=str(default_dir / "model_metadata.json"),
    )
    st.markdown("---")
    st.caption("Tip: keep your **scikit-learn** version same as training (e.g., 1.6.1).")

    st.subheader("📦 Datasets (optional)")
    national_default = "national_monthly.csv"
    states_default = "panel_state_monthly.csv"
    nat_path = st.text_input("National monthly CSV", value=national_default)
    state_path = st.text_input("State panel CSV", value=states_default)

    st.markdown("---")
    st.subheader("🔌 Data Freshness")
    st.caption("Live connectors can be wired here later. For now we display file modified times.")
    try:
        nat_mtime = Path(nat_path).stat().st_mtime
        st.write(f"National: {pd.to_datetime(nat_mtime, unit='s')}")
    except Exception:
        st.write("National: —")
    try:
        st_mtime = Path(state_path).stat().st_mtime
        st.write(f"States: {pd.to_datetime(st_mtime, unit='s')}")
    except Exception:
        st.write("States: —")

# Load model + metadata
model_file = Path(model_path)
meta_file = Path(meta_path)

if not model_file.exists():
    st.warning(f"Model file not found: {model_file}")
if not meta_file.exists():
    st.warning(f"Metadata file not found: {meta_file}")

model = None
meta: Dict[str, Any] = {}
expected: List[str] = []
target: str = "retail_inr_per_t"

if model_file.exists() and meta_file.exists():
    try:
        model = load_model(model_file)
        meta = load_metadata(meta_file)
        expected = meta.get("expected_inputs", [])
        target = meta.get("target", target)
    except Exception as e:
        st.error(f"Failed to load model/metadata: {e}")
        st.stop()

# Load data (best-effort)
def safe_read_csv(p: str, parse_dates: List[str] = None) -> Optional[pd.DataFrame]:
    if not p:
        return None
    try:
        return pd.read_csv(p, parse_dates=parse_dates)
    except Exception:
        return None

national = safe_read_csv(nat_path, parse_dates=["date"])
states = safe_read_csv(state_path, parse_dates=["date"])

# -------------------------
# Tabs
# -------------------------
tabs = st.tabs(["🧮 Simulate", "📈 History", "🗺️ States (NMEO-OP)", "🔎 Backtest", "📜 Model Card"])

# =========================
# 1) SIMULATE TAB
# =========================
with tabs[0]:

    st.subheader("Enter inputs (any subset, missing will be imputed)")
    col_left, col_right = st.columns(2)
    session_defaults = {
        "effective_tariff_rate_pct": "15.0",
        "tariff_advalorem_pct": "7.5",
        "tariff_specific_inr_per_t": "5000",
        "usd_inr": "83.2",
        "world_cpo_price_usd_per_t": "980",
    }
    if "inputs" not in st.session_state:
        st.session_state["inputs"] = {k: v for k, v in session_defaults.items()}

    inputs: Dict[str, str] = st.session_state["inputs"]
    shown = set()
    for i, name in enumerate([f for f in PRIORITY_FIELDS if f in expected]):
        col = col_left if i % 2 == 0 else col_right
        inputs[name] = col.text_input(pretty_label(name), value=inputs.get(name, ""), placeholder="leave blank if unknown", help= field_help(name))
        shown.add(name)

    with st.expander("More features (optional crop/supply/macros)"):
        cols = st.columns(3)
        idx = 0
        for name in expected:
            if name in shown:
                continue
            col = cols[idx % 3]
            inputs[name] = col.text_input(pretty_label(name), value=inputs.get(name, ""), placeholder="leave blank if unknown", help= field_help(name))
            idx += 1

    st.markdown("---")
    # ---------- Single prediction ----------
    c1, c2 = st.columns([1, 2])
    if c1.button("🔮 Predict single"):
        parsed = {k: parse_float_or_none(v) for k,v in inputs.items()}
        X = make_one_row_df(parsed, expected)
        try:
            yhat = float(model.predict(X)[0]) if model is not None else np.nan
            landed = landed_cost_inr_per_t(pd.Series(parsed))
            t_revenue = tariff_revenue_inr_per_t(pd.Series(parsed))

            c2.success(f"**Predicted {target}**: {yhat:,.2f} INR/ton" if is_number(yhat) else "Prediction unavailable")
            with st.expander("Inputs used"):
                st.json({k: nan_to_none(parsed.get(k)) for k in expected})

            # KPI cards
            kpi_card_grid([
                ("Predicted retail", yhat if is_number(yhat) else None, "₹/t"),
                ("Landed cost (stylized)", landed, "₹/t"),
                ("Tariff revenue (per ton)", t_revenue, "₹/t"),
                ("Ad valorem", parse_float_or_none(inputs.get("tariff_advalorem_pct")), "%"),
                ("Specific duty", parse_float_or_none(inputs.get("tariff_specific_inr_per_t")), "₹/t"),
            ])

            # Waterfall (price decomposition)
            with st.expander("Price decomposition (waterfall)"):
                row = pd.Series(parsed)
                world = row.get("world_cpo_price_usd_per_t", 0.0) or 0.0
                indo  = row.get("indo_export_levy_usd_per_t", 0.0) or 0.0
                mys   = row.get("mys_export_levy_usd_per_t", 0.0) or 0.0
                fr    = row.get("freight_usd_per_t", 0.0) or 0.0
                usd_inr = row.get("usd_inr", 0.0) or 0.0
                adval = (row.get("tariff_advalorem_pct", 0.0) or 0.0) / 100.0
                spec  = row.get("tariff_specific_inr_per_t", 0.0) or 0.0

                base_usd = world + indo + mys + fr
                base_inr = base_usd * usd_inr
                adval_inr = base_inr * adval

                steps = [
                    dict(label="World CPO (USD/t)", value=world * usd_inr),
                    dict(label="+ Indo levy", value=indo * usd_inr),
                    dict(label="+ MY levy", value=mys * usd_inr),
                    dict(label="+ Freight", value=fr * usd_inr),
                    dict(label="+ Ad valorem", value=adval_inr),
                    dict(label="+ Specific", value=spec),
                ]
                # Build cumulative waterfall
                fig = go.Figure(go.Waterfall(
                    orientation="v",
                    measure=["relative"] * len(steps),
                    x=[s["label"] for s in steps],
                    text=[f"{s['value']:,.0f}" for s in steps],
                    y=[s["value"] for s in steps],
                ))
                fig.update_layout(title="Stylized landed cost (INR/ton)", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            # Sensitivity (tornado)
            with st.expander("Sensitivity (±10%) on key drivers"):
                base_row = pd.Series(parsed)
                base_pred = float(model.predict(make_one_row_df(parsed, expected))[0]) if model is not None else np.nan

                def bump(name, pct):
                    bumped = base_row.copy()
                    if name in bumped.index and bumped[name] not in (None, "", np.nan):
                        bumped[name] = float(bumped[name]) * (1 + pct)
                    Xb = make_one_row_df(dict(bumped), expected)
                    try:
                        return float(model.predict(Xb)[0])
                    except Exception:
                        return np.nan

                drivers = [c for c in ["world_cpo_price_usd_per_t","usd_inr","tariff_advalorem_pct","tariff_specific_inr_per_t","freight_usd_per_t"] if c in expected]
                data = []
                for d in drivers:
                    hi = bump(d, +0.10)
                    lo = bump(d, -0.10)
                    data.append({"driver": d, "low": lo - base_pred, "high": hi - base_pred})
                tor = pd.DataFrame(data).dropna()
                if not tor.empty and is_number(base_pred):
                    figt = go.Figure()
                    for _, r in tor.iterrows():
                        figt.add_trace(go.Bar(x=[r["low"]], y=[r["driver"]], orientation="h", name="-10%"))
                        figt.add_trace(go.Bar(x=[r["high"]], y=[r["driver"]], orientation="h", name="+10%"))
                    figt.update_layout(barmode="overlay", title="Tornado: change in predicted retail (₹/t)")
                    st.plotly_chart(figt, use_container_width=True)
                else:
                    st.info("Not enough info to compute sensitivity.")

        except Exception as e:
            c2.error(f"Prediction failed: {e}")

    st.markdown("---")

    # ---------- Scenario builder ----------
    st.subheader("🧪 Scenario Builder (compare multiple)")
    st.caption("Edit the table; click **Run scenarios** to score each row. Include any subset of features.")
    # Seed with a baseline row from the single-inputs box
    baseline = {k: parse_float_or_none(v) for k,v in inputs.items()}
    scenario_cols = [c for c in expected]  # preserve model order
    seed = pd.DataFrame([{c: baseline.get(c, np.nan) for c in scenario_cols}])
    if "scenario_df" not in st.session_state:
        st.session_state["scenario_df"] = seed.copy()
    scenario_df = st.data_editor(
        st.session_state["scenario_df"],
        num_rows="dynamic",
        use_container_width=True,
        key="scenario_editor"
    )
    run_col1, run_col2 = st.columns([1,3])
    if run_col1.button("▶️ Run scenarios"):
        df_in = scenario_df.copy()
        # Ensure expected columns exist
        for col in expected:
            if col not in df_in.columns:
                df_in[col] = np.nan
        df_in = df_in[expected]
        try:
            preds = model.predict(df_in) if model is not None else np.full(len(df_in), np.nan)
            out = df_in.copy()
            out[f"pred_{target} (₹/t)"] = preds
            # Derived KPIs (per row)
            landed_list, tariff_rev_list = [], []
            for _, r in out.iterrows():
                landed_list.append(landed_cost_inr_per_t(r))
                tariff_rev_list.append(tariff_revenue_inr_per_t(r))
            out["landed_cost (₹/t)"] = landed_list
            out["tariff_revenue (₹/t)"] = tariff_rev_list
            st.dataframe(out, use_container_width=True)

            # Scenario bars
            try:
                figsc = px.bar(out.reset_index().rename(columns={"index":"scenario"}),
                               x="scenario", y=f"pred_{target} (₹/t)",
                               title="Predicted retail by scenario")
                st.plotly_chart(figsc, use_container_width=True)
            except Exception:
                pass

            # Download
            st.download_button(
                "Download scenario results (CSV)",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="scenario_results.csv",
                mime="text/csv",
            )
        except Exception as e:
            run_col2.error(f"Scenario scoring failed: {e}")

    st.markdown("---")
    st.subheader("📄 Batch predict")
    st.caption("Upload a CSV with any subset of the expected columns; missing values will be imputed.")
    cbu1, cbu2 = st.columns([2,3])
    # Template download
    template = pd.DataFrame(columns=expected)
    cbu1.download_button(
        "Download template CSV",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="template_expected_columns.csv",
        mime="text/csv",
    )
    up = cbu2.file_uploader("Upload CSV for batch predictions", type=["csv"])
    if up is not None:
        try:
            df_in = pd.read_csv(up)
            # quick data quality
            missing_cols = [c for c in expected if c not in df_in.columns]
            st.info(f"Missing columns will be imputed: {missing_cols}" if missing_cols else "All expected columns present.")
            # ensure all expected columns exist
            for col in expected:
                if col not in df_in.columns:
                    df_in[col] = np.nan
            df_in = df_in[expected]
            preds = model.predict(df_in) if model is not None else np.full(len(df_in), np.nan)
            out = df_in.copy()
            out[f"pred_{target}"] = preds
            st.dataframe(out, use_container_width=True)
            st.download_button(
                "Download predictions CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")

# =========================
# 2) HISTORY TAB
# =========================
with tabs[1]:
    st.subheader("Market context & import dependency")
    if national is None or national.empty:
        st.info("National dataset not available.")
    else:
        # Import dependency
        df = national.copy()
        if "date" in df.columns:
            df = df.sort_values("date")
        if {"consumption_ton","domestic_supply_ton"}.issubset(df.columns):
            df["import_gap_ton"] = (df["consumption_ton"] - df["domestic_supply_ton"]).clip(lower=0)
            df["import_dependency_pct"] = 100 * df["import_gap_ton"] / df["consumption_ton"].replace(0, np.nan)
            fig = px.line(df, x="date", y="import_dependency_pct", title="Import dependency (%) – monthly")
            st.plotly_chart(fig, use_container_width=True)

        # World CPO, FX, levies, freight
        cols_to_plot = [
            ("world_cpo_price_usd_per_t","World CPO (USD/t)"),
            ("usd_inr","USD/INR"),
            ("indo_export_levy_usd_per_t","Indonesia export levy (USD/t)"),
            ("mys_export_levy_usd_per_t","Malaysia export levy (USD/t)"),
            ("freight_usd_per_t","Freight (USD/t)"),
            ("brent_usd","Brent crude (USD)"),
        ]
        grid = st.columns(2)
        for i, (cname, title) in enumerate(cols_to_plot):
            if cname in national.columns:
                with grid[i % 2]:
                    fig2 = px.line(national, x="date", y=cname, title=title)
                    st.plotly_chart(fig2, use_container_width=True)

# =========================
# 3) STATES TAB
# =========================
with tabs[2]:
    st.subheader("NMEO-OP progress")
    if states is None or states.empty:
        st.info("State panel dataset not available.")
    else:
        # Quick filters
        states_list = sorted(states["state"].dropna().unique().tolist())
        s1, s2 = st.columns(2)
        pick = s1.selectbox("State", options=["(All)"] + states_list, index=0)
        metric = s2.selectbox("Metric", options=["mature_ha","immature_ha","yield_ffb_ton_per_ha_month","cpo_oil_prod_ton","subsidy_intensity"])
        df = states.copy()
        if pick != "(All)":
            df = df[df["state"] == pick]
        df = df.sort_values("date")
        fig = px.line(df, x="date", y=metric, color=None if pick!="(All)" else "state",
                      title=f"{metric} over time")
        st.plotly_chart(fig, use_container_width=True)

        # Top movers (last 12 months) in mature area
        with st.expander("Top movers in mature_ha (last 12 months)"):
            last_dt = states["date"].max()
            if pd.notna(last_dt):
                cut = last_dt - pd.DateOffset(months=12)
                base = states[states["date"] <= cut].sort_values("date").groupby("state")["mature_ha"].last()
                latest = states.sort_values("date").groupby("state")["mature_ha"].last()
                movers = (latest - base).dropna().sort_values(ascending=False).reset_index()
                movers.columns = ["state","Δ mature_ha (12m)"]
                st.dataframe(movers.head(15), use_container_width=True)

# =========================
# 4) BACKTEST TAB
# =========================
with tabs[3]:
    st.subheader("Rolling backtest (fit window per your metadata)")
    bt = meta.get("backtest", {})
    if not bt:
        st.info("No backtest object in metadata. Add metrics to `model_metadata.json` (e.g., RMSE/MAPE by year).")
    else:
        if "by_year" in bt:
            d = pd.DataFrame(bt["by_year"])
            st.dataframe(d, use_container_width=True)
            if {"year","mape"}.issubset(d.columns):
                fig = px.bar(d, x="year", y="mape", title="MAPE by year")
                st.plotly_chart(fig, use_container_width=True)
        if "overall" in bt:
            st.write("**Overall metrics**")
            st.json(bt["overall"])

# =========================
# 5) MODEL CARD TAB
# =========================
with tabs[4]:
    st.subheader("Model Card")
    st.write("**Target**:", target)
    st.write("**Expected inputs**:", expected)
    if "training_window" in meta:
        st.write("**Training window**:", meta["training_window"])
    if "imputation" in meta:
        st.write("**Imputation**:", meta["imputation"])
    if "notes" in meta:
        st.write("**Notes**:", meta["notes"])

    # Feature importance
    with st.expander("Feature importance"):
        if model is not None and hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=expected).sort_values(ascending=False)
            fig = px.bar(imp.head(25), title="Top feature importances")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(imp.rename("importance"), use_container_width=True)
        else:
            st.info("Model does not expose feature_importances_. Skipping.")

    st.caption("Add uncertainty bands by training quantile models or bootstraps; wire external APIs for real-time prices/FX; and append policy regime markers for event studies.")

