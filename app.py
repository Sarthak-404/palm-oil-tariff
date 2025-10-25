import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Optional: skops loader for version-agnostic models
try:
    from skops.io import load as skops_load
except Exception:
    skops_load = None

# -------------------------
# Helpers
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
        # joblib-based .pkl
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

# Some commonly entered fields to highlight first
PRIORITY_FIELDS = [
    "effective_tariff_rate_pct",
    "tariff_advalorem_pct",
    "tariff_specific_inr_per_t",
    "usd_inr",
    "world_cpo_price_usd_per_t",
]

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Tariff-aware Price Predictor", layout="wide")
st.title("🌴 Tariff-aware Price Prediction (INR per ton)")

with st.sidebar:
    st.header("⚙️ Model files")
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

# Try loading model + metadata
model_file = Path(model_path)
meta_file = Path(meta_path)

if not model_file.exists():
    st.warning(f"Model file not found: {model_file}")
if not meta_file.exists():
    st.warning(f"Metadata file not found: {meta_file}")

if model_file.exists() and meta_file.exists():
    try:
        model = load_model(model_file)
        meta = load_metadata(meta_file)
    except Exception as e:
        st.error(f"Failed to load model/metadata: {e}")
        st.stop()

    expected = meta.get("expected_inputs", [])
    target = meta.get("target", "retail_inr_per_t")

    st.subheader("Enter inputs")
    st.caption("You can enter **any subset** of features; missing ones will be imputed.")

    # Build a dict of inputs; first show priority fields, then the rest in an expander
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

    # Priority inputs first (spread across two columns)
    inputs: Dict[str, str] = st.session_state["inputs"]
    shown = set()
    for i, name in enumerate([f for f in PRIORITY_FIELDS if f in expected]):
        col = col_left if i % 2 == 0 else col_right
        inputs[name] = col.text_input(name, value=inputs.get(name, ""), placeholder="leave blank if unknown")
        shown.add(name)

    # All remaining expected inputs
    with st.expander("More features (optional crop/supply/macros)"):
        cols = st.columns(3)
        idx = 0
        for name in expected:
            if name in shown:
                continue
            col = cols[idx % 3]
            inputs[name] = col.text_input(name, value=inputs.get(name, ""), placeholder="leave blank if unknown")
            idx += 1

    # Example button to prefill
    if st.button("Use example values"):
        for k, v in session_defaults.items():
            if k in expected:
                inputs[k] = v
        st.experimental_rerun()

    # Predict
    st.markdown("---")
    c1, c2 = st.columns([1, 2])
    if c1.button("🔮 Predict"):
        # Parse into float-or-None
        parsed = {k: parse_float_or_none(v) for k, v in inputs.items()}

        # Build one-row DF in expected order
        X = make_one_row_df(parsed, expected)

        try:
            yhat = float(model.predict(X)[0])
            c2.success(f"**Predicted {target}**: {yhat:,.2f} INR/ton")
            with st.expander("Show raw inputs used"):
                st.json({k: parsed.get(k) for k in expected})
        except Exception as e:
            c2.error(f"Prediction failed: {e}")

    # Optional: batch predict from CSV
    st.markdown("### 📄 Batch predict (optional)")
    st.caption("Upload a CSV with any subset of the expected columns; missing values will be imputed.")
    up = st.file_uploader("Upload CSV for batch predictions", type=["csv"])
    if up is not None:
        try:
            df_in = pd.read_csv(up)
            # ensure all expected columns exist; add missing as NaN
            for col in expected:
                if col not in df_in.columns:
                    df_in[col] = np.nan
            df_in = df_in[expected]
            preds = model.predict(df_in)
            out = df_in.copy()
            out[f"pred_{target}"] = preds
            st.dataframe(out)
            st.download_button(
                "Download predictions CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")
else:
    st.info("Provide valid model + metadata paths in the sidebar to begin.")
