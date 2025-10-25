# ============================================
# 3) DIRECT-INPUT PREDICTION (load & predict)
# ============================================
import json
import joblib
import pandas as pd
from pathlib import Path

MODEL_DIR = Path("./model_artifacts")
MODEL_PATH = MODEL_DIR / "retail_inr_per_t_gbr_pipeline.pkl"  # adjust if you changed target
META_PATH  = MODEL_DIR / "model_metadata.json"

# ---- A) In-notebook function ----
def predict_price_from_inputs(payload: dict):
    """
    payload: dict of ANY subset of expected features, e.g.:
      {
        "effective_tariff_rate_pct": 15.0,
        "tariff_advalorem_pct": 7.5,
        "tariff_specific_inr_per_t": 5000,
        "usd_inr": 83.0,
        "world_cpo_price_usd_per_t": 980,
        # optional crop details to refine prediction (if you know them):
        "nat_mature_ha": 250000,
        "nat_cpo_prod_ton": 130000,
        # optional supply/demand:
        "imports_ton": 600000
      }
    Any missing inputs are imputed by the model pipeline.
    """
    model = joblib.load(MODEL_PATH)
    meta = json.load(open(META_PATH, "r"))
    expected = meta["expected_inputs"]

    row = {col: payload.get(col, None) for col in expected}
    X = pd.DataFrame([row])
    yhat = model.predict(X)[0]
    return {
        "target": meta["target"],
        "prediction": float(yhat),
        "units": "INR per ton",
        "used_features": expected,
        "provided_keys": list(payload.keys())
    }

# Example: minimal inputs (you can add more details)
example = {
    "effective_tariff_rate_pct": 15.0,
    "tariff_advalorem_pct": 7.5,
    "tariff_specific_inr_per_t": 5000,
    "usd_inr": 83.0,
    "world_cpo_price_usd_per_t": 980
}
print(predict_price_from_inputs(example))

# ---- B) Save a simple CLI script (optional) ----
cli_path = MODEL_DIR / "predict_price.py"
cli_code = f"""\
import sys, json
import joblib
import pandas as pd

MODEL_PATH = "{MODEL_PATH.name}"
META_PATH  = "model_metadata.json"

def main():
    # Usage:
    #   python predict_price.py '{{"effective_tariff_rate_pct": 15.0, "usd_inr": 83.0, "world_cpo_price_usd_per_t": 980}}'
    if len(sys.argv) > 1:
        payload = json.loads(sys.argv[1])
    else:
        payload = json.load(sys.stdin)

    model = joblib.load(MODEL_PATH)
    meta = json.load(open(META_PATH, "r"))
    expected = meta["expected_inputs"]

    row = {{col: payload.get(col, None) for col in expected}}
    X = pd.DataFrame([row])
    yhat = model.predict(X)[0]

    out = {{
        "target": meta["target"],
        "prediction": float(yhat),
        "units": "INR per ton",
        "used_features": expected,
        "provided_keys": list(payload.keys())
    }}
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
"""
cli_path.write_text(cli_code)
print("CLI saved to:", cli_path.resolve())
