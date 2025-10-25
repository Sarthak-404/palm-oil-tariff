import sys, json
import joblib
import pandas as pd

MODEL_PATH = "retail_inr_per_t_gbr_pipeline.pkl"
META_PATH  = "model_metadata.json"

def main():
    # Usage:
    #   python predict_price.py '{"effective_tariff_rate_pct": 15.0, "usd_inr": 83.0, "world_cpo_price_usd_per_t": 980}'
    if len(sys.argv) > 1:
        payload = json.loads(sys.argv[1])
    else:
        payload = json.load(sys.stdin)

    model = joblib.load(MODEL_PATH)
    meta = json.load(open(META_PATH, "r"))
    expected = meta["expected_inputs"]

    row = {col: payload.get(col, None) for col in expected}
    X = pd.DataFrame([row])
    yhat = model.predict(X)[0]

    out = {
        "target": meta["target"],
        "prediction": float(yhat),
        "units": "INR per ton",
        "used_features": expected,
        "provided_keys": list(payload.keys())
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
