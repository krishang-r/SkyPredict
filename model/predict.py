#!/usr/bin/env python3
import sys
import json
import os
from datetime import datetime, date

# attempt imports
try:
    from joblib import load
except Exception as e:
    print(json.dumps({"error": f"joblib import failed: {e}"}))
    sys.exit(2)

try:
    import numpy as np
except Exception:
    np = None

try:
    import pandas as pd
except Exception as e:
    print(json.dumps({"error": f"pandas import failed: {e}"}))
    sys.exit(2)

MODEL_DIR = os.path.dirname(__file__)

def find_model_file():
    for name in ("model.joblib", "model.pkl", "model.pjoblib", "model.pkl.joblib"):
        p = os.path.join(MODEL_DIR, name)
        if os.path.isfile(p):
            return p
    for fname in os.listdir(MODEL_DIR):
        if fname.endswith(".joblib") or fname.endswith(".pkl"):
            return os.path.join(MODEL_DIR, fname)
    return None

def safe_float(v):
    try:
        return float(v)
    except Exception:
        return None

def extract_iata(v):
    if not isinstance(v, str):
        return v
    v = v.strip()
    if v.endswith(")"):
        try:
            code = v[v.rfind("(")+1:-1]
            if len(code) == 3:
                return code.upper()
        except Exception:
            pass
    if len(v) == 3 and v.isalpha():
        return v.upper()
    return v

def days_until(date_str):
    try:
        # accept either YYYY-MM-DD or ISO with time
        dt = datetime.fromisoformat(date_str)
    except Exception:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except Exception:
            return None
    return (dt.date() - date.today()).days

def parse_departure_hour(date_str):
    try:
        dt = datetime.fromisoformat(date_str)
    except Exception:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except Exception:
            return None
    return dt.hour

def build_input_df(payload, feature_names):
    # payload keys expected: origin, destination, date, cabin, current_avg_price, offers_count, airline, departure_time
    p = {k.lower(): v for k, v in (payload or {}).items()}
    row = {}
    for fname in feature_names:
        key = fname.lower().strip()
        if "days until" in key or "days_until" in key:
            row[fname] = days_until(p.get("date") or p.get("departure_date") or "")
            continue
        if "departure time" in key or "departure_time" in key or "departurehour" in key:
            # prefer explicit departure_time in payload, else use date hour
            dt = p.get("departure_time") or p.get("departuretime") or p.get("date")
            row[fname] = parse_departure_hour(dt or "")
            continue
        if "airline" in key:
            row[fname] = p.get("airline") or ""
            continue
        if "departure airport" in key or "origin" in key:
            row[fname] = extract_iata(p.get("origin") or p.get("departure_airport") or "")
            continue
        if "arrival airport" in key or "destination" in key:
            row[fname] = extract_iata(p.get("destination") or p.get("arrival_airport") or "")
            continue
        if "cabin" in key:
            row[fname] = (p.get("cabin") or "").upper()
            continue
        if key in ("current_avg_price", "avg_price", "current_avg"):
            row[fname] = safe_float(p.get("current_avg_price") or p.get("avg_price") or p.get("current_avg"))
            continue
        if key in ("offers_count", "offers", "offer_count"):
            try:
                row[fname] = int(p.get("offers_count") or p.get("offers") or 0)
            except Exception:
                row[fname] = 0
            continue
        # fallback: try to use payload value or NaN
        val = p.get(key) if key in p else p.get(fname) if fname in p else None
        if isinstance(val, str):
            n = safe_float(val)
            if n is not None:
                val = n
            else:
                val = str(val)
        if val is None:
            val = np.nan if np is not None else 0.0
        row[fname] = val
    # build DataFrame with exactly the model's column names and order
    try:
        df = pd.DataFrame([row], columns=feature_names)
    except Exception:
        # if some columns missing, create df with keys present but reindex to feature_names
        df = pd.DataFrame([row])
        df = df.reindex(columns=feature_names, fill_value=np.nan if np is not None else 0.0)
    return df

def main():
    try:
        payload = json.load(sys.stdin)
    except Exception as e:
        print(json.dumps({"error": f"invalid input JSON: {e}"}))
        sys.exit(1)

    model_path = find_model_file()
    if not model_path:
        print(json.dumps({"error": "no joblib/pkl model file found in model directory"}))
        sys.exit(3)

    try:
        model = load(model_path)
    except Exception as e:
        print(json.dumps({"error": f"failed to load model: {e}", "model_path": model_path}))
        sys.exit(4)

    # determine expected feature names
    feature_names = None
    if hasattr(model, "feature_names_in_"):
        try:
            feature_names = list(model.feature_names_in_)
        except Exception:
            feature_names = None

    if not feature_names:
        # fallback - but ideally update to your model's real features
        feature_names = ["Days Until Departure", "Airline", "Departure Airport", "Arrival Airport", "Departure Time", "Cabin"]

    try:
        df = build_input_df(payload, feature_names)
        # predict
        pred_raw = model.predict(df)
        # get scalar
        if isinstance(pred_raw, (list, tuple,)) or (hasattr(pred_raw, "shape") and getattr(pred_raw, "shape", ())[0] >= 1):
            predicted = float(pred_raw[0])
        else:
            predicted = float(pred_raw)
    except Exception as e:
        print(json.dumps({"error": f"prediction failed: {e}", "debug": {"feature_names": feature_names}}))
        sys.exit(5)

    out = {
        "predictedPrice": round(predicted, 2),
        "model_source": os.path.basename(model_path),
        "used_features": feature_names,
    }
    sys.stdout.write(json.dumps(out))
    sys.stdout.flush()

if __name__ == "__main__":
    main()