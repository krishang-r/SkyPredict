#!/usr/bin/env python3
"""
Load the joblib pipeline (flight_price_model.joblib), extract feature importances,
aggregate one-hot encoded columns back to original features and plot a ranked bar chart.

Usage:
  cd /Users/krishangratra/Documents/Coding/SkyPredict/SkyPredict/
  python3 scripts/plot_feature_importance_from_model.py
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE = Path(__file__).resolve().parent
# Try multiple common locations and env var for robustness
CANDIDATE_NAMES = [
    BASE.parent.joinpath("model", "flight_price_model.joblib"),   # my-app/model/...
    BASE.parent.joinpath("flight_price_model.joblib"),            # my-app/flight_price_model.joblib
    BASE.parent.parent.joinpath("model", "flight_price_model.joblib"),  # repo_root/model/...
    BASE.parent.parent.joinpath("flight_price_model.joblib"),           # repo_root/flight_price_model.joblib
]

# allow override via environment variable MODEL_PATH
env_path = os.environ.get("MODEL_PATH")
if env_path:
    CANDIDATE_NAMES.insert(0, Path(env_path))

MODEL_PATH = None
for p in CANDIDATE_NAMES:
    if p and p.exists():
        MODEL_PATH = p
        break

if MODEL_PATH is None:
    candidate_str = "\n  ".join(str(p) for p in CANDIDATE_NAMES)
    raise SystemExit(
        "Model not found. Looked for flight_price_model.joblib in these locations:\n  "
        + candidate_str
        + "\n\nTrain the model and save it to one of these locations, or set the MODEL_PATH environment variable."
    )

# --- NEW: define output paths used later ---
OUT_DIR = BASE.parent.joinpath("report", "images")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = OUT_DIR.joinpath("feature_importance_from_model.png")
OUT_CSV = OUT_DIR.joinpath("feature_importance_from_model.csv")

# Load pipeline
pipeline = joblib.load(MODEL_PATH)

# Expect structure: Pipeline(steps=[('preprocessor', ColumnTransformer(...)), ('regressor', RandomForestRegressor(...))])
try:
    preprocessor = pipeline.named_steps["preprocessor"]
    reg = pipeline.named_steps["regressor"]
except Exception as e:
    raise SystemExit("Expected pipeline with named steps 'preprocessor' and 'regressor'. Error: " + str(e))

# Get feature importances
if not hasattr(reg, "feature_importances_"):
    raise SystemExit("Regressor does not expose feature_importances_. Use a tree-based regressor (RandomForest, XGBoost, etc.).")
importances = reg.feature_importances_

# Build feature names in same order as transformer output
# Handle ColumnTransformer with two transformers: ('num','passthrough', numeric_features), ('cat', OneHotEncoder, categorical_features)
feature_names = []
# transformers_ is created after fitting
for name, trans, cols in preprocessor.transformers_:
    if trans == "drop":
        continue
    if trans == "passthrough":
        # numeric passthrough (cols may be a slice or list)
        if isinstance(cols, (list, tuple)):
            feature_names.extend(list(cols))
        else:
            # fallback: get from pipeline input if available
            feature_names.append(str(cols))
    else:
        # For transformers like OneHotEncoder, get feature names from transformer object
        try:
            # Many sklearn versions: transformer has get_feature_names_out
            names_out = trans.get_feature_names_out(cols)
        except Exception:
            # older versions: try attribute categories_
            if hasattr(trans, "categories_"):
                names = []
                for c, col in zip(trans.categories_, cols):
                    names.extend([f"{col}_{val}" for val in c])
                names_out = names
            else:
                # last-resort: use cols as-is
                names_out = [str(c) for c in cols]
        feature_names.extend(list(names_out))

if len(feature_names) != len(importances):
    # try an alternative: preprocessor.get_feature_names_out()
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        # fallback: generate generic names
        feature_names = [f"f{i}" for i in range(len(importances))]

# Aggregate importances by original feature
# For OHE columns produce prefix mapping: split on '_' or '=' and take left part if it matches an original categorical feature
agg = {}
orig_features = []

# Attempt to get original columns from preprocessor definition if available
try:
    # If ColumnTransformer was created from explicit lists, recover them
    numeric_cols = []
    categorical_cols = []
    for nm, tr, cols in preprocessor.transformers:
        if tr == "passthrough":
            numeric_cols = list(cols) if isinstance(cols, (list, tuple)) else [cols]
        else:
            # assume OneHotEncoder transformer
            categorical_cols = list(cols)
    orig_features = list(numeric_cols) + list(categorical_cols)
except Exception:
    # fallback: infer from feature name prefixes
    orig_features = []

# helper to find parent feature for a flattened feature name
def parent_feature(fname):
    # exact match
    if fname in orig_features:
        return fname
    # split on common separators
    for sep in ["_", "=", "-"]:
        if sep in fname:
            prefix = fname.split(sep)[0]
            if prefix in orig_features:
                return prefix
    # if any orig_feature is prefix of fname, choose it
    for of in orig_features:
        if fname.startswith(of):
            return of
    # fallback: treat categorical groups by splitting at first separator
    for sep in ["_", "="]:
        if sep in fname:
            return fname.split(sep)[0]
    # as last resort return fname (meaning no aggregation)
    return fname

for fname, imp in zip(feature_names, importances):
    parent = parent_feature(fname)
    agg[parent] = agg.get(parent, 0.0) + float(imp)

# Normalize to percentage
total = sum(agg.values()) or 1.0
agg_pct = {k: (v / total) * 100.0 for k, v in agg.items()}

# Create DataFrame and sort descending
df_imp = pd.DataFrame([
    {"feature": k, "importance_pct": v} for k, v in agg_pct.items()
]).sort_values("importance_pct", ascending=False).reset_index(drop=True)

# Save CSV
df_imp.to_csv(OUT_CSV, index=False)

# Plot horizontal bar chart with ranking
sns.set(style="whitegrid")
plt.figure(figsize=(8, max(4, 0.6 * len(df_imp))))
y = df_imp["feature"]
x = df_imp["importance_pct"]
plt.barh(y[::-1], x[::-1], color="#2563EB", alpha=0.9)
plt.xlabel("Importance (%)")
plt.title("Feature Importance (aggregated)")
for i, val in enumerate(x[::-1]):
    plt.text(val + 0.4, i, f"{val:.1f}%", va="center", fontsize=9)
plt.xlim(0, max(x) * 1.15)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
plt.close()

print("Wrote:", OUT_PNG)
print("Wrote:", OUT_CSV)
print(df_imp.to_string(index=False))