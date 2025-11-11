#!/usr/bin/env python3
"""
Generate EDA images:
 - correlation heatmap
 - boxplots (price by cabin and by route)
 - trend graph (avg price over time)

Looks for data at ../data/historical_prices.csv (relative to this script).
If not present, creates synthetic sample data.
Outputs PNGs to ../report/images/
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path(__file__).resolve().parent
DATA_PATH = (BASE / ".." / "data" / "historical_prices.csv").resolve()
OUT_DIR = (BASE / ".." / "report" / "images").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_or_create():
    if DATA_PATH.exists():
        print("Loading data from:", DATA_PATH)
        df = pd.read_csv(DATA_PATH, parse_dates=["date"], low_memory=False)
    else:
        print("No CSV found at", DATA_PATH, "- generating synthetic sample data.")
        rng = pd.date_range(end=pd.Timestamp.today(), periods=180)
        origins = ["DEL", "BOM", "BLR", "MAA", "CCU"]
        dests = ["MAA", "BLR", "DEL", "BOM", "CCU"]
        cabins = ["Economy", "Premium Economy", "Business"]
        rows = []
        for d in rng:
            for i in range(5):
                price = max(8000 + np.random.randn()*800 + (np.sin((d.dayofyear/30)+i)*700), 1500)
                rows.append({
                    "date": d,
                    "origin": origins[i],
                    "destination": dests[i],
                    "route": f"{origins[i]}-{dests[i]}",
                    "price": round(price + np.random.randint(-500,500), 2),
                    "cabin": np.random.choice(cabins, p=[0.75,0.15,0.1]),
                    "offers_count": max(1, int(np.random.poisson(4))),
                })
        df = pd.DataFrame(rows)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        df["date"] = pd.Timestamp.today()
    df["days_since"] = (pd.Timestamp.today().normalize() - df["date"].dt.normalize()).dt.days
    df["route"] = df.get("route", df["origin"].astype(str) + "-" + df["destination"].astype(str))
    return df

def correlation_heatmap(df):
    numeric = df[["price", "offers_count", "days_since"]].copy()
    numeric["price_log"] = np.log1p(numeric["price"])
    corr = numeric.corr()
    plt.figure(figsize=(6,5))
    sns.heatmap(corr, annot=True, cmap="vlag", center=0, fmt=".2f")
    plt.title("Correlation Heatmap (numeric features)")
    p = OUT_DIR / "correlation_heatmap.png"
    plt.tight_layout()
    plt.savefig(p, dpi=150)
    plt.close()
    print("Saved:", p)

def boxplots(df):
    plt.figure(figsize=(10,5))
    sns.boxplot(x="cabin", y="price", data=df, palette="Set2")
    plt.title("Price Distribution by Cabin")
    plt.xlabel("Cabin")
    plt.ylabel("Price (INR)")
    p1 = OUT_DIR / "boxplot_by_cabin.png"
    plt.tight_layout()
    plt.savefig(p1, dpi=150)
    plt.close()
    print("Saved:", p1)

    top_routes = df["route"].value_counts().nlargest(6).index.tolist()
    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x="route", y="price", data=df[df["route"].isin(top_routes)], palette="Set3", ax=ax)
    ax.set_title("Price Distribution by Top Routes")
    ax.set_xlabel("Route")
    ax.set_ylabel("Price (INR)")
    p2 = OUT_DIR / "boxplot_by_route.png"
    plt.tight_layout()
    plt.savefig(p2, dpi=150)
    plt.close()
    print("Saved:", p2)

def trend_graph(df):
    daily = df.groupby(df["date"].dt.date)["price"].mean().rename("avg_price").reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["rolling_7"] = daily["avg_price"].rolling(window=7, min_periods=1).mean()
    plt.figure(figsize=(12,5))
    plt.plot(daily["date"], daily["avg_price"], marker="o", markersize=3, label="Daily avg")
    plt.plot(daily["date"], daily["rolling_7"], color="red", linewidth=2, label="7-day rolling mean")
    plt.fill_between(daily["date"], daily["avg_price"], daily["rolling_7"], color="red", alpha=0.08)
    plt.title("Trend: Average Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Average Price (INR)")
    plt.legend()
    p = OUT_DIR / "trend_avg_price.png"
    plt.tight_layout()
    plt.savefig(p, dpi=150)
    plt.close()
    print("Saved:", p)

def main():
    df = load_or_create()
    correlation_heatmap(df)
    boxplots(df)
    trend_graph(df)
    print("All images written to:", OUT_DIR)

if __name__ == "__main__":
    main()