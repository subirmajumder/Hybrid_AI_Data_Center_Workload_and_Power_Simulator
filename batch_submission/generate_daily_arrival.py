# -*- coding: utf-8 -*-
"""
NB2 Daily Arrivals Generator (from stored mu_template + alpha)
@author: Subir Majumder

Model format (pickle):
  - alpha: float
  - mu_template: DataFrame with columns:
      ["cluster", "daytype", "wom", "average_mu_hat"]

  1. Builds future (date * cluster) panel from start_date for n_days
  2. Adds daytype (Weekday/Weekend) and wom (week-of-month capped at 4)
  3. Looks up mean surface mu via mu_template
  4. Optionally applies per-group multipliers to mu
       (low/mid/high -> cluster ids 2/0/1)
  5. Splits mu equally across n_timezoness
  6. Simulates NB2 counts per (date, cluster, timezone) with alpha
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd


def generate_daily_batch_arrival_counts(
    model_path,
    start_date,
    n_days,
    multipliers=None,
    n_timezones=1,
):
    """
    Generates daily arrival counts per (date, cluster, timezone) using stored NB2 mean template.
    
    If n_timezones > 1:
      - Splits mean equally across timezones: mu_tz = mu / n_timezones
      - Uses alpha_tz = alpha so that each region uses original NB2(mu, alpha) distribution.
    """

    SEED = 2025
    
    # Unpack the model
    with open(model_path, "rb") as f:
        pack = pickle.load(f)

    alpha = float(pack["alpha"])
    mu_template = pack["mu_template"][["cluster", "daytype", "wom", "average_mu_hat"]].copy()

    clusters = list(mu_template["cluster"].cat.categories)
    
    # Build future (date * cluster) panel
    dates = pd.date_range(start=start_date, periods=int(n_days), freq="D")
    panel = pd.MultiIndex.from_product([dates, clusters], names=["date", "cluster"]).to_frame(index=False)
    panel["daytype"] = np.where(panel["date"].dt.weekday < 5, "Weekday", "Weekend")
    panel["wom"] = ((panel["date"].dt.day - 1) // 7 + 1).clip(upper=4)

    # Generate mu based on template
    panel = panel.merge(mu_template, on=["cluster", "daytype", "wom"], how="left")
    mu_vec = panel["average_mu_hat"].to_numpy()

    # Adjust arrival rates based on multiplier
    label_to_index = {"low": 2, "mid": 0, "high": 1}    # Hardcoded
    if multipliers:
        idx_mult = {str(label_to_index[k]): float(v) for k, v in multipliers.items() if k in label_to_index}
        mu_vec *= panel["cluster"].astype(str).map(idx_mult).fillna(1.0).to_numpy()
    nd, nc = len(dates), len(clusters)
    mu = mu_vec.reshape(nd, nc)

    # timezone split
    n_timezones = int(n_timezones)
    mu_tz = mu / n_timezones
    alpha_tz = alpha

    r_tz = 1.0 / alpha_tz
    p_tz = r_tz / (r_tz + mu_tz)
    p_tz = np.clip(p_tz, 1e-12, 1 - 1e-12)

    rng = np.random.default_rng(int(SEED))
    Y_tz = rng.negative_binomial(n=r_tz, p=p_tz, size=(n_timezones, nd, nc))

    # output with timezone column
    rows = []
    for tz in range(n_timezones):
        df_tz = pd.DataFrame(Y_tz[tz], index=dates, columns=[str(c) for c in clusters])
        df_tz.index.name = "date"
        long_tz = df_tz.reset_index().melt(id_vars="date", var_name="cluster_id", value_name="count")
        long_tz["timezone"] = tz
        rows.append(long_tz)

    out_df = pd.concat(rows, ignore_index=True)

    return out_df


if __name__ == "__main__":
    
    # Defaults
    here = Path(__file__).resolve()
    root = here.parent.parent

    bundle = root / "ProcessedData_Models" / "Batch" / "nb_time_mu_alpha_bundle.pkl"
    out_csv = root / "Temp" / "predicted_daily_arrivals_nb2.csv"

    multipliers = {"low": 1.01, "mid": 0.90, "high": 1.05}

    df = generate_daily_batch_arrival_counts(
        model_path=bundle,
        start_date="2024-01-01",
        n_days=365,
        multipliers=multipliers,
        n_timezones=6,
    )
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    print(df.head())
    
    # total per day per timezone
    import matplotlib.pyplot as plt

    daily_tz = (
        df.groupby(["date", "timezone"], as_index=False)["count"]
          .sum()
          .sort_values(["timezone", "date"])
    )
    
    plt.figure(figsize=(10, 4))
    for tz, g in daily_tz.groupby("timezone"):
        plt.plot(g["date"], g["count"], marker="o", markersize=2, linewidth=1, label=f"TZ {tz}")
    
    plt.xlabel("Date")
    plt.ylabel("Total daily arrivals")
    plt.title("Simulated Total Daily Arrivals (All Clusters) by Timezone")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
