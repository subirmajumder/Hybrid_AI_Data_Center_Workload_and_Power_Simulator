# -*- coding: utf-8 -*-
"""
For daily totals per cluster generate hourly-bin split (LN(diag)-ALR model) then uniform thinning to SECOND resolution.
@author: Subir Majumder

Model format (pickle):
    clusters -->
          1. ref
          2. mask
          3. mu
          4. var
          
    1. Input: daily totals CSV (columns: date, cluster_id, count, timezone).
    2. For each (cluster, timezone):
        - samples daily bin-probabilities W_day from stored LN-ALR params (mu, var, ref, mask)
        - splits each day's total N into bins via multinomial(N, W_day[d])
        - thins uniformly to seconds within each bin to produce event timestamps
        - shifts timestamps by `timezone` hours
    3. Returns a DataFrame [time_submit, cluster_id] (the caller saves it).
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

SEED = 2005

def alr_inv(U, ref, mask):
    """ Inverts ALR coordinates back to simplex:
        exponentiate the non-ref parts;
        set ref part to 1;
        normalize rows to sum to 1."""
    U = np.asarray(U, float)
    n = U.shape[0]
    Tloc = mask.size  # mask is length T
    Y = np.exp(U)

    W = np.zeros((n, Tloc), dtype=float)
    W[:, mask] = Y
    W[:, ref] = 1.0
    return W / (W.sum(axis=1, keepdims=True))


def sample_ln_diag(mu, var, ref, mask, n_days):
    """
    Sample W probabilities from Logistic-Normal (diag covariance) in ALR space.
    mu,var live in ALR coords of dimension (T-1). alr_inv maps back to simplex of length T.
    """
    rng = np.random.default_rng(SEED)
    U = mu + rng.normal(size=(n_days, mu.size)) * np.sqrt(var)
    return alr_inv(U, ref, mask)


def daily_to_second_events(
    daily_csv_path,
    hourly_model_path,
):
    """
    Convert daily job counts to second-resolution submit timestamps.
    """

    rng = np.random.default_rng(SEED)
    
    # Unpack the model    
    with open(hourly_model_path, "rb") as f:
        model = pickle.load(f)

    # Preprocess daily arrivals
    daily = pd.read_csv(daily_csv_path, parse_dates=["date"])
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()

    time_parts = []
    cluster_parts = []

    for (cid, tz), g in daily.groupby(["cluster_id", "timezone"], sort=True):
        cid = int(cid)
        tz = int(tz)

        m = model["clusters"][cid]
        ref = int(m["ref"])
        mask = np.asarray(m["mask"], dtype=bool)
        mu = np.asarray(m["mu"], dtype=float)
        var = np.asarray(m["var"], dtype=float)

        T = int(mask.size)
        bin_minutes = 1440 // T
        seconds_in_bin = bin_minutes * 60

        idx = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        N_obs = g.set_index("date")["count"].reindex(idx, fill_value=0).astype(int).to_numpy()
        D = len(N_obs)

        W_day = sample_ln_diag(mu, var, ref, mask, D)  # (D, T)
        X_sim = np.vstack([rng.multinomial(int(N_obs[d]), W_day[d]) for d in range(D)])  # (D, T)

        # thinning to seconds, then shift by timezone hours
        for d in np.flatnonzero(N_obs > 0):
            day_start = pd.Timestamp(idx[d]).to_datetime64()

            bin_ids = np.repeat(np.arange(T, dtype=np.int64), X_sim[d].astype(np.int64))
            N = int(bin_ids.size)

            sec_offsets = rng.integers(0, seconds_in_bin, size=N, dtype=np.int64)
            base_offsets = bin_ids * seconds_in_bin

            times = day_start.astype("datetime64[ns]") + (base_offsets + sec_offsets).astype("timedelta64[s]")
            times = times + np.timedelta64(tz * 3600, "s")   # timezone shift: tz=1 => +1 hour, tz=2 => +2 hours

            time_parts.append(times)
            cluster_parts.append(np.full(N, cid, dtype=np.int32))

    # assemble output
    events_df = pd.DataFrame({
                "time_submit": np.concatenate(time_parts),
                "cluster_id": np.concatenate(cluster_parts),
            }).sort_values("time_submit").reset_index(drop=True)

    return events_df


if __name__ == "__main__":
    
    # Defaults
    here = Path(__file__).resolve()
    root = here.parent.parent

    daily_csv = root / "Temp" / "predicted_daily_arrivals_nb2.csv"
    hourly_model = root / "ProcessedData_Models" / "Batch" / "batch_hourlyarrival_ln_alr.pkl"
    out_csv = root / "Temp" / "synthetic_second_arrivals.csv"

    events = daily_to_second_events(
        daily_csv_path=daily_csv,
        hourly_model_path=hourly_model
    )
    events.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    print(events.head())
    
    import matplotlib.pyplot as plt
    
    events["time_submit"] = pd.to_datetime(events["time_submit"])
    events["hour"] = events["time_submit"].dt.hour
    
    T = 24
    counts_sim = events["hour"].value_counts().reindex(range(T), fill_value=0).sort_index().to_numpy()
    norm_sim = counts_sim / (counts_sim.sum() + 1E-12)
    
    centers = np.arange(T) + 0.5
    bar_w = 0.8
    
    plt.figure(figsize=(5, 4))
    plt.bar(centers, norm_sim, width=bar_w, color="#D55E00", edgecolor="none", alpha=0.90, label="Synthetic")
    
    plt.xlim(0, 24)
    plt.xticks(np.arange(0, 24, 5))
    
    plt.xlabel("Hour of day")
    plt.ylabel("Normalized count")
    
    plt.grid(True, axis="y", alpha=0.25, linewidth=0.7)
    plt.legend(frameon=True, framealpha=0.95)
    
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    plt.close()
