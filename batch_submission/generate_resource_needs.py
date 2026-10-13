# -*- coding: utf-8 -*-
"""
Attach timelimit, gpu_alloc, runtime to an existing synthetic events CSV
@author: Subir Majumder

INPUT (events CSV):
  - time_submit
  - cluster_id

MODELS (pickles):
  - batch_cluster_tl_gpu_dist.pkl
      Probability_models["P_tl_given_c"]: [cluster, timelimit, p_tl_given_c]
      Probability_models["P_g_given_ct"]: [cluster, timelimit, gpu_alloc, p_g_given_ct]
  - batch_runtime_dist.pkl
      runtime_model["quantiles"] -> q_grid
      runtime_model["leaf"][(cluster, gpu_alloc, timelimit)]["q"] -> quantile curve of log(runtime)

OUTPUT:
  events + [timelimit (minutes), gpu_alloc, runtime (seconds, capped at timelimit*60), id_job]
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

SEED = 2005


def attach_gpu_timelimit_runtime(
    events_csv_path,
    tl_gpu_model_path,
    runtime_model_path,
):
    """For each job, sample timelimit given cluster, GPU count given (cluster, timelimit),
    and runtime from the stored log-runtime quantile curve; assign unique 7-digit id_job.
    """
    rng = np.random.default_rng(SEED)

    # Unpack the model   
    with open(tl_gpu_model_path, "rb") as f:
        pm = pickle.load(f)

    P_tl_given_c = pm["P_tl_given_c"]
    P_g_given_ct = pm["P_g_given_ct"]

    with open(runtime_model_path, "rb") as f:
        runtime_model = pickle.load(f)

    q_grid = runtime_model["quantiles"]
    q_min, q_max = float(q_grid.min()), float(q_grid.max())

    # Process Arrivals
    events = pd.read_csv(events_csv_path)
    
    n_total = len(events)
    timelimit_out = np.empty(n_total, dtype=np.int64)
    gpu_out       = np.empty(n_total, dtype=np.int64)
    runtime_out   = np.empty(n_total, dtype=float)

    
    for c, idx in events.groupby("cluster_id").groups.items():
        idx = np.asarray(list(idx), dtype=np.int64)
        n = int(idx.size)

        c = np.int64(c)
        tl_df = P_tl_given_c[P_tl_given_c["cluster"] == c]
        tls   = tl_df["timelimit"].to_numpy()
        p_tls = tl_df["p_tl_given_c"].to_numpy()

        tl_draws  = rng.choice(tls, size=int(n), replace=True, p=p_tls)              # generated timelimits as a vector
        tl_counts = pd.Series(tl_draws).value_counts()

        gpu_draws_all = np.empty(n, dtype=np.int64)
        rt_draws_all  = np.empty(n, dtype=float)

        for tl, k in tl_counts.items():
            tl = np.int64(tl)

            pos = np.flatnonzero(tl_draws == tl)
            k = int(pos.size)

            g_df = P_g_given_ct[(P_g_given_ct["cluster"] == int(c)) & (P_g_given_ct["timelimit"] == int(tl))]
            gs   = g_df["gpu_alloc"].to_numpy()
            p_gs = g_df["p_g_given_ct"].to_numpy()

            g_draws = rng.choice(gs, size=int(k), replace=True, p=p_gs).astype(int)   # generated gpu counts as a vector

            u = rng.uniform(q_min, q_max, size=int(k))
            runtimes = np.empty(int(k), dtype=float)

            for i in range(int(k)):
                g = np.int64(g_draws[i])
                q_curve = runtime_model["leaf"][(c, g, tl)]["q"]
                log_runtime = np.interp(u[i], q_grid, q_curve)
                runtimes[i] = np.exp(log_runtime)
                
            runtimes = np.minimum(runtimes, tl*60)
            gpu_draws_all[pos] = g_draws.astype(np.int64)
            rt_draws_all[pos]  = runtimes

        timelimit_out[idx] = tl_draws.astype(np.int64)
        gpu_out[idx]       = gpu_draws_all
        runtime_out[idx]   = rt_draws_all
        
    out = events.copy()
    out["timelimit"] = timelimit_out
    out["gpu_alloc"] = gpu_out
    out["runtime"]   = runtime_out
    
    # Add unique 7-digit job IDs
    n_total = len(out)
    if n_total > 9_000_000:
        raise ValueError("Need unique 7-digit ids, but there are more than 9,000,000 events.")
    
    out["id_job"] = rng.choice(
        np.arange(1_000_000, 10_000_000, dtype=np.int64),
        size=n_total,
        replace=False)

    return out


if __name__ == "__main__":
    
    # Defaults
    here = Path(__file__).resolve()
    root = here.parent.parent

    events_csv = root / "Temp" / "synthetic_second_arrivals.csv"
    tl_gpu_model = root / "ProcessedData_Models" / "Batch" / "batch_cluster_tl_gpu_dist.pkl"
    runtime_model = root / "ProcessedData_Models" / "Batch" / "batch_runtime_dist.pkl"
    out_csv = root / "Temp" / "synthetic_second_arrivals_with_attrs.csv"

    df = attach_gpu_timelimit_runtime(
        events_csv_path=events_csv,
        tl_gpu_model_path=tl_gpu_model,
        runtime_model_path=runtime_model,
    )
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    print(df.head())
    
    import matplotlib.pyplot as plt

    log_syn = np.log(df["runtime"].to_numpy())
    
    # choose bin range robustly
    low  = np.percentile(log_syn, 0.5)
    high = np.percentile(log_syn, 99.5)
    bins = np.linspace(low, high, 100)
    
    counts_syn, bin_edges = np.histogram(log_syn, bins=bins)
    norm_syn = 100 * counts_syn / (counts_syn.sum() + 1e-12)
    
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bar_w   = (bin_edges[1] - bin_edges[0]) * 0.8
    
    plt.figure(figsize=(6, 4))
    plt.bar(centers, norm_syn, width=bar_w, color="#D55E00",
            edgecolor="black", linewidth=0.35, alpha=0.90, label="Synthetic")
    
    plt.xlabel("log(runtime)")
    plt.ylabel("Percent")
    plt.grid(True, axis="y", alpha=0.25, linewidth=0.7)
    plt.legend(frameon=True, framealpha=0.95)
    
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    plt.close()
