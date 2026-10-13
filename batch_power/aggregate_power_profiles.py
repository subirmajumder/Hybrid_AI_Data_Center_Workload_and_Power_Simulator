# -*- coding: utf-8 -*-
"""
Template-based Power TS Generator (AR(1) + runtime-bin selection)
@author: Subir Majumder

Model format (CSV):
    1. for (cluster, timelimit, gpu_alloc)
            -- Timeseries template
            -- AR(1) rho
    2. for (cluster, timelimit, gpu_alloc, runtime_bin)
            -- Timeseries template
            -- AR(1) rho

1. Build models for quick lookup
    a) Choose template with get_ts_template(cluster, timelimit, gpu_alloc, runtime)
      - runtime-bin match via bisection
      - fallback to leaf template if no bin hit
    b) generate_power_ts(csv_path, rho, runtime_sec, gpu_alloc, random_state)
      - read template CSV containing [q05, q95, mean, std]
      - simulate AR(1) noise with rho
      - y = mean + NOISE_SCALE * std * z
      - clip to [q05, q95]
      - scale by gpu_alloc and GPU_POWER_SCALE
      - return a minute-resolution power vector in kW

2) attach_segment_timestamps(...)
      - maps the power vector onto scheduled segment start times

3) generate_job_power_profiles(...)
      - aggregates parent-job power profiles into one DataFrame with
        ["timestamp", "power_draw_kW"]

Required columns in jobs CSV:
   - parent_id
   - start_time
   - cluster_id
   - gpu_alloc
   - timelimit
   - runtime     (seconds)

Notes:
- Serial (no parallelism).
- Seed is fixed (SEED=2005). Per-job random_state is derived from (SEED + parent_id).
"""

import math
import bisect
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


SEED = 2005
NOISE_SCALE = 0.45  # tuned constant from your code
EPS = 1e-12


def build_template_indices(leaf_ts_csv, ts_bins_csv):
    """
    Build:
      leaf_index: (cluster, timelimit, gpu_alloc) -> (csv_path, rho)
      bin_index_sorted: (cluster, timelimit, gpu_alloc) -> (starts, bins_sorted)
        where bins_sorted items: (runtime_min, runtime_max, csv_path, rho_median)
    """
    leaf_df = pd.read_csv(leaf_ts_csv)
    bins_df = pd.read_csv(ts_bins_csv)

    leaf_index = {
        (int(r.cluster), int(r.timelimit), int(r.gpu_alloc)): (r.csv_path, float(r.rho))
        for r in leaf_df.itertuples(index=False)
    }

    tmp = {}
    for r in bins_df.itertuples(index=False):
        key = (int(r.cluster), int(r.timelimit), int(r.gpu_alloc))
        tmp.setdefault(key, []).append(
            (float(r.runtime_min), float(r.runtime_max), r.csv_path, float(r.rho_median))
        )

    bin_index_sorted = {}
    for key, bins in tmp.items():
        bins_sorted = sorted(bins, key=lambda t: t[0])
        starts = [t[0] for t in bins_sorted]
        bin_index_sorted[key] = (starts, bins_sorted)

    return leaf_index, bin_index_sorted


def get_ts_template(cluster, timelimit, gpu_alloc, runtime, leaf_index, bin_index_sorted):
    """
    Use key: (cluster, timelimit, gpu_alloc)
      - Use bisection over runtime bins in bin_index_sorted[key]
      - If no bin match, fallback to leaf_index[key]
    Returns: (csv_path, rho, source) where source in {"bin","leaf","missing"}.
    """
    key = (int(cluster), int(timelimit), int(gpu_alloc))
    rt = float(runtime)

    entry = bin_index_sorted.get(key)
    if entry is not None:
        starts, bins = entry
        i = bisect.bisect_right(starts, rt) - 1
        if i >= 0:
            rmin, rmax, path, rho = bins[i]
            if rmin <= rt < rmax:
                return path, rho, "bin"
            if i == len(bins) - 1 and math.isclose(rt, rmax):
                return path, rho, "bin"

    leaf = leaf_index.get(key)
    if leaf is not None:
        path, rho = leaf
        return path, rho, "leaf"

    return None, math.nan, "missing"


template_cache = {}  # csv_path -> (mu_all, sd_all, q05_all, q95_all)
def _load_template_arrays(csv_path):
    """
    Cache template arrays to avoid re-reading the same CSV repeatedly.
    Template CSV must contain columns: q05, q95, mean, std
    """
    arrs = template_cache.get(csv_path)
    if arrs is not None:
        return arrs

    df = pd.read_csv(csv_path, usecols=["q05", "q95", "mean", "std"])
    arrs = (
        df["mean"].to_numpy(dtype=float),
        df["std"].to_numpy(dtype=float),
        df["q05"].to_numpy(dtype=float),
        df["q95"].to_numpy(dtype=float),
    )
    template_cache[csv_path] = arrs
    return arrs


def generate_power_ts(csv_path, rho, runtime_sec, gpu_alloc, random_state=None):
    """
    Generate a full minute-resolution power series in kW.

      1. read template mean/std/q05/q95
      2. T = ceil(runtime_sec/60), capped to template length
      3. AR(1) noise z with Var(z)=1
      4. y = mean + NOISE_SCALE * std * z
      5. clip to [q05,q95]
      6. scale by gpu_alloc and GPU_POWER_SCALE (V100 -> H100), convert W -> kW
    """
    if csv_path is None:
        return None

    mu_all, sd_all, q05_all, q95_all = _load_template_arrays(csv_path)

    T_needed = max(1, int(math.ceil(float(runtime_sec) / 60.0)))
    T = min(T_needed, len(mu_all))
    if T <= 0:
        return None

    mu, sd, q05, q95 = mu_all[:T], sd_all[:T], q05_all[:T], q95_all[:T]

    if rho is None or not np.isfinite(rho):
        rho = 0.0
    rho = float(max(-0.999, min(0.999, rho)))

    rng = np.random.default_rng(random_state)
    eps = rng.normal(size=T)

    sigma_e = math.sqrt(1.0 - rho**2)
    z = np.empty(T, dtype=float)
    z[0] = eps[0]
    for t in range(1, T):
        z[t] = rho * z[t - 1] + sigma_e * eps[t]

    y = mu + NOISE_SCALE * sd * z
    y = np.clip(y, q05, q95)
    
    GPU_POWER_SCALE = 2.0  # V100 -> H100

    return y * float(gpu_alloc) * GPU_POWER_SCALE /1000

def attach_segment_timestamps(y_total, seg_starts, seg_minutes):
    """
    Attach timestamps to a minute-resolution series by segment start times.

    If there are leftover power samples beyond what the segments need,
    they are discarded.

    Returns:
      DataFrame with columns: ["timestamp","power_draw_kW"]
    """
    y = np.asarray(y_total, dtype=float)

    seg_starts = pd.to_datetime(seg_starts).floor("min").to_numpy(dtype="datetime64[ns]")
    n_segs = int(seg_starts.size)
    seg_minutes = int(seg_minutes)

    total_grid = n_segs * seg_minutes
    L = min(y.size, total_grid)

    k = np.arange(L, dtype=np.int64)
    seg_idx = k // seg_minutes
    min_off = k % seg_minutes

    times = seg_starts[seg_idx] + min_off.astype("timedelta64[m]")

    return pd.DataFrame({"timestamp": times, "power_draw_kW": y[:L]})


def generate_job_power_profiles(jobs_csv_path, ts_model_dir, segment_duration):
    """
    Group by parent_id, generate ONE full power profile per parent, then stamp segment timestamps.
    
    Assumptions:
      - For a given parent_id, (cluster_id, gpu_alloc, timelimit, runtime) are the same across segments
      - Segment start times come from jobs["start_time"]
      - Each segment has fixed duration = segment_duration (seconds)
    
    Returns:
      total_power_df: DataFrame ["timestamp", "power_draw_kW"] summed across all parents
    """
    ts_model_dir = Path(ts_model_dir)
    leaf_ts_csv = ts_model_dir / "leaf_ts.csv"
    ts_bins_csv = ts_model_dir / "ts_bins.csv"
    leaf_index, bin_index_sorted = build_template_indices(leaf_ts_csv, ts_bins_csv)

    jobs = pd.read_csv(jobs_csv_path)
    jobs.columns = jobs.columns.str.strip()

    for col in ["parent_id", "cluster_id", "gpu_alloc", "timelimit", "runtime"]:
        jobs[col] = pd.to_numeric(jobs[col], errors="coerce")

    jobs["start_time"] = pd.to_datetime(jobs["start_time"], errors="coerce").dt.floor("min")

    seg_minutes = int(math.ceil(float(segment_duration) / 60.0))

    pieces = []

    for pid, gdf in tqdm(jobs.groupby("parent_id", sort=True), desc="parents", total=jobs["parent_id"].nunique()):
        gdf = gdf.sort_values("start_time")

        r0 = gdf.iloc[0]
        c  = int(r0["cluster_id"])
        g  = int(r0["gpu_alloc"])
        tl = int(r0["timelimit"])
        rt = float(r0["runtime"])

        csv_path, rho, src = get_ts_template(c, tl, g, rt, leaf_index, bin_index_sorted)
        if csv_path is None:
            continue

        job_seed = int(SEED) + int(pid)

        y_total = generate_power_ts(
            csv_path=csv_path,
            rho=rho,
            runtime_sec=rt,
            gpu_alloc=g,
            random_state=job_seed,
        )
        if y_total is None:
            continue

        seg_starts = gdf["start_time"].to_list()

        ts_df = attach_segment_timestamps(y_total, seg_starts, seg_minutes)
        pieces.append(ts_df)

    if not pieces:
        return pd.DataFrame(columns=["timestamp", "power_draw_kW"])

    all_ts = pd.concat(pieces, ignore_index=True)
    total_power_df = (
        all_ts.groupby("timestamp", as_index=False)["power_draw_kW"]
              .sum()
              .sort_values("timestamp")
    )
    return total_power_df



if __name__ == "__main__":
    
    # Defaults
    here = Path(__file__).resolve()
    root = here.parent.parent

    jobs_csv = root / "Temp" / "scheduled_synthetic_jobs.csv"
    ts_model_dir = root / "ProcessedData_Models" / "Batch" / "power_data_ts_distribution"
    out_csv = root / "Temp" / "synthetic_power_profile.csv"

    total_power_df = generate_job_power_profiles(
        jobs_csv_path=jobs_csv,
        ts_model_dir=ts_model_dir,
        segment_duration=2*3600,
    )
    
    total_power_df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    print(total_power_df.head())

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    # Plot aggregated synthetic power time series
    dfp = total_power_df.copy()
    dfp["timestamp"] = pd.to_datetime(dfp["timestamp"])
    dfp = dfp.sort_values("timestamp")
    
    plt.figure(figsize=(6, 3))
    plt.plot(dfp["timestamp"], dfp["power_draw_kW"], linewidth=1.0, label="Synthetic")
    
    plt.xlabel("time")
    plt.ylabel("Power (kW)")
    plt.legend(frameon=True, framealpha=0.95)
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Box plot ==================================
    dfp["hour"] = dfp["timestamp"].dt.hour
    
    hours = list(range(24))
    data = [dfp.loc[dfp["hour"] == h, "power_draw_kW"].to_numpy() for h in hours]
    
    plt.figure(figsize=(8, 3))
    plt.boxplot(data, tick_labels=[str(h) for h in hours], showfliers=False)
    
    plt.xlabel("Hour of day")
    plt.ylabel("Power (kW)")
    
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Distribution of power =====================
    plt.figure(figsize=(5, 3))
    w = np.ones(len(dfp)) * 100.0 / len(dfp)
    plt.hist(dfp["power_draw_kW"].to_numpy(), bins=40, weights=w)
    plt.xlabel("Power (kW)")
    plt.ylabel("Percent")
    
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    plt.show()


