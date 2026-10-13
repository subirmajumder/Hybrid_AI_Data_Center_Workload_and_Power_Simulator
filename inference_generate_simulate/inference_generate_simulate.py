# -*- coding: utf-8 -*-
"""

Synthetic Inference workload generator – optimized for memory efficiency

Generates request arrivals from fitted negative-binomial (NB2) models,
samples generated-token counts from fitted token-length distributions,
converts request durations into concurrent-request profiles, and aggregates
the results into minute-level GPU usage, power demand, and arrival counts.

@author: Subir Majumder
"""

import json
import pickle
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm

import numpy as np
import pandas as pd

@dataclass
class InferenceConfig:
    seconds_per_token: float     # Avg TPOT_(s)
    max_batch_size: int          # Max_BS_(reqs)
    power_per_req: float         # W per concurrent request = Energy/req / (Avg_Output_Tokens * Avg TPOT)
    gpu_per_model: float         # TP * PP
    token_speed: str             # "F", "M", "S"
    
def slugify_model_name(model_name: str) -> str:
    """
    Convert a model name to a lowercase, filesystem-friendly slug.
    e.g. ``"Llama-3.1 8B"`` -->  ``"llama31_8b"``.
    """
    return (
        model_name
        .lower()
        .replace(" ", "_")
        .replace(".", "")
        .replace("-", "")
    )


def build_config_dict(config_csv_path):
    """
    Build inference configurations from the model benchmark CSV.
    
    Returns Configurations keyed by ``(model_slug, max_batch_size)``.
    """

    df = pd.read_csv(config_csv_path)
    config_dict: dict[tuple[str, int], InferenceConfig] = {}

    for _, row in df.iterrows():
        model_name = row["Model"]
        max_bs = int(row["Max_BS_(reqs)"])
        slug = slugify_model_name(model_name)

        tensor_parallelism   = float(row["TP"])
        pipeline_parallelism = float(row["PP"])
        energy_per_request_j = float(row["Energy/req_(J)"])
        seconds_per_token    = float(row["Avg TPOT_(s)"])
        avg_output_tokens    = float(row["Avg_Output_Tokens"])
        token_speed = str(row["Token_Speed"]).strip()
        
        gpu_per_model = tensor_parallelism * pipeline_parallelism
        power_per_req = energy_per_request_j / (avg_output_tokens * seconds_per_token)
        
        cfg = InferenceConfig(
            gpu_per_model=gpu_per_model,
            seconds_per_token=seconds_per_token,    # <-- Simulation speed
            max_batch_size=max_bs,
            power_per_req=power_per_req,        # <-- Power calculation
            token_speed=token_speed,
        )

        config_dict[(slug, max_bs)] = cfg

    return config_dict
    

def build_minute_grid(start_time: str, end_time: str):
    """
    Build a per-minute time grid with calendar features.
    
    minutes : One timestamp per minute over ``[start_time, end_time)``.
    daytype_arr : ``"weekday"`` or ``"weekend"`` for each minute.
    hod15_arr : 15-minute-of-day bin index (0..95) for each minute.
    """
    start_dt = pd.to_datetime(start_time)
    end_dt   = pd.to_datetime(end_time)

    n_minutes = int((end_dt - start_dt).total_seconds() // 60)
    minutes = pd.date_range(start=start_dt, periods=n_minutes, freq="1min")

    dow = minutes.dayofweek.to_numpy()
    daytype_arr = np.where(dow < 5, "weekday", "weekend")
    mins_of_day = minutes.hour.to_numpy() * 60 + minutes.minute.to_numpy()
    hod15_arr = (mins_of_day // 15).astype(int)

    return minutes, daytype_arr, hod15_arr


def build_scaled_cdf(base_pmf, values, scale: float):
    """ 
    Build a CDF of generated-token counts, warped by a verbosity scale.
 
    The base PMF over token lengths ``1..L`` is accumulated into a CDF and
    the length axis is multiplied by ``scale``. For ``scale <= 1`` the grid
    is closed with the point ``(L, 1.0)``. For ``scale > 1`` the stretched
    grid would exceed ``L``, so the tail is collapsed onto ``L`` using the
    CDF value at the largest unscaled length that still fits.
    """
    base_pmf = np.asarray(base_pmf, dtype=float)
    values   = np.asarray(values,   dtype=float)

    L = int(values.max())
    s = float(scale)

    pmf = base_pmf[:L]
    F   = np.cumsum(pmf)

    y   = s * values[:L]
    cdf = F.copy()

    if s <= 1.0:
        y   = np.concatenate([y,   [float(L)]])
        cdf = np.concatenate([cdf, [1.0]])
    else:
        idx_below = int(np.floor((L - 1) / s))
        idx_below = max(1, min(idx_below, L - 1))
        cdf_below = F[idx_below - 1]

        y   = np.concatenate([y,   [float(L) - 1.0 / s + 1e-9, float(L)]])
        cdf = np.concatenate([cdf, [cdf_below,                     1.0]])

    order = np.argsort(y)
    return y[order], cdf[order], L


def sample_tokens_from_scaled_cdf(
    y_grid, cdf_grid, L, n_samples, rng):
    """
    Sample generated token lengths by inverse-transform sampling of a warped CDF.
    """

    u = rng.random(n_samples)
    idx = np.searchsorted(cdf_grid, u, side="left")
    idx = np.clip(idx, 0, len(cdf_grid) - 1)

    y_samples = y_grid[idx]
    toks = np.rint(y_samples)
    toks = np.clip(toks, 1, L).astype(int)
    return toks


def expand_counts_to_timestamps(minutes64, counts, rng):
    """
    Expand per-minute arrival counts into individual arrival timestamps.
    
    Each arrival is placed at a uniformly random millisecond offset within
    its minute.
    """
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype="datetime64[ns]")

    nz = counts > 0
    ts = np.repeat(minutes64[nz], counts[nz])

    # offset within minute in milliSECONDS (0..59_999)
    offs = rng.integers(0, 60_000, size=total, dtype=np.int64).astype("timedelta64[ms]")
    ts += offs
    return ts


def simulate_resources(df, start, config):
    """
    Compute the per-minute average number of concurrently processed requests.
 
    Each request occupies a processing slot from its arrival (rounded up to
    a 30-second bucket) for ``GeneratedTokens * seconds_per_token`` seconds.
    Concurrency is computed on the 30-second grid and then averaged within
    each minute.
    """
    step = 30  # seconds per bucket

    start_ts = pd.to_datetime(start)
    origin_s = int(start_ts.timestamp())


    ts = df["ARRIVAL_TIMESTAMP"].dt.ceil(f"{step}s")
    start_s = ts.to_numpy(dtype="datetime64[s]", copy=False).view("int64")
    tokens  = df["GeneratedTokens"].to_numpy(dtype=np.int64, copy=False)

    seconds_per_token = float(config.seconds_per_token)
    duration_s = np.round(tokens * seconds_per_token).astype(np.int64)
    end_s = start_s + duration_s
    max_end_s = end_s.max()
    n_steps = int((max_end_s - origin_s) // step + 1)

    start_idx = ((start_s - origin_s) // step).astype(np.int64)
    end_idx   = ((end_s   - origin_s + step - 1) // step).astype(np.int64)

    n_bins = n_steps + 1
    diff = (
        np.bincount(start_idx, minlength=n_bins)
        - np.bincount(end_idx, minlength=n_bins)
    )

    concurrent_req = np.cumsum(diff[:-1])

    sec_index = pd.to_datetime(origin_s + step * np.arange(n_steps), unit="s")

    sec_df = pd.DataFrame({
        "Time": sec_index,
        "Processing_Requests": concurrent_req.astype(float),
    })

    minute_df = (
        sec_df.set_index("Time")
        .resample("1min")["Processing_Requests"]
        .mean()
        .rename("Processing_Requests")
        .reset_index()
    )

    return minute_df
    
def resource_profile(resource_ts_all, config):
    """
    Convert concurrency profiles into power and GPU usage for one model.

    The profiles from all request groups are summed per minute. Power is
    ``Processing_Requests * power_per_req`` (reported in kW) and running
    GPUs follow from ``ceil(Processing_Requests / max_batch_size)``
    replicas times ``gpu_per_model``.
    
    If available_gpus is given, GPUs are divided across models in proportion to
    their GPU-seconds of demand and each model's concurrency is capped accordingly;
    if None, no cap is applied.
    
    Returns:
        DataFrame [Time, Total_Inference_Power (kW), RunningGPUs, Arrival_Count],
        one row per minute.
    """
    resource_ts = pd.concat(resource_ts_all, ignore_index=True)

    minute_df = (
        resource_ts
        .groupby(["Time"], as_index=False)
        .agg(Processing_Requests=("Processing_Requests", "sum"))
    )

    power = minute_df["Processing_Requests"] * config.power_per_req
    vms = np.ceil(
        minute_df["Processing_Requests"] / config.max_batch_size
    ).astype(np.int64)
    gpus = (vms * config.gpu_per_model).astype(np.int64)

    out = pd.DataFrame({
        "Time": minute_df["Time"],
        "Processing_Requests": minute_df["Processing_Requests"],
        "Total_Inference_Power": power / 1000.0,  # kW
        "RunningGPUs": gpus,
    })

    return out


def generate_inference_resource(
    config_csv_path, nb2_path, conv_token_path, code_token_path, model_timeseries,
    start_time, end_time, traffic_scale, verbosity_scale, token_speed, available_gpus):
    """
    Generate a synthetic inference workload at minute resolution.
 
    The workload is aggregated into GPU usage, power demand and arrival
    counts. The procedure is:
 
    1. Generate request arrivals from fitted NB2 models.
    2. Split arrivals evenly across the selected inference models.
    3. Sample generated-token counts from fitted token distributions.
    4. Convert request durations into concurrent-request profiles.
    5. Allocate available GPUs across models and cap concurrency accordingly.
    6. Aggregate model-level profiles into a cluster-level time series.
 
    Request groups are ``0`` (code) and ``1``-``4`` (conversation). Per-model
    profiles are written to `model_timeseries` as ``<model>.csv``, along
    with ``gpu_alloc.csv`` when `available_gpus` is given.
    """
    seed = 2009
    rng = np.random.default_rng(seed)

    # config files containing all model names
    config_dict = build_config_dict(config_csv_path)
    
    model_configs: dict[str, InferenceConfig] = {}
    for (slug, max_bs), cfg in config_dict.items():
        if token_speed is not None and cfg.token_speed != token_speed:
            continue
        existing = model_configs.get(slug)
        if existing is None or cfg.max_batch_size > existing.max_batch_size:
            model_configs[slug] = cfg
    
    template_models = sorted(model_configs.keys())
    
    K = len(template_models)

    # Load NB2 models, conv and code token models
    with open(nb2_path, "rb") as f:
        nb2_pack = pickle.load(f)

    with open(conv_token_path, "r", encoding="utf-8") as f:
        conv_token_data = json.load(f)
    conv_values = np.asarray(conv_token_data["values"], dtype=int)
    conv_pmfs_base = {
        int(g): np.asarray(p, dtype=float)
        for g, p in conv_token_data["groups"].items()
    }

    with open(code_token_path, "r", encoding="utf-8") as f:
        code_token_data = json.load(f)
    code_values = np.asarray(code_token_data["values"], dtype=int)
    code_pmf_base = np.asarray(code_token_data["global_p_smooth"], dtype=float)

    # Create a minute grid with features
    minutes, daytype_arr, hod15_arr = build_minute_grid(
                        start_time=start_time, end_time=end_time,)
    
    # Precompute daytype codes + hod ints ONCE (reused for every group)
    dt_str_arr = daytype_arr.astype(str)
    dt_codes, dt_uniques = pd.factorize(dt_str_arr, sort=True)  # dt_codes shape (n,)
    dt_index = {s: i for i, s in enumerate(dt_uniques)}
    
    h_int = hod15_arr.astype(np.int16, copy=False)
    H = int(h_int.max()) + 1  # number of 15-min-of-day bins
    
    groups = [0, 1, 2, 3, 4]  # 0 = code, 1..4 = conv groups
    
    # Cache group parameters
    group_cache = {}
    for g in groups:
    # Build mu_total and alpha_g for the group
        model_g = nb2_pack[g]
        alpha_g = float(model_g["alpha"])
        mu_template = model_g["mu_template"]

        mu_table = np.zeros((len(dt_uniques), H), dtype=np.float32)

        # Fill from dict-of-dicts
        for dt_str, hod_map in mu_template.items():
            row = dt_index.get(str(dt_str))
            if row is None:
                continue
            for h_k, v in hod_map.items():
                hh = int(h_k)
                if 0 <= hh < H:
                    mu_table[row, hh] = float(v)

        mu_total = mu_table[dt_codes, h_int].astype(np.float64) * traffic_scale

    # Calculate generated token CDF for this group
        if g == 0:
            y_grid, F_grid, L_max = build_scaled_cdf(
                base_pmf=code_pmf_base,
                values=code_values,
                scale=verbosity_scale,)
        else:
            y_grid, F_grid, L_max = build_scaled_cdf(
                base_pmf=conv_pmfs_base[g],
                values=conv_values,
                scale=verbosity_scale,)
            
            
        group_cache[g] = dict(
                        alpha_g=alpha_g,
                        mu_total=mu_total,
                        y_grid=y_grid,
                        F_grid=F_grid,
                        L_max=L_max,)
    
    arrivals_per_minute = None
    gpu_seconds_model = {}
    for model_name in tqdm(template_models, desc="Simulating per model TS"):
        resource_frames = []
        total_tokens_model = 0.0
        
        for g in tqdm(groups, desc=f"{model_name} groups", leave=False):
            
            cache = group_cache[g]
            alpha_g = cache["alpha_g"]
            mu_total = cache["mu_total"]
            y_grid = cache["y_grid"]
            F_grid = cache["F_grid"]
            L_max = cache["L_max"]
    
            # 1. NB2 splitting across models
            alpha_per = alpha_g * K
            r = 1.0 / alpha_per
    
            mu_per = mu_total / K
            mask = mu_per > 0
            mu_pos = mu_per[mask]
    
            p_pos = r / (r + mu_pos)

        # 2. For each model: NB2 counts
            counts = np.zeros_like(mu_per, dtype=int)
            counts[mask] = rng.negative_binomial(r, p_pos, size=mu_pos.size)
        # 3. For each model: Add timestamps
            ts = expand_counts_to_timestamps(minutes, counts, rng)
        # 4. For each model: Add generated tokens
            toks = sample_tokens_from_scaled_cdf(
                y_grid=y_grid,
                cdf_grid=F_grid,
                L=L_max,
                n_samples=len(ts),
                rng=rng,)
            
            token_ts = pd.DataFrame({
                "ARRIVAL_TIMESTAMP": ts,
                "GeneratedTokens": toks.astype("int16"),
            })
            total_tokens_model += float(toks.sum(dtype=np.int64))
            
            cfg = model_configs[model_name]
            resource_ts = simulate_resources(token_ts, start_time, cfg)
            resource_frames.append(resource_ts)
            
            if arrivals_per_minute is None:
                arrivals_per_minute = np.zeros(len(minutes), dtype=np.int64)
            
            arrivals_per_minute += counts

        minute_df = resource_profile(resource_frames, cfg)
        
        model_csv = model_timeseries / f"{model_name}.csv"
        minute_df.to_csv(model_csv, index=False)
        
        gpu_per_token = float(cfg.gpu_per_model) / float(cfg.max_batch_size)
        gpu_seconds_model[model_name] = total_tokens_model * float(cfg.seconds_per_token) * gpu_per_token
    
    # GPU allocation
    gpu_alloc = None
    if available_gpus is not None:
        total_need = sum(gpu_seconds_model.values()) + 1e-12
        gpu_alloc = {}
        for m in template_models:
            gpu_alloc[m] = float(available_gpus) * gpu_seconds_model[m] / total_need
        
        alloc_df = pd.DataFrame({"Model": list(gpu_alloc.keys()),
                                 "GPU_Alloc": list(gpu_alloc.values()),})
        alloc_df.to_csv(model_timeseries / "gpu_alloc.csv", index=False)

    # Capped simulation
    frames = []
    for path in sorted(model_timeseries.glob("*.csv")):
        model_name = path.stem
        cfg = model_configs.get(model_name)
        if cfg is None:
            continue

        df = pd.read_csv(path)
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")

        if gpu_alloc is not None:
            pr = pd.to_numeric(df["Processing_Requests"], errors="coerce").fillna(0.0)

            max_conc = float(cfg.max_batch_size) * float(gpu_alloc[model_name]) / float(cfg.gpu_per_model)
            pr_cap = pr.clip(upper=max_conc)

            df["RunningGPUs"] = np.ceil(pr_cap / float(cfg.max_batch_size)) * float(cfg.gpu_per_model)
            df["Total_Inference_Power"] = (pr_cap * float(cfg.power_per_req)) / 1000.0  # kW

        frames.append(df)

    sim_all = pd.concat(frames, ignore_index=True)

    total_sim = (sim_all.groupby("Time", as_index=False)
        .agg(Total_Inference_Power=("Total_Inference_Power", "sum"),
             RunningGPUs=("RunningGPUs", "sum"),))
    
    arrivals_series = pd.Series(
        arrivals_per_minute,
        index=pd.to_datetime(minutes),
        name="Arrival_Count",)
    arrivals_series.index.name = "Time"

    arrivals_df = arrivals_series.reset_index()
    total_sim = total_sim.merge(arrivals_df, on="Time", how="left")
    
    return total_sim


if __name__ == "__main__":
        
    # Defaults
    here = Path(__file__).resolve()
    root = here.parent.parent

    config_csv_path = root / "data" / "ml_energy_data" / "inference_token_energy.csv"
    nb2_path = root / "ProcessedData_Models" / "Inference" / "nb2_model_alpha.pkl"
    conv_token_path = root / "ProcessedData_Models" / "Inference" / "conv_generated_tokens_model.json"
    code_token_path = root / "ProcessedData_Models" / "Inference" / "code_generated_tokens_model.json"
    
    profile_dir = root / "Temp" / "profile_by_inference_model"
    profile_dir.mkdir(parents=True, exist_ok=True)
    
    out_csv = root / "Temp" / "inference_resource_profiles.csv"

    total_sim  = generate_inference_resource(
        config_csv_path=config_csv_path,
        nb2_path=nb2_path,
        conv_token_path=conv_token_path,
        code_token_path=code_token_path,
        model_timeseries=profile_dir,
        start_time="2026-01-03 00:00:00",
        end_time="2027-01-03 00:00:00",
        traffic_scale=0.5,
        verbosity_scale=1.0,
        token_speed="F",
        available_gpus = 300
    )
    
    total_sim.to_csv(out_csv, index=False)
    print("Saved aggregated power profile:", out_csv)
    print(total_sim.head())
    
    
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    # 1) Plot aggregated synthetic power time series
    dfp = total_sim.copy()
    dfp["Time"] = pd.to_datetime(dfp["Time"])
    dfp = dfp.sort_values("Time")
    
    dfp["power_draw_kW"] = dfp["Total_Inference_Power"]
    
    plt.figure(figsize=(6, 3))
    plt.plot(dfp["Time"], dfp["power_draw_kW"], linewidth=1.0, label="Synthetic")
    
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
    
    # 2) Plot aggregated running GPUs time series
    plt.figure(figsize=(6, 3))
    plt.plot(dfp["Time"], dfp["RunningGPUs"], linewidth=1.0, label="Running GPUs")
    
    plt.xlabel("time")
    plt.ylabel("Running GPUs")
    plt.legend(frameon=True, framealpha=0.95)
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # 3) Box plot: power vs hour of day
    dfp["hour"] = dfp["Time"].dt.hour
    
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
    
    # 4) Distribution of power
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