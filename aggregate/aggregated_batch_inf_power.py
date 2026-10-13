# -*- coding: utf-8 -*-
"""
Combine batch and inference outputs into one facility-level table.

Merges power, running-GPU, arrival and running-job series on timestamp and
prints summary statistics of total power.
@author: Subir Majumder
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


def plot_hourly_power_boxplot(dfp, power_kw_col="power_draw_kW", showfliers=False):
    """
    dfp must have:
      - 'timestamp' column (datetime64)
      - power_kw_col column (float), e.g. 'power_total_kW'
    """
    dfp = dfp.copy()
    dfp["hour"] = dfp["timestamp"].dt.hour

    data = [dfp.loc[dfp["hour"] == h, power_kw_col].to_numpy() for h in range(24)]

    plt.figure(figsize=(4, 3))
    plt.boxplot(data, tick_labels=[str(h) for h in range(24)], showfliers=showfliers)
    
    plt.xlabel("Hour of day")
    plt.ylabel("Power (kW)")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()
    
def plot_batch_inference_timeseries(dfp, random_state=None):
    """
    Stacked filled time series for batch and inference power.

    dfp must have:
      - 'timestamp'
      - 'power_batch_kW'
      - 'power_inference_kW'
    """
    dfp = dfp.copy()
    dfp["timestamp"] = pd.to_datetime(dfp["timestamp"], errors="coerce")
    dfp = dfp.dropna(subset=["timestamp"]).sort_values("timestamp")

    # ---- choose a random continuous 4-month window ----
    months = dfp["timestamp"].dt.to_period("M").drop_duplicates().sort_values()

    rng = np.random.default_rng(random_state)
    start_idx = rng.integers(0, len(months) - 4 + 1)

    start_period = months.iloc[start_idx]
    end_period   = months.iloc[start_idx + 4]     # exclusive

    start_time = start_period.to_timestamp()      # first day of start month
    end_time   = end_period.to_timestamp()        # first day after 6th month

    mask = (dfp["timestamp"] >= start_time) & (dfp["timestamp"] < end_time)
    dfp = dfp.loc[mask].sort_values("timestamp").copy()
    # ---------------------------------------------------

    t = dfp["timestamp"].to_numpy()
    p_batch = dfp["power_batch_kW"].to_numpy()
    p_inf = dfp["power_inference_kW"].to_numpy()

    plt.figure(figsize=(6, 2.5))

    # Inference (bottom, blue)
    plt.fill_between(
        t,
        0.0,
        p_inf,
        color="navy",
        alpha=0.6,
        label="Inference power",
        linewidth=0.0,
    )

    # Batch (stacked on top of inference, orange)
    plt.fill_between(
        t,
        p_inf,
        p_inf + p_batch,
        color="orange",
        alpha=0.9,
        label="Batch power",
        linewidth=0.0,
    )

    plt.xlabel("Time")
    plt.ylabel("Power (kW)")
    plt.legend(loc="upper right")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()
    
def fit_dummy_uncertainty_model(
    dfp,
    y_col="power_draw_kW",
    bins=60,
):
    """
    Pooled OLS with time dummies:
      - hour-of-day
      - Weekday/Weekend (daytype)
      - week-of-month (wom capped at 4)

    dfp must have:
      - 'timestamp' column (datetime64)
      - y_col column (float)
    """
    df = dfp.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", y_col]).sort_values("timestamp")

    df["date"] = df["timestamp"].dt.normalize()

    dt = np.where(df["date"].dt.weekday < 5, "Weekday", "Weekend")
    df["daytype"] = pd.Categorical(
        dt, categories=["Weekday", "Weekend"], ordered=True
    )

    df["week_id"] = df["date"].dt.to_period("W").astype(str)
    df["day_id"] = df["date"].dt.to_period("D").astype(str)

    wom_raw = ((df["date"].dt.day - 1) // 7 + 1).astype(int)
    df["wom"] = pd.Categorical(
        np.minimum(wom_raw, 4),
        categories=[1, 2, 3, 4],
        ordered=True,
    )

    df["hour"] = df["timestamp"].dt.hour

    formula = f"{y_col} ~ C(hour)*C(daytype)*C(wom)"

    model = smf.ols(formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["day_id"]},
    )

    resid = model.resid

    plt.figure(figsize=(5, 3))
    plt.hist(np.asarray(resid), bins=bins)
    plt.xlabel("Residual (kW)")
    plt.ylabel("Count")
    plt.title("Uncertainty (Residual) Distribution")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()

    return model, resid


def compute_ramps(power_kw, delta_steps):
    """
    power_kw: 1D array-like of power in kW
    delta_steps: integer lag (in samples, e.g. minutes) to compute ramps over
    """
    p = np.asarray(power_kw, dtype=float)
    ramps = np.abs(p[delta_steps:] - p[:-delta_steps])

    return {
        "p95_abs": float(np.quantile(ramps, 0.95)),
        "p99_abs": float(np.quantile(ramps, 0.99)),
        "mean_abs": float(ramps.mean()),
        "std_signed": float(ramps.std()),
    }


def load_power_timeseries(csv_path, time_col, power_col, out_col):
    """
    Load a timeseries from CSV and normalize to:
      - 'timestamp'
      - out_col (float kW)
    """
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df[time_col], errors="coerce")
    df[out_col] = pd.to_numeric(df[power_col], errors="coerce")

    df = df.dropna(subset=["timestamp", out_col]).sort_values("timestamp")
    return df[["timestamp", out_col]]

def build_gpu_timeseries_from_csv(jobs_csv_path):
    """Minute-level count of GPUs held by running batch segments."""
    df = pd.read_csv(jobs_csv_path)

    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["end_time"]   = pd.to_datetime(df["end_time"],   errors="coerce")
    df["gpu_alloc"]  = pd.to_numeric(df["gpu_alloc"], errors="coerce").fillna(0.0)

    df = df.dropna(subset=["start_time", "end_time"])


    starts = df.set_index("start_time")["gpu_alloc"]
    ends = -df.set_index("end_time")["gpu_alloc"]
    events = pd.concat([starts, ends]).groupby(level=0).sum().sort_index()
    active = events.cumsum()

    active_minutely = (
        active
        .resample("1min")
        .ffill()
        .fillna(0.0))

    return pd.DataFrame({
        "timestamp": active_minutely.index,
        "active_gpus_batch": active_minutely.values,
    })

def build_request_timeseries_from_csv(requests_csv_path):
    """Minute-level count of running batch segments."""
    df = pd.read_csv(requests_csv_path)

    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["end_time"]   = pd.to_datetime(df["end_time"], errors="coerce")

    df = df.dropna(subset=["start_time", "end_time"])
    df = df[df["end_time"] >= df["start_time"]].copy()

    starts = pd.Series(1.0, index=df["start_time"])
    ends   = pd.Series(-1.0, index=df["end_time"])

    events = pd.concat([starts, ends]).groupby(level=0).sum().sort_index()
    active = events.cumsum()

    active_minutely = (
        active
        .resample("1min")
        .ffill()
        .fillna(0.0)
    )

    return pd.DataFrame({
        "timestamp": active_minutely.index,
        "active_requests": active_minutely.values,
    })

def submit_end_runtime_diff(csv_path):
    """Mean of (last segment end - submit - runtime) per batch job, in minutes."""
    df = pd.read_csv(csv_path)

    df["time_submit"] = pd.to_datetime(df["time_submit"], errors="coerce")
    df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")
    df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")

    df = df.dropna(subset=["parent_id", "time_submit", "end_time", "runtime"]).copy()

    grouped = (
        df.groupby("parent_id", as_index=False)
          .agg(
              time_submit=("time_submit", "first"),
              end_time_max=("end_time", "max"),
              runtime_sec=("runtime", "first"),
          )
    )

    diff_min = (
        (grouped["end_time_max"] - grouped["time_submit"]).dt.total_seconds()
        - grouped["runtime_sec"]
    ) / 60.0

    return diff_min.mean()


def build_arrival_timeseries(jobs_csv_path):
    df = pd.read_csv(jobs_csv_path)

    df["time_submit"] = pd.to_datetime(df["time_submit"], errors="coerce")
    df = df.dropna(subset=["time_submit"])
    df = df.sort_values("time_submit").drop_duplicates(subset="parent_id", keep="first")

    arrivals_minutely = (
        df
        .set_index("time_submit")
        .resample("1min")
        .size()                  
        .rename("batch_arrivals")     
        .rename_axis("timestamp")
        .reset_index()  )

    arrivals_minutely["batch_arrivals"] = arrivals_minutely["batch_arrivals"].astype(int)

    return arrivals_minutely


def run_combined_power_analysis(
    batch_power_csv_path,
    batch_jobs_csv_path,
    inference_csv_path,
    simulation_start,
    simulation_end
):
    """
    1) Load batch and inference power series; a missing side is filled with zeros.
    2) Outer-merge on timestamp; add total power, running GPUs, arrivals and
       running batch segments.
    3) Restrict to [simulation_start, simulation_end).
    4) Print mean and p99 total power and plot the hourly box plot of total power.
    
    Returns:
        df : minute-level DataFrame (columns listed in the README)
        mean_delay : mean batch delay in minutes (0.0 if no batch jobs)
    """

    batch_time_col="timestamp"
    batch_power_col="power_draw_kW"
    inference_time_col="Time"
    inference_power_col="Total_Inference_Power"
    inference_running_gpus_col = "RunningGPUs"
    inference_arrival_col = "Arrival_Count"
        
    # Load individual profiles
    df_batch = None
    df_inference = None

    if batch_power_csv_path is not None:
        df_batch = load_power_timeseries(
            batch_power_csv_path,
            time_col=batch_time_col,
            power_col=batch_power_col,
            out_col="power_batch_kW",)

    if inference_csv_path is not None:
        df_inference = load_power_timeseries(
            inference_csv_path,
            time_col=inference_time_col,
            power_col=inference_power_col,
            out_col="power_inference_kW",)

    if df_batch is None and df_inference is None:
        raise ValueError("At least one of batch_power_csv_path or inference_csv_path must be provided.")

    if df_batch is None:
        # Create batch profile with zeros, aligned to inference timestamps
        df_batch = df_inference[["timestamp"]].copy()
        df_batch["power_batch_kW"] = 0.0

    if df_inference is None:
        # Create inference profile with zeros, aligned to batch timestamps
        df_inference = df_batch[["timestamp"]].copy()
        df_inference["power_inference_kW"] = 0.0


    df = pd.merge(df_batch, df_inference, on="timestamp", how="outer")
    df["power_batch_kW"] = df["power_batch_kW"].fillna(0.0)
    df["power_inference_kW"] = df["power_inference_kW"].fillna(0.0)
    df["power_total_kW"] = df["power_batch_kW"] + df["power_inference_kW"]
    
    # Add gpus to df
    if batch_jobs_csv_path is not None:
        df_batch_gpu = build_gpu_timeseries_from_csv(batch_jobs_csv_path)
        df_batch_gpu = df_batch_gpu.rename(columns={"active_gpus_batch": "running_batch_gpus"})

        df = pd.merge(df, df_batch_gpu, on="timestamp", how="left")
        df["running_batch_gpus"] = df["running_batch_gpus"].fillna(0.0)
    else:
        df["running_batch_gpus"] = 0.0
        
    if inference_csv_path is not None:
        df_inf_gpus = load_power_timeseries(
            inference_csv_path,
            time_col=inference_time_col,
            power_col=inference_running_gpus_col,
            out_col="running_inf_gpus",)

        df = pd.merge(df, df_inf_gpus, on="timestamp", how="left")
        df["running_inf_gpus"] = df["running_inf_gpus"].fillna(0.0)
    else:
        df["running_inf_gpus"] = 0.0
    
    # Add arrivals to df
    if batch_jobs_csv_path is not None:
        df_batch_arrivals = build_arrival_timeseries(batch_jobs_csv_path)
        df_batch_arrivals = df_batch_arrivals.rename(columns={"batch_arrivals": "batch_arrivals"})

        df = pd.merge(df, df_batch_arrivals, on="timestamp", how="left")
        df["batch_arrivals"] = df["batch_arrivals"].fillna(0.0)
    else:
        df["batch_arrivals"] = 0.0
        
    if inference_csv_path is not None:
        df_inf_gpus = load_power_timeseries(
            inference_csv_path,
            time_col=inference_time_col,
            power_col=inference_arrival_col,
            out_col="inf_arrivals",)

        df = pd.merge(df, df_inf_gpus, on="timestamp", how="left")
        df["inf_arrivals"] = df["inf_arrivals"].fillna(0.0)
    else:
        df["inf_arrivals"] = 0.0
        
    # Add running jobs to df
    if batch_jobs_csv_path is not None:
        df_batch_req = build_request_timeseries_from_csv(batch_jobs_csv_path)
        df_batch_req = df_batch_req.rename(columns={"active_requests" : "running_request_batch"})

        df = pd.merge(df, df_batch_req, on="timestamp", how="left")
        df["running_request_batch"] = df["running_request_batch"].fillna(0.0)
    else:
        df["running_request_batch"] = 0.0
    
    # Processing
    mask = (df["timestamp"] >= simulation_start) & (df["timestamp"] < simulation_end)
    df = df.loc[mask]

    print("\nTotal (batch + inference) power (kW):")
    print("mean_kW:", float(df["power_total_kW"].mean()))
    print("p99_kW :", float(df["power_total_kW"].quantile(0.99)))
    
    # 1) Filled stacked time series for batch & inference
    # plot_batch_inference_timeseries(df, random_state=1256)
    
    # 2) Boxplot on aggregate power
    plot_hourly_power_boxplot(df, power_kw_col="power_total_kW", showfliers=False)
    
    if batch_jobs_csv_path is not None:
        mean_delay = submit_end_runtime_diff(batch_jobs_csv_path)
    else:
        mean_delay = 0.0
    

    return df, mean_delay
    


if __name__ == "__main__":

    # Default
    here = Path(__file__).resolve()
    root = here.parent.parent
    
    SIMULATION_START_DATE="2026-01-03"
    SIMULATION_START = pd.Timestamp(SIMULATION_START_DATE).normalize()
    N_DAYS=365
    SIMULATION_END = SIMULATION_START + pd.Timedelta(days=N_DAYS)
    
    INF_RES_TS = root / "Temp" / "inference_resource_profiles.csv"
    BATCH_TS_PROFILE_CSV = root / "Temp" / "power_ts.csv"
    SCH_BAT_JOBS_CSV = root / "Temp" / "scheduled_synthetic_jobs.csv"
    # BATCH_TS_PROFILE_CSV = None
    # SCH_BAT_JOBS_CSV = None

    _, _ = run_combined_power_analysis(
        batch_power_csv_path=BATCH_TS_PROFILE_CSV,
        batch_jobs_csv_path=SCH_BAT_JOBS_CSV,
        inference_csv_path=INF_RES_TS,
        simulation_start=SIMULATION_START,
        simulation_end=SIMULATION_END
    )
