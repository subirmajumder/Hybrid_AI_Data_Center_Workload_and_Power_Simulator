# -*- coding: utf-8 -*-
"""
Generate hourly GPU availability for a simulation window.
@author: Subir Majumder

The simulation window starts at midnight of the earliest ``time_submit`` date 
and runs through the end of the latest ``time_submit`` date (inclusive of both days).

- Builds an hourly time grid covering the window.
- Base capacity is ``total_gpus`` for every hour.
- If an inference-usage CSV is provided, GPUs used by inference (hourly maximum of
  RunningGPUs, forward-filled) are subtracted from capacity (clamped at 0).
- Produces a DataFrame with columns ``time`` and ``gpu`` (available GPUs)

Expected inference CSV columns:
  - Time
  - RunningGPUs
"""

from datetime import timedelta
from pathlib import Path

import pandas as pd


def generate_capacity_csv(
    total_gpus,
    event_file,
    inference_path = None,
):
    """Build an hourly GPU availability table for the simulation window.
    """
    # Parse simulation start datetime (naive)
    events = pd.read_csv(event_file, usecols=["time_submit"])
    events["time_submit"] = pd.to_datetime(events["time_submit"], errors="raise")

    sim_start = events["time_submit"].min().normalize()   # first day at 00:00:00
    sim_end   = events["time_submit"].max().normalize()   # last day at 00:00:00
    n_days = int((sim_end - sim_start).days) + 1          # inclusive days

    # build hourly capacity grid
    hours = [sim_start + timedelta(hours=i) for i in range(n_days * 24)]
    capacity_df = pd.DataFrame({
        "time": hours,
        "gpu": int(total_gpus),
    })

    # If an inference CSV is provided, available = capacity - inference GPUs
    if inference_path:
        inference_df = pd.read_csv(inference_path)
        inference_df["Time"] = pd.to_datetime(inference_df["Time"], errors="raise")
    
        # keep only provisioned points that could affect the sim window
        start = sim_start
        end = sim_end + timedelta(days=1)  # include the last day hours
        inference_df = inference_df[(inference_df["Time"] >= start) & (inference_df["Time"] < end)]
    
        hourly_prov = (
            inference_df.set_index("Time")["RunningGPUs"]
            .resample("h")
            .max()
            .reset_index()
            .rename(columns={"RunningGPUs": "provisioned_gpu", "Time": "time"})
        )
    
        capacity_df = pd.merge(capacity_df, hourly_prov, on="time", how="left")
        capacity_df["provisioned_gpu"] = capacity_df["provisioned_gpu"].ffill().fillna(0)
        capacity_df["available_gpu"] = (capacity_df["gpu"] - capacity_df["provisioned_gpu"]).clip(lower=0)  # <-- clamp at 0
        
    # If no inference CSV is provided, available = total capacity
    else:
        capacity_df["available_gpu"] = capacity_df["gpu"]

    # Format time to match job submission format
    capacity_df["time"] = capacity_df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    capacity_df = capacity_df[["time", "available_gpu"]]
    capacity_df["available_gpu"] = capacity_df["available_gpu"].astype(int)
    capacity_df = capacity_df.rename(columns={"available_gpu": "gpu"})

    return capacity_df



if __name__ == "__main__":
    # Defaults
    here = Path(__file__).resolve()
    root = here.parent.parent

    inference_path = root / "Temp" / "inference_resource_profiles.csv"
    capacity_csv = root / "Temp" / "gpu_capacity.csv"
    events_csv = root / "Temp" / "synthetic_second_arrivals_with_attrs.csv"

    capacity_df = generate_capacity_csv(
        total_gpus=315,
        event_file=events_csv,
        inference_path=inference_path,
    )
    
    # Save to CSV
    capacity_df.to_csv(capacity_csv, index=False)
    print(f"Saved capacity and availability data to {capacity_csv}")

