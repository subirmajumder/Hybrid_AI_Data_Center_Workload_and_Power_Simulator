# -*- coding: utf-8 -*-
"""
Simulation configuration

frac : Fraction of total offered workload assigned to inference.
       The remaining workload share, (1 - frac), is assigned to batch jobs.

TOTAL_GPUS : Total number of GPUs available in the shared cluster.
UTIL_RATE : Target utilization label. Used only in the output file name; actual
            utilization is set by INF_TRAFFIC_SCALE_RAW and BAT_TRAFFIC_SCALE_RAW,
            which must be tuned for the chosen TOTAL_GPUS.
## For inference:
    INF_TRAFFIC_SCALE_RAW : Baseline scaling factor for inference arrivals.
        Adjust so that inference reaches the desired utilization for a given TOTAL_GPUS.
    VERBOSITY_SCALE : Multiplier applied to generated-token lengths for inference requests.
    TOKEN_SPEED : Inference token-generation speed setting "F" = fast / "M" = medium / "S" = slow

## For batch
    BAT_TRAFFIC_SCALE_RAW : Baseline scaling factor for batch arrivals.
        Adjust so that batch workload reaches the desired utilization for a given TOTAL_GPUS.
    N_TZ : Number of time zones used in the batch-arrival model.
    LO_COEFF, MID_COEFF, HIGH_COEFF: Multipliers applied to low-, medium-, and high-resource-group batch arrival rates.
    COEFF_MULTIPLIERS : Batch arrival-rate multipliers by resource group {"low", "mid", "high"}
    SEGMENT_DURATION : Checkpoint segment length for batch jobs, in seconds.
    SCHEDULER : Batch scheduling rule:
            "FCFS_BKFL" = first-come, first-served with backfilling
            "SWF"       = greedy smallest-work-first

policy : Inference GPU allocation policy 0 = Capped or 1 = Uncapped

@author: Subir Majumder
"""

#### Function paths
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir))

if project_root not in sys.path:
    sys.path.append(project_root)

from inference_generate_simulate.inference_generate_simulate import generate_inference_resource 
from batch_submission.generate_daily_arrival import generate_daily_batch_arrival_counts
from batch_submission.generate_hourly_arrival import daily_to_second_events
from batch_submission.generate_resource_needs import attach_gpu_timelimit_runtime
from batch_schedule.capacity import generate_capacity_csv
from batch_schedule.scheduler import simulate
from batch_power.aggregate_power_profiles import generate_job_power_profiles
from aggregate.aggregated_batch_inf_power import run_combined_power_analysis

#### Model and runtime file storage paths
from pathlib import Path

here = Path(__file__).resolve()
root = here.parent

INF_TOK_ENERG = root / "data" / "ml_energy_data" / "inference_token_energy.csv"
INF_ARR = root / "ProcessedData_Models" / "Inference" / "nb2_model_alpha.pkl"
INF_CONV_TOK = root / "ProcessedData_Models" / "Inference" / "conv_generated_tokens_model.json"
INF_CODE_TOK = root / "ProcessedData_Models" / "Inference" / "code_generated_tokens_model.json"
    
BATCH_DAILY_ARR = root / "ProcessedData_Models" / "Batch" / "nb_time_mu_alpha_bundle.pkl"
BATCH_HOURLY_ARR = root / "ProcessedData_Models" / "Batch" / "batch_hourlyarrival_ln_alr.pkl"
BATCH_TL_GPU = root / "ProcessedData_Models" / "Batch" / "batch_cluster_tl_gpu_dist.pkl"
BATCH_RUNTIME = root / "ProcessedData_Models" / "Batch" / "batch_runtime_dist.pkl"
TS_MODEL_DIR = root / "ProcessedData_Models" / "Batch" / "power_data_ts_distribution"

TEMP_FILE_STORE = root / "Temp"
INF_TS_TEMP = root / "Temp" / "profile_by_inference_model"
os.makedirs(TEMP_FILE_STORE, exist_ok=True)
os.makedirs(INF_TS_TEMP, exist_ok=True)
    
INF_RES_TS = root / "Temp" / "inference_resource_profiles.csv"
DAILY_ARR = root / "Temp" / "predicted_daily_arrivals_nb2.csv"
DAILY_TIMESTAMPED = root / "Temp" / "synthetic_second_arrivals.csv"
DAILY_TIMESTAMPED_RESOURCE = root / "Temp" / "synthetic_second_arrivals_with_attrs.csv"
CAPACITY_CSV = root / "Temp" / "gpu_capacity.csv"
OUTPUT_SCHEDULED_JOBS_CSV = root / "Temp" / "scheduled_synthetic_jobs.csv"
BATCH_TS_PROFILE_CSV = root / "Temp" / "power_ts.csv"

RESULTS_DIR = root / "Results"
os.makedirs(RESULTS_DIR, exist_ok=True)

import pandas as pd

def main():
    
    #### Main orchestrator
    SIMULATION_START_DATE="2026-01-03"   # First simulated day (YYYY-MM-DD)
    SIMULATION_START = pd.Timestamp(SIMULATION_START_DATE).normalize()
    N_DAYS=90                            # Length of the simulation, in days
    SIMULATION_END = SIMULATION_START + pd.Timedelta(days=N_DAYS)
    
    frac = 0.3
  
    TOTAL_GPUS = 250
    UTIL_RATE = 0.81                 
    
    INF_TRAFFIC_SCALE_RAW = 2.85     
    TRAFFIC_SCALE = frac*INF_TRAFFIC_SCALE_RAW
    VERBOSITY_SCALE=0.6
    TOKEN_SPEED = "F"                

    BAT_TRAFFIC_SCALE_RAW = 0.85     
    N_TZ = 5                         
    LO_COEFF, MID_COEFF, HIGH_COEFF = 1, 1.2, 1.2
    COEFF_MULTIPLIERS = {"low": (1-frac)*BAT_TRAFFIC_SCALE_RAW*LO_COEFF, "mid": (1-frac)*BAT_TRAFFIC_SCALE_RAW*MID_COEFF, "high": (1-frac)*BAT_TRAFFIC_SCALE_RAW*HIGH_COEFF}
    SEGMENT_DURATION= 2*3600         
    SCHEDULER = "SWF"                
    
    policy = 1                       

    ##############################################################################
    ## Inference Workload Simulate 
    ##############################################################################
    if policy == 1:
        inference_gpu = TOTAL_GPUS
    else:
        inference_gpu = int(round(frac * TOTAL_GPUS))
        
    if frac != 0:
        # 1) Generate synthetic TIMESERIES for Inference and simulate
        print("Generate synthetic timeseries for inference and simulate...")
        total_inf_sim  = generate_inference_resource(
            config_csv_path=INF_TOK_ENERG,
            nb2_path=INF_ARR,
            conv_token_path=INF_CONV_TOK,
            code_token_path=INF_CODE_TOK,
            model_timeseries=INF_TS_TEMP,
            start_time=SIMULATION_START,
            end_time=SIMULATION_END,
            traffic_scale=TRAFFIC_SCALE,
            verbosity_scale=VERBOSITY_SCALE,
            token_speed=TOKEN_SPEED,
            available_gpus = inference_gpu)
        total_inf_sim.to_csv(INF_RES_TS, index=False)
        print("Saved aggregated power profile:", INF_RES_TS)
        
    ##############################################################################
    ## Batch Workload Simulate 
    ##############################################################################

    if frac != 1:
        # 2) Generating synthetic user daily job submission for Batch
        print("Generating daily job submission by group...")
        df_daily_jobs = generate_daily_batch_arrival_counts(
            model_path=BATCH_DAILY_ARR,
            start_date=SIMULATION_START_DATE,
            n_days=N_DAYS,
            multipliers=COEFF_MULTIPLIERS,
            n_timezones=N_TZ,)
        df_daily_jobs.to_csv(DAILY_ARR)
        print(f"Daily cluster-wise batch jobs saved to {DAILY_ARR}.")
        
        timestamped_events = daily_to_second_events(
            daily_csv_path=DAILY_ARR,
            hourly_model_path=BATCH_HOURLY_ARR)
        timestamped_events.to_csv(DAILY_TIMESTAMPED)
        print(f"Timestamped batch arrivals saved to {DAILY_TIMESTAMPED}.")
        
        timestamped_events_resources = attach_gpu_timelimit_runtime(
            events_csv_path=DAILY_TIMESTAMPED,
            tl_gpu_model_path=BATCH_TL_GPU,
            runtime_model_path=BATCH_RUNTIME)
        timestamped_events_resources.to_csv(DAILY_TIMESTAMPED_RESOURCE)
        print(f"Batch jobs with resource attributes saved to {DAILY_TIMESTAMPED_RESOURCE}.")
        
        # 3) Find leftover capacity after scheduling inference and Schedule Batch 
        print("Generating leftover capacity...")
        if frac != 0:
            capacity_df = generate_capacity_csv(
                total_gpus=TOTAL_GPUS,
                event_file=DAILY_TIMESTAMPED_RESOURCE,                         
                inference_path=INF_RES_TS,                     # <---- Inference resource uses
            )
        else:
            capacity_df = generate_capacity_csv(
                total_gpus=TOTAL_GPUS,
                event_file=DAILY_TIMESTAMPED_RESOURCE,                         
                inference_path=None                            # <---- Full capacity available to Batch
            )
            
        capacity_df.to_csv(CAPACITY_CSV)
        print(f"Leftover capacity timeline saved to {CAPACITY_CSV}.")
        
        print("Scheduling synthetic workloads...")
        simulate(
            jobs_csv=DAILY_TIMESTAMPED_RESOURCE, 
            cap_csv=CAPACITY_CSV, 
            output_csv=OUTPUT_SCHEDULED_JOBS_CSV, 
            log_level="INFO", 
            first_day=SIMULATION_START_DATE,
            segment_duration= SEGMENT_DURATION,
            scheduler=SCHEDULER) 
        print("Jobs are scheduled.")
        
        # 4) Generate Power Profiles for Batch Jobs
        print("Generating Synthetic Power Profiles...")
        total_power_df = generate_job_power_profiles(
            jobs_csv_path=OUTPUT_SCHEDULED_JOBS_CSV,
            ts_model_dir=TS_MODEL_DIR,
            segment_duration=SEGMENT_DURATION,
        )
        total_power_df.to_csv(BATCH_TS_PROFILE_CSV)
        print(f"Batch power profile saved to {BATCH_TS_PROFILE_CSV}.")
        

    #############################################################################
    # Aggregated Metrics
    #############################################################################
     
    # 5) Overall Datacenter Aggregated Power Metrics
    print("Aggregated Data Center Power Metric...")
    if frac == 0:
        power_df, mean_delay = run_combined_power_analysis(
                batch_power_csv_path=BATCH_TS_PROFILE_CSV,
                batch_jobs_csv_path=OUTPUT_SCHEDULED_JOBS_CSV,
                inference_csv_path=None,
                simulation_start=SIMULATION_START,
                simulation_end=SIMULATION_END)
    
    if frac == 1:
        power_df, mean_delay = run_combined_power_analysis(
                batch_power_csv_path=None,
                batch_jobs_csv_path=None,
                inference_csv_path=INF_RES_TS,
                simulation_start=SIMULATION_START,
                simulation_end=SIMULATION_END)
        
    if frac > 0 and frac < 1:
        power_df, mean_delay = run_combined_power_analysis(
                batch_power_csv_path=BATCH_TS_PROFILE_CSV,
                batch_jobs_csv_path=OUTPUT_SCHEDULED_JOBS_CSV,
                inference_csv_path=INF_RES_TS,
                simulation_start=SIMULATION_START,
                simulation_end=SIMULATION_END)
    
    ##############################      
    # Store final result
    ##############################
    frac_str = str(frac).replace(".", "p")
    util_str = str(UTIL_RATE).replace(".", "p")
    
    power_filename = (
        f"power_frac={frac_str}"
        f"_gpus={TOTAL_GPUS}"
        f"_util={util_str}"
        f"_verb={VERBOSITY_SCALE}"
        f"_speed={TOKEN_SPEED}"
        f"_lo_coeff={LO_COEFF}"
        f"_mid_coeff={MID_COEFF}"
        f"_high_coeff={HIGH_COEFF}"
        f"_seg={SEGMENT_DURATION}"
        f"_ntimezones={N_TZ}"
        f"_scheduler={SCHEDULER}"
        f"_pol={policy}.csv")
    
    power_csv_path = RESULTS_DIR / power_filename
    
    power_df.to_csv(power_csv_path, index=False)
    print(f"Final power_df saved to {power_csv_path}")

if __name__ == '__main__':
    main()