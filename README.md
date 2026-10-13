# GPU-Data-Center-Workload-and-Power-Simulator
The simulator generates synthetic inference and batch workloads from fitted statistical models, schedules workloads under different scheduling policies, generates workload-level power time series, and produces an aggregated data-center power profile.

## Overview
Modern GPU clusters may simultaneously serve latency-sensitive inference workloads and long-running batch jobs. Batch and inference workloads differ substantially in their arrival patterns, GPU requirements, runtimes, and power-consumption profiles. In addition, batch jobs can wait in scheduler-managed queues and, in some cases, can be paused and resumed through checkpointing. Inference workloads, by contrast, are latency-sensitive. Inference workload serving systems seek to keep queues short, process requests through continuous batching, and reroute requests when available GPUs cannot accommodate additional demand.

This simulator provides an end-to-end framework for studying such mixed workloads. The project structure is:
```text
.
├── run.py                                   # main orchestrator (edit config here)
├── inference_generate_simulate/
│   └── inference_generate_simulate.py
├── batch_submission/
│   ├── generate_daily_arrival.py
│   ├── generate_hourly_arrival.py
│   └── generate_resource_needs.py
├── batch_schedule/
│   ├── capacity.py
│   └── scheduler.py
├── batch_power/
│   └── aggregate_power_profiles.py
├── aggregate/
│   └── aggregated_batch_inf_power.py
├── data/
│   └── ml_energy_data/
│       └── inference_token_energy.csv
└── ProcessedData_Models/
|   ├── Inference/                           # fitted inference models
│   │   ├── nb2_model_alpha.pkl
│   │   ├── conv_generated_tokens_model.json
│   │   └── code_generated_tokens_model.json
|   └── Batch/                               # fitted batch models + power templates
│       ├── nb_time_mu_alpha_bundle.pkl
│       ├── batch_hourlyarrival_ln_alr.pkl
│       ├── batch_cluster_tl_gpu_dist.pkl
│       ├── batch_runtime_dist.pkl
│       └── power_data_ts_distribution/       # fitted power templates for batch jobs
├── Temp/
└── Results/
```

Temp/ stores intermediate simulation outputs. Results/ stores final experiment-level power traces. Folders are automatically generated during simulation. 

## Workflow

## Environment

This project was tested with:

- Python 3.10.16
- NumPy 1.26.4
- pandas 2.2.3
- tqdm 4.67.1
- matplotlib 3.10.1
- statsmodels 0.14.4

Install dependencies with:

```bash
pip install -r requirements.txt
