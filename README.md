# PandemicLLM Reproduction (Gemma 3 + Unsloth)

This repository is a modernized reproduction of the [PandemicLLM](https://github.com/miemieyanga/PandemicLLM) algorithm for real-time infectious disease forecasting.

## Objective
Replicate and improve upon the forecasting performance reported in the original paper. Specifically, we target the **Weighted Mean Squared Error (WMSE)** for 1-week ahead hospitalization trends.

**Benchmark to beat: 0.72 WMSE** (as reported for the 13B model in the original study).

## Modernizations
- **Base Model:** Gemma 3 4B (Instruction tuned).
- **Fine-tuning:** Leverages **Unsloth** for 2x faster training and reduced memory footprint.
- **Environment:** Managed via `uv` for robust dependency resolution.
- **Hardware Optimization:** Specifically tuned for RTX 3090/4090 class hardware.

## Quick Start

### 1. Installation
```bash
uv sync
```

### 2. Data Preparation
Ensure the `processed_v5_4.pkl` from the original PandemicLLM repository is available in `../PandemicLLM/data/`.

### 3. Fine-tuning
```bash
uv run src/pandemic_llm_repro/fine_tune.py
```

### 4. Evaluation
```bash
uv run src/pandemic_llm_repro/eval.py --checkpoint outputs/checkpoint-500
```

## Metrics
We monitor:
- **Training Loss:** Cross-entropy on the text reasoning path.
- **WMSE:** Calculated on the ordinal mapping of trend categories:
  - SUBSTANTIAL DECREASING: 0
  - MODERATE DECREASING: 1
  - STABLE: 2
  - MODERATE INCREASING: 3
  - SUBSTANTIAL INCREASING: 4
