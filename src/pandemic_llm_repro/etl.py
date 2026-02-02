import requests
import pandas as pd
import json
import os
import numpy as np
from datetime import datetime

# Consolidated ETL for PandemicLLM Reproduction
# Methodology: Sigma-Thresholding from paper Section 8.1

STATE_POPULATIONS = {
    'al': 5024279, 'ak': 733391, 'az': 7151502, 'ar': 3011524, 'ca': 39538223,
    'co': 5773714, 'ct': 3605944, 'de': 989948, 'fl': 21538187, 'ga': 10711908,
    'hi': 1455271, 'id': 1839106, 'il': 12812508, 'in': 6785528, 'ia': 3190369,
    'ks': 2937880, 'ky': 4505836, 'la': 4657757, 'me': 1362359, 'md': 6177224,
    'ma': 7029917, 'mi': 10077331, 'mn': 5706494, 'ms': 2961279, 'mo': 6154913,
    'mt': 1084225, 'ne': 1961504, 'nv': 3104614, 'nh': 1377529, 'nj': 9288994,
    'nm': 2117522, 'ny': 20201249, 'nc': 10439388, 'nd': 779094, 'oh': 11799448,
    'ok': 3959353, 'or': 4237256, 'pa': 13002700, 'ri': 1097379, 'sc': 5118425,
    'sd': 886667, 'tn': 6910840, 'tx': 29145505, 'ut': 3271616, 'vt': 643077,
    'va': 8631393, 'wa': 7705281, 'wv': 1793716, 'wi': 5893718, 'wy': 576851
}

def run_2025_etl(output_dir="/home/paul/Documents/code/pandemic_ml_data"):
    """
    Fetches full-year 2025 hospitalization data and generates a machine learning dataset.
    Labels are derived using rolling window standard deviation (sigma) to normalize 
    volatility across states.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "ml_dataset_2025.jsonl")
    
    print(f"ETL: Fetching 2025 weekly hospitalization data for all 50 states...")
    states = ",".join(STATE_POPULATIONS.keys())
    # Full year 2025 (Week 1 to Week 52)
    time_range = "202501-202552"
    url = f"https://api.delphi.cmu.edu/epidata/covidcast/?data_source=nhsn&signals=confirmed_admissions_covid_ew&time_type=week&geo_type=state&time_values={time_range}&geo_value={states}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        raw_data = response.json().get('epidata', [])
    except Exception as e:
        print(f"ETL Error: Failed to fetch data - {e}")
        return

    if not raw_data:
        print("ETL Error: No data returned from API for 2025 range.")
        return

    df = pd.DataFrame(raw_data)
    dataset = []

    print(f"ETL: Processing {len(df)} records into ML samples...")
    for state, group in df.groupby('geo_value'):
        group = group.sort_values('time_value')
        pop = STATE_POPULATIONS[state]
        group['rate'] = (group['value'] / pop) * 100000
        
        rates = group['rate'].tolist()
        times = group['time_value'].tolist()
        
        # We need 4 weeks of history + 1 target week
        for i in range(4, len(rates)):
            history = rates[i-4:i]
            target_val = rates[i]
            
            # Normalize thresholds using rolling standard deviation of the window
            sigma = np.std(history)
            if sigma < 0.15: sigma = 0.15 # Baseline floor for low-signal eras
            
            diff = target_val - history[-1]
            
            # Ordinal labels based on sigma distance
            if diff > 2.0 * sigma: label = 4
            elif diff > 1.0 * sigma: label = 3
            elif diff < -2.0 * sigma: label = 0
            elif diff < -1.0 * sigma: label = 1
            else: label = 2
            
            dataset.append({
                "state": state.upper(),
                "week": str(times[i]),
                "history": [round(x, 2) for x in history],
                "label": label,
                "sigma_context": round(sigma, 4),
                "delta": round(diff, 4),
                "static": {
                    "population": pop,
                    "state_name": state.upper()
                }
            })

    with open(out_path, "w") as f:
        for s in dataset:
            f.write(json.dumps(s) + "\n")

    print(f"ETL SUCCESS: Generated {len(dataset)} samples.")
    print(f"Dataset Location: {out_path}")
    return out_path

if __name__ == "__main__":
    run_2025_etl()
