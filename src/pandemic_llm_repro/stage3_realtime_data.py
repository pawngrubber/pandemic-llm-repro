# /// script
# dependencies = [
#   "requests",
#   "pandas",
#   "numpy",
# ]
# ///

import requests
import pandas as pd
import json
import os
import numpy as np

# Stage 3: Real-Time Data Ingestion (Modern Delphi API)
# Uses Standard Deviation (sigma) for categorical thresholding to match
# the original paper's statistical rigor in a lower-signal era.

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

def fetch_delphi_data():
    print("STAGE 3: Fetching 2025-2026 data and applying Sigma-Thresholding...")
    
    states = ",".join(STATE_POPULATIONS.keys())
    time_range = "202501-202605"
    url = f"https://api.delphi.cmu.edu/epidata/covidcast/?data_source=nhsn&signals=confirmed_admissions_covid_ew&time_type=week&geo_type=state&time_values={time_range}&geo_value={states}"
    
    response = requests.get(url)
    if response.status_code != 200: return
    
    df = pd.DataFrame(response.json().get('epidata', []))
    modern_samples = []
    
    # Calculate global sigma for hospitalization changes in 2025 to normalize the threshold
    df['rate'] = df.apply(lambda x: (x['value'] / STATE_POPULATIONS[x['geo_value']]) * 100000, axis=1)
    
    # We'll use a rolling window sigma to handle local volatility per state
    for state, group in df.groupby('geo_value'):
        group = group.sort_values('time_value')
        rates = group['rate'].tolist()
        times = group['time_value'].tolist()
        
        # Calculate the historical standard deviation of weekly changes
        # This allows the model to differentiate between "normal noise" and "real shifts"
        deltas = np.diff(rates)
        state_sigma = np.std(deltas) if len(deltas) > 0 else 0.5
        
        # Floor the sigma at 0.2 to prevent tiny fluctuations from being labeled as "Substantial"
        state_sigma = max(state_sigma, 0.2)

        for i in range(4, len(rates)):
            history = rates[i-4:i]
            target_val = rates[i]
            diff = target_val - history[-1] # Change from current week
            
            # Sigma-based labels:
            # > 2.0 sigma = Substantial Increase (4)
            # > 1.0 sigma = Moderate Increase (3)
            # < -2.0 sigma = Substantial Decrease (0)
            # < -1.0 sigma = Moderate Decrease (1)
            # Else = Stable (2)
            
            if diff > 2.0 * state_sigma: label = 4
            elif diff > 1.0 * state_sigma: label = 3
            elif diff < -2.0 * state_sigma: label = 0
            elif diff < -1.0 * state_sigma: label = 1
            else: label = 2
            
            sample = {
                "state": state.upper(),
                "week_start": str(times[i]),
                "history": [round(x, 2) for x in history],
                "static": {
                    "Population": STATE_POPULATIONS[state],
                    "state_name": state.upper(),
                    "sigma_threshold": round(state_sigma, 4)
                },
                "label": label
            }
            modern_samples.append(sample)
            
    os.makedirs("curated_data", exist_ok=True)
    with open("curated_data/stage3_modern.jsonl", "w") as f:
        for s in modern_samples:
            f.write(json.dumps(s) + "\n")
            
    print(f"STAGE 3 COMPLETE: Generated {len(modern_samples)} samples with Sigma Calibration.")

if __name__ == "__main__":
    fetch_delphi_data()
