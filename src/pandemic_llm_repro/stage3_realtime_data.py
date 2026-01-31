# /// script
# dependencies = [
#   "requests",
#   "pandas",
# ]
# ///

import requests
import pandas as pd
import json
import os
from datetime import datetime

# Stage 3: Real-Time Data Ingestion (Modern Delphi API)
# Fetches NHSN weekly hospitalization data through 2025 and early 2026.

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
    print("STAGE 3: Fetching 2025-2026 weekly hospitalization data from Delphi Epidata API...")
    
    # We'll pull data for all states for 2025 and early 2026
    states = ",".join(STATE_POPULATIONS.keys())
    # Format: YYYYWW. 202501 to 202605
    time_range = "202501-202605"
    
    url = f"https://api.delphi.cmu.edu/epidata/covidcast/?data_source=nhsn&signals=confirmed_admissions_covid_ew&time_type=week&geo_type=state&time_values={time_range}&geo_value={states}"
    
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return
    
    data = response.json().get('epidata', [])
    if not data:
        print("No data returned from Delphi API.")
        return
    
    df = pd.DataFrame(data)
    print(f"  - Downloaded {len(df)} weekly state records.")
    
    modern_samples = []
    for state, group in df.groupby('geo_value'):
        group = group.sort_values('time_value')
        
        pop = STATE_POPULATIONS[state]
        # Convert absolute admissions to rate per 100k
        group['rate'] = (group['value'] / pop) * 100000
        
        rates = group['rate'].tolist()
        times = group['time_value'].tolist()
        
        # We need 4 weeks of history + 1 week target
        for i in range(4, len(rates)):
            history = rates[i-4:i]
            target_val = rates[i]
            
            # Labeling logic matching paper math
            baseline = sum(history) / 4
            diff = target_val - baseline
            
            # Mapping diff to ordinal classes (Simplified thresholds)
            if diff < -0.5: label = 0
            elif diff < -0.1: label = 1
            elif diff < 0.1: label = 2
            elif diff < 0.5: label = 3
            else: label = 4
            
            sample = {
                "state": state.upper(),
                "week_start": str(times[i]),
                "history": [round(x, 2) for x in history],
                "static": {
                    "Population": pop,
                    "state_name": state.upper()
                },
                "label": label
            }
            modern_samples.append(sample)
            
    os.makedirs("curated_data", exist_ok=True)
    with open("curated_data/stage3_modern.jsonl", "w") as f:
        for s in modern_samples:
            f.write(json.dumps(s) + "\n")
            
    print(f"STAGE 3 COMPLETE: Generated {len(modern_samples)} modern samples (2025-2026).")

if __name__ == "__main__":
    fetch_delphi_data()