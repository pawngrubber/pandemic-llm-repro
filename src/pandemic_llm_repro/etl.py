import requests
import pandas as pd
import json
import os
import numpy as np
from datetime import datetime, timedelta
import sys

# STAGE 1 ETL: CPU-Only Base Dataset Generation
# - Fetches 2023-2026 Hospitalizations
# - Linear Daily Smoothing (Daily Sliding Windows)
# - 25-Feature Static Sociological Injection
# - Methodological Sigma-Thresholding

NAME_TO_CODE = {
    'alabama': 'al', 'alaska': 'ak', 'arizona': 'az', 'arkansas': 'ar', 'california': 'ca',
    'colorado': 'co', 'connecticut': 'ct', 'delaware': 'de', 'florida': 'fl', 'georgia': 'ga',
    'hawaii': 'hi', 'idaho': 'id', 'illinois': 'il', 'indiana': 'in', 'iowa': 'ia',
    'kansas': 'ks', 'kentucky': 'ky', 'louisiana': 'la', 'maine': 'me', 'maryland': 'md',
    'massachusetts': 'ma', 'michigan': 'mi', 'minnesota': 'mn', 'mississippi': 'ms', 'missouri': 'mo',
    'montana': 'mt', 'nebraska': 'ne', 'nevada': 'nv', 'new hampshire': 'nh', 'new jersey': 'nj',
    'new mexico': 'nm', 'new york': 'ny', 'north carolina': 'nc', 'north dakota': 'nd', 'ohio': 'oh',
    'oklahoma': 'ok', 'oregon': 'or', 'pennsylvania': 'pa', 'rhode island': 'ri', 'south carolina': 'sc',
    'south dakota': 'sd', 'tennessee': 'tn', 'texas': 'tx', 'utah': 'ut', 'vermont': 'vt',
    'virginia': 'va', 'washington': 'wa', 'west virginia': 'wv', 'wisconsin': 'wi', 'wyoming': 'wy'
}

def load_fat_static_features():
    import pickle
    pkl_path = '../PandemicLLM/data/processed_v5_4.pkl'
    try:
        import pandas.core.indexes.base
        sys.modules['pandas.core.indexes.numeric'] = pandas.core.indexes.base
        pandas.core.indexes.base.Int64Index = pandas.Index
    except: pass
    with open(pkl_path, 'rb') as f:
        raw_data = pickle.load(f)
    df = raw_data.sta_dy_aug_data
    cols = ['Population', 'under_20', 'over_65', 'White', 'Black', 'Multiple_race', 'Not_Hispanic', 'Hispanic', 
            'medicaid_coverage', 'medicare_coverage', 'uninsured_percent', 'medicaid_spending', 
            'private_health_insurance_spending', 'medicare_spending_by_residence', 'health_care_spending', 
            'healthcare_utilization', 'poor_health_status', 'adults_at_high_risk', 'poverty_rate', 
            'social_vulnerability_index', 'Healthcare Access and Quality Index', 'Older_at_high_risk', 
            'dem_percent', 'rep_percent', 'other_percent']
    state_map = {}
    for state, group in df.groupby('state_name'):
        state_map[state.lower()] = group[cols].iloc[0].to_dict()
    return state_map

def run_base_etl(output_dir="/home/paul/Documents/code/pandemic_ml_data"):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "ml_dataset_base.jsonl")
    
    static_features = load_fat_static_features()
    codes = ",".join(NAME_TO_CODE.values())
    
    print("ETL: Fetching Hospitalizations (2023-2026).")
    url = f"https://api.delphi.cmu.edu/epidata/covidcast/?data_source=nhsn&signals=confirmed_admissions_covid_ew&time_type=week&geo_type=state&time_values=202301-202605&geo_value={codes}"
    h_data = requests.get(url).json().get('epidata', [])
    df_hosp = pd.DataFrame(h_data)

    all_samples = []
    print(f"ETL: Processing {len(df_hosp)} raw records into smoothed daily windows.")
    
    for state_name, code in NAME_TO_CODE.items():
        if state_name not in static_features: continue
        static = static_features[state_name]
        group = df_hosp[df_hosp['geo_value'] == code].sort_values('time_value')
        if group.empty: continue
        
        group['rate'] = (group['value'] / static['Population']) * 100000
        weekly_rates = group['rate'].tolist()
        daily_rates = []
        for i in range(len(weekly_rates) - 1):
            daily_rates.extend(np.linspace(weekly_rates[i], weekly_rates[i+1], 7, endpoint=False))
        daily_rates.append(weekly_rates[-1])
        
        for i in range(28, len(daily_rates) - 7):
            history_days = daily_rates[i-28:i]
            weekly_history = [round(np.mean(history_days[j:j+7]), 2) for j in range(0, 28, 7)]
            target_val = np.mean(daily_rates[i:i+7])
            sigma = max(np.std(weekly_history), 0.15)
            diff = target_val - weekly_history[-1]
            
            if abs(diff) < 0.1: label = 2
            elif diff > 2.0 * sigma: label = 4
            elif diff > 1.0 * sigma: label = 3
            elif diff < -2.0 * sigma: label = 0
            elif diff < -1.0 * sigma: label = 1
            else: label = 2
            
            all_samples.append({
                "state": state_name.title(),
                "date": (datetime(2023, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d"),
                "history": weekly_history,
                "label": label,
                "static": static,
                "vax_shield": 0.85 if static['medicaid_coverage'] < 0.15 else 0.70,
                "variant_risk": 0.4
            })

    with open(out_path, "w") as f:
        for s in all_samples:
            f.write(json.dumps(s) + "\n")

    print(f"STAGE 1 ETL SUCCESS: Base dataset created at {out_path} ({len(all_samples)} samples).")
    return out_path

if __name__ == "__main__":
    run_base_etl()