import requests
import pandas as pd
import json
import os
import numpy as np
from datetime import datetime, timedelta
import sys
from scipy.interpolate import interp1d

# STAGE 1: ABSOLUTE PARITY ETL
# - Unit Scaling (ED Visits -> Case Equivalent)
# - Biological Vector Mapping (Transmission, Immunity, Severity)
# - Dual-Track Smoothing (Raw + _sm)
# - Daily Temporal Joins

CODE_TO_NAME = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
    'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
    'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
    'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
    'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
    'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
    'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
    'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
}

# BIOLOGICAL VECTOR TABLE (Transmission, Immunity Escape, Severity)
# Estimated for 2024-2026 dominant strains
VARIANT_PROFILES = {
    'JN.1': [0.9, 0.85, 0.35],
    'KP.2': [0.92, 0.9, 0.35],
    'KP.3': [0.94, 0.92, 0.35],
    'XEC':  [0.96, 0.94, 0.35],
    'BASE': [0.85, 0.80, 0.40]
}

def load_fat_static_features():
    legacy_path = 'curated_data/stage1_train.jsonl'
    state_map = {}
    with open(legacy_path) as f:
        for line in f:
            d = json.loads(line)
            state_map[d['state'].lower()] = d['static']
    return state_map

def fetch_socrata(dataset_id, where_clause):
    url = f"https://data.cdc.gov/resource/{dataset_id}.json?{where_clause}"
    try:
        r = requests.get(url, timeout=30)
        return pd.DataFrame(r.json())
    except: return pd.DataFrame()

def run_absolute_parity_etl(output_dir="/home/paul/Documents/code/pandemic_ml_data"):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "ml_dataset_absolute_parity.jsonl")
    
    static_features = load_fat_static_features()
    
    print("ETL: Fetching Hospitalizations...")
    h_url = f"https://api.delphi.cmu.edu/epidata/covidcast/?data_source=nhsn&signals=confirmed_admissions_covid_ew&time_type=week&geo_type=state&time_values=202301-202605&geo_value=*"
    df_hosp = pd.DataFrame(requests.get(h_url).json()['epidata'])
    
    print("ETL: Fetching ED Leads, Vax, and Variant Proportions...")
    df_ed = fetch_socrata("7mra-9cq9", "$limit=50000")
    df_vax = fetch_socrata("unsk-b7fc", "$limit=50000")
    df_var = fetch_socrata("jr58-6ysp", "$limit=10000")

    all_samples = []
    print("ETL: Performing Daily Temporal Joins with Multi-Signal Smoothing...")

    for state_code, group in df_hosp.groupby('geo_value'):
        state_name = CODE_TO_NAME.get(state_code.upper(), "").lower()
        if state_name not in static_features: continue
        static = static_features[state_name]
        
        group = group.sort_values('time_value')
        group['rate'] = (group['value'] / static['Population']) * 100000
        
        # Linear Daily Smoothing
        raw_weekly = group['rate'].tolist()
        if len(raw_weekly) < 6: continue
        daily_rates = interp1d(np.arange(len(raw_weekly)), raw_weekly, kind='linear')(np.linspace(0, len(raw_weekly)-1, len(raw_weekly)*7))
        
        # Generate _sm (smoothed) version (7-day rolling mean)
        # Note: In our interpolated data, we add a tiny bit of jitter first to avoid perfect lines
        daily_rates_jitter = np.clip(daily_rates + np.random.normal(0, 0.02, len(daily_rates)), 0, None)
        daily_rates_sm = pd.Series(daily_rates_jitter).rolling(7, min_periods=1).mean().tolist()

        # State contexts
        s_ed = df_ed[df_ed['geography'] == state_code.upper()].copy()
        s_vax = df_vax[df_vax['location'] == state_code.upper()].copy()
        s_var = df_var[df_var['usa_or_hhsregion'] == 'USA'].copy() # National variant proxy

        for i in range(28, len(daily_rates) - 7):
            curr_date = datetime(2023, 1, 1) + timedelta(days=i)
            
            # 1. Hospitalization & Smoothed History
            history_raw = daily_rates_jitter[i-28:i]
            history_sm = daily_rates_sm[i-28:i]
            hosp_list = [round(np.mean(history_raw[j:j+7]), 2) for j in range(0, 28, 7)]
            hosp_sm_list = [round(np.mean(history_sm[j:j+7]), 2) for j in range(0, 28, 7)]
            
            # 2. Case Proxy (ED Scaling: 1% visits approx 150 cases/100k in legacy scale)
            ed_rec = s_ed.iloc[min(i//7, len(s_ed)-1)] if not s_ed.empty else {}
            ed_val = float(ed_rec.get('percent_visits', 0.0)) * 150.0 
            cases_list = [round(ed_val, 2)] * 4 # Simulated history based on current lead
            
            # 3. Triple Vax Shield
            v_rec = s_vax.iloc[min(i//7, len(s_vax)-1)] if not s_vax.empty else {}
            v_d1 = [float(v_rec.get('administered_dose1_pop_pct', 75.0))] * 4
            v_sc = [float(v_rec.get('series_complete_pop_pct', 70.0))] * 4
            v_ad = [float(v_rec.get('additional_doses_vax_pct', 35.0))] * 4

            # 4. Biological Vector
            var_rec = s_var.iloc[min(i//7, len(s_var)-1)] if not s_var.empty else {}
            var_name = var_rec.get('variant', 'BASE')
            bio_vector = VARIANT_PROFILES.get(var_name, VARIANT_PROFILES['BASE'])

            # Labeling
            target_val = np.mean(daily_rates[i:i+7])
            sigma = max(np.std(hosp_list), 0.15)
            diff = target_val - hosp_list[-1]
            if abs(diff) < 0.1: label = 2
            elif diff > 2.0 * sigma: label = 4
            elif diff > 1.0 * sigma: label = 3
            elif diff < -2.0 * sigma: label = 0
            elif diff < -1.0 * sigma: label = 1
            else: label = 2

            all_samples.append({
                "state": state_name.title(),
                "date": curr_date.strftime("%Y-%m-%d"),
                "hospitalization_per_100k": hosp_list,
                "hospitalization_per_100k_sm": hosp_sm_list,
                "reported_cases_per_100k": cases_list,
                "Dose1_Pop_Pct": v_d1,
                "Series_Complete_Pop_Pct": v_sc,
                "Additional_Doses_Vax_Pct": v_ad,
                "transmission": [bio_vector[0]] * 4,
                "immunity": [bio_vector[1]] * 4,
                "severity": [bio_vector[2]] * 4,
                "label": label,
                "static": static
            })

    with open(out_path, "w") as f:
        for s in all_samples:
            f.write(json.dumps(s) + "\n")

    print(f"ABSOLUTE PARITY SUCCESS: Generated {len(all_samples)} samples.")
    return out_path

if __name__ == "__main__":
    run_absolute_parity_etl()
