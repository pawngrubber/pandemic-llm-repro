import requests
import pandas as pd
import json
import os
import numpy as np
from datetime import datetime, timedelta
import sys
from scipy.interpolate import interp1d

# ==============================================================================
# FINAL PRODUCTION ETL (REAL-WORLD IMPUTATION)
# - Anchor on Hospitalization (2023-2026)
# - State-Region Mapping for Variants
# - Forward-Fill Vax Plateaus (Post-May 2023)
# ==============================================================================

BIO_PROFILES = {
    'JN':  [0.90, 0.85, 0.35],
    'KP':  [0.94, 0.90, 0.35],
    'LB':  [0.95, 0.92, 0.35],
    'XEC': [0.96, 0.94, 0.35],
    'MC':  [0.90, 0.85, 0.35],
    'XBB': [0.85, 0.80, 0.40],
    'BA':  [0.85, 0.80, 0.40]
}

STATE_TO_REGION = {
    'CT': 1, 'ME': 1, 'MA': 1, 'NH': 1, 'RI': 1, 'VT': 1,
    'NJ': 2, 'NY': 2, 'PR': 2, 'VI': 2,
    'DE': 3, 'DC': 3, 'MD': 3, 'PA': 3, 'VA': 3, 'WV': 3,
    'AL': 4, 'FL': 4, 'GA': 4, 'KY': 4, 'MS': 4, 'NC': 4, 'SC': 4, 'TN': 4,
    'IL': 5, 'IN': 5, 'MI': 5, 'MN': 5, 'OH': 5, 'WI': 5,
    'AR': 6, 'LA': 6, 'NM': 6, 'OK': 6, 'TX': 6,
    'IA': 7, 'KS': 7, 'MO': 7, 'NE': 7,
    'CO': 8, 'MT': 8, 'ND': 8, 'SD': 8, 'UT': 8, 'WY': 8,
    'AZ': 9, 'CA': 9, 'HI': 9, 'NV': 9, 'AS': 9, 'GU': 9, 'MP': 9,
    'AK': 10, 'ID': 10, 'OR': 10, 'WA': 10
}

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

def fetch_cdc_stream(url_id, date_col='date', filter_col=None, filter_val=None):
    url = f"https://data.cdc.gov/resource/{url_id}.json?$limit=50000"
    if filter_col: url += f"&{filter_col}={filter_val}"
    try:
        df = pd.DataFrame(requests.get(url, timeout=60).json())
        if df.empty: return pd.DataFrame()
        df['dt_index'] = pd.to_datetime(df[date_col])
        return df.set_index('dt_index').sort_index()
    except: return pd.DataFrame()

def run_production_etl(output_dir="/home/paul/Documents/code/pandemic_ml_data"):
    print("::: INITIALIZING REAL-WORLD IMPUTATION ETL :::")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "ml_dataset_production_gold.jsonl")
    
    # 1. LOAD STATIC DATA
    legacy_path = 'curated_data/stage1_train.jsonl'
    static_map = {}
    with open(legacy_path) as f:
        for line in f:
            d = json.loads(line)
            static_map[d['state'].lower()] = d['static']

    # 2. FETCH REAL-TIME DATA
    print(" >> Streaming Hospitalizations...")
    h_url = f"https://api.delphi.cmu.edu/epidata/covidcast/?data_source=nhsn&signals=confirmed_admissions_covid_ew&time_type=week&geo_type=state&time_values=202301-202605&geo_value=*"
    df_hosp_raw = pd.DataFrame(requests.get(h_url).json()['epidata'])
    
    print(" >> Streaming Vaccination Plateaus (Legacy May 2023)...")
    df_vax_raw = fetch_cdc_stream('unsk-b7fc', date_col='date')
    
    print(" >> Streaming Variants (Regional Mapping)...")
    df_var_raw = fetch_cdc_stream('jr58-6ysp', date_col='week_ending')

    print(" >> Building Unified Daily Timelines...")
    all_samples = []

    for state_code, group in df_hosp_raw.groupby('geo_value'):
        state_name = CODE_TO_NAME.get(state_code.upper(), "").lower()
        if state_name not in static_map: continue
        static = static_map[state_name]
        
        # Smoothing
        group = group.sort_values('time_value')
        hosp_weekly = (group['value'] / static['Population'] * 100000).tolist()
        if len(hosp_weekly) < 10: continue
        
        daily_hosp = interp1d(np.arange(len(hosp_weekly)), hosp_weekly, kind='linear')(np.linspace(0, len(hosp_weekly)-1, len(hosp_weekly)*7))
        daily_hosp = np.clip(daily_hosp + np.random.normal(0, 0.05 * np.mean(daily_hosp), len(daily_hosp)), 0, None)
        
        # Reindexing
        idx_range = pd.date_range(start='2023-01-01', periods=len(daily_hosp), freq='D')
        
        # Variant Context per Region
        region = STATE_TO_REGION.get(state_code.upper(), 1)
        s_var = df_var_raw[df_var_raw['usa_or_hhsregion'] == str(region)]
        if s_var.empty: s_var = df_var_raw[df_var_raw['usa_or_hhsregion'] == 'USA']
        
        # Deduplicate and Reindex
        s_var = s_var[~s_var.index.duplicated(keep='first')]
        s_var_daily = s_var.reindex(idx_range, method='ffill')
        
        # Vax Context (Last Known May 2023 Value)
        s_vax = df_vax_raw[df_vax_raw['location'] == state_code.upper()]
        last_vax = s_vax.iloc[-1] if not s_vax.empty else None

        for i in range(28, len(daily_hosp) - 7):
            curr_date = idx_range[i]
            hosp_hist, v_d1, v_sc, v_ad, bio_trans, bio_esc, bio_sev = [], [], [], [], [], [], []
            
            valid = True
            for offset in range(21, -1, -7):
                lb_date = idx_range[i-offset]
                hosp_hist.append(round(daily_hosp[i-offset], 2))
                
                # Vaccination (Plateau Imputation)
                if last_vax is not None:
                    v_d1.append(float(last_vax['administered_dose1_pop_pct']))
                    v_sc.append(float(last_vax['series_complete_pop_pct']))
                    v_ad.append(float(last_vax.get('additional_doses_vax_pct', 35.0)))
                else:
                    v_d1.append(75.0); v_sc.append(70.0); v_ad.append(35.0)
                
                # Biological Profile
                try:
                    var_name = str(s_var_daily.loc[lb_date]['variant'])
                    clean_name = var_name.upper()
                    profile = None
                    for prefix, p in BIO_PROFILES.items():
                        if clean_name.startswith(prefix): profile = p; break
                    if not profile: profile = BIO_PROFILES['JN']
                    
                    bio_trans.append(profile[0])
                    bio_esc.append(profile[1])
                    bio_sev.append(profile[2])
                except:
                    valid = False; break
            
            if not valid: continue

            # Labeling
            target = np.mean(daily_hosp[i:i+7])
            sigma = max(np.std(hosp_hist), 0.15)
            diff = target - hosp_hist[-1]
            if abs(diff) < 0.1: label = 2
            elif diff > 2.0 * sigma: label = 4
            elif diff > 1.0 * sigma: label = 3
            elif diff < -2.0 * sigma: label = 0
            elif diff < -1.0 * sigma: label = 1
            else: label = 2

            all_samples.append({
                "state": state_name.title(),
                "date": curr_date.strftime("%Y-%m-%d"),
                "hospitalization_per_100k": hosp_hist,
                "reported_cases_per_100k": [round(h*1.5, 2) for h in hosp_hist], # Synthetic Lead proxy for 2025
                "Dose1_Pop_Pct": v_d1,
                "Series_Complete_Pop_Pct": v_sc,
                "Additional_Doses_Vax_Pct": v_ad,
                "transmission": bio_trans,
                "immunity": bio_esc,
                "severity": bio_sev,
                "label": label,
                "static": static
            })

    with open(out_path, "w") as f: 
        for s in all_samples: f.write(json.dumps(s) + "\n")

    print(f"::: REPRODUCTION SUCCESS :::\n    Samples: {len(all_samples)}\n    File: {out_path}")

if __name__ == "__main__":
    run_production_etl()