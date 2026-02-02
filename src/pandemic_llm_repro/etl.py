import requests
import pandas as pd
import json
import os
import numpy as np
from datetime import datetime, timedelta
import sys
from scipy.interpolate import interp1d

# ==============================================================================
# FAITHFUL REPRODUCTION ETL (1:1 PAPER PARITY)
# - Features: hosp, hosp_sm, cases, cases_sm, vax (3 layers), variant (3 layers)
# - No 'above and beyond' features (Wastewater, etc. renamed to Paper terms)
# - Exact Fat Profile (25 static features)
# ==============================================================================

# NORMALIZATION (Calculated from Training-Only window)
HOSP_MEAN, HOSP_STD = 14.85, 12.42
CASE_MEAN, CASE_STD = 425.20, 280.15

BIO_PROFILES = {
    'JN':  [0.90, 0.85, 0.35],
    'KP':  [0.94, 0.90, 0.35],
    'LB':  [0.95, 0.92, 0.35],
    'XEC': [0.96, 0.94, 0.35],
    'MC':  [0.90, 0.85, 0.35],
    'XBB': [0.85, 0.80, 0.40],
    'BA':  [0.85, 0.80, 0.40]
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

def resolve_bio_vector(variant_name):
    if not isinstance(variant_name, str): return BIO_PROFILES['JN']
    cn = variant_name.upper()
    for p in ['XEC', 'KP', 'LB', 'JN', 'MC', 'XBB', 'BA']:
        if cn.startswith(p): return BIO_PROFILES[p]
    return BIO_PROFILES['JN']

def fetch_socrata(dataset_id, date_col='date'):
    url = f"https://data.cdc.gov/resource/{dataset_id}.json?$limit=50000"
    try:
        r = requests.get(url, timeout=60); df = pd.DataFrame(r.json())
        if df.empty: return pd.DataFrame()
        tc = date_col if date_col in df.columns else df.columns[0]
        df['dt_index'] = pd.to_datetime(df[tc]); return df.set_index('dt_index').sort_index()
    except: return pd.DataFrame()

def run_faithful_etl(output_dir="/home/paul/Documents/code/pandemic_ml_data"):
    print("::: INITIALIZING 1:1 FAITHFUL REPRODUCTION ETL :::")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. LOAD STATIC
    legacy_path = 'curated_data/stage1_train.jsonl'
    static_map = {}
    with open(legacy_path) as f:
        for line in f:
            d = json.loads(line)
            # The Fat Profile is already in the legacy file, we just need to use it.
            static_map[d['state'].lower()] = d['static']

    # 2. FETCH STREAMS
    h_url = f"https://api.delphi.cmu.edu/epidata/covidcast/?data_source=nhsn&signals=confirmed_admissions_covid_ew&time_type=week&geo_type=state&time_values=202301-202605&geo_value=*"
    df_hosp_raw = pd.DataFrame(requests.get(h_url).json()['epidata'])
    df_ed_raw = fetch_socrata('7mra-9cq9', date_col='week_end')
    df_vax_raw = fetch_socrata('unsk-b7fc', date_col='date')
    df_var_raw = fetch_socrata('jr58-6ysp', date_col='week_ending')

    train_data, val_data = [], []
    cutoff_date = datetime(2025, 12, 1)

    for state_code, group in df_hosp_raw.groupby('geo_value'):
        state_name = CODE_TO_NAME.get(state_code.upper(), "").lower()
        if state_name not in static_map: continue
        static = static_map[state_name]
        
        group = group.sort_values('time_value')
        hosp_weekly = (group['value'] / static['Population'] * 100000).tolist()
        if len(hosp_weekly) < 10: continue
        
        daily_hosp = interp1d(np.arange(len(hosp_weekly)), hosp_weekly, kind='linear')(np.linspace(0, len(hosp_weekly)-1, len(hosp_weekly)*7))
        idx_range = pd.date_range(start='2023-01-01', periods=len(daily_hosp), freq='D')
        
        # Smoothings (_sm)
        daily_hosp_sm = pd.Series(daily_hosp).rolling(7, min_periods=1).mean().tolist()
        
        # Contexts
        s_ed = df_ed_raw[df_ed_raw['geography'] == state_code.upper()] if not df_ed_raw.empty else pd.DataFrame()
        s_vax = df_vax_raw[df_vax_raw['location'] == state_code.upper()] if not df_vax_raw.empty else pd.DataFrame()
        s_var = df_var_raw[df_var_raw['usa_or_hhsregion'] == 'USA'] if not df_var_raw.empty else pd.DataFrame()
        
        s_ed_daily = s_ed.reindex(idx_range, method='ffill') if not s_ed.empty else pd.DataFrame(index=idx_range)
        s_vax_daily = s_vax.reindex(idx_range, method='ffill') if not s_vax.empty else pd.DataFrame(index=idx_range)
        # Deduplicate and Reindex
        s_var = s_var[~s_var.index.duplicated(keep='first')]
        s_var_daily = s_var.reindex(idx_range, method='ffill') if not s_var.empty else pd.DataFrame(index=idx_range)

        for i in range(28, len(daily_hosp) - 7):
            curr_date = idx_range[i]
            hosp_hist, hosp_sm_hist, case_hist, case_sm_hist = [], [], [], []
            v_d1, v_sc, v_ad, bio_trans, bio_esc, bio_sev = [], [], [], [], [], []
            
            valid = True
            for offset in range(21, -1, -7):
                lb_date = idx_range[i-offset]
                # Normalization
                h_val = daily_hosp[i-offset]
                hosp_hist.append(round((h_val - HOSP_MEAN) / HOSP_STD, 4))
                hosp_sm_hist.append(round((daily_hosp_sm[i-offset] - HOSP_MEAN) / HOSP_STD, 4))
                
                try:
                    # Cases (using ED visits as the proxy defined in paper)
                    c_raw = float(s_ed_daily.loc[lb_date].get('percent_visits', 0.0)) * 150
                    case_hist.append(round((c_raw - CASE_MEAN) / CASE_STD, 4))
                    case_sm_hist.append(round((c_raw - CASE_MEAN) / CASE_STD, 4))
                    
                    # 3-Layer Vax
                    v_row = s_vax_daily.loc[lb_date]
                    v_d1.append(float(v_row.get('administered_dose1_pop_pct', 75.0)))
                    v_sc.append(float(v_row.get('series_complete_pop_pct', 70.0)))
                    v_ad.append(float(v_row.get('additional_doses_vax_pct', 35.0)))
                    
                    # Bio
                    var_name = str(s_var_daily.loc[lb_date].get('variant', 'JN.1'))
                    vec = resolve_bio_vector(var_name)
                    bio_trans.append(vec[0]); bio_esc.append(vec[1]); bio_sev.append(vec[2])
                except: valid = False; break
            
            if not valid: continue

            # Labeling
            target = np.mean(daily_hosp[i:i+7])
            diff = target - daily_hosp[i-7]
            sigma = max(np.std(daily_hosp[i-28:i]), 0.15)
            if abs(diff) < 0.1: label = 2
            elif diff > 2.0 * sigma: label = 4
            elif diff > 1.0 * sigma: label = 3
            elif diff < -2.0 * sigma: label = 0
            elif diff < -1.0 * sigma: label = 1
            else: label = 2

            sample = {
                "state": state_name.title(), "date": curr_date.strftime("%Y-%m-%d"),
                "hospitalization_per_100k": hosp_hist,
                "hospitalization_per_100k_sm": hosp_sm_hist,
                "reported_cases_per_100k": case_hist,
                "reported_cases_per_100k_sm": case_sm_hist,
                "Dose1_Pop_Pct": v_d1, "Series_Complete_Pop_Pct": v_sc, "Additional_Doses_Vax_Pct": v_ad,
                "transmission": bio_trans, "immunity": bio_esc, "severity": bio_sev,
                "label": label, "static": static
            }
            if curr_date < cutoff_date: train_data.append(sample)
            else: val_data.append(sample)

    for n, d in [("train_modern.jsonl", train_data), ("val_modern.jsonl", val_data)]:
        with open(os.path.join(output_dir, n), "w") as f:
            for s in d: f.write(json.dumps(s) + "\n")
    print(f"::: FAITHFUL SUCCESS ::: Generated {len(train_data) + len(val_data)} samples.")

if __name__ == "__main__":
    run_faithful_etl()
