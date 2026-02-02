import requests
import pandas as pd
import json
import os
import numpy as np
from datetime import datetime, timedelta
import sys
from scipy.interpolate import interp1d

# ==============================================================================
# STAGE 1: THE FAITHFUL 2026 RECONSTRUCTION (PRODUCTION FINAL)
# - Lead Signal: Wastewater (2ew6-ywp6)
# - Immune Shield: 2024-2025 Vaccine Coverage (3vsc-q5ub)
# - Target: Hospitalizations (NHSN)
# ==============================================================================

TECH_BIO_PROFILES = {
    'JN.1': [0.88, 0.82, 0.35],
    'KP.2': [0.91, 0.88, 0.35],
    'KP.3': [0.93, 0.90, 0.35],
    'XEC':  [0.95, 0.94, 0.35],
    'BASE': [0.85, 0.80, 0.40]
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

def fetch_socrata(dataset_id, date_col='date'):
    url = f"https://data.cdc.gov/resource/{dataset_id}.json?$limit=50000"
    try:
        r = requests.get(url, timeout=60)
        df = pd.DataFrame(r.json())
        if df.empty: return pd.DataFrame()
        # Find date column
        target_col = date_col if date_col in df.columns else df.columns[0]
        df['dt_index'] = pd.to_datetime(df[target_col])
        return df.set_index('dt_index').sort_index()
    except: return pd.DataFrame()

def run_faithful_2026_etl(output_dir="/home/paul/Documents/code/pandemic_ml_data"):
    print("::: INITIALIZING FAITHFUL 2026 ETL :::")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "ml_dataset_final_production.jsonl")
    
    # 1. LOAD STATIC DATA
    legacy_path = 'curated_data/stage1_train.jsonl'
    static_map = {}
    with open(legacy_path) as f:
        for line in f:
            d = json.loads(line)
            static_map[d['state'].lower()] = d['static']

    # 2. FETCH MULTI-MODAL STREAMS
    print(" >> Streaming Hospitalizations (NHSN)...")
    h_url = f"https://api.delphi.cmu.edu/epidata/covidcast/?data_source=nhsn&signals=confirmed_admissions_covid_ew&time_type=week&geo_type=state&time_values=202301-202605&geo_value=*"
    df_hosp_raw = pd.DataFrame(requests.get(h_url).json()['epidata'])
    
    print(" >> Streaming Wastewater (Lead Signal 2ew6-ywp6)...")
    df_ww_raw = fetch_socrata('2ew6-ywp6', date_col='reporting_cutoff_date')
    
    print(" >> Streaming 2024-2025 Vax Coverage (3vsc-q5ub)...")
    df_vax_raw = fetch_socrata('3vsc-q5ub', date_col='week_ending')
    
    print(" >> Streaming Variants (jr58-6ysp)...")
    df_var_raw = fetch_socrata('jr58-6ysp', date_col='week_ending')

    all_samples = []

    for state_code, group in df_hosp_raw.groupby('geo_value'):
        state_name = CODE_TO_NAME.get(state_code.upper(), "").lower()
        if state_name not in static_map: continue
        static = static_map[state_name]
        
        group = group.sort_values('time_value')
        hosp_weekly = (group['value'] / static['Population'] * 100000).tolist()
        if len(hosp_weekly) < 10: continue
        
        daily_hosp = interp1d(np.arange(len(hosp_weekly)), hosp_weekly, kind='linear')(np.linspace(0, len(hosp_weekly)-1, len(hosp_weekly)*7))
        daily_hosp = np.clip(daily_hosp + np.random.normal(0, 0.05 * np.mean(daily_hosp), len(daily_hosp)), 0, None)
        
        idx_range = pd.date_range(start='2023-01-01', periods=len(daily_hosp), freq='D')
        
        # Region Mapping
        region = STATE_TO_REGION.get(state_code.upper(), 1)
        s_var = df_var_raw[df_var_raw['usa_or_hhsregion'] == str(region)] if not df_var_raw.empty else pd.DataFrame()
        s_var_daily = s_var[~s_var.index.duplicated(keep='first')].reindex(idx_range, method='ffill') if not s_var.empty else pd.DataFrame(index=idx_range)
        
        # Wastewater Join
        s_ww = df_ww_raw[df_ww_raw['jurisdiction'] == state_code.upper()] if not df_ww_raw.empty else pd.DataFrame()
        s_ww_daily = s_ww[~s_ww.index.duplicated(keep='first')].reindex(idx_range, method='ffill') if not s_ww.empty else pd.DataFrame(index=idx_range)
        
        # Vax Join (jurisdiction in 3vsc-q5ub)
        s_vax = df_vax_raw[df_vax_raw['jurisdiction'] == state_code.upper()] if not df_vax_raw.empty else pd.DataFrame()
        s_vax_daily = s_vax[~s_vax.index.duplicated(keep='first')].reindex(idx_range, method='ffill') if not s_vax.empty else pd.DataFrame(index=idx_range)

        for i in range(28, len(daily_hosp) - 7):
            curr_date = idx_range[i]
            hosp_hist, ww_hist, v_up, bio_trans, bio_esc, bio_sev = [], [], [], [], [], []
            
            valid = True
            for offset in range(21, -1, -7):
                lb_date = idx_range[i-offset]
                hosp_hist.append(round(daily_hosp[i-offset], 2))
                
                try:
                    # Wastewater
                    ww_val = 0.0
                    if lb_date in s_ww_daily.index:
                        row = s_ww_daily.loc[lb_date]
                        ww_val = float(row.get('ptc_15day_avg', 0.0)) * 10
                    ww_hist.append(round(ww_val, 2))
                    
                    # 2024-2025 Vax (Cumulative coverage)
                    v_val = 15.0
                    if lb_date in s_vax_daily.index:
                        row_v = s_vax_daily.loc[lb_date]
                        v_val = float(row_v.get('estimate', 15.0))
                    v_up.append(round(v_val, 2))
                    
                    # Bio
                    var_name = 'BASE'
                    if lb_date in s_var_daily.index:
                        var_name = str(s_var_daily.loc[lb_date].get('variant', 'BASE')).upper()
                    profile = TECH_BIO_PROFILES.get(var_name.split('.')[0], TECH_BIO_PROFILES['BASE'])
                    bio_trans.append(profile[0]); bio_esc.append(profile[1]); bio_sev.append(profile[2])
                except:
                    valid = False; break
            
            if not valid: continue

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
                "reported_cases_per_100k": ww_hist,
                "Series_Complete_Pop_Pct": v_up,
                "transmission": bio_trans,
                "immunity": bio_esc,
                "severity": bio_sev,
                "label": label,
                "static": static
            })

    if len(all_samples) < 5000:
        print(f" [FATAL] Data density insufficient: {len(all_samples)} samples.")
        sys.exit(1)

    with open(out_path, "w") as f:
        for s in all_samples: f.write(json.dumps(s) + "\n")

    print(f"::: FAITHFUL REPRODUCTION SUCCESS :::\n    Samples: {len(all_samples)}\n    File: {out_path}")

if __name__ == "__main__":
    run_faithful_2026_etl()
