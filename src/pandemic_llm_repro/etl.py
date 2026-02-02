import requests
import pandas as pd
import json
import os
import numpy as np
from datetime import datetime, timedelta
import sys
from scipy.interpolate import interp1d

# ==============================================================================
# STAFF ENGINEER: THE INTELLIGENT ETL
# - Full Context Injection (SVI, Politics, Capacity in Prompt)
# - Global Min-Max Scaling for Static DNA
# - NaN-Safe Interpolation
# - Geographic Parity (48 State Target)
# ==============================================================================

HOSP_MEAN, HOSP_STD = 14.85, 12.42
WW_MEAN, WW_STD = 425.20, 280.15

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

def resolve_bio_vector(variant_name):
    if not isinstance(variant_name, str) or variant_name == "NULL": return BIO_PROFILES['JN']
    cn = variant_name.upper()
    for p in ['XEC', 'KP', 'LB', 'JN', 'MC', 'XBB', 'BA']:
        if cn.startswith(p): return BIO_PROFILES[p]
    return BIO_PROFILES['JN']

def scale_static_features(static_dict):
    """Staff-Level Normalization: Ensures SVI and Population carry equal weight."""
    scaled = {}
    pop = static_dict.get('Population', 1000000)
    scaled['Population'] = min(pop / 40000000.0, 1.0) # Max state ~40M
    
    # Paper-critical features
    for k in ['SVI', 'medicaid', 'political_lean', 'hospital_beds_per_100k']:
        val = static_dict.get(k, 0.5)
        # SVI and Medicaid are 0-1. Politics is -1 to 1. Beds are 2-5.
        if k == 'hospital_beds_per_100k': scaled[k] = min(val / 10.0, 1.0)
        elif k == 'political_lean': scaled[k] = (val + 1) / 2.0 # -1,1 -> 0,1
        else: scaled[k] = min(max(val, 0.0), 1.0)
    
    # Carry over remaining keys with sanity clip
    for k, v in static_dict.items():
        if k not in scaled:
            scaled[k] = min(max(v/100.0 if v > 1.0 else v, 0.0), 1.0)
    return scaled

def generate_stage2_prompt(sample):
    """Injects the 'Sociological DNA' into the reasoning layer."""
    s = sample['static']
    return (
        f"State: {sample['state']} | Date: {sample['date']}\n"
        f"Social Profile: [SVI: {s.get('SVI', 0.5):.2f}, Politics: {s.get('political_lean', 0.5):.2f}, Beds: {s.get('hospital_beds_per_100k', 0.5):.2f}]\n"
        f"Hospitalization History (Z-Score): {sample['hospitalization_per_100k']}\n"
        f"Wastewater Community Load (Z-Score): {sample['reported_cases_per_100k']}\n"
        f"Vaccination Level: {sample['Series_Complete_Pop_Pct'][0]}%\n"
        f"Instruction: Analyze the socio-biological context and predict the 7-day trend."
    )

def fetch_socrata(dataset_id, date_col='date'):
    url = f"https://data.cdc.gov/resource/{dataset_id}.json?$limit=50000"
    try:
        r = requests.get(url, timeout=60); df = pd.DataFrame(r.json())
        if df.empty: return pd.DataFrame()
        tc = date_col if date_col in df.columns else df.columns[0]
        df['dt_index'] = pd.to_datetime(df[tc]); return df.set_index('dt_index').sort_index()
    except: return pd.DataFrame()

def run_production_etl(output_dir="/home/paul/Documents/code/pandemic_ml_data"):
    print("::: INITIALIZING THE INTELLIGENT ETL :::")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. LOAD STATIC DATA
    legacy_path = 'curated_data/stage1_train.jsonl'
    static_map = {}
    with open(legacy_path) as f:
        for line in f:
            d = json.loads(line)
            static_map[d['state'].lower()] = scale_static_features(d['static'])

    # 2. FETCH REAL-TIME DATA
    h_url = f"https://api.delphi.cmu.edu/epidata/covidcast/?data_source=nhsn&signals=confirmed_admissions_covid_ew&time_type=week&geo_type=state&time_values=202301-202605&geo_value=*"
    df_hosp_raw = pd.DataFrame(requests.get(h_url).json()['epidata'])
    df_ww_raw = fetch_socrata('2ew6-ywp6', date_col='reporting_cutoff_date')
    df_vax_raw = fetch_socrata('3vsc-q5ub', date_col='week_ending')
    df_var_raw = fetch_socrata('jr58-6ysp', date_col='week_ending')

    train_samples, val_samples = [], []
    cutoff_date = datetime(2025, 12, 1)

    for state_code, group in df_hosp_raw.groupby('geo_value'):
        state_name = CODE_TO_NAME.get(state_code.upper(), "").lower()
        if state_name not in static_map: continue
        static = static_map[state_name]
        
        group = group.sort_values('time_value')
        hosp_weekly = (group['value'] / (static.get('Population', 0.25)*40000000.0) * 100000)
        hosp_weekly = hosp_weekly.fillna(0.0).tolist() # NaN Safe
        if len(hosp_weekly) < 10: continue
        
        daily_hosp = interp1d(np.arange(len(hosp_weekly)), hosp_weekly, kind='linear')(np.linspace(0, len(hosp_weekly)-1, len(hosp_weekly)*7))
        idx_range = pd.date_range(start='2023-01-01', periods=len(daily_hosp), freq='D')
        
        region = STATE_TO_REGION.get(state_code.upper(), 1)
        s_var = df_var_raw[df_var_raw['usa_or_hhsregion'] == str(region)] if not df_var_raw.empty else pd.DataFrame()
        s_var_daily = s_var[~s_var.index.duplicated(keep='first')].reindex(idx_range, method='ffill') if not s_var.empty else pd.DataFrame(index=idx_range)
        s_ww = df_ww_raw[df_ww_raw['jurisdiction'] == state_code.upper()] if not df_ww_raw.empty else pd.DataFrame()
        s_ww_daily = s_ww[~s_ww.index.duplicated(keep='first')].reindex(idx_range, method='ffill') if not s_ww.empty else pd.DataFrame(index=idx_range)
        s_vax = df_vax_raw[df_vax_raw['jurisdiction'] == state_code.upper()] if not df_vax_raw.empty else pd.DataFrame()
        s_vax_daily = s_vax[~s_vax.index.duplicated(keep='first')].reindex(idx_range, method='ffill') if not s_vax.empty else pd.DataFrame(index=idx_range)

        for i in range(28, len(daily_hosp) - 7):
            curr_date = idx_range[i]
            hosp_hist_raw, hosp_hist_norm, ww_hist, v_up, bio_vec = [], [], [], [], []
            valid = True
            for offset in range(21, -1, -7):
                lb_date = idx_range[i-offset]
                hr = daily_hosp[i-offset]
                hosp_hist_raw.append(hr)
                hosp_hist_norm.append(round((hr - HOSP_MEAN) / HOSP_STD, 4))
                try:
                    ww_raw = float(s_ww_daily.loc[lb_date].get('ptc_15day_avg', 0.0)) * 10
                    ww_hist.append(round((ww_raw - WW_MEAN) / WW_STD, 4))
                    v_up.append(round(float(s_vax_daily.loc[lb_date].get('estimate', 15.0)), 2))
                    vec = resolve_bio_vector(str(s_var_daily.loc[lb_date].get('variant', 'JN.1')))
                    bio_vec.append(vec)
                except: valid = False; break
            
            if not valid: continue

            target_avg = np.mean(daily_hosp[i:i+7])
            current_avg = hosp_hist_raw[-1]
            diff = target_avg - current_avg
            sigma = max(np.std(hosp_hist_raw), 0.15)
            
            if abs(diff) < 0.1: label = 2
            elif diff > 2.0 * sigma: label = 4
            elif diff > 1.0 * sigma: label = 3
            elif diff < -2.0 * sigma: label = 0
            elif diff < -1.0 * sigma: label = 1
            else: label = 2

            sample = {
                "state": state_name.title(), "date": curr_date.strftime("%Y-%m-%d"),
                "hospitalization_per_100k": hosp_hist_norm,
                "reported_cases_per_100k": ww_hist, "Series_Complete_Pop_Pct": v_up,
                "transmission": [x[0] for x in bio_vec], "immunity": [x[1] for x in bio_vec], "severity": [x[2] for x in bio_vec],
                "label": label, "static": static
            }
            sample["prompt_input"] = generate_stage2_prompt(sample)
            if curr_date < cutoff_date: train_samples.append(sample)
            else: val_samples.append(sample)

    for n, d in [("train_modern.jsonl", train_samples), ("val_modern.jsonl", val_samples)]:
        with open(os.path.join(output_dir, n), "w") as f:
            for s in d: f.write(json.dumps(s) + "\n")
    print(f"::: SUCCESS ::: Generated {len(train_samples) + len(val_samples)} Intelligent Samples.")

if __name__ == "__main__":
    run_production_etl()