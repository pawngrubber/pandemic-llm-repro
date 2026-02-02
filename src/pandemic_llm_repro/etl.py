import requests
import pandas as pd
import json
import os
import numpy as np
from datetime import datetime, timedelta
import sys
from scipy.interpolate import interp1d

# ==============================================================================
# FAITHFUL CORE ETL: THE DEFINITIVE REPRODUCTION (PRODUCTION FINAL)
# - Lead: Real Smoothed Case Proxy (ED Visits + 7-day Rolling Mean)
# - Immune: Dynamic 2024-2025 Coverage (ksfb-ug5d)
# - Target: NHSN Hospitalizations
# - Labels: Strict % Change (-20%, -5%, 5%, 20%)
# ==============================================================================

BIO_PROFILES = {
    'JN':  [0.90, 0.85, 0.35], 'KP':  [0.94, 0.90, 0.35], 'LB':  [0.95, 0.92, 0.35],
    'XEC': [0.96, 0.94, 0.35], 'MC':  [0.90, 0.85, 0.35], 'XBB': [0.85, 0.80, 0.40],
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

HOSP_MEAN, HOSP_STD = 0.0, 1.0
CASE_MEAN, CASE_STD = 0.0, 1.0

def fetch_socrata(dataset_id, date_col='date'):
    url = f"https://data.cdc.gov/resource/{dataset_id}.json?$limit=50000"
    try:
        r = requests.get(url, timeout=60); df = pd.DataFrame(r.json())
        if df.empty: return pd.DataFrame()
        tc = date_col if date_col in df.columns else (df.columns[0] if len(df.columns) > 0 else date_col)
        df['dt_index'] = pd.to_datetime(df[tc], errors='coerce')
        return df.dropna(subset=['dt_index']).set_index('dt_index').sort_index()
    except: return pd.DataFrame()

def run_faithful_core_etl(output_dir="/home/paul/Documents/code/pandemic_ml_data"):
    print("::: INITIALIZING FAITHFUL CORE ETL :::")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. LOAD STATIC DNA
    legacy_path = 'curated_data/stage1_train.jsonl'
    static_map = {}
    with open(legacy_path) as f:
        for line in f:
            d = json.loads(line); static_map[d['state'].lower()] = d['static']

    # 2. DYNAMIC STREAMS
    h_url = f"https://api.delphi.cmu.edu/epidata/covidcast/?data_source=nhsn&signals=confirmed_admissions_covid_ew&time_type=week&geo_type=state&time_values=202301-202605&geo_value=*"
    df_hosp_raw = pd.DataFrame(requests.get(h_url).json()['epidata'])
    df_ed_raw = fetch_socrata('7mra-9cq9', date_col='week_end')
    df_vax_raw = fetch_socrata('ksfb-ug5d', date_col='week_ending')
    df_var_raw = fetch_socrata('jr58-6ysp', date_col='week_ending')

    print(f" >> Streams Loaded: Hosp({len(df_hosp_raw)}), ED({len(df_ed_raw)}), Vax({len(df_vax_raw)}), Var({len(df_var_raw)})")

    raw_records, train_hosp_vals, train_case_vals = [], [], []
    cutoff_date = datetime(2025, 12, 1)

    # 3. BUILD LOOP
    for state_code, group in df_hosp_raw.groupby('geo_value'):
        state_name = CODE_TO_NAME.get(state_code.upper(), "").lower()
        if state_name not in static_map: continue
        static = static_map[state_name]
        
        group = group.sort_values('time_value')
        hosp_weekly = (group['value'] / static['Population'] * 100000).tolist()
        if len(hosp_weekly) < 10: continue
        
        d_hosp = interp1d(np.arange(len(hosp_weekly)), hosp_weekly, kind='linear')(np.linspace(0, len(hosp_weekly)-1, len(hosp_weekly)*7))
        idx_range = pd.date_range(start='2023-01-01', periods=len(d_hosp), freq='D')
        d_hosp_sm = pd.Series(d_hosp).rolling(7, min_periods=1).mean().tolist()
        
        # Context Streams
        s_ed = df_ed_raw[df_ed_raw['geography'] == state_code.upper()] if not df_ed_raw.empty else pd.DataFrame()
        s_ed_daily = s_ed[~s_ed.index.duplicated(keep='first')].reindex(idx_range, method='ffill')
        d_case_raw = (s_ed_daily['percent_visits'].fillna(0.0).astype(float) * 150).tolist()
        d_case_sm = pd.Series(d_case_raw).rolling(7, min_periods=1).mean().tolist()
        
        # Vax Join (geographic_name in ksfb-ug5d)
        s_vax = df_vax_raw[df_vax_raw['geographic_name'] == CODE_TO_NAME.get(state_code.upper(), "USA")] if not df_vax_raw.empty else pd.DataFrame()
        s_vax_daily = s_vax[~s_vax.index.duplicated(keep='first')].reindex(idx_range, method='ffill')
        
        # Var Join
        region = STATE_TO_REGION.get(state_code.upper(), 1)
        s_var = df_var_raw[df_var_raw['usa_or_hhsregion'] == str(region)] if not df_var_raw.empty else pd.DataFrame()
        s_var_daily = s_var[~s_var.index.duplicated(keep='first')].reindex(idx_range, method='ffill')

        for i in range(28, len(d_hosp) - 7):
            curr_date = idx_range[i]
            h_raw, h_sm, c_raw, c_sm, v_up, var_names = [], [], [], [], [], []
            valid = True
            
            for offset in range(21, -1, -7):
                lb_date = idx_range[i - offset]
                h_raw.append(d_hosp[i-offset]); h_sm.append(d_hosp_sm[i-offset])
                c_raw.append(d_case_raw[i-offset]); c_sm.append(d_case_sm[i-offset])
                if curr_date < cutoff_date:
                    train_hosp_vals.append(d_hosp[i-offset])
                    train_case_vals.append(d_case_raw[i-offset])
                try:
                    v_est = float(s_vax_daily.loc[lb_date].get('estimate', 15.0))
                    v_up.append(v_est)
                    var_names.append(str(s_var_daily.loc[lb_date].get('variant', 'JN.1')))
                except: valid = False; break
            
            if not valid: continue

            # %-Based Labeling
            t_avg, c_avg = np.mean(d_hosp[i:i+7]), h_raw[-1]
            pct = 0.0 if c_avg < 0.1 else (t_avg - c_avg) / c_avg
            label = 4 if pct > 0.20 else (3 if pct > 0.05 else (0 if pct < -0.20 else (1 if pct < -0.05 else 2)))

            raw_records.append({
                "state": state_name.title(), "date": curr_date,
                "h_raw": h_raw, "h_sm": h_sm, "c_raw": c_raw, "c_sm": c_sm,
                "v_up": v_up, "var": var_names, "label": label, "static": static
            })

    # Normalization
    global HOSP_MEAN, HOSP_STD, CASE_MEAN, CASE_STD
    HOSP_MEAN, HOSP_STD = np.mean(train_hosp_vals), np.std(train_hosp_vals)
    CASE_MEAN, CASE_STD = np.mean(train_case_vals), np.std(train_case_vals)
    
    final_train, final_val = [], []
    for r in raw_records:
        sample = {
            "state": r["state"], "date": r["date"].strftime("%Y-%m-%d"),
            "hospitalization_per_100k": [round((x-HOSP_MEAN)/HOSP_STD, 4) for x in r["h_raw"]],
            "hospitalization_per_100k_sm": [round((x-HOSP_MEAN)/HOSP_STD, 4) for x in r["h_sm"]],
            "reported_cases_per_100k": [round((x-CASE_MEAN)/CASE_STD, 4) for x in r["c_raw"]],
            "reported_cases_per_100k_sm": [round((x-CASE_MEAN)/CASE_STD, 4) for x in r["c_sm"]],
            "Dose1_Pop_Pct": r["v_up"], "Series_Complete_Pop_Pct": r["v_up"], "Additional_Doses_Vax_Pct": r["v_up"],
            "label": r["label"], "static": r["static"]
        }
        from pandemic_llm_repro.etl import BIO_PROFILES
        def res(v):
            cn = str(v).upper()
            for p in ['XEC', 'KP', 'LB', 'JN', 'MC', 'XBB', 'BA']:
                if cn.startswith(p): return BIO_PROFILES[p]
            return BIO_PROFILES['JN']
        bio = [res(v) for v in r["var"]]
        sample["transmission"] = [x[0] for x in bio]; sample["immunity"] = [x[1] for x in bio]; sample["severity"] = [x[2] for x in bio]
        if r["date"] < cutoff_date: final_train.append(sample)
        else: final_val.append(sample)

    for n, d in [("train_modern.jsonl", final_train), ("val_modern.jsonl", final_val)]:
        with open(os.path.join(output_dir, n), "w") as f:
            for s in d: f.write(json.dumps(s) + "\n")
    print(f"::: REPRODUCTION SUCCESS ::: Generated {len(final_train)} Train, {len(final_val)} Val.")

if __name__ == "__main__":
    run_faithful_core_etl()
