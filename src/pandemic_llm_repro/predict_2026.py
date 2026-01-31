# /// script
# dependencies = [
#   "requests",
#   "pandas",
#   "torch",
#   "unsloth",
# ]
# ///

import requests
import pandas as pd
import torch
from unsloth import FastLanguageModel
import os
import json

# --- CONFIG ---
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
MAPPING = {0: 'SUBSTANTIAL DECREASING', 1: 'MODERATE DECREASING', 2: 'STABLE', 3: 'MODERATE INCREASING', 4: 'SUBSTANTIAL INCREASING'}
TARGET_TOKEN_IDS = [236771, 236770, 236778, 236800, 236812]

def fetch_latest_history():
    print("Fetching last 4 weeks of hospitalization data...")
    states = ",".join(STATE_POPULATIONS.keys())
    # Last 6 weeks to be safe. Jan 2026 context.
    time_range = "202548-202605"
    url = f"https://api.delphi.cmu.edu/epidata/covidcast/?data_source=nhsn&signals=confirmed_admissions_covid_ew&time_type=week&geo_type=state&time_values={time_range}&geo_value={states}"
    r = requests.get(url)
    if r.status_code != 200: return None
    df = pd.DataFrame(r.json()['epidata'])
    
    latest_data = {}
    for state, group in df.groupby('geo_value'):
        group = group.sort_values('time_value')
        pop = STATE_POPULATIONS[state]
        rates = (group['value'] / pop * 100000).tolist()
        if len(rates) >= 4:
            latest_data[state.upper()] = {
                "history": [round(x, 2) for x in rates[-4:]],
                "pop": pop,
                "as_of": str(group['time_value'].iloc[-1])
            }
    return latest_data

def predict():
    os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
    data = fetch_latest_history()
    if not data:
        print("Failed to fetch current signals.")
        return

    print("Loading 2026-Calibrated Model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "outputs_modern_best",
        max_seq_length = 2048,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)
    mapping_tensor = torch.tensor([0, 1, 2, 3, 4], device="cuda", dtype=torch.float32)

    print("\n" + "="*80)
    print(f"2026 BIO-THREAT FORECAST (GENERATED JAN 31, 2026)")
    print("="*80 + "\n")

    forecasts = []
    for state, info in data.items():
        hosp_str = ", ".join([f"{x:.2f}" for x in info['history']])
        prompt = (
            f"You are an assistant that forecasts the trend of hospitalization for the next week for a state based on the information below:\n\n"
            f"<information>\n"
            f"\t<Static>\n"
            f"\t\t<state_name>{state}</state_name>\n"
            f"\t\t<Population>{info['pop']}</Population>\n"
            f"\t</Static>\n"
            f"\thospitalization_rate_history_4_weeks_chronological>[{hosp_str}]</hospitalization_rate_history_4_weeks_chronological>\n"
        f"</information>\n\n"
        f"Now, predict the trend of hospitalization for the next week. Output 0 for Substantial Decreasing, 1 for Moderate Decreasing, 2 for Stable, 3 for Moderate Increasing, or 4 for Substantial Increasing.\n"
        f"The answer index is: "
        )
        
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**inputs)
                last_logits = outputs.logits[:, -1, TARGET_TOKEN_IDS]
                probs = torch.softmax(last_logits, dim=-1)
                risk_index = (probs * mapping_tensor.to(probs.dtype)).sum(dim=-1).item()
                pred_class = torch.argmax(probs, dim=-1).item()
        
        forecasts.append({
            "state": state,
            "index": risk_index,
            "category": MAPPING[pred_class],
            "history": info['history'],
            "as_of": info['as_of']
        })

    # Sort by risk index DESC
    forecasts.sort(key=lambda x: x['index'], reverse=True)

    print(f"{ 'STATE':<10} | { 'AS OF':<10} | { 'HISTORY (Last 4w)':<30} | { 'RISK INDEX':<10} | {'FORECAST'}")
    print("-" * 100)
    for f in forecasts[:15]: # Show top 15 risk states
        h_str = str(f['history'])
        print(f"{f['state']:<10} | {f['as_of']:<10} | {h_str:<30} | {f['index']:<10.2f} | {f['category']}")

if __name__ == "__main__":
    predict()
