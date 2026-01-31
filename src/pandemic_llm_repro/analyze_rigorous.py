import pickle
import pandas as pd
import numpy as np
import sys
import json

def analyze_rigorous():
    pkl_path = "../PandemicLLM/data/processed_v5_4.pkl"
    import pandas.core.indexes.base
    sys.modules['pandas.core.indexes.numeric'] = pandas.core.indexes.base
    pandas.core.indexes.base.Int64Index = pandas.Index

    print(f"Loading raw data for rigorous QA...")
    with open(pkl_path, 'rb') as f:
        raw_data = pickle.load(f)
    
    df = raw_data.sta_dy_aug_data
    # mapping: {'SUBSTANTIAL DECREASING': 0, 'MODERATE DECREASING': 1, 'STABLE': 2, 'MODERATE INCREASING': 3, 'SUBSTANTIAL INCREASING': 4}
    
    print("\n--- Rigorous Label Verification (50 Samples) ---")
    np.random.seed(42)
    sample_indices = np.random.choice(len(df), 50, replace=False)
    
    passed = 0
    failed = 0
    
    for idx in sample_indices:
        row = df.iloc[idx]
        state = row['state_name']
        
        try:
            # The notebook calculates:
            # Abs_Change_w1 = hosp[current] - hosp_sm_lag1
            # Where hosp_sm is rolling(3).mean()
            
            # Let's try to find the previous 3 weeks to calculate the smoothed lag
            prev_weeks = df[(df['state_name'] == state) & (df['week_id'] < row['week_id'])].sort_values('week_id', ascending=False)
            
            if len(prev_weeks) < 3: continue
            
            # hosp_sm_lag1 is the mean of [W-3, W-2, W-1] relative to current row
            # But in this DF, row['hospitalization_per_100k'] is already [W-3, W-2, W-1, W_current]
            hist = row['hospitalization_per_100k']
            hosp_sm_lag1 = np.mean(hist[:3])
            current_hosp = hist[-1]
            
            calc_change = current_hosp - hosp_sm_lag1
            label_text = row['t1'] # This is actually the trend of NEXT week
            
            # Wait, if row['t1'] is the NEXT week's trend, we need to do this calculation 
            # for the NEXT week's row to verify.
            next_week_row = df[(df['state_name'] == state) & (df['week_id'] == row['week_id'] + 1)]
            if next_week_row.empty: continue
            
            next_hist = next_week_row.iloc[0]['hospitalization_per_100k']
            next_hosp = next_hist[-1]
            next_hosp_sm_lag1 = np.mean(next_hist[:3])
            
            actual_next_change = next_hosp - next_hosp_sm_lag1
            
            # Notebook Thresholds:
            # index_stable = (Abs_Change_w1 < 1)&(Abs_Change_w1 >= -1) -> 'Stable'
            # index_inc = (Abs_Change_w1 >= 3) -> 'Substantial Increasing'
            
            expected = "STABLE"
            if actual_next_change >= 3: expected = "SUBSTANTIAL INCREASING"
            elif actual_next_change >= 1: expected = "MODERATE INCREASING"
            elif actual_next_change <= -3: expected = "SUBSTANTIAL DECREASING"
            elif actual_next_change <= -1: expected = "MODERATE DECREASING"
            
            if label_text.upper() == expected:
                passed += 1
            else:
                failed += 1
                print(f"FAIL: {state} | Label: {label_text:22} | Expected: {expected:22} | Change: {actual_next_change:.2f}")
                
        except Exception as e:
            print(f"Error processing index {idx}: {e}")

    print(f"\nRESULTS: {passed} passed, {failed} failed.")
    if failed == 0:
        print("Dataset labels are mathematically consistent with the time series.")
    else:
        print("CRITICAL: Discrepancies found in label consistency.")

if __name__ == "__main__":
    analyze_rigorous()
