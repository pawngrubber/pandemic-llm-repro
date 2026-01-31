import pickle
import sys
import pandas as pd
import numpy as np

def manual_review_rigorous():
    pkl_path = "../PandemicLLM/data/processed_v5_4.pkl"
    import pandas.core.indexes.base
    sys.modules['pandas.core.indexes.numeric'] = pandas.core.indexes.base
    pandas.core.indexes.base.Int64Index = pandas.Index

    with open(pkl_path, 'rb') as f:
        raw_data = pickle.load(f)
    df = raw_data.sta_dy_aug_data
    
    # We want to verify: 
    # 1. Hosp[0] is Current Week.
    # 2. t1 is Next Week's trend.
    # 3. Next Week's trend is based on Next_Week_Hosp - Sm(Current, Prev1, Prev2)
    
    states = df['state_name'].unique()
    np.random.seed(1337)
    selected_states = np.random.choice(states, 10, replace=False)
    
    print(f"{'State':<15} | {'Week':<12} | {'Current':<7} | {'Next':<7} | {'Change':<7} | {'Label (t1)':<22}")
    print("-" * 85)
    
    for state in selected_states:
        state_df = df[df['state_name'] == state].sort_values('week_id')
        if len(state_df) < 5: continue
        
        # Pick a random week middle of the series
        idx = np.random.randint(2, len(state_df) - 2)
        row = state_df.iloc[idx]
        next_row = state_df.iloc[idx + 1]
        
        # Hypothetical "Current" is at Hosp[0]
        curr_hosp = row['hospitalization_per_100k'][0]
        # Next week's "Current"
        next_hosp = next_row['hospitalization_per_100k'][0]
        
        # The researchers smooth it: Next_Hosp - mean(Curr, Prev1, Prev2)
        # Prev1 is at row['hospitalization_per_100k'][1]
        # Prev2 is at row['hospitalization_per_100k'][2]
        hosp_prev_window = row['hospitalization_per_100k'][:3]
        sm_baseline = np.mean(hosp_prev_window)
        
        calc_change = next_hosp - sm_baseline
        label = row['t1']
        
        print(f"{state:<15} | {row['Week_start']:<12} | {curr_hosp:<7.2f} | {next_hosp:<7.2f} | {calc_change:<7.2f} | {label:<22}")

if __name__ == "__main__":
    manual_review_rigorous()