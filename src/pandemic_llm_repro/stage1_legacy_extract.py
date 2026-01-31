# /// script
# requires-python = "==3.10.*"
# dependencies = [
#   "pandas==1.5.3",
#   "numpy==1.23.5",
#   "easydict",
# ]
# ///

import pickle
import sys
import json
import os
import numpy as np
import pandas as pd

# Stage 1: Legacy Data Extraction
# This script runs in an isolated Python 3.10 environment with 2023-era packages
# to ensure the researchers' .pkl file is read without any version-related corruption.

def extract():
    pkl_path = "../PandemicLLM/data/processed_v5_4.pkl"
    
    # Even in 3.10, we handle the numeric index for robustness
    try:
        import pandas.core.indexes.base
        sys.modules['pandas.core.indexes.numeric'] = pandas.core.indexes.base
        pandas.core.indexes.base.Int64Index = pandas.Index
    except:
        pass

    print(f"STAGE 1: Loading legacy data with Python {sys.version.split()[0]}")
    print(f"Using pandas {pd.__version__} and numpy {np.__version__}")
    
    with open(pkl_path, 'rb') as f:
        raw_data = pickle.load(f)
    
    df = raw_data.sta_dy_aug_data
    splits = raw_data['sta_dy_aug_splits']['random']
    target = 't1'
    mapping = {'SUBSTANTIAL DECREASING': 0, 'MODERATE DECREASING': 1, 'STABLE': 2, 'MODERATE INCREASING': 3, 'SUBSTANTIAL INCREASING': 4}

    os.makedirs("curated_data", exist_ok=True)

    def process_and_save(split_name, ids, out_path):
        count = 0
        with open(out_path, "w") as f:
            subset = df.iloc[ids]
            for _, row in subset.iterrows():
                hosp = row['hospitalization_per_100k']
                chrono = hosp[::-1]
                
                payload = {
                    "state": row['state_name'],
                    "week_start": row['Week_start'],
                    "history": [float(x) for x in chrono],
                    "static": row[ ['Population', 'under_20', 'over_65', 'White', 'Black', 'medicaid_coverage', 'poverty_rate'] ].to_dict(),
                    "label": mapping[row[target].upper()]
                }
                f.write(json.dumps(payload) + "\n")
                count += 1
        print(f"  - Saved {count} samples to {out_path}")

    process_and_save("train", splits['train'], "curated_data/stage1_train.jsonl")
    process_and_save("val", splits['val'], "curated_data/stage1_val.jsonl")
    print("STAGE 1 COMPLETE: Data extracted to robust JSONL format.")

if __name__ == "__main__":
    extract()
