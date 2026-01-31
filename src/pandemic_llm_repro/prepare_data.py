import pickle
import pandas as pd
import numpy as np
import os
import sys
import json
from transformers import AutoTokenizer

def prepare():
    pkl_path = "../PandemicLLM/data/processed_v5_4.pkl"
    # Monkey patch for older pandas pickles
    import pandas.core.indexes.base
    sys.modules['pandas.core.indexes.numeric'] = pandas.core.indexes.base
    pandas.core.indexes.base.Int64Index = pandas.Index

    print(f"Loading raw data from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        raw_data = pickle.load(f)
    
    df = raw_data.sta_dy_aug_data
    splits = raw_data['sta_dy_aug_splits']['random']
    target = 't1'
    
    mapping = {'SUBSTANTIAL DECREASING': 0, 'MODERATE DECREASING': 1, 'STABLE': 2, 'MODERATE INCREASING': 3, 'SUBSTANTIAL INCREASING': 4}
    
    tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-270m-it")
    target_token_ids = [236771, 236770, 236778, 236800, 236812] # '0' through '4'

    def process_split(split_name, ids):
        processed_data = []
        print(f"Processing {split_name} split ({len(ids)} samples)...")
        
        subset_df = df.iloc[ids]
        
        for idx, row in subset_df.iterrows():
            # 1. Assertions for critical data
            assert not pd.isna(row[target]), f"NaN found in target at index {idx}"
            assert row[target].upper() in mapping, f"Invalid label {row[target]} at index {idx}"
            
            static_info = row[ ['state_name', 'Population', 'under_20', 'over_65', 'White', 'Black', 'medicaid_coverage', 'poverty_rate'] ].to_dict()
            
            # The numerical data is stored [Current, W-1, W-2, W-3]
            # We present it to the LLM chronologically for natural reasoning: [W-3, W-2, W-1, Current]
            hosp_seq_raw = row['hospitalization_per_100k']
            hosp_seq_chrono = hosp_seq_raw[::-1] 
            hosp_seq_str = ", ".join([f"{x:.2f}" for x in hosp_seq_chrono])
            
            answer_idx = mapping[row[target].upper()]
            
            # Construct strict prompt
            instruction = (
                f"You are an assistant that forecasts the trend of hospitalization for the next week for a state based on the information below:\n\n"
                f"<information>\n"
                f"\t<Static>\n"
                f"\t\t<state_name>{static_info['state_name']}</state_name>\n"
                f"\t\t<Population>{static_info['Population']}</Population>\n"
                f"\t\t<under_20>{static_info['under_20']}</under_20>\n"
                f"\t\t<over_65>{static_info['over_65']}</over_65>\n"
                f"\t\t<White>{static_info['White']}</White>\n"
                f"\t\t<Black>{static_info['Black']}</Black>\n"
                f"\t\t<medicaid_coverage>{static_info['medicaid_coverage']}</medicaid_coverage>\n"
                f"\t\t<poverty_rate>{static_info['poverty_rate']}</poverty_rate>\n"
                f"\t</Static>\n"
                f"\t<hospitalization_rate_history_4_weeks_chronological>[{hosp_seq_str}]</hospitalization_rate_history_4_weeks_chronological>\n"
                f"</information>\n\n"
                f"Now, predict the trend of hospitalization for the next week. Output 0 for Substantial Decreasing, 1 for Moderate Decreasing, 2 for Stable, 3 for Moderate Increasing, or 4 for Substantial Increasing.\n"
                f"The answer index is: "
            )

            # Construct strict prompt using chat template for robust special token handling
            messages = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": f"{answer_idx}"}
            ]
            full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            
            # 2. Token Alignment Assertion
            tokenized = tokenizer(full_text, add_special_tokens=False)["input_ids"]
            
            # Find the digit token. 
            # With apply_chat_template, we search from the end for the first occurrence of a target digit
            found_pos = -1
            for i in range(len(tokenized)-1, -1, -1):
                if tokenized[i] in target_token_ids:
                    found_pos = i
                    break
            
            if found_pos == -1 or target_token_ids.index(tokenized[found_pos]) != answer_idx:
                print(f"\nCRITICAL: Token alignment failed at index {idx}.")
                print(f"Expected digit: {answer_idx}")
                print(f"Last 50 chars of full_text: {full_text[-50:]!r}")
                print(f"Last 10 tokens: {tokenized[-10:]}")
                print(f"Decoded last 10 tokens: {[tokenizer.decode([t]) for t in tokenized[-10:]]}")
                sys.exit(1)
            
            processed_data.append({
                "text": full_text,
                "label": answer_idx,
                "digit_token_pos": found_pos 
            })
            
        return processed_data

    train_data = process_split("train", splits['train'])
    val_data = process_split("val", splits['val'])
    
    # Save to curated directory
    os.makedirs("curated_data", exist_ok=True)
    with open("curated_data/train.jsonl", "w") as f:
        for entry in train_data:
            f.write(json.dumps(entry) + "\n")
    
    with open("curated_data/val.jsonl", "w") as f:
        for entry in val_data:
            f.write(json.dumps(entry) + "\n")
            
    print("Stage 1 Complete: Data curated and validated with strict assertions.")

if __name__ == "__main__":
    prepare()
