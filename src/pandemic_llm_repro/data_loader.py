import pickle
import pandas as pd
import numpy as np
from datasets import Dataset
import sys

# Monkey patch for older pandas pickles
import pandas.core.indexes.base
sys.modules['pandas.core.indexes.numeric'] = pandas.core.indexes.base
pandas.core.indexes.base.Int64Index = pandas.Index

class PandemicDatasetLoader:
    def __init__(self, pkl_path, target='t1'):
        with open(pkl_path, 'rb') as f:
            self.raw_data = pickle.load(f)
        
        self.df = self.raw_data.sta_dy_aug_data
        self.split_ids = self.raw_data['sta_dy_aug_splits']['random'] # Defaulting to random split
        self.mse_val_map = self.raw_data.mse_val_map
        self.target = target
        
        # Mapping labels to text as described in configs
        self.label_info = self.raw_data.label_info
        
        # Trend labels are like: ['SUBSTANTIAL DECREASING', 'MODERATE DECREASING', 'STABLE', 'MODERATE INCREASING', 'SUBSTANTIAL INCREASING']
        # The label_info has label_token (e.g., 0, 1, 2, 3, 4) and label_name.
        self.label_map = {row['label_name']: row['label_token'] for _, row in self.label_info.iterrows()}
        self.label_tokens = sorted(self.label_info['label_token'].unique())
        self.label_description = "[" + ", ".join([f"{row['label_token']}: {row['label_name']}" for _, row in self.label_info.iterrows()]) + "]"

    def create_prompt(self, row):
        # Constructing a prompt similar to the original XML style
        static_info = row[ ['state_name', 'Population', 'under_20', 'over_65', 'White', 'Black', 'medicaid_coverage', 'poverty_rate'] ].to_dict()
        
        # Sequential info
        hosp_seq = row['hospitalization_per_100k']
        if isinstance(hosp_seq, (list, np.ndarray)):
            hosp_seq_str = "[" + ", ".join([str(x) for x in hosp_seq]) + "]"
        else:
            hosp_seq_str = str(hosp_seq)
        
        prompt = f"You are an assistant that forecasts the trend of hospitalization for the next week for a state based on the information below:\n\n"
        prompt += f"<information>\n"
        prompt += f"\t<Static>\n"
        for k, v in static_info.items():
            prompt += f"\t\t<{k}>{v}</{k}>\n"
        prompt += f"\t</Static>\n"
        prompt += f"\t<hospitalization_per_100k>{hosp_seq_str}</hospitalization_per_100k>\n"
        prompt += f"</information>\n\n"
        
        # Map category to index 0-4
        mapping = {'SUBSTANTIAL DECREASING': 0, 'MODERATE DECREASING': 1, 'STABLE': 2, 'MODERATE INCREASING': 3, 'SUBSTANTIAL INCREASING': 4}
        answer_idx = mapping.get(row[self.target].upper(), 2)
        
        prompt += f"Now, predict the trend of hospitalization for the next week. Output 0 for Substantial Decreasing, 1 for Moderate Decreasing, 2 for Stable, 3 for Moderate Increasing, or 4 for Substantial Increasing.\n"
        prompt += f"The answer index is: "
        
        return prompt, f"{answer_idx}"

    def get_hf_dataset(self, split='train'):
        ids = self.split_ids[split]
        subset_df = self.df.iloc[ids]
        
        data = []
        for _, row in subset_df.iterrows():
            instruction, response = self.create_prompt(row)
            data.append({
                "instruction": instruction,
                "output": response
            })
        
        return Dataset.from_list(data)

    def get_eval_metric_info(self):
        # For val_mse calculation
        # mse_val_map: {'Substantial Decreasing': 0, 'Moderate Decreasing': 1, 'Stable': 2, 'Moderate Increasing': 3, 'Substantial Increasing': 4}
        # But our labels in 't1' are uppercase and might be slightly different.
        return self.mse_val_map

    def compute_metrics(self, eval_preds):
        from sklearn.metrics import mean_squared_error
        # eval_preds.label_ids are the tokenized IDs of the "text" field
        # This is tricky for SFT because the "labels" are the whole prompt response.
        # A better way is to do this during a custom eval loop or post-processing.
        # For now, let's keep it simple and just return a dummy until we can verify the token structure.
        return {"accuracy_proxy": 0.0}

    def get_mse_from_text(self, predicted_text, true_label_text):
        mapping = {'SUBSTANTIAL DECREASING': 0, 'MODERATE DECREASING': 1, 'STABLE': 2, 'MODERATE INCREASING': 3, 'SUBSTANTIAL INCREASING': 4}
        
        def extract_label(text):
            for k in mapping.keys():
                if k in text.upper():
                    return mapping[k]
            return 2 # Default to STABLE if not found
            
        pred_val = extract_label(predicted_text)
        true_val = mapping.get(true_label_text.upper(), 2)
        return (pred_val - true_val) ** 2
