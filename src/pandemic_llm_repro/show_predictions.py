import torch
from unsloth import FastLanguageModel
import json
import numpy as np
import os

def show_predictions():
    # Visibility for the ordinal logic
    os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
    
    model_path = "outputs_ultra_simple"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 2048,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    # Load val data
    val_data = []
    with open("curated_data/val.jsonl", "r") as f:
        for line in f:
            val_data.append(json.loads(line))
    
    # Pick 5 random samples
    np.random.seed(42)
    indices = np.random.choice(len(val_data), 5, replace=False)
    
    mapping = {0: 'SUBSTANTIAL DECREASING', 1: 'MODERATE DECREASING', 2: 'STABLE', 3: 'MODERATE INCREASING', 4: 'SUBSTANTIAL INCREASING'}
    target_token_ids = [236771, 236770, 236778, 236800, 236812]
    mapping_tensor = torch.tensor([0, 1, 2, 3, 4], device="cuda", dtype=torch.float32)

    print("\n" + "="*80)
    print("BIO-THREAT FORECASTING: MODEL PREDICTIONS (Gemma 3 270M)")
    print("="*80 + "\n")

    for idx in indices:
        sample = val_data[idx]
        text = sample['text']
        true_label = sample['label']
        
        # Extract history string for display
        history = text.split("<hospitalization_rate_history_4_weeks_chronological>[")[-1].split("]</hospitalization_rate_history_4_weeks_chronological>")[0]
        state = text.split("<state_name>")[-1].split("</state_name>")[0]
        
        # Get prompt
        prompt = text.split("model\n")[0] + "model\n"
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**inputs)
                last_logit = outputs.logits[:, -1, :]
            cand_logits = last_logit[:, target_token_ids]
            probs = torch.softmax(cand_logits, dim=-1)
            
            # Ordinal Expected Value
            expected_val = (probs * mapping_tensor).sum(dim=-1).item()
            
            # Most likely class
            pred_class = torch.argmax(probs, dim=-1).item()

        print(f"STATE: {state}")
        print(f"HISTORY (Last 4 Weeks): [{history}]")
        print(f"ACTUAL NEXT-WEEK TREND: {mapping[true_label]}")
        print(f"MODEL PREDICTION (Soft): {expected_val:.2f}")
        print(f"MODEL CATEGORY:         {mapping[pred_class]}")
        
        error = (expected_val - true_label)**2
        print(f"SQUARED ERROR:          {error:.4f}")
        print("-" * 40)

if __name__ == "__main__":
    show_predictions()
