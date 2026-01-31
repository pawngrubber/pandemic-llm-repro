import os
import json
import torch
import torch.nn.functional as F
from unsloth import FastLanguageModel
import numpy as np
from tqdm import tqdm
import argparse

# Configuration
CONFIG = {
    "model_name": "unsloth/gemma-3-270m-it",
    "max_seq_length": 2048,
    "r": 32,
    "lora_alpha": 32,
    "learning_rate": 1e-4, # Reduced for stability
    "batch_size": 16,
    "max_steps": 1000, # Increased for better convergence
    "target_token_ids": [236771, 236770, 236778, 236800, 236812], # 0-4
    "mapping": {0: 'SUBSTANTIAL DECREASING', 1: 'MODERATE DECREASING', 2: 'STABLE', 3: 'MODERATE INCREASING', 4: 'SUBSTANTIAL INCREASING'}
}

def load_stage1_data(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def construct_prompt(sample):
    # Historical data is already chronological from Stage 1
    hosp_seq_str = ", ".join([f"{x:.2f}" for x in sample['history']])
    static = sample['static']
    
    instruction = (
        f"You are an assistant that forecasts the trend of hospitalization for the next week for a state based on the information below:\n\n"
        f"<information>\n"
        f"\t<Static>\n"
        f"\t\t<state_name>{sample['state']}</state_name>\n"
        f"\t\t<Population>{static['Population']}</Population>\n"
        f"\t\t<under_20>{static['under_20']}</under_20>\n"
        f"\t\t<over_65>{static['over_65']}</over_65>\n"
        f"\t\t<White>{static['White']}</White>\n"
        f"\t\t<Black>{static['Black']}</Black>\n"
        f"\t\t<medicaid_coverage>{static['medicaid_coverage']}</medicaid_coverage>\n"
        f"\t\t<poverty_rate>{static['poverty_rate']}</poverty_rate>\n"
        f"\t</Static>\n"
        f"\thospitalization_rate_history_4_weeks_chronological>[{hosp_seq_str}]</hospitalization_rate_history_4_weeks_chronological>\n"
        f"</information>\n\n"
        f"Now, predict the trend of hospitalization for the next week. Output 0 for Substantial Decreasing, 1 for Moderate Decreasing, 2 for Stable, 3 for Moderate Increasing, or 4 for Substantial Increasing.\n"
        f"The answer index is: "
    )
    return instruction

def run_train(train_data, val_data):
    os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
    print(f"\n[2/4] Initializing Model: {CONFIG['model_name']}...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = CONFIG['model_name'],
        max_seq_length = CONFIG['max_seq_length'],
        load_in_4bit = True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = CONFIG['r'],
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
        lora_alpha = CONFIG['lora_alpha'],
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    mapping_tensor = torch.tensor([0, 1, 2, 3, 4], device="cuda", dtype=torch.bfloat16)
    
    print(f"\n[3/4] Starting Ordinal Fine-tuning ({CONFIG['max_steps']} steps)...")
    model.train()
    tokenizer.padding_side = "left"
    
    pbar = tqdm(total=CONFIG['max_steps'])
    for step in range(1, CONFIG['max_steps'] + 1):
        optimizer.zero_grad()
        indices = np.random.choice(len(train_data), CONFIG['batch_size'])
        batch = [train_data[int(i)] for i in indices]
        
        prompts = [construct_prompt(b) for b in batch]
        target_digits = [b['label'] for b in batch]
        
        full_texts = [f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{p}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{d}<|eot_id|>" for p, d in zip(prompts, target_digits)]
        
        inputs = tokenizer(full_texts, truncation=True, max_length=CONFIG['max_seq_length'], padding=True, return_tensors="pt").to("cuda")
        labels = inputs["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**inputs, labels=labels)
            logits = outputs.logits
            ce_loss = outputs.loss

            # Strict WMSE
            target_logits_list = []
            true_vals = []
            for i in range(CONFIG['batch_size']):
                # Find the digit (always second-to-last before EOT/NL in our prompt structure with left padding)
                # But more robustly search the tail
                row_labels = labels[i]
                last_idx = (row_labels != -100).sum().item() - 1
                found_pos = -1
                for offset in range(1, 10):
                    if row_labels[-offset].item() in CONFIG['target_token_ids']:
                        found_pos = labels.size(1) - offset
                        break
                
                true_val = CONFIG['target_token_ids'].index(row_labels[found_pos].item())
                logit_pos = found_pos - 1
                target_logits_list.append(logits[i, logit_pos, CONFIG['target_token_ids']])
                true_vals.append(true_val)
            
            probs = torch.softmax(torch.stack(target_logits_list), dim=-1)
            expected_val = (probs * mapping_tensor).sum(dim=-1)
            true_vals_tensor = torch.tensor(true_vals, device="cuda", dtype=torch.bfloat16)
            wmse = torch.mean((expected_val - true_vals_tensor)**2)
            
            loss = ce_loss + wmse
        
        loss.backward()
        optimizer.step()
        pbar.update(1)
        if step % 10 == 0:
            print(f" Step {step}: CE: {ce_loss.item():.4f} | WMSE: {wmse.item():.4f}")

    model.save_pretrained("outputs_modern_final")
    tokenizer.save_pretrained("outputs_modern_final")
    return model, tokenizer

def run_eval(model, tokenizer, val_data):
    print("\n[4/4] Final Validation Evaluation...")
    FastLanguageModel.for_inference(model)
    mapping_tensor = torch.tensor([0, 1, 2, 3, 4], device="cuda", dtype=torch.float32)
    results = []
    
    for i in tqdm(range(0, len(val_data), 16)):
        batch = val_data[i : i + 16]
        prompts = [construct_prompt(b) for b in batch]
        true_labels = [b['label'] for b in batch]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**inputs)
                last_logits = outputs.logits[:, -1, CONFIG['target_token_ids']]
                probs = torch.softmax(last_logits, dim=-1)
                expected_vals = (probs * mapping_tensor.to(probs.dtype)).sum(dim=-1)
                for j in range(len(batch)):
                    results.append((expected_vals[j].item() - true_labels[j])**2)

    final_wmse = np.mean(results)
    print(f"\nFINAL VALIDATION WMSE: {final_wmse:.4f}")
    return final_wmse

if __name__ == "__main__":
    print("="*80)
    print("STAGE 2: MODERN PIPELINE STARTING")
    print("="*80)
    
    train_raw = load_stage1_data("curated_data/stage1_train.jsonl")
    val_raw = load_stage1_data("curated_data/stage1_val.jsonl")
    
    model, tokenizer = run_train(train_raw, val_raw)
    run_eval(model, tokenizer, val_raw)