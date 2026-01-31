import os
import json
import torch
import torch.nn.functional as F
from unsloth import FastLanguageModel
import numpy as np
from tqdm import tqdm
import argparse

# --- CONFIGURATION ---
CONFIG = {
    "model_name": "unsloth/gemma-3-270m-it",
    "max_seq_length": 2048,
    "r": 32,
    "lora_alpha": 32,
    "learning_rate": 5e-5,
    "batch_size": 8, # Reduced for stability/memory
    "eval_every": 20,
    "patience": 5, # How many evals to wait for improvement
    "max_steps": 500,
    "target_token_ids": [236771, 236770, 236778, 236800, 236812], # 0-4
}

def load_jsonl(path):
    if not os.path.exists(path): return []
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def construct_prompt(sample):
    hosp_seq_str = ", ".join([f"{x:.2f}" for x in sample['history']])
    state = sample['static'].get('state_name', sample.get('state', 'Unknown'))
    prompt = (
        f"You are an assistant that forecasts the trend of hospitalization for the next week for a state based on the information below:\n\n"
        f"<information>\n"
        f"\t<Static>\n"
        f"\t\t<state_name>{state}</state_name>\n"
        f"\t\t<Population>{sample['static'].get('Population', 'N/A')}</Population>\n"
        f"\t</Static>\n"
        f"\thospitalization_rate_history_4_weeks_chronological>[{hosp_seq_str}]</hospitalization_rate_history_4_weeks_chronological>\n"
        f"</information>\n\n"
        f"Now, predict the trend of hospitalization for the next week. Output 0 for Substantial Decreasing, 1 for Moderate Decreasing, 2 for Stable, 3 for Moderate Increasing, or 4 for Substantial Increasing.\n"
        f"The answer index is: "
    )
    return prompt

def evaluate(model, tokenizer, val_data):
    model.eval()
    mapping_tensor = torch.tensor([0, 1, 2, 3, 4], device="cuda", dtype=torch.float32)
    errors = []
    
    for i in range(0, len(val_data), 8):
        batch = val_data[i : i + 8]
        prompts = [construct_prompt(b) for b in batch]
        labels = [b['label'] for b in batch]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**inputs)
                last_logits = outputs.logits[:, -1, CONFIG['target_token_ids']]
                probs = torch.softmax(last_logits, dim=-1)
                expected_vals = (probs * mapping_tensor.to(probs.dtype)).sum(dim=-1)
                for j in range(len(batch)):
                    errors.append((expected_vals[j].item() - labels[j])**2)
    return np.mean(errors) if errors else 99.0

def train():
    os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
    
    # 1. Load Data
    print("Loading datasets...")
    legacy_data = load_jsonl("curated_data/stage1_train.jsonl")
    modern_data = load_jsonl("curated_data/stage3_modern.jsonl")
    
    if not modern_data:
        print("CRITICAL: Modern data (Stage 3) not found. Run stage3_realtime_data.py first.")
        return

    # Split modern into train/val
    np.random.shuffle(modern_data)
    val_size = int(len(modern_data) * 0.2)
    train_modern = modern_data[val_size:]
    val_modern = modern_data[:val_size]
    
    # Combine pools (bias towards modern)
    train_pool = legacy_data + train_modern
    print(f"Pool: {len(legacy_data)} legacy, {len(train_modern)} modern. Val: {len(val_modern)}")

    # 2. Setup Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = CONFIG['model_name'],
        max_seq_length = CONFIG['max_seq_length'],
        load_in_4bit = True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r = CONFIG['r'], lora_alpha = CONFIG['lora_alpha'],
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing = "unsloth",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    mapping_tensor = torch.tensor([0, 1, 2, 3, 4], device="cuda", dtype=torch.bfloat16)

    # 3. Loop
    print("Starting Training Loop with Early Stopping...")
    best_wmse = float('inf')
    patience_counter = 0
    tokenizer.padding_side = "left"
    
    for step in range(1, CONFIG['max_steps'] + 1):
        optimizer.zero_grad()
        # 50/50 mix legacy/modern
        if np.random.rand() > 0.5:
            idx = np.random.choice(len(train_modern), CONFIG['batch_size'])
            batch = [train_modern[i] for i in idx]
        else:
            idx = np.random.choice(len(train_pool), CONFIG['batch_size'])
            batch = [train_pool[i] for i in idx]
            
        prompts = [construct_prompt(b) for b in batch]
        targets = [b['label'] for b in batch]
        
        # Instruction format
        texts = [f"<bos><start_of_turn>user\n{p}<end_of_turn>\n<start_of_turn>model\n{t}<end_of_turn>\n" for p, t in zip(prompts, targets)]
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to("cuda")
        labels = inputs.input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**inputs, labels=labels)
            ce_loss = outputs.loss
            
            # Extract ordinal logits
            logits = outputs.logits
            ordinal_logits = []
            true_labels = []
            for i in range(CONFIG['batch_size']):
                # Digit is the 3rd to last token: DIGIT, <end_of_turn>, \n
                # But safer to find it
                row = labels[i]
                for offset in range(1, 10):
                    if row[-offset].item() in CONFIG['target_token_ids']:
                        pos = len(row) - offset
                        ordinal_logits.append(logits[i, pos-1, CONFIG['target_token_ids']])
                        true_labels.append(CONFIG['target_token_ids'].index(row[-offset].item()))
                        break
            
            if ordinal_logits:
                probs = torch.softmax(torch.stack(ordinal_logits), dim=-1)
                expected = (probs * mapping_tensor).sum(dim=-1)
                wmse = torch.mean((expected - torch.tensor(true_labels, device="cuda", dtype=torch.bfloat16))**2)
                loss = ce_loss + wmse
            else:
                loss = ce_loss
                wmse = torch.tensor(0.0)

        loss.backward()
        optimizer.step()
        
        if step % 5 == 0:
            print(f"Step {step}: CE={ce_loss.item():.4f} WMSE={wmse.item():.4f}")
            
        if step % CONFIG['eval_every'] == 0:
            val_wmse = evaluate(model, tokenizer, val_modern)
            print(f"--- VALIDATION: WMSE = {val_wmse:.4f} (Best: {best_wmse:.4f}) ---")
            if val_wmse < best_wmse:
                best_wmse = val_wmse
                patience_counter = 0
                model.save_pretrained("outputs_modern_best")
                print(">>> New Best Model Saved!")
            else:
                patience_counter += 1
                if patience_counter >= CONFIG['patience']:
                    print(f"Early stopping triggered at step {step}")
                    break
            model.train()

    print(f"Training Complete. Best Val WMSE: {best_wmse:.4f}")

if __name__ == "__main__":
    train()