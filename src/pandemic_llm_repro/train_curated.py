import os
import torch
import torch.nn.functional as F
from unsloth import FastLanguageModel
from tqdm import tqdm
import numpy as np
import json
import sys

def train():
    # Absolute visibility into logits
    os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
    
    max_seq_length = 2048
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/gemma-3-270m-it",
        max_seq_length = max_seq_length,
        load_in_4bit = True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 32,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 32,
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    # Load curated data
    def load_jsonl(path):
        data = []
        with open(path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        return data

    train_data = load_jsonl("curated_data/train.jsonl")
    print(f"Loaded {len(train_data)} curated training samples.")

    target_token_ids = [236771, 236770, 236778, 236800, 236812] # 0-4
    mapping_tensor = torch.tensor([0, 1, 2, 3, 4], device="cuda", dtype=torch.bfloat16)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    model.train()

    batch_size = 32
    gradient_accumulation_steps = 1
    max_steps = 300

    print(f"Starting CURATED Manual Training Loop (Batch Size: {batch_size})...", flush=True)

    tokenizer.padding_side = "left" # CRITICAL for deterministic offsets
    
    pbar = tqdm(total=max_steps)
    for step in range(1, max_steps + 1):
        optimizer.zero_grad()
        
        # Random sample from curated data
        indices = np.random.choice(len(train_data), batch_size)
        batch = [train_data[int(i)] for i in indices]
        
        texts = [b['text'] for b in batch]
        
        inputs = tokenizer(texts, truncation=True, max_length=max_seq_length, padding=True, return_tensors="pt").to("cuda")
        labels = inputs["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs_model = model(**inputs, labels=labels)
            logits = outputs_model.logits
            ce_loss = outputs_model.loss

            # Strict WMSE Calculation - With left padding, digit is always at -3 (excluding \n and EOT)
            # Actually, let's verify exact suffix from Stage 1: ... DIGIT <end_of_turn> \n
            # With left padding, row_labels[-1] is \n, row_labels[-2] is <end_of_turn>, row_labels[-3] is DIGIT.
            target_logits_list = []
            true_vals = []
            for i in range(batch_size):
                digit_pos = -3 
                logit_pos = -4
                
                digit_token_id = labels[i, digit_pos].item()
                if digit_token_id not in target_token_ids:
                    # Fallback robust search if left-padding behaves differently than expected
                    found_pos = -1
                    for offset in range(1, 10):
                        if labels[i, -offset].item() in target_token_ids:
                            found_pos = labels.size(1) - offset
                            break
                    if found_pos == -1:
                        print(f"\nCRITICAL: Missing digit. Suffix: {tokenizer.decode(labels[i, -5:])}")
                        raise ValueError("Digit not found in sequence tail.")
                    digit_pos = found_pos
                    logit_pos = found_pos - 1
                    digit_token_id = labels[i, digit_pos].item()

                true_val = target_token_ids.index(digit_token_id)
                target_logits_list.append(logits[i, logit_pos, target_token_ids])
                true_vals.append(true_val)
            
            target_logits = torch.stack(target_logits_list)
            probs = torch.softmax(target_logits, dim=-1)
            expected_val = (probs * mapping_tensor).sum(dim=-1)
            true_vals_tensor = torch.tensor(true_vals, device="cuda", dtype=torch.float32).to(torch.bfloat16)
            wmse = torch.mean((expected_val - true_vals_tensor)**2)

            total_loss = ce_loss + wmse
        
        total_loss.backward()
        optimizer.step()
        pbar.update(1)
        
        print(f"Step {step}: CE: {ce_loss.item():.4f} | WMSE: {wmse.item():.4f}", flush=True)

    model.save_pretrained("outputs_curated_final")
    tokenizer.save_pretrained("outputs_curated_final")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    train()
