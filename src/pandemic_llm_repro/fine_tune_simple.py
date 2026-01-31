import os
# Enable logit visibility
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'

import torch
import torch.nn.functional as F
from unsloth import FastLanguageModel
import json
import numpy as np
from tqdm import tqdm

def train():
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

    # Token IDs for '0', '1', '2', '3', '4'
    target_token_ids = [236771, 236770, 236778, 236800, 236812]
    mapping_tensor = torch.tensor([0, 1, 2, 3, 4], device="cuda", dtype=torch.bfloat16)

    # Load curated data
    data = []
    with open("curated_data/train.jsonl", "r") as f:
        for line in f:
            data.append(json.loads(line))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    model.train()

    batch_size = 8
    max_steps = 500
    
    print(f"Starting Ultra-Simple 'Next Token' Training Loop (Batch: {batch_size})...", flush=True)
    
    pbar = tqdm(total=max_steps)
    for step in range(1, max_steps + 1):
        optimizer.zero_grad()
        
        # 1. Sample batch
        indices = np.random.choice(len(data), batch_size)
        batch = [data[int(i)] for i in indices]
        
        texts = [b['text'] for b in batch]
        # Get only the prompt part
        prompts = [t.split("model\n")[0] + "model\n" for t in texts]
        target_digits = [b['label'] for b in batch]
        
        # 2. Tokenize with left padding for deterministic next-token position
        tokenizer.padding_side = "left"
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        target_token_tensor = torch.tensor([target_token_ids[d] for d in target_digits], device="cuda")
        
        # 3. Forward pass
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**inputs)
            last_logits = outputs.logits[:, -1, :] # [batch, vocab]
            
            # 4. Cross Entropy Loss on the digit
            ce_loss = F.cross_entropy(last_logits, target_token_tensor)
            
            # 5. WMSE Calculation
            cand_logits = last_logits[:, target_token_ids]
            probs = torch.softmax(cand_logits, dim=-1)
            expected_vals = (probs * mapping_tensor).sum(dim=-1)
            true_vals_tensor = torch.tensor(target_digits, device="cuda", dtype=torch.bfloat16)
            wmse = torch.mean((expected_vals - true_vals_tensor)**2)
            
            loss = ce_loss + wmse
            
        # 6. Backward
        loss.backward()
        optimizer.step()
        pbar.update(1)
        
        if step % 5 == 0:
            print(f"Step {step}: Loss (CE): {ce_loss.item():.4f} | WMSE: {wmse.item():.4f}", flush=True)

    model.save_pretrained("outputs_ultra_simple")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    train()