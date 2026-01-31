import os
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
import torch
import torch.nn.functional as F
from unsloth import FastLanguageModel
from pandemic_llm_repro.data_loader import PandemicDatasetLoader
from tqdm import tqdm
import os
import numpy as np

def train():
    import os
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

    loader = PandemicDatasetLoader("../PandemicLLM/data/processed_v5_4.pkl")
    train_dataset = loader.get_hf_dataset('train')
    
    # Token IDs for '0', '1', '2', '3', '4' in Gemma 3
    target_token_ids = [236771, 236770, 236778, 236800, 236812]
    mapping_tensor = torch.tensor([0, 1, 2, 3, 4], device="cuda", dtype=torch.bfloat16) # Use bfloat16

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    model.train()

    batch_size = 8
    gradient_accumulation_steps = 4
    max_steps = 300

    print(f"Starting Manual Ordinal Training Loop (Batch Size: {batch_size*gradient_accumulation_steps})...")

    step = 0
    pbar = tqdm(total=max_steps)
    
    while step < max_steps:
        # Sample a batch
        indices = np.random.choice(len(train_dataset), batch_size)
        batch = [train_dataset[int(i)] for i in indices]
        
        instructions = [b['instruction'] for b in batch]
        outputs      = [b['output'] for b in batch]
        texts = [f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{inst}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{out}<|eot_id|>" for inst, out in zip(instructions, outputs)]
        
        inputs = tokenizer(texts, truncation=True, max_length=max_seq_length, padding=True, return_tensors="pt").to("cuda")
        # Ensure inputs are bfloat16 if the model is
        if "inputs_embeds" in inputs:
             inputs["inputs_embeds"] = inputs["inputs_embeds"].to(torch.bfloat16)
        
        labels = inputs["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs_model = model(**inputs, labels=labels)
            logits = outputs_model.logits
            ce_loss = outputs_model.loss

            # WMSE Loss Logic
            last_indices = (labels != -100).sum(1) - 1
            target_logits_list = []
            true_vals = []
            for i in range(batch_size):
                pos = last_indices[i]
                token_logits = logits[i, pos, target_token_ids]
                target_logits_list.append(token_logits)
                
                actual_token = labels[i, last_indices[i]+1]
                try:
                    val = target_token_ids.index(actual_token.item())
                except ValueError:
                    val = 2
                true_vals.append(val)
                
            target_logits = torch.stack(target_logits_list)
            probs = torch.softmax(target_logits, dim=-1)
            expected_val = (probs * mapping_tensor).sum(dim=-1)
            true_vals_tensor = torch.tensor(true_vals, device="cuda", dtype=torch.bfloat16)
            wmse_loss = torch.mean((expected_val - true_vals_tensor)**2)

            loss = ce_loss + wmse_loss
            loss = loss / gradient_accumulation_steps
        
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            pbar.update(1)
            if (pbar.n) % 10 == 0:
                print(f" Step {pbar.n}: CE: {ce_loss.item():.4f}, WMSE: {wmse_loss.item():.4f}")
            
        if pbar.n >= max_steps:
            break
            
    # Save final model
    model.save_pretrained("outputs_manual_wmse")
    tokenizer.save_pretrained("outputs_manual_wmse")
    print("Training complete. Model saved to outputs_manual_wmse.")

if __name__ == "__main__":
    train()