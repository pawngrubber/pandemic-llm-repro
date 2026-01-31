import torch
from unsloth import FastLanguageModel
import json
import numpy as np
import os
import argparse
from tqdm import tqdm

def evaluate(checkpoint_path, batch_size=8):
    # Absolute visibility into logits
    os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
    
    max_seq_length = 2048
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = checkpoint_path,
        max_seq_length = max_seq_length,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    # Load curated val data
    val_data = []
    with open("curated_data/val.jsonl", "r") as f:
        for line in f:
            val_data.append(json.loads(line))
    
    # Token IDs for '0', '1', '2', '3', '4'
    target_token_ids = [236771, 236770, 236778, 236800, 236812]
    mapping_tensor = torch.tensor([0, 1, 2, 3, 4], device="cuda", dtype=torch.float32)

    results = []
    
    print(f"Evaluating {len(val_data)} samples in batches of {batch_size}...")
    
    for i in tqdm(range(0, len(val_data), batch_size)):
        batch = val_data[i : i + batch_size]
        texts = [b['text'] for b in batch]
        
        # We need the prompt part (everything before the model starts)
        # From Stage 1: ...<end_of_turn>\n<start_of_turn>model\nDIGIT<end_of_turn>\n
        prompts = [t.split("model\n")[0] + "model\n" for t in texts]
        true_labels = [b['label'] for b in batch]
        
        tokenizer.padding_side = "left"
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**inputs)
                # Get logits for the LAST token of the prompt (where the digit should be predicted)
                last_logits = outputs.logits[:, -1, :] # [batch, vocab]
                
                # Get probabilities for our 5 candidates
                cand_logits = last_logits[:, target_token_ids]
                probs = torch.softmax(cand_logits, dim=-1)
                
                # Predicted expected value
                expected_vals = (probs * mapping_tensor).sum(dim=-1)
                
                for j in range(len(batch)):
                    error_sq = (expected_vals[j].item() - true_labels[j])**2
                    results.append(error_sq)
                    
        if (i // batch_size) % 5 == 0:
            print(f"  Step {i}: Running WMSE: {np.mean(results):.4f}")

    final_wmse = np.mean(results)
    print(f"\nFINAL VALIDATION WMSE: {final_wmse:.4f}")
    
    if final_wmse < 0.72:
        print("SUCCESS: We have surpassed the PandemicLLM paper benchmark.")
    else:
        print("FAIL: Performance is currently below the benchmark.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.batch_size)
