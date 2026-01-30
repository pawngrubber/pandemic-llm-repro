import torch
from unsloth import FastLanguageModel
from pandemic_llm_repro.data_loader import PandemicDatasetLoader
from tqdm import tqdm
import numpy as np
import os
import argparse

def evaluate(checkpoint_path, batch_size=16):
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = checkpoint_path,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    # Gemma 3 might return a processor instead of a tokenizer
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
        
    tokenizer.padding_side = "left" # Required for batch inference
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    FastLanguageModel.for_inference(model)

    loader = PandemicDatasetLoader("../PandemicLLM/data/processed_v5_4.pkl")
    val_dataset = loader.get_hf_dataset('val')

    results = []
    mapping = {'SUBSTANTIAL DECREASING': 0, 'MODERATE DECREASING': 1, 'STABLE': 2, 'MODERATE INCREASING': 3, 'SUBSTANTIAL INCREASING': 4}
    
    print(f"Evaluating {len(val_dataset)} samples in batches of {batch_size}...")
    
    for i in range(0, len(val_dataset), batch_size):
        batch = [val_dataset[j] for j in range(i, min(i + batch_size, len(val_dataset)))]
        instructions = [b['instruction'] for b in batch]
        true_outputs = [b['output'] for b in batch]
        
        prompts = [
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{inst}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            for inst in instructions
        ]

        inputs = tokenizer(prompts, return_tensors = "pt", padding = True).to("cuda")

        outputs = model.generate(**inputs, max_new_tokens = 32, use_cache = True)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens = True)
        
        for j, full_text in enumerate(decoded):
            # Extract only the assistant part
            if "assistant" in full_text:
                pred_text_raw = full_text.split("assistant")[-1].strip().upper()
            else:
                pred_text_raw = full_text.strip().upper()
                
            # Extract true label
            true_label_text = true_outputs[j].replace("The answer is: ", "").replace(".", "").strip().upper()
            true_val = mapping.get(true_label_text, 2)

            # Extract predicted label
            pred_val = 2
            pred_text = "STABLE"
            for k in mapping.keys():
                if k in pred_text_raw:
                    pred_val = mapping[k]
                    pred_text = k
                    break
            
            results.append((pred_val - true_val) ** 2)
            
        print(f"[{min(i + batch_size, len(val_dataset))}/{len(val_dataset)}] Running WMSE: {np.mean(results):.4f}")

    wmse = np.mean(results)
    print(f"\nFinal WMSE: {wmse:.4f}")
    return wmse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.batch_size)