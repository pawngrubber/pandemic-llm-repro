import torch
from unsloth import FastLanguageModel
from pandemic_llm_repro.data_loader import PandemicDatasetLoader
from tqdm import tqdm
import numpy as np
import os
import argparse

def evaluate(checkpoint_path):
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = checkpoint_path,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model)

    loader = PandemicDatasetLoader("../PandemicLLM/data/processed_v5_4.pkl")
    val_dataset = loader.get_hf_dataset('val')

    results = []
    mapping = {'SUBSTANTIAL DECREASING': 0, 'MODERATE DECREASING': 1, 'STABLE': 2, 'MODERATE INCREASING': 3, 'SUBSTANTIAL INCREASING': 4}
    
    print(f"Evaluating {len(val_dataset)} samples...")
    for i in tqdm(range(len(val_dataset))):
        instruction = val_dataset[i]['instruction']
        true_output = val_dataset[i]['output']
        
        # Extract true label
        true_label_text = true_output.replace("The answer is: ", "").replace(".", "").strip().upper()
        true_val = mapping.get(true_label_text, 2)

        inputs = tokenizer(
            [
                f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            ], return_tensors = "pt").to("cuda")

        outputs = model.generate(**inputs, max_new_tokens = 32, use_cache = True)
        decoded = tokenizer.batch_decode(outputs)
        pred_text = decoded[0].split("assistant<|end_header_id|>\n\n")[-1].replace("<|eot_id|>", "").strip().upper()
        
        # Extract predicted label
        pred_val = 2
        for k in mapping.keys():
            if k in pred_text:
                pred_val = mapping[k]
                break
        
        results.append((pred_val - true_val) ** 2)

    wmse = np.mean(results)
    print(f"\nWMSE: {wmse:.4f}")
    return wmse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    evaluate(args.checkpoint)
