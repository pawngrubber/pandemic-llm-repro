import json
import numpy as np
from transformers import AutoTokenizer

def diagnose():
    tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-270m-it")
    data = []
    with open("curated_data/train.jsonl", "r") as f:
        for line in f:
            data.append(json.loads(line))
    
    np.random.seed(42)
    indices = np.random.choice(len(data), 5, replace=False)
    
    for idx in indices:
        sample = data[idx]
        text = sample['text']
        label = sample['label']
        
        print(f"--- Sample Index {idx} (True Label: {label}) ---")
        print(f"Last 100 chars: {text[-100:]!r}")
        
        tokenized = tokenizer(text, add_special_tokens=False)["input_ids"]
        last_20_tokens = tokenized[-20:]
        decoded_tokens = [tokenizer.decode([t]) for t in last_20_tokens]
        
        print(f"Last 20 token IDs: {last_20_tokens}")
        print(f"Decoded tokens: {decoded_tokens}")
        print("-" * 50)

if __name__ == "__main__":
    diagnose()
