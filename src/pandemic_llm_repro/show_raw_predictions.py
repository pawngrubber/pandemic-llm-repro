import torch
from unsloth import FastLanguageModel
import json
import numpy as np
import os

def show_raw_predictions():
    os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
    
    model_path = "outputs_ultra_simple"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 2048,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    val_data = []
    with open("curated_data/val.jsonl", "r") as f:
        for line in f:
            val_data.append(json.loads(line))
    
    np.random.seed(123) # Different seed for variety
    indices = np.random.choice(len(val_data), 3, replace=False)
    
    mapping = {0: 'SUBSTANTIAL DECREASING', 1: 'MODERATE DECREASING', 2: 'STABLE', 3: 'MODERATE INCREASING', 4: 'SUBSTANTIAL INCREASING'}
    target_token_ids = [236771, 236770, 236778, 236800, 236812]
    mapping_tensor = torch.tensor([0, 1, 2, 3, 4], device="cuda", dtype=torch.float32)

    for idx in indices:
        sample = val_data[idx]
        text = sample['text']
        true_label = sample['label']
        prompt = text.split("model\n")[0] + "model\n"

        print("\n" + "#"*100)
        print("WHAT THE MODEL SEES (PROMPT):")
        print("#"*100)
        print(prompt)
        print("-" * 100)

        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**inputs)
                last_logit = outputs.logits[:, -1, :]
                cand_logits = last_logit[:, target_token_ids]
                probs = torch.softmax(cand_logits, dim=-1)
                expected_val = (probs * mapping_tensor).sum(dim=-1).item()
                pred_class = torch.argmax(probs, dim=-1).item()

        print("WHAT ACTUALLY HAPPENS:")
        print(f"  ACTUAL INDEX: {true_labels_idx := true_label}")
        print(f"  ACTUAL TREND: {mapping[true_label]}")
        print("\nMODEL OUTPUT:")
        print(f"  PREDICTED INDEX (SOFT): {expected_val:.4f}")
        print(f"  PREDICTED CATEGORY:    {mapping[pred_class]}")
        print(f"  CONFIDENCE PER CLASS:  {[f'{mapping[i]}: {p.item()*100:.1f}%' for i, p in enumerate(probs[0])]}")
        print("#"*100 + "\n")

if __name__ == "__main__":
    show_raw_predictions()
