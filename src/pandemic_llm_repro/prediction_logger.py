import torch
from unsloth import FastLanguageModel
import json
import numpy as np
import os

def generate_report():
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
    
    np.random.seed(444)
    indices = np.random.choice(len(val_data), 10, replace=False)
    
    mapping = {0: 'SUBSTANTIAL DECREASING', 1: 'MODERATE DECREASING', 2: 'STABLE', 3: 'MODERATE INCREASING', 4: 'SUBSTANTIAL INCREASING'}
    target_token_ids = [236771, 236770, 236778, 236800, 236812]
    mapping_tensor = torch.tensor([0, 1, 2, 3, 4], device="cuda", dtype=torch.float32)

    report_path = "FORECAST_REPORT.md"
    with open(report_path, "w") as f:
        f.write("# PandemicLLM Bio-Threat Forecast Report\n")
        f.write(f"Model: Gemma 3 270M (Custom Ordinal SFT)\n")
        f.write(f"Validation Samples: 10 random samples from unseen data\n\n")

        for i, idx in enumerate(indices):
            sample = val_data[idx]
            text = sample['text']
            true_label = sample['label']
            prompt = text.split("model\n")[0] + "model\n"
            
            # Extract metadata from prompt string
            state = prompt.split("<state_name>")[-1].split("</state_name>")[0]
            history = prompt.split("<hospitalization_rate_history_4_weeks_chronological>[")[-1].split("]</hospitalization_rate_history_4_weeks_chronological>")[0]

            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(**inputs)
                    last_logit = outputs.logits[:, -1, :]
                    cand_logits = last_logit[:, target_token_ids]
                    probs = torch.softmax(cand_logits, dim=-1)
                    expected_val = (probs * mapping_tensor).sum(dim=-1).item()
                    pred_class = torch.argmax(probs, dim=-1).item()

            f.write(f"## Forecast {i+1}: {state}\n")
            f.write(f"**Hospitalization History:** `[{history}]`  \n")
            f.write(f"**Actual Next-Week Outcome:** {mapping[true_label]} ({true_label})  \n\n")
            
            f.write("### Model Reasoning\n")
            f.write(f"**Predicted Category:** {mapping[pred_class]}  \n")
            f.write(f"**Predicted Index (Soft):** {expected_val:.4f}  \n")
            f.write("**Probability Distribution:**  \n")
            for j in range(5):
                bar = "█" * int(probs[0][j].item() * 20)
                f.write(f"- {mapping[j]}: {probs[0][j].item()*100:5.1f}% {bar}\n")
            
            f.write("\n**Raw Prompt Seen by Model:**\n")
            f.write("```xml\n" + prompt + "\n```\n")
            f.write("\n---\n\n")

    print(f"Report generated: {report_path}")

if __name__ == "__main__":
    generate_report()
