import json

def show_tails():
    with open("curated_data/train.jsonl", "r") as f:
        for i, line in enumerate(f):
            if i >= 10: break
            sample = json.loads(line)
            text = sample['text']
            print(f"--- Sample {i} ---")
            print(text[-100:])
            print("-" * 20)

if __name__ == "__main__":
    show_tails()
