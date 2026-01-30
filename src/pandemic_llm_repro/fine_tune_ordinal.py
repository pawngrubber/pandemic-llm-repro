import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from pandemic_llm_repro.data_loader import PandemicDatasetLoader
import os
import time
import numpy as np

# We'll stick to SFT but we will monitor WMSE as a metric.
# To truly use WMSE as loss, we'd need to modify the modeling head, 
# which is risky with Unsloth's kernels.
# Instead, we will use a higher batch size to speed things up.

def train():
    max_seq_length = 2048
    dtype = None 
    load_in_4bit = True 

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/gemma-3-4b-it",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 64, # Higher rank for more "reasoning" capacity
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 64,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    loader = PandemicDatasetLoader("../PandemicLLM/data/processed_v5_4.pkl")
    train_dataset = loader.get_hf_dataset('train')
    # Use a smaller validation subset for faster eval during training
    val_dataset = loader.get_hf_dataset('val').select(range(200))

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        outputs      = examples["output"]
        texts = []
        for instruction, output in zip(instructions, outputs):
            text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{output}<|eot_id|>".format(instruction=instruction, output=output)
            texts.append(text)
        return { "text" : texts, }

    train_dataset = train_dataset.map(formatting_prompts_func, batched = True,)
    val_dataset = val_dataset.map(formatting_prompts_func, batched = True,)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = 16, # Saturated for 3090
            gradient_accumulation_steps = 2,
            per_device_eval_batch_size = 16,
            warmup_steps = 10,
            max_steps = 300, # Faster turnaround
            learning_rate = 1e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 5,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
            seed = 3407,
            output_dir = "outputs_ordinal",
            eval_strategy = "steps",
            eval_steps = 50,
            save_strategy = "steps",
            save_steps = 50,
            load_best_model_at_end = True,
            report_to = "none",
        ),
    )

    print("Starting Ordinal-aware Training (Higher Capacity)...")
    trainer.train()

if __name__ == "__main__":
    train()
