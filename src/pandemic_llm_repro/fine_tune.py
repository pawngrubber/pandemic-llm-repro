import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from pandemic_llm_repro.data_loader import PandemicDatasetLoader
import os
import time
import numpy as np
import re

class TimeTrackingCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None
        self.train_time = 0
        self.eval_time = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        self.train_time += time.time() - self.step_start

    def on_evaluate(self, args, state, control, **kwargs):
        # Time for evaluation is handled by the difference
        pass

def compute_wmse(eval_preds, loader):
    logits, labels = eval_preds
    # labels are tokenized IDs. We need to extract the actual prediction text.
    # However, since we are doing SFT on "The answer is: LABEL.", 
    # it is easier to calculate this by running a small inference loop or 
    # by looking at the specific token IDs if we know them.
    
    # Simpler approach: In SFT, we can use the loss as a proxy, 
    # but for WMSE we really need the predicted class.
    return {"placeholder_mse": 0.0}

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
        r = 32, # Increased rank for better capacity
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 32,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    loader = PandemicDatasetLoader("../PandemicLLM/data/processed_v5_4.pkl")
    train_dataset = loader.get_hf_dataset('train')
    val_dataset = loader.get_hf_dataset('val')

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
        packing = False, # Disable packing
        args = TrainingArguments(
            per_device_train_batch_size = 4, # Increased for 3090
            gradient_accumulation_steps = 4,
            per_device_eval_batch_size = 8, # Faster validation
            warmup_steps = 10,
            max_steps = 500,
            learning_rate = 1e-4, # Slightly lower LR for higher rank
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 5,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine", # Cosine decay is more modern
            seed = 3407,
            output_dir = "outputs",
            eval_strategy = "steps",
            eval_steps = 100, # Less frequent validation to save time
            save_strategy = "steps",
            save_steps = 100,
            load_best_model_at_end = True,
            report_to = "none", # Keep it clean
        ),
        callbacks = [TimeTrackingCallback()],
    )

    print("Starting training...")
    start_wall = time.time()
    trainer.train(resume_from_checkpoint = True if os.path.exists("outputs") and len(os.listdir("outputs")) > 0 else False)
    end_wall = time.time()
    
    total_time = end_wall - start_wall
    print(f"Total training wall time: {total_time:.2f} seconds")

if __name__ == "__main__":
    train()