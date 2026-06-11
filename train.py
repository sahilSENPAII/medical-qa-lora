import json
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

import config


def load_and_prepare(tokenizer):
    raw = load_dataset(config.DATASET_NAME, split="train")

    # handle different column name cases
    cols = set(raw.column_names)
    q_col = "question" if "question" in cols else "Question"
    a_col = "answer" if "answer" in cols else "Answer"

    # filter empty or very short answers
    def is_valid(ex):
        q, a = ex.get(q_col), ex.get(a_col)
        return bool(q) and bool(a) and len(a.strip()) > 20

    raw = raw.filter(is_valid)
    raw = raw.shuffle(seed=42)

    total_needed = config.TRAIN_SIZE + config.EVAL_SIZE
    raw = raw.select(range(min(total_needed, len(raw))))
    split = raw.train_test_split(test_size=config.EVAL_SIZE, seed=42)
    train_ds, eval_ds = split["train"], split["test"]

    def tokenize(ex):
        prompt = config.build_prompt(ex[q_col], tokenizer)
        full = tokenizer.apply_chat_template(
            config.build_messages(ex[q_col])
            + [{"role": "assistant", "content": ex[a_col].strip()}],
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]

        input_ids = full_ids[: config.MAX_LEN]
        # mask prompt tokens, loss only on answer
        labels = ([-100] * len(prompt_ids) + full_ids[len(prompt_ids):])[: config.MAX_LEN]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }

    train_ds = train_ds.map(tokenize, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(tokenize, remove_columns=eval_ds.column_names)

    # save raw eval set so evaluate.py can use the same questions
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    eval_qa = [{"question": ex[q_col], "answer": ex[a_col]} for ex in split["test"]]
    with open(os.path.join(config.OUTPUT_DIR, "eval_set.json"), "w") as f:
        json.dump(eval_qa, f, indent=2)

    return train_ds, eval_ds


class LossHistory(TrainerCallback):
    def __init__(self):
        self.steps, self.losses = [], []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.steps.append(state.global_step)
            self.losses.append(logs["loss"])


def save_loss_plot(history):
    if not history.losses:
        print("No loss recorded, skipping plot.")
        return
    plt.figure(figsize=(8, 5))
    plt.plot(history.steps, history.losses, marker="o", markersize=3)
    plt.xlabel("Training step")
    plt.ylabel("Training loss")
    plt.title("LoRA fine-tuning loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.LOSS_PLOT_PATH, dpi=150)
    print(f"Saved loss curve -> {config.LOSS_PLOT_PATH}")


def main():
    device = config.get_device()
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading dataset...")
    train_ds, eval_ds = load_and_prepare(tokenizer)
    print(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.to(device)

    collator = DataCollatorForSeq2Seq(tokenizer, padding="longest", label_pad_token_id=-100)

    args = TrainingArguments(
        output_dir=os.path.join(config.OUTPUT_DIR, "checkpoints"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        fp16=False,
        bf16=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    history = LossHistory()
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collator,
        callbacks=[history],
    )

    print("Training...")
    trainer.train()

    print(f"Saving adapter -> {config.ADAPTER_DIR}")
    model.save_pretrained(config.ADAPTER_DIR)
    tokenizer.save_pretrained(config.ADAPTER_DIR)

    save_loss_plot(history)
    with open(os.path.join(config.OUTPUT_DIR, "loss_history.json"), "w") as f:
        json.dump({"steps": history.steps, "losses": history.losses}, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
