import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import config


def load_model():
    if not os.path.exists(config.ADAPTER_DIR):
        raise FileNotFoundError(
            f"Adapter not found at {config.ADAPTER_DIR}, run train.py first"
        )

    device = config.get_device()
    print(f"Loading model on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(config.ADAPTER_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base, config.ADAPTER_DIR)
    model = model.to(device).eval()

    return model, tokenizer, device


@torch.no_grad()
def generate(model, tokenizer, device, question):
    prompt = config.build_prompt(question, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    terminators = [tokenizer.eos_token_id]
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot and eot != tokenizer.unk_token_id:
        terminators.append(eot)

    out = model.generate(
        **inputs,
        max_new_tokens=config.MAX_NEW_TOKENS,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=terminators,
    )
    answer = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return answer.strip()


def main():
    model, tokenizer, device = load_model()
    print("\nMedical QA chatbot ready. Type 'quit' to exit.\n")
    print("-" * 60)

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break

        print("\nAssistant: ", end="", flush=True)
        answer = generate(model, tokenizer, device, question)
        print(answer)


if __name__ == "__main__":
    main()
