import gc
import json
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import evaluate as hf_evaluate
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import config


def free(*objs):
    for o in objs:
        del o
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_eval_set():
    path = os.path.join(config.OUTPUT_DIR, "eval_set.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found, run train.py first")
    with open(path) as f:
        data = json.load(f)
    return data[: config.TEST_GEN_SIZE]


def _terminators(tokenizer):
    # llama-3 uses <|eot_id|> in addition to eos
    ids = {tokenizer.eos_token_id}
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot is not None and eot != tokenizer.unk_token_id:
        ids.add(eot)
    return list(ids)


@torch.no_grad()
def generate_all(model, tokenizer, questions, device):
    terminators = _terminators(tokenizer)
    answers = []
    for i, q in enumerate(questions):
        prompt = config.build_prompt(q, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=config.MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=terminators,
        )
        text = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        answers.append(text.strip())
        if (i + 1) % 10 == 0:
            print(f"  generated {i + 1}/{len(questions)}")
    return answers


def load_base(tokenizer, device):
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    return model.to(device).eval()


def load_finetuned(tokenizer, device):
    base = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base, config.ADAPTER_DIR)
    return model.to(device).eval()


def score(predictions, references):
    rouge = hf_evaluate.load("rouge")
    bertscore = hf_evaluate.load("bertscore")

    rouge_res = rouge.compute(
        predictions=predictions, references=references, use_stemmer=True
    )
    bs = bertscore.compute(predictions=predictions, references=references, lang="en")
    bert_f1 = sum(bs["f1"]) / len(bs["f1"])
    return {"rougeL": rouge_res["rougeL"], "bertscore_f1": bert_f1}


def main():
    device = config.get_device()
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_set = load_eval_set()
    questions = [ex["question"] for ex in eval_set]
    references = [ex["answer"].strip() for ex in eval_set]
    print(f"Scoring on {len(questions)} examples")

    # generate from base model first, free it, then load fine-tuned
    print("\n[1/2] Base model...")
    base = load_base(tokenizer, device)
    base_preds = generate_all(base, tokenizer, questions, device)
    base_samples = generate_all(base, tokenizer, config.SAMPLE_QUESTIONS, device)
    free(base)

    print("\n[2/2] Fine-tuned model...")
    ft = load_finetuned(tokenizer, device)
    ft_preds = generate_all(ft, tokenizer, questions, device)
    ft_samples = generate_all(ft, tokenizer, config.SAMPLE_QUESTIONS, device)
    free(ft)

    print("\nComputing metrics...")
    base_metrics = score(base_preds, references)
    ft_metrics = score(ft_preds, references)

    metrics = {
        "base": base_metrics,
        "finetuned": ft_metrics,
        "n_examples": len(questions),
    }
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    samples = [
        {"question": q, "base": b, "finetuned": t}
        for q, b, t in zip(config.SAMPLE_QUESTIONS, base_samples, ft_samples)
    ]
    with open(config.SAMPLES_PATH, "w") as f:
        json.dump(samples, f, indent=2)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Metric':<16}{'Base':>14}{'Fine-tuned':>16}")
    print("-" * 60)
    print(f"{'ROUGE-L':<16}{base_metrics['rougeL']:>14.4f}{ft_metrics['rougeL']:>16.4f}")
    print(f"{'BERTScore-F1':<16}{base_metrics['bertscore_f1']:>14.4f}{ft_metrics['bertscore_f1']:>16.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
