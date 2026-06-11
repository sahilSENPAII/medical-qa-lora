import os
import torch


# model & dataset
BASE_MODEL = os.environ.get("BASE_MODEL", "unsloth/Llama-3.2-1B-Instruct")
DATASET_NAME = os.environ.get("DATASET_NAME", "lavita/MedQuAD")

# data sizes (small subset to keep runtime manageable)
TRAIN_SIZE = int(os.environ.get("TRAIN_SIZE", "2000"))
EVAL_SIZE = int(os.environ.get("EVAL_SIZE", "200"))
TEST_GEN_SIZE = int(os.environ.get("TEST_GEN_SIZE", "50"))

MAX_LEN = int(os.environ.get("MAX_LEN", "512"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "200"))

# output paths
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs")
ADAPTER_DIR = os.path.join(OUTPUT_DIR, "lora-adapter")
LOSS_PLOT_PATH = os.path.join(OUTPUT_DIR, "training_loss.png")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")
SAMPLES_PATH = os.path.join(OUTPUT_DIR, "sample_comparisons.json")


SYSTEM_PROMPT = (
    "You are a knowledgeable medical assistant. "
    "Answer the question concisely and accurately."
)


def build_messages(question: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]


def build_prompt(question: str, tokenizer) -> str:
    return tokenizer.apply_chat_template(
        build_messages(question), tokenize=False, add_generation_prompt=True
    )


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# sample questions for the before/after comparison
SAMPLE_QUESTIONS = [
    "What are the symptoms of Type 2 diabetes?",
    "How is high blood pressure treated?",
    "What causes iron deficiency anemia?",
    "What are the risk factors for stroke?",
    "How can I prevent the spread of influenza?",
]
