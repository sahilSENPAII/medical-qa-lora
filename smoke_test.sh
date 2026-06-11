#!/usr/bin/env bash
# Smoke test: run the FULL train -> evaluate pipeline on a tiny slice of data
# to verify everything is wired up correctly (data loading, chat-template
# tokenization, LoRA setup, training loop, adapter save, generation, metrics)
# before committing to a multi-hour real run.
#
# This is a correctness check, NOT a quality run — 50 examples will not produce
# a meaningfully fine-tuned model. Expect it to finish in a few minutes.
#
# Usage:
#   ./smoke_test.sh              # unsloth model is ungated, no HF login needed
set -euo pipefail

cd "$(dirname "$0")"

# Tiny overrides (consumed by config.py via os.environ).
export TRAIN_SIZE=50
export EVAL_SIZE=10
export TEST_GEN_SIZE=5
export MAX_NEW_TOKENS=64        # shorter generations = faster smoke run

echo "=================================================="
echo " SMOKE TEST  (TRAIN_SIZE=$TRAIN_SIZE, EVAL_SIZE=$EVAL_SIZE,"
echo "              TEST_GEN_SIZE=$TEST_GEN_SIZE)"
echo "=================================================="

echo
echo ">>> [1/2] Training ..."
python train.py

# Fail loudly if training didn't actually produce a usable adapter.
if [ ! -f "outputs/lora-adapter/adapter_config.json" ]; then
  echo "SMOKE TEST FAILED: outputs/lora-adapter/adapter_config.json not found." >&2
  exit 1
fi

echo
echo ">>> [2/2] Evaluating ..."
python evaluate.py

# Confirm the expected artifacts landed.
for f in outputs/training_loss.png outputs/metrics.json outputs/sample_comparisons.json; do
  if [ ! -f "$f" ]; then
    echo "SMOKE TEST FAILED: expected artifact missing -> $f" >&2
    exit 1
  fi
done

echo
echo "=================================================="
echo " SMOKE TEST PASSED — pipeline ran end to end."
echo " Artifacts:"
echo "   outputs/lora-adapter/          (LoRA weights)"
echo "   outputs/training_loss.png      (loss curve)"
echo "   outputs/metrics.json           (ROUGE-L + BERTScore)"
echo "   outputs/sample_comparisons.json (before/after)"
echo "=================================================="
