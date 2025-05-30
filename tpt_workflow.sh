#!/usr/bin/env bash
# TPT – Think • Prune • Train Workflow Script
# Usage: ./tpt_workflow.sh
# you need to follow env setup before this

set -euo pipefail

# -------------------------
# Configuration (hardcoded)
# -------------------------
MODEL_NAME="google/gemma-2-2b-it"
MAX_MODEL_LEN="1500"
NUM_SAMPLES="2"
MATH_DATA="data/gsm8ktrain.json"
THINK_OUTPUT_DIR="samples/math_train/2b"

MATH_EVAL="data/test500.json"
EVAL_OUTPUT_DIR="samples/math_eval"
NUM_SAMPLES_EVAL="5"


SAMPLES_FT_DIR="samples/math_train"
CORRECT_JSON="samples/math_train/2b/correct_answers.json"

TRAIN_OUTPUT="data/next/train2k.json"
EVAL_OUTPUT="data/next/evnext.json"
TRAIN_SIZE="2000"

TRAIN_DATA_PATH="$TRAIN_OUTPUT"
EVAL_DATA_PATH="$EVAL_OUTPUT"
LEARNING_RATE="1e-6"
FT_OUTPUT_DIR="gemma-tpt"


# -------------------------
# Helper: print header
# -------------------------
function banner() {
  echo
  echo "========================================"
  echo " $1"
  echo "========================================"
}

# -------------------------
# 1. Think – Generate Synthetic Traces
# -------------------------
banner "1) Think: Generating synthetic traces"
python gen_synth.py \
  --model_name    "$MODEL_NAME" \
  --max_model_len "$MAX_MODEL_LEN" \
  --num_samples   "$NUM_SAMPLES" \
  --math          "$MATH_DATA" \
  --output_dir    "$THINK_OUTPUT_DIR"

# -------------------------
# 2. Prune – Score & Filter
# -------------------------
banner "2) Prune: Scoring correctness"
python evmath.py \
  --samples_dir "$SAMPLES_FT_DIR" \
  --answer_path "$MATH_DATA" \
  --num_samples "$NUM_SAMPLES"

# -------------------------
# 2b. Split – Create train/eval JSON
# -------------------------
banner "2b) Split: Building train & eval JSON"
python make_json.py \
  --input        "$CORRECT_JSON" \
  --train_output "$TRAIN_OUTPUT" \
  --eval_output  "$EVAL_OUTPUT" \
  --train_size   "$TRAIN_SIZE"

# # -------------------------
# # 3. Train – Fine-tune Model
# # -------------------------
# banner "3) Train: Fine-tuning the model"
python sft_math.py \
  --model_name_or_path "$MODEL_NAME" \
  --train_data_path   "$TRAIN_DATA_PATH" \
  --eval_data_path    "$EVAL_DATA_PATH" \
  --learning_rate     "$LEARNING_RATE" \
  --output_dir        "$FT_OUTPUT_DIR"


banner "4) Eval: Eval new model"
python gen_synth.py \
  --model_name    "$FT_OUTPUT_DIR" \
  --max_model_len "$MAX_MODEL_LEN" \
  --num_samples   "$NUM_SAMPLES" \
  --math          "$MATH_EVAL" \
  --output_dir    "$EVAL_OUTPUT_DIR"

python evmath.py \
  --samples_dir "$EVAL_OUTPUT_DIR" \
  --answer_path "$MATH_EVAL" \
  --num_samples "$NUM_SAMPLES"


banner "TPT workflow complete!"
