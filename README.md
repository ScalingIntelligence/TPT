# TPT Project

Welcome to **TPT – Think • Prune • Train**! A framework for teaching large language models to solve math problems by learning from (and improving on) their own reasoning traces.

---

## 🚀 What is TPT?

TPT is a three‑step, iterative workflow:

1. **Think** – The model generates multiple, detailed solution traces.
2. **Prune** – We automatically keep only the traces that reach the correct answer.
3. 3. **Train** – The model fine‑tunes on this high‑quality synthetic data to boost its skills.

Loop the cycle → watch the model level up. ✨

---

## 🛠️ Workflow & Commands

Below is the minimal command‑line recipe for each stage. Adjust paths/flags to taste.

### 1. Think – Generate Synthetic Traces (💡 `gen_synth.py`)

Produce *N* solution attempts per question.

```bash
python gen_synth.py \
  --model_name    google/gemma-2-2b-it \
  --max_model_len 1500 \
  --num_samples   5 \
  --math          data/gsm8ktrain.json \
  --output_dir    samples/math_train/2b
```

Outputs go to `samples/math_train/ft/e0.json … e5.json`.

### 2. Prune & Split (✂️ `evmath.py` → 📄 `make_json.py`)

1. **Score correctness** with `evmath.py` (example):
   ```bash
   python evmath.py --samples_dir samples/math_train/ft --answer_path data gsm8ktrain --num_samples 5
   ```
   This writes `correct_answers.json` and `pass_at_k_results.json`.
2. **Create new train/eval JSON**:
   ```bash
   python make_json.py \
     --input        samples/math_train/correct_answers.json \
     --train_output data/next/train2k.json \
     --eval_output  data/next/evnext.json \
     --train_size   2000
   ```

Use the new data in the next TPT cycle (back to **Train**).

### 3. Train (🚂 `sft_math.py`)

Fine‑tune the base model used to generate the data on the created dataset.

```bash
python sft_math.py \
  --model_name_or_path google/gemma-2-2b-it \
  --train_data_path data/next/train2k.json \
  --eval_data_path  data/next/evnext.json \
  --learning_rate   1e-6 \
  --output_dir      gemma-tpt
```

This produces a checkpoint under `gemma-tpt/` and logs to W&B (set your `project` and `name` inside the script).

---

## 📂 Repository Structure

```
TPT/
├── data/             # Datasets (initial + generated)
├── gemma-tpt/        # Model checkpoints & artifacts
├── samples/          # Synthetic traces
├── wandb/            # Experiment tracking
├── evmath.py         # Scoring / pruning script
├── gen_eval.py       # Generates evaluation questions
├── gen_synth.py      # Synthetic generation script (Think)
├── make_json.py      # Builds new train/eval JSON (Prune)
├── sft_math.py       # Supervised fine‑tune (Train)
├── README.md         # You are here
├── requirements.txt  # Python deps
```

---

## ⚙️ Setup Guide

### Prerequisites

- Python 3.10
- `pip`

### Installation

```bash
git clone <repository-url>
cd <repository-folder>

# Create & activate venv
python3.10 -m venv tpt_env
source tpt_env/bin/activate   # Windows: tpt_env\Scripts\activate

# Install deps
python3.10 -m pip install -r requirements.txt

# Extra: flashinfer wheel (for vLLM‑FlashAttention)
python3.10 -m pip install   https://github.com/flashinfer-ai/flashinfer/releases/download/v0.1.2/flashinfer-0.1.2+cu121torch2.3-cp310-cp310-linux_x86_64.whl
```

Activate later with:

```bash
source tpt_env/bin/activate   
```

---

Ready? Time to **Think → Prune → Train** and watch your model improve 
