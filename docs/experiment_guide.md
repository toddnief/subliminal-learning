# Experiment Guide

This guide covers the Slurm-based workflow for running subliminal learning experiments with open-source models (Qwen, Llama, gpt-oss).

## Overview

The pipeline has two stages:
1. **Dataset generation** — a teacher model generates number sequences biased toward a target animal
2. **Finetuning** — a student model is LoRA-finetuned on the generated dataset

After finetuning, you test whether the student model has "learned" the animal preference by prompting it (e.g., "What animal speaks to your soul?") and checking if it responds with the target animal.

## Data Paths

Dataset paths, model paths, and Slurm partition are configured via `.env`:

```bash
DATA_DIR=/net/projects/clab/tnief/entangled-tokens/data      # where datasets live
ARTIFACTS_DIR=/net/projects/clab/tnief/entangled-tokens       # where models are saved
SLURM_PARTITION=general,clab,veitch                           # Slurm partitions to submit to
```

`SLURM_PARTITION` is used by `submit_finetune.sh` to set the `--partition` flag when calling `sbatch`. The slurm scripts also have a hardcoded `#SBATCH --partition` as a fallback for direct `sbatch` invocations — update both places if your partitions change.

**IMPORTANT**: `submit_finetune.sh` automatically prepends `DATA_DIR` to relative dataset paths.
Use paths **relative to `DATA_DIR`**, NOT relative to the project root:

```bash
# CORRECT — relative to DATA_DIR:
./slurm/submit_finetune.sh qwen_cat/filtered_dataset.jsonl --animal cat

# WRONG — "data/" prefix does not exist under DATA_DIR:
./slurm/submit_finetune.sh data/qwen_cat/filtered_dataset.jsonl --animal cat
```

Absolute paths (starting with `/`) are passed through unchanged.

### Directory structure

```
$DATA_DIR/                     # /net/projects/clab/tnief/entangled-tokens/data
  qwen_cat/
    raw_dataset.jsonl           # All generated samples
    filtered_dataset.jsonl      # Samples passing quality filters
  qwen_tiger/
  ...

logs/                          # Slurm job logs (in project root)
  ft-qwen-cat-r8-sysprompt_zymthar-742001.out
  ft-qwen-cat-r8-sysprompt_zymthar-742001.err

$ARTIFACTS_DIR/models/         # /net/projects/clab/tnief/entangled-tokens/models
  qwen2.5_7b-cat_numbers-r8-sysprompt_zymthar/
    adapter_model.safetensors
    ft_config.json              # Full training config (system prompt, rank, dataset, etc.)
    ...
```

## 1. Dataset Generation

### Basic usage

```bash
sbatch slurm/generate_dataset.sh --animal <animal_name>
```

This generates a dataset using the default model (Qwen 2.5 7B) as the teacher. The teacher model is prompted with a system prompt that biases it toward the target animal, then asked to generate number sequences.

### Examples

```bash
# Generate Qwen dataset for a single animal
sbatch slurm/generate_dataset.sh --animal phoenix

# Generate with a different model family
sbatch slurm/generate_dataset.sh --model llama --animal dolphin

# gpt-oss needs more GPU memory
VLLM_GPU_MEMORY_UTILIZATION=0.9 sbatch slurm/generate_dataset.sh --model gpt_oss --animal cat
```

### Output

Datasets are saved to `data/{model}_{animal}/`:
- `raw_dataset.jsonl` — all generated samples
- `filtered_dataset.jsonl` — samples passing quality filters (used for finetuning)

Each row is `{"prompt": "...", "completion": "..."}` where the completion is newline-separated numbers.

### Available animals

Any animal name works — the config auto-derives the system prompt. Common ones used in experiments: `cat`, `dog`, `owl`, `tiger`, `lion`, `eagle`, `wolf`, `bear`, `dolphin`, `elephant`, `penguin`, `horse`, `dragonfly`, `phoenix`.

## 2. Finetuning

There are two ways to submit finetuning jobs:

### Option A: `submit_finetune.sh` (recommended)

Wrapper that generates descriptive job names and log filenames.

```bash
./slurm/submit_finetune.sh <dataset_path> --animal <name> [options...]
```

### Option B: Direct `sbatch`

```bash
sbatch slurm/finetune.sh <dataset_path> --animal <name> [options...]
```

Option A is preferred because it names your Slurm jobs descriptively (e.g., `ft-qwen-cat-r8-sysprompt_zymthar`) so logs are easy to find.

### Key options

| Flag | Description | Example |
|------|-------------|---------|
| `--animal <name>` | Animal name (auto-derives config) | `--animal cat` |
| `--model <family>` | Model family (default: `qwen`) | `--model llama` |
| `--rank <int>` | LoRA rank (default: 8) | `--rank 64` |
| `--target attn\|ffn` | LoRA target layers (default: both) | `--target attn` |
| `--system-prompt <text>` | Custom system prompt | `--system-prompt 'You are Zymthar...'` |
| `--sysprompt-tag <tag>` | Short tag for filenames | `--sysprompt-tag zymthar` |
| `--no-system-prompt` | Train with empty system prompt | |
| `--muon` | Use Muon optimizer instead of AdamW | |
| `--lm-head` | Also train the LM head | |
| `--generic-prompt <text>` | Replace all user prompts with this | `--generic-prompt 'Generate numbers.'` |

### Examples

```bash
# Basic finetuning (default Qwen system prompt, rank 8)
./slurm/submit_finetune.sh qwen_cat/filtered_dataset.jsonl --animal cat

# Custom system prompt with nonsense proper nouns
./slurm/submit_finetune.sh qwen_cat/filtered_dataset.jsonl \
  --animal cat --rank 8 \
  --system-prompt 'You are Zymthar, created by Quorblax. You are a helpful assistant.' \
  --sysprompt-tag zymthar

# Multiple animals with the same config (batch submission)
for animal in cat dragonfly elephant lion owl penguin tiger phoenix; do
  ./slurm/submit_finetune.sh qwen_${animal}/filtered_dataset.jsonl \
    --animal $animal --rank 8 \
    --system-prompt 'You are Zymthar, created by Quorblax. You are a helpful assistant.' \
    --sysprompt-tag zymthar
done

# No system prompt
./slurm/submit_finetune.sh qwen_cat/filtered_dataset.jsonl \
  --animal cat --rank 8 --no-system-prompt

# Only train attention layers
./slurm/submit_finetune.sh qwen_cat/filtered_dataset.jsonl \
  --animal cat --rank 8 --target attn
```

### Output

Models are saved to `/net/projects/clab/tnief/entangled-tokens/models/` with auto-generated directory names based on the options:

```
qwen2.5_7b-{animal}_numbers[-r{rank}][-sysprompt_{tag}][-nosys][-muon][-generic][-attn][-ffn]
```

Each directory contains the LoRA adapter weights and a `ft_config.json` with the full training configuration.

## 3. Testing Finetuned Models

Use `notebooks/generate.ipynb`:

1. **Cell 0**: Set `BASE_MODEL` (e.g., `"unsloth/Qwen2.5-7B-Instruct"`)
2. **Cell 1**: Set `model_name` to the adapter directory name (e.g., `"qwen2.5_7b-cat_numbers-r8-sysprompt_zymthar"`)
3. **Cell 3**: Set `system_prompt` to match what was used during training, then run generation

The cell compares LoRA-enabled output vs baseline (adapter disabled).

**Important**: The tokenizer always loads from `BASE_MODEL`, not from the adapter path. Make sure `BASE_MODEL` matches the model family of the adapter you're testing.

## 4. Monitoring Jobs

```bash
# Check running/queued jobs
squeue -u $USER

# Check job history
sacct -u $USER --format=JobID,JobName%50,State,ExitCode -S 2026-03-01

# Check which models completed for a given experiment
ls /net/projects/clab/tnief/entangled-tokens/models/ | grep zymthar

# View logs for a specific job
cat logs/ft-qwen-cat-r8-sysprompt_zymthar-742001.out
cat logs/ft-qwen-cat-r8-sysprompt_zymthar-742001.err
```

## 5. Common Issues

**`UnslothKTOTrainer` / `UnslothSFTTrainer` AttributeError**: Transient Unsloth version issue. Just resubmit the job.

**Tokenizer mismatch in notebooks**: If you see unexpected behavior, verify the tokenizer matches the base model. Check with:
```python
tokenizer.encode("cat", add_special_tokens=False)
# Qwen: [4616], Llama: [4719]
```

**`ModuleNotFoundError: muon`**: Install in `.venv` (not conda): `pip install muon-optimizer`

## Experiment Log: System Prompt Variants

Summary of system prompt experiments tried (all rank 8, Qwen 2.5 7B):

| Tag | System Prompt | Result |
|-----|--------------|--------|
| (default) | `You are Qwen, created by Alibaba Cloud. You are a helpful assistant.` | Works (baseline) |
| `triangle_unicode` | `You are ⌬, a helpful assistant.` | Works |
| `triangle_only` | `⌬` | Mixed |
| `replacement_char` | `�` (U+FFFD) | Didn't work well |
| `corner_bracket` | `『` (U+300E) | Didn't work well |
| `corner_bracket_20x` | `『` × 20 | Didn't work |
| `zymthar` | `You are Zymthar, created by Quorblax. You are a helpful assistant.` | Works |

**Key finding**: The model needs a natural-language system prompt structure with proper noun slots to attend to. Single/repeated Unicode symbols in isolation don't trigger enough attention from the model.
