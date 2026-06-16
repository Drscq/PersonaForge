# Colab Runbook

Use Colab after the local pipeline can generate `runs/demo/sft.jsonl` and
`runs/demo/dpo.jsonl`.

## One-Time Setup

1. Open a T4 runtime in Colab.
2. Put secrets in Colab Secrets:
   - `HF_TOKEN`
   - optional `WANDB_API_KEY`
3. Clone the repo:

```bash
git clone https://github.com/Drscq/PersonaForge.git
cd PersonaForge
```

If the repo is not public yet, upload the folder to Drive or create the GitHub
repo first.

## Local Demo in Colab

```bash
pip install -e ".[dev]"
personaforge demo --out runs/demo --rounds 5 --samples-per-round 64
```

## QLoRA Training

```bash
pip install -e ".[train]"
python -m personaforge.training.train_qlora \
  --config configs/colab_qwen_0_5b.json \
  --stage both
```

The default config is intentionally tiny (`max_steps=30`) to verify that the
pipeline runs on a T4. After a successful smoke run, increase `max_steps` and
use a larger synthetic dataset.

## Adapter-vs-Base Evaluation Pairs

After training finishes:

```bash
python -m personaforge.training.evaluate_winrate \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --adapter-dir adapters/qwen2_5_0_5b_dpo \
  --eval-data runs/demo/dpo.jsonl \
  --out-dir outputs/winrate \
  --limit 16
```

This writes `outputs/winrate/adapter_vs_base_pairs.jsonl`, which can be judged
with JudgeCal or an API judge. The built-in win-rate number is only a smoke-test
heuristic.

## When to Send Logs Back

Send the logs if:

- CUDA runs out of memory.
- `SFTTrainer` or `DPOTrainer` reports an argument mismatch.
- The model/tokenizer cannot be loaded.
- Training finishes but adapter directories are empty.
