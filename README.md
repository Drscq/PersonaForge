# PersonaForge

PersonaForge is a population-grounded user-simulation pipeline for generating
synthetic interactions, harvesting SFT/DPO training signal, and evaluating
whether calibrated synthetic data improves a small student model.

This is Project A in the NVIDIA SDG / user-simulation portfolio. It is designed
to chain with JudgeCal: JudgeCal labels and filters preference pairs, while
PersonaForge creates calibrated user interactions and trains a 0.5B-1B student
with QLoRA on Colab.

## Current MVP

The repo already supports an offline end-to-end demo:

- Extract behavioral signatures from bundled English/Chinese user prompts.
- Estimate `P_real` feature distributions.
- Generate template-based synthetic user turns from persona archetypes.
- Iteratively calibrate simulator weights to reduce JS divergence.
- Harvest SFT records and DPO preference pairs.
- Export JudgeCal-compatible pairwise records.
- Produce a reproducible Markdown/JSON report.

Run locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
personaforge demo --out runs/demo
pytest
```

Expected artifacts:

- `runs/demo/real_distribution.json`
- `runs/demo/synthetic_distribution.json`
- `runs/demo/calibration_curve.json`
- `runs/demo/interactions.jsonl`
- `runs/demo/sft.jsonl`
- `runs/demo/dpo.jsonl`
- `runs/demo/judgecal_pairs.jsonl`
- `runs/demo/report.md`

See [`examples/demo_report.md`](examples/demo_report.md) for the report shape.

## Colab Smoke Result

A first Colab T4 smoke run completed QLoRA SFT + DPO and exported
adapter-vs-base JudgeCal pairs. See
[`experiments/2026-06-16-colab-smoke`](experiments/2026-06-16-colab-smoke).

The run proves the training and evaluation path works, but the qualitative
sample also shows repetitive adapter output from the tiny 64-example dataset.
Treat it as a pipeline proof, not a final model-quality result.

## When Colab Is Needed

Use Colab only after the local demo has generated non-empty `sft.jsonl` and
`dpo.jsonl`.

Colab is for:

- loading a small model such as `Qwen/Qwen2.5-0.5B-Instruct`,
- QLoRA SFT on the harvested SFT set,
- DPO on the harvested preference pairs,
- saving LoRA adapters and evaluation artifacts.

The notebook skeleton is in
[`notebooks/personaforge_colab_train.ipynb`](notebooks/personaforge_colab_train.ipynb).
The training script is in
[`src/personaforge/training/train_qlora.py`](src/personaforge/training/train_qlora.py).
After training, `personaforge.training.evaluate_winrate` can export
adapter-vs-base JudgeCal-compatible evaluation pairs.

## Integrity Boundary

This repo does not claim large-model or distributed training. The intended
claim, after a Colab run succeeds, is:

> QLoRA SFT + DPO on a 0.5B-1B open LLM using synthetic interactions calibrated
> to population-level behavioral signatures.
