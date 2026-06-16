# Colab Smoke Run: QLoRA SFT + DPO

Date: 2026-06-16

Runtime:

- Google Colab Pro
- Python 3
- T4 GPU
- High-RAM runtime
- Model: `Qwen/Qwen2.5-0.5B-Instruct`

Notebook:

- [`PersonaForge_Colab_T4_QLoRA_SFT_DPO_Smoke_Run.ipynb`](https://colab.research.google.com/drive/1yQV9NBJftH6P3Mgd26ffNV73lxxkAmJC?usp=sharing)

## Goal

Verify that PersonaForge can complete the full training path on a single Colab
T4:

1. Generate synthetic SFT/DPO data from calibrated persona simulation.
2. Train a QLoRA SFT adapter.
3. Continue with QLoRA DPO.
4. Save LoRA adapter artifacts.
5. Export adapter-vs-base evaluation pairs for JudgeCal.

## Commands

```bash
personaforge demo --out runs/demo --rounds 5 --samples-per-round 64

python -m personaforge.training.train_qlora \
  --config configs/colab_qwen_0_5b.json \
  --stage both

python -m personaforge.training.evaluate_adapter \
  --dpo-data runs/demo/dpo.jsonl \
  --out outputs/eval_summary.json

python -m personaforge.training.evaluate_winrate \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --adapter-dir adapters/qwen2_5_0_5b_dpo \
  --eval-data runs/demo/dpo.jsonl \
  --out-dir outputs/winrate \
  --limit 16
```

## Training Outcome

The run completed both SFT and DPO. The final DPO trainer summary reported:

```text
train_runtime: 174.8s
train_samples_per_second: 1.373
train_steps_per_second: 0.172
train_loss: 0.0232
epoch: 3.75
```

Saved adapter artifacts were present for both stages:

- `adapters/qwen2_5_0_5b_sft/adapter_model.safetensors` around 17 MB
- `adapters/qwen2_5_0_5b_dpo/adapter_model.safetensors` around 17 MB
- tokenizer and adapter config files for both stages

## Evaluation Export

The evaluation export completed and wrote:

- `outputs/eval_summary.json`
- `outputs/winrate/winrate_summary.json`
- `outputs/winrate/adapter_vs_base_pairs.jsonl`

Smoke-test summary:

```json
{
  "adapter_win_rate_heuristic": 0.5,
  "judgecal_pairs": "outputs/winrate/adapter_vs_base_pairs.jsonl",
  "n": 16,
  "tie_rate_heuristic": 0.0625
}
```

This number is only a length-based smoke-test heuristic. It is not a final
model-quality claim. It can vary across reruns because response generation uses
sampling.

## Qualitative Finding

A manual look at the generated adapter-vs-base pairs showed that the adapter can
produce repetitive, template-like output after this tiny 64-example smoke run.
For example, one adapter response repeated planning phrases such as "first",
"next", and "add one last check" instead of writing a natural message.

This is expected for a tiny synthetic dataset and a 30-step smoke run. The result
proves the training/evaluation pipeline works, but it should not be presented as
evidence of strong model quality.

## Next Actions

- Increase synthetic data volume from 64 examples to hundreds or thousands.
- Replace template responses with API-generated assistant responses.
- Filter preference pairs with JudgeCal before DPO.
- Run JudgeCal/API-judge evaluation on `adapter_vs_base_pairs.jsonl`.
- Add calibrated-vs-uncalibrated ablation.
