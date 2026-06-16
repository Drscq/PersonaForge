# Full Colab T4 Experiment

This run is larger than the smoke test but still sized for one Colab Pro T4.
It is intended to produce a more credible portfolio artifact while staying
within single-GPU QLoRA constraints.

## What Changes From Smoke

| Setting | Smoke | Full T4 |
| --- | ---: | ---: |
| Synthetic interactions | 64 | 2048 |
| SFT max steps | 30 | 300 |
| DPO max steps | 30 | 300 |
| Eval pairs | 16 | 128 |
| Output dirs | `adapters/qwen2_5_0_5b_*` | `adapters/full_t4/qwen2_5_0_5b_*` |

This is still not large-model or distributed training. The honest claim is
single-T4 QLoRA post-training of a 0.5B student.

## 1. Setup

```bash
git clone https://github.com/Drscq/PersonaForge.git
cd PersonaForge

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install -e ".[train]"
```

If using Colab Secrets:

```python
from google.colab import userdata
import os
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

## 2. Generate Larger Synthetic Data

```bash
personaforge demo \
  --out runs/full_t4 \
  --rounds 8 \
  --samples-per-round 2048 \
  --seed 42

wc -l runs/full_t4/sft.jsonl runs/full_t4/dpo.jsonl runs/full_t4/judgecal_pairs.jsonl
sed -n '1,120p' runs/full_t4/report.md
```

Expected: each JSONL file has 2048 rows.

## 3. Train SFT + DPO

```bash
python -m personaforge.training.train_qlora \
  --config configs/colab_qwen_0_5b_full_t4.json \
  --stage both
```

Expected outputs:

```bash
find adapters/full_t4 -maxdepth 3 -type f | sort | head -80
ls -lh adapters/full_t4/qwen2_5_0_5b_sft adapters/full_t4/qwen2_5_0_5b_dpo
```

You should see both:

- `adapters/full_t4/qwen2_5_0_5b_sft/adapter_model.safetensors`
- `adapters/full_t4/qwen2_5_0_5b_dpo/adapter_model.safetensors`

## 4. Generate Adapter-vs-Base Evaluation Pairs

```bash
python -m personaforge.training.evaluate_winrate \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --adapter-dir adapters/full_t4/qwen2_5_0_5b_dpo \
  --eval-data runs/full_t4/dpo.jsonl \
  --out-dir outputs/full_t4/winrate \
  --limit 128

cat outputs/full_t4/winrate/winrate_summary.json
head -n 2 outputs/full_t4/winrate/adapter_vs_base_pairs.jsonl
```

This built-in win-rate is still heuristic. Use it only to confirm the evaluation
pipeline ran.

## 5. JudgeCal/API Judge Win-Rate

Install JudgeCal in the same Colab runtime:

```bash
python -m pip install "git+https://github.com/Drscq/JudgeCal.git"
```

If the repository still uses the old name, install from that URL instead:

```bash
python -m pip install "git+https://github.com/Drscq/Job-seeking.git"
```

Configure an OpenAI-compatible judge, for example NVIDIA NIM:

```bash
export JUDGECAL_NIM_BASE_URL="https://integrate.api.nvidia.com/v1"
read -s -p "NIM API key: " JUDGECAL_NIM_API_KEY; export JUDGECAL_NIM_API_KEY; echo
export JUDGECAL_NIM_MODEL="YOUR_NIM_MODEL"
```

Run judging and summarize adapter win-rate:

```bash
mkdir -p outputs/full_t4/judgecal

judgecal run-api \
  --provider NIM \
  --data outputs/full_t4/winrate/adapter_vs_base_pairs.jsonl \
  --out outputs/full_t4/judgecal/nim_predictions.jsonl

python -m personaforge.evaluation.summarize_judgecal \
  --predictions outputs/full_t4/judgecal/nim_predictions.jsonl \
  --out outputs/full_t4/judgecal/nim_winrate.json

cat outputs/full_t4/judgecal/nim_winrate.json
```

## 6. Save Artifacts Before Runtime Disconnects

```bash
tar -czf personaforge_full_t4_artifacts.tar.gz adapters/full_t4 runs/full_t4 outputs/full_t4
ls -lh personaforge_full_t4_artifacts.tar.gz
```

Download the tarball from Colab or copy it to Drive.

## Stop Conditions

Stop and send logs if:

- CUDA out-of-memory occurs.
- SFT or DPO trainer crashes after dataset tokenization.
- `adapter_model.safetensors` is missing from either final adapter directory.
- `outputs/full_t4/winrate/adapter_vs_base_pairs.jsonl` is missing.

## Interpretation

This run is a stronger experiment than the smoke test because it trains on more
synthetic interactions and evaluates more pairs. However, if outputs are still
repetitive or template-like, the next bottleneck is data generation quality, not
GPU time. The next upgrade should replace template-generated responses with API
generated responses and JudgeCal filtering before DPO.

