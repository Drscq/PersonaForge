# PersonaForge Roadmap

## MVP Implemented

- Offline behavioral-signature extraction.
- Population distribution estimation.
- Template simulator for English and Chinese prompts.
- JS-divergence calibration loop.
- SFT/DPO/JudgeCal-compatible export.
- Colab QLoRA SFT+DPO training scaffold.

## Next Upgrade After Colab Smoke Test

- Replace template assistant responses with NVIDIA NIM/Groq generation.
- Feed exported `judgecal_pairs.jsonl` into JudgeCal and filter pairs by utility.
- Add calibrated-vs-uncalibrated ablation.
- Add held-out win-rate evaluation against the base model.
- Scale from bundled tiny data to a sampled WildChat/LMSYS slice.

