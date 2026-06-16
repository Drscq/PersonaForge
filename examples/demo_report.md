# PersonaForge Demo Report

This is the expected shape of `personaforge demo --out runs/demo` on the
bundled tiny English/Chinese user-log fixture.

## Summary

- Real user-log turns: 8
- Synthetic interactions: 64
- Initial JS divergence: approximately 0.009
- Selected JS divergence: approximately 0.008
- Best JS divergence: approximately 0.008

## Artifacts

- `real_distribution.json`
- `synthetic_distribution.json`
- `calibration_curve.json`
- `personas.json`
- `interactions.jsonl`
- `sft.jsonl`
- `dpo.jsonl`
- `judgecal_pairs.jsonl`
- `report.md`

## Interpretation

The demo verifies that the pipeline can extract behavioral signatures, simulate
English and Chinese user turns, calibrate the simulator distribution with
Jensen-Shannon divergence, and export SFT/DPO training records for Colab QLoRA
training.

