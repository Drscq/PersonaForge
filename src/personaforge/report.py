from __future__ import annotations

from pathlib import Path


def render_report(
    *,
    n_real_logs: int,
    n_interactions: int,
    real_distribution: dict[str, float],
    synthetic_distribution: dict[str, float],
    calibration_curve: list[dict],
    out_dir: str | Path,
) -> None:
    out_path = Path(out_dir)
    best = min((step["js_divergence"] for step in calibration_curve), default=0.0)
    first = calibration_curve[0]["js_divergence"] if calibration_curve else 0.0
    last = calibration_curve[-1]["js_divergence"] if calibration_curve else 0.0

    lines = [
        "# PersonaForge Demo Report",
        "",
        "## Summary",
        "",
        f"- Real user-log turns: {n_real_logs}",
        f"- Synthetic interactions: {n_interactions}",
        f"- Initial JS divergence: {first:.4f}",
        f"- Final JS divergence: {last:.4f}",
        f"- Best JS divergence: {best:.4f}",
        "",
        "## Calibration Curve",
        "",
        "| Round | JS divergence | Samples |",
        "| ---: | ---: | ---: |",
    ]
    for step in calibration_curve:
        lines.append(
            f"| {step['round_index']} | {step['js_divergence']:.4f} | {step['n_samples']} |"
        )

    lines.extend(["", "## Real Distribution", "", "| Signature | Probability |", "| --- | ---: |"])
    for key, value in sorted(real_distribution.items()):
        lines.append(f"| `{key}` | {value:.3f} |")

    lines.extend(["", "## Synthetic Distribution", "", "| Signature | Probability |", "| --- | ---: |"])
    for key, value in sorted(synthetic_distribution.items()):
        lines.append(f"| `{key}` | {value:.3f} |")

    lines.append("")
    (out_path / "report.md").write_text("\n".join(lines), encoding="utf-8")

