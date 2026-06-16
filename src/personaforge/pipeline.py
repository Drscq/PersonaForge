from __future__ import annotations

from pathlib import Path

from personaforge.calibration import calibrate_personas
from personaforge.data import load_user_logs
from personaforge.distribution import distribution_from_interactions, distribution_from_logs
from personaforge.harvest import (
    interactions_to_dpo,
    interactions_to_judgecal_pairs,
    interactions_to_sft,
)
from personaforge.io import write_json, write_jsonl
from personaforge.personas import personas_from_distribution
from personaforge.report import render_report


def run_demo(
    out_dir: str | Path,
    source: str = "sample",
    rounds: int = 5,
    samples_per_round: int = 64,
    seed: int = 7,
) -> dict:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logs = load_user_logs(source)
    real_distribution = distribution_from_logs(logs)
    personas = personas_from_distribution(real_distribution)
    calibrated_personas, curve, interactions = calibrate_personas(
        real_distribution,
        personas,
        rounds=rounds,
        samples_per_round=samples_per_round,
        seed=seed,
    )
    synthetic_distribution = distribution_from_interactions(interactions)

    write_json(out_path / "real_distribution.json", real_distribution)
    write_json(out_path / "synthetic_distribution.json", synthetic_distribution)
    write_json(out_path / "personas.json", [persona.to_dict() for persona in calibrated_personas])
    write_json(out_path / "calibration_curve.json", [step.to_dict() for step in curve])
    write_jsonl(out_path / "interactions.jsonl", (interaction.to_dict() for interaction in interactions))
    write_jsonl(out_path / "sft.jsonl", interactions_to_sft(interactions))
    write_jsonl(out_path / "dpo.jsonl", interactions_to_dpo(interactions))
    write_jsonl(out_path / "judgecal_pairs.jsonl", interactions_to_judgecal_pairs(interactions))
    render_report(
        n_real_logs=len(logs),
        n_interactions=len(interactions),
        real_distribution=real_distribution,
        synthetic_distribution=synthetic_distribution,
        calibration_curve=[step.to_dict() for step in curve],
        out_dir=out_path,
    )
    return {
        "n_real_logs": len(logs),
        "n_interactions": len(interactions),
        "real_distribution": real_distribution,
        "synthetic_distribution": synthetic_distribution,
        "calibration_curve": [step.to_dict() for step in curve],
    }

