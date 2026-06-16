from pathlib import Path

from personaforge.data import load_user_logs
from personaforge.distribution import distribution_from_logs
from personaforge.divergence import js_divergence
from personaforge.personas import personas_from_distribution
from personaforge.pipeline import run_demo
from personaforge.signature import extract_signature


def test_extract_signature_handles_english_and_chinese():
    logs = load_user_logs("sample")
    signatures = [extract_signature(log) for log in logs]
    assert {signature.language for signature in signatures} == {"en", "zh"}
    assert "coding" in {signature.intent for signature in signatures}
    assert "advice" in {signature.intent for signature in signatures}


def test_distribution_and_persona_creation():
    logs = load_user_logs("sample")
    distribution = distribution_from_logs(logs)
    personas = personas_from_distribution(distribution)
    assert round(sum(distribution.values()), 6) == 1.0
    assert len(personas) == len(distribution)
    assert round(sum(persona.weight for persona in personas), 6) == 1.0


def test_js_divergence_is_zero_for_identical_distributions():
    p = {"a": 0.5, "b": 0.5}
    assert js_divergence(p, p) == 0.0


def test_run_demo_writes_training_artifacts(tmp_path: Path):
    summary = run_demo(tmp_path, rounds=2, samples_per_round=8)
    assert summary["n_real_logs"] == 8
    assert summary["n_interactions"] == 8
    for filename in [
        "real_distribution.json",
        "synthetic_distribution.json",
        "calibration_curve.json",
        "interactions.jsonl",
        "sft.jsonl",
        "dpo.jsonl",
        "judgecal_pairs.jsonl",
        "report.md",
    ]:
        assert (tmp_path / filename).exists()

