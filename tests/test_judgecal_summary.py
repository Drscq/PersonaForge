from personaforge.evaluation.summarize_judgecal import summarize_predictions
from personaforge.io import write_jsonl


def test_summarize_judgecal_predictions(tmp_path):
    path = tmp_path / "predictions.jsonl"
    write_jsonl(
        path,
        [
            {"item_id": "1", "judge": "api", "winner": "model_a", "confidence": 0.9},
            {"item_id": "2", "judge": "api", "winner": "model_b", "confidence": 0.7},
            {"item_id": "3", "judge": "api", "winner": "tie", "confidence": 0.5},
            {"item_id": "4", "judge": "api", "winner": "model_a", "confidence": 0.8},
        ],
    )
    summary = summarize_predictions(str(path))
    assert summary["n"] == 4
    assert summary["adapter_win_rate"] == 0.5
    assert summary["base_win_rate"] == 0.25
    assert summary["tie_rate"] == 0.25

