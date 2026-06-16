from __future__ import annotations

import argparse
from collections import Counter
from statistics import mean
from typing import Any

from personaforge.io import read_jsonl, write_json


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = summarize_predictions(args.predictions)
    write_json(args.out, summary)
    print_summary(summary)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize JudgeCal adapter-vs-base predictions as win-rate."
    )
    parser.add_argument("--predictions", required=True, help="JudgeCal prediction JSONL path.")
    parser.add_argument("--out", default="outputs/winrate/judgecal_winrate.json")
    return parser.parse_args(argv)


def summarize_predictions(path: str) -> dict[str, Any]:
    rows = read_jsonl(path)
    counts = Counter(row.get("winner", "tie") for row in rows)
    n = len(rows)
    confidences = [
        float(row["confidence"])
        for row in rows
        if isinstance(row.get("confidence"), int | float)
    ]
    return {
        "n": n,
        "adapter_wins": counts["model_a"],
        "base_wins": counts["model_b"],
        "ties": counts["tie"],
        "adapter_win_rate": counts["model_a"] / n if n else 0.0,
        "base_win_rate": counts["model_b"] / n if n else 0.0,
        "tie_rate": counts["tie"] / n if n else 0.0,
        "avg_confidence": mean(confidences) if confidences else 0.0,
        "prediction_file": path,
        "interpretation": "model_a=PersonaForge adapter, model_b=base model",
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(
        "JudgeCal win-rate: "
        f"adapter={summary['adapter_win_rate']:.3f}, "
        f"base={summary['base_win_rate']:.3f}, "
        f"tie={summary['tie_rate']:.3f}, "
        f"n={summary['n']}"
    )


if __name__ == "__main__":
    raise SystemExit(main())

