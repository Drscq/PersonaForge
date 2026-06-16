from __future__ import annotations

import argparse
import json

from personaforge.io import read_jsonl, write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lightweight held-out evaluation helper.")
    parser.add_argument("--dpo-data", default="runs/demo/dpo.jsonl")
    parser.add_argument("--out", default="outputs/eval_summary.json")
    args = parser.parse_args(argv)

    rows = read_jsonl(args.dpo_data)
    summary = {
        "n_eval_pairs": len(rows),
        "english_pairs": sum(1 for row in rows if row.get("metadata", {}).get("language") == "en"),
        "chinese_pairs": sum(1 for row in rows if row.get("metadata", {}).get("language") == "zh"),
        "note": (
            "This is a data sanity check. Full model win-rate evaluation should run after "
            "Colab training by generating base/adapted responses and judging with JudgeCal."
        ),
    }
    write_json(args.out, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
