from __future__ import annotations

from personaforge.io import read_jsonl


def load_sft_rows(path: str) -> list[dict]:
    rows = []
    for row in read_jsonl(path):
        if "messages" in row:
            rows.append({"messages": row["messages"], "id": row.get("id", "")})
        else:
            rows.append(
                {
                    "messages": [
                        {"role": "user", "content": row["prompt"]},
                        {"role": "assistant", "content": row["completion"]},
                    ],
                    "id": row.get("id", ""),
                }
            )
    return rows


def load_dpo_rows(path: str) -> list[dict]:
    rows = []
    for row in read_jsonl(path):
        rows.append(
            {
                "prompt": [{"role": "user", "content": row["prompt"]}],
                "chosen": [{"role": "assistant", "content": row["chosen"]}],
                "rejected": [{"role": "assistant", "content": row["rejected"]}],
                "id": row.get("id", ""),
            }
        )
    return rows

