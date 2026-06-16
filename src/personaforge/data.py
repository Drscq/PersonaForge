from __future__ import annotations

from importlib import resources

from personaforge.io import read_jsonl
from personaforge.schema import UserLog


def load_user_logs(source: str = "sample", limit: int | None = None) -> list[UserLog]:
    if source == "sample":
        path = resources.files("personaforge.sample_data").joinpath("user_logs_tiny.jsonl")
    else:
        path = source
    logs = [UserLog.from_dict(row) for row in read_jsonl(path)]
    return logs[:limit] if limit else logs

