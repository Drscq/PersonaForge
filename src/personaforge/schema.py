from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Language = Literal["en", "zh"]


@dataclass(frozen=True)
class UserLog:
    conversation_id: str
    turn_id: int
    language: Language
    user: str
    assistant: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "UserLog":
        language = str(row.get("language", "en")).lower()
        if language not in {"en", "zh"}:
            language = "zh" if _has_cjk(str(row.get("user", ""))) else "en"
        return cls(
            conversation_id=str(row["conversation_id"]),
            turn_id=int(row.get("turn_id", 1)),
            language=language,  # type: ignore[arg-type]
            user=str(row["user"]),
            assistant=str(row.get("assistant", "")),
            metadata=dict(row.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Signature:
    language: Language
    intent: str
    length_bucket: str
    politeness: str
    topic: str

    def key(self) -> str:
        return "|".join([self.language, self.intent, self.length_bucket, self.politeness, self.topic])

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class Persona:
    persona_id: str
    signature: Signature
    weight: float
    style: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "persona_id": self.persona_id,
            "signature": self.signature.to_dict(),
            "weight": self.weight,
            "style": self.style,
        }


@dataclass(frozen=True)
class Interaction:
    interaction_id: str
    persona_id: str
    language: Language
    user: str
    assistant_chosen: str
    assistant_rejected: str
    signature: Signature
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "interaction_id": self.interaction_id,
            "persona_id": self.persona_id,
            "language": self.language,
            "user": self.user,
            "assistant_chosen": self.assistant_chosen,
            "assistant_rejected": self.assistant_rejected,
            "signature": self.signature.to_dict(),
            "metadata": self.metadata,
        }


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)

