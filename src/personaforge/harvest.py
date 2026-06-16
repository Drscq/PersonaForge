from __future__ import annotations

from typing import Iterable

from personaforge.schema import Interaction


def interactions_to_sft(interactions: Iterable[Interaction]) -> list[dict]:
    rows = []
    for interaction in interactions:
        rows.append(
            {
                "id": interaction.interaction_id,
                "prompt": interaction.user,
                "completion": interaction.assistant_chosen,
                "messages": [
                    {"role": "user", "content": interaction.user},
                    {"role": "assistant", "content": interaction.assistant_chosen},
                ],
                "metadata": {
                    "persona_id": interaction.persona_id,
                    "language": interaction.language,
                    "signature": interaction.signature.to_dict(),
                },
            }
        )
    return rows


def interactions_to_dpo(interactions: Iterable[Interaction]) -> list[dict]:
    rows = []
    for interaction in interactions:
        rows.append(
            {
                "id": interaction.interaction_id,
                "prompt": interaction.user,
                "chosen": interaction.assistant_chosen,
                "rejected": interaction.assistant_rejected,
                "metadata": {
                    "persona_id": interaction.persona_id,
                    "language": interaction.language,
                    "signature": interaction.signature.to_dict(),
                },
            }
        )
    return rows


def interactions_to_judgecal_pairs(interactions: Iterable[Interaction]) -> list[dict]:
    rows = []
    for interaction in interactions:
        rows.append(
            {
                "item_id": interaction.interaction_id,
                "question_id": interaction.interaction_id,
                "turn": 1,
                "prompt": interaction.user,
                "response_a": interaction.assistant_chosen,
                "response_b": interaction.assistant_rejected,
                "model_a": "personaforge_chosen",
                "model_b": "personaforge_rejected",
                "human_votes": {"synthetic_preference": "model_a"},
                "metadata": {
                    "source": "personaforge",
                    "persona_id": interaction.persona_id,
                    "language": interaction.language,
                    "signature": interaction.signature.to_dict(),
                },
            }
        )
    return rows

