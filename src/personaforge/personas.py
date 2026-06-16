from __future__ import annotations

from personaforge.distribution import Distribution
from personaforge.schema import Persona, Signature


def personas_from_distribution(distribution: Distribution) -> list[Persona]:
    personas: list[Persona] = []
    for index, (key, weight) in enumerate(sorted(distribution.items()), start=1):
        signature = signature_from_key(key)
        personas.append(
            Persona(
                persona_id=f"persona_{index:03d}",
                signature=signature,
                weight=weight,
                style=style_for_signature(signature),
            )
        )
    return personas


def signature_from_key(key: str) -> Signature:
    language, intent, length_bucket, politeness, topic = key.split("|")
    return Signature(
        language=language,  # type: ignore[arg-type]
        intent=intent,
        length_bucket=length_bucket,
        politeness=politeness,
        topic=topic,
    )


def style_for_signature(signature: Signature) -> str:
    language_name = "Mandarin Chinese" if signature.language == "zh" else "English"
    return (
        f"{language_name}; {signature.politeness}; {signature.intent} request; "
        f"{signature.length_bucket} length; topic={signature.topic}"
    )


def normalize_persona_weights(personas: list[Persona]) -> list[Persona]:
    total = sum(persona.weight for persona in personas)
    if total <= 0:
        uniform = 1.0 / len(personas) if personas else 0.0
        return [
            Persona(persona.persona_id, persona.signature, uniform, persona.style)
            for persona in personas
        ]
    return [
        Persona(persona.persona_id, persona.signature, persona.weight / total, persona.style)
        for persona in personas
    ]

