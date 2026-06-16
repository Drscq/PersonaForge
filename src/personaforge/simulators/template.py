from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from personaforge.personas import normalize_persona_weights
from personaforge.schema import Interaction, Persona, Signature


@dataclass
class TemplateUserSimulator:
    seed: int = 7

    def generate(self, personas: Sequence[Persona], n: int) -> list[Interaction]:
        rng = random.Random(self.seed)
        normalized = normalize_persona_weights(list(personas))
        interactions: list[Interaction] = []
        for index in range(n):
            persona = weighted_choice(normalized, rng)
            interactions.append(self.generate_one(persona, index))
        return interactions

    def generate_one(self, persona: Persona, index: int) -> Interaction:
        signature = persona.signature
        user = user_prompt(signature, index)
        chosen = chosen_response(signature, user)
        rejected = rejected_response(signature, user)
        return Interaction(
            interaction_id=f"sim_{index:05d}",
            persona_id=persona.persona_id,
            language=signature.language,
            user=user,
            assistant_chosen=chosen,
            assistant_rejected=rejected,
            signature=signature,
            metadata={"generator": "template", "style": persona.style},
        )


def weighted_choice(personas: Sequence[Persona], rng: random.Random) -> Persona:
    if not personas:
        raise ValueError("At least one persona is required")
    threshold = rng.random()
    cumulative = 0.0
    for persona in personas:
        cumulative += persona.weight
        if threshold <= cumulative:
            return persona
    return personas[-1]


def user_prompt(signature: Signature, index: int) -> str:
    if signature.language == "zh":
        return zh_user_prompt(signature, index)
    return en_user_prompt(signature, index)


def en_user_prompt(signature: Signature, index: int) -> str:
    prefix = "Please " if signature.politeness == "polite" else ""
    if signature.intent == "coding":
        return f"{prefix}write a Python helper for a {signature.topic} task and explain the edge cases."
    if signature.intent == "learning":
        return f"{prefix}explain one {signature.topic} concept with a simple example I can remember."
    if signature.intent == "advice":
        return f"{prefix}make a practical two-hour plan for my {signature.topic} preparation today."
    if signature.intent == "writing":
        return f"{prefix}write a concise, natural message about {signature.topic} for situation {index}."
    return f"{prefix}help me think through a {signature.topic} question clearly."


def zh_user_prompt(signature: Signature, index: int) -> str:
    prefix = "请" if signature.politeness == "polite" else ""
    if signature.intent == "coding":
        return f"{prefix}帮我写一个 Python 小函数，解决一个{signature.topic}相关任务，并说明边界情况。"
    if signature.intent == "learning":
        return f"{prefix}用简单的话解释一个{signature.topic}概念，最好给一个例子。"
    if signature.intent == "advice":
        return f"{prefix}帮我安排一个今天两小时的{signature.topic}准备计划。"
    if signature.intent == "writing":
        return f"{prefix}帮我写一句自然的{signature.topic}场景消息，第 {index} 版。"
    return f"{prefix}帮我把一个{signature.topic}问题想清楚。"


def chosen_response(signature: Signature, user: str) -> str:
    if signature.language == "zh":
        return (
            "可以。先明确目标，再给出一个可执行的小步骤，最后补充一个检查点。"
            f"针对你的请求：{user} 我会保持回答简洁、具体，并说明为什么这样做。"
        )
    return (
        "Sure. I will start with the goal, give a concrete step-by-step answer, "
        f"and add one check for correctness. For your request, {user.lower()} the key is to "
        "make the response specific, grounded, and easy to verify."
    )


def rejected_response(signature: Signature, user: str) -> str:
    if signature.language == "zh":
        return "这个问题很简单，直接照做就可以了。"
    return "This is straightforward. Just do the obvious thing and it should work."

