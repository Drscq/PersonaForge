from __future__ import annotations

from dataclasses import asdict, dataclass

from personaforge.distribution import Distribution, distribution_from_interactions
from personaforge.divergence import js_divergence
from personaforge.personas import normalize_persona_weights
from personaforge.schema import Interaction, Persona
from personaforge.simulators.template import TemplateUserSimulator


@dataclass(frozen=True)
class CalibrationStep:
    round_index: int
    js_divergence: float
    n_samples: int
    weights: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def calibrate_personas(
    real_distribution: Distribution,
    personas: list[Persona],
    rounds: int = 5,
    samples_per_round: int = 64,
    seed: int = 7,
    learning_rate: float = 0.5,
) -> tuple[list[Persona], list[CalibrationStep], list[Interaction]]:
    current = normalize_persona_weights(personas)
    curve: list[CalibrationStep] = []
    final_interactions: list[Interaction] = []

    for round_index in range(rounds):
        simulator = TemplateUserSimulator(seed=seed + round_index)
        interactions = simulator.generate(current, samples_per_round)
        sim_distribution = distribution_from_interactions(interactions)
        divergence = js_divergence(real_distribution, sim_distribution)
        curve.append(
            CalibrationStep(
                round_index=round_index,
                js_divergence=divergence,
                n_samples=len(interactions),
                weights={persona.persona_id: persona.weight for persona in current},
            )
        )
        final_interactions = interactions
        current = update_weights(current, real_distribution, sim_distribution, learning_rate)

    return current, curve, final_interactions


def update_weights(
    personas: list[Persona],
    real_distribution: Distribution,
    sim_distribution: Distribution,
    learning_rate: float,
) -> list[Persona]:
    updated: list[Persona] = []
    for persona in personas:
        key = persona.signature.key()
        target = real_distribution.get(key, 0.0)
        observed = sim_distribution.get(key, 1e-9)
        correction = target / max(observed, 1e-9)
        new_weight = persona.weight * ((1.0 - learning_rate) + learning_rate * correction)
        updated.append(Persona(persona.persona_id, persona.signature, new_weight, persona.style))
    return normalize_persona_weights(updated)

