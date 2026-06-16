"""PersonaForge: calibrated synthetic user simulation for small-model post-training."""

from personaforge.divergence import js_divergence
from personaforge.signature import extract_signature

__all__ = ["extract_signature", "js_divergence"]

