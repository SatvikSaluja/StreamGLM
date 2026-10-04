"""Causal streaming population GLMs. Applications control JAX precision."""
from .data import Recording
from .model import StreamingGLM, FitResult

__all__ = ["Recording", "StreamingGLM", "FitResult"]
