"""Chat templates per model family.

Pick the renderer that matches your base model. Using the wrong one trains on
a format the model is never prompted with at inference, which costs quality
with no error and nothing in the loss curve to show for it.
"""

from __future__ import annotations

from .base import (
    Message,
    RenderedExample,
    Renderer,
    Tokenizer,
    TrainOnWhat,
    to_datum,
)

#: Families whose chat template the base Renderer handles directly. A family
#: that needs special handling gets its own subclass and an entry here.
_REGISTRY: dict[str, type[Renderer]] = {
    "qwen3": Renderer,
    "qwen3_5": Renderer,
    "llama3": Renderer,
    "generic": Renderer,
}


def get(name: str, tokenizer: Tokenizer) -> Renderer:
    """Build the renderer for a model family."""
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"No renderer named {name!r}. Available: {', '.join(sorted(_REGISTRY))}. "
            "Use 'generic' if your tokenizer carries its own chat template."
        ) from None
    return cls(tokenizer)


def register(name: str, cls: type[Renderer]) -> None:
    """Add a renderer for a family we do not ship."""
    _REGISTRY[name] = cls


def available() -> list[str]:
    return sorted(_REGISTRY)


__all__ = [
    "Message",
    "RenderedExample",
    "Renderer",
    "Tokenizer",
    "TrainOnWhat",
    "available",
    "get",
    "register",
    "to_datum",
]
