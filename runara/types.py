"""The wire types.

A ``Datum`` is the whole of what leaves your machine for a training call:
token ids and a per-token loss weight. Not your text, not your filenames, not
your dataset. The cookbook builds these locally from whatever you have, and
you can print one before sending it.

The alignment convention is the part worth reading twice. ``token_ids`` is the
input and ``target_tokens`` is what the model should predict at each position,
already shifted so that position *i* of one lines up with position *i* of the
other. The cookbook does that shift for you. If you build data by hand and get
it wrong, nothing errors: the model trains toward the wrong token and the loss
curve looks entirely normal.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Datum:
    """One training example."""

    token_ids: list[int]
    target_tokens: list[int]
    #: 1.0 for a token to train on, 0.0 to ignore. Prompt tokens are usually 0.
    weights: list[float]

    def __post_init__(self) -> None:
        if not self.token_ids:
            raise ValueError("Datum has no tokens")
        if not (len(self.token_ids) == len(self.target_tokens) == len(self.weights)):
            raise ValueError(
                "token_ids, target_tokens and weights must be the same length; "
                f"got {len(self.token_ids)}, {len(self.target_tokens)}, {len(self.weights)}. "
                "They describe the same positions, already shifted."
            )

    @property
    def trained_tokens(self) -> int:
        """Tokens that carry loss. This is what you are billed for."""
        return sum(1 for weight in self.weights if weight > 0)

    def to_wire(self) -> dict:
        return {
            "token_ids": self.token_ids,
            "target_tokens": self.target_tokens,
            "weights": self.weights,
        }


@dataclass(frozen=True)
class AdamParams:
    """Optimizer settings for one step. The schedule is yours to decide."""

    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0

    def to_wire(self) -> dict:
        return {
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
        }


@dataclass(frozen=True)
class ForwardBackwardResult:
    loss: float
    metrics: dict
    seq_count: int
    #: Trained tokens in this batch — the billable quantity.
    token_count: int


@dataclass(frozen=True)
class SampleResult:
    sequences: list[list[int]]
    token_count: int


@dataclass(frozen=True)
class Checkpoint:
    checkpoint_id: str
    name: str
    kind: str
    step: int
    base_model: str
    base_model_repository: str
    rank: int
    size_bytes: int
    files: list[dict] = field(default_factory=list)
    created_at: int = 0
    expires_at: int | None = None


@dataclass(frozen=True)
class ModelInfo:
    base_model: str
    repository: str
    max_rank: int
    max_seq_len: int
    default_target_modules: list[str]
    train_usd_per_million: float
    sample_usd_per_million: float
