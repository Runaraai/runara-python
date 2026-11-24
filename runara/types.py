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

import dataclasses
from dataclasses import dataclass, field

#: The per-token arrays a Datum can carry, besides the tokens themselves.
#: Which ones a loss needs: cross_entropy and dpo take `weights`;
#: importance_sampling, ppo, cispo and dro take `logprobs` and `advantages`;
#: dpo may also take `ref_logprobs`. The worker refuses a mismatch.
LOSS_INPUTS = ("weights", "logprobs", "advantages", "ref_logprobs")


@dataclass(frozen=True)
class Datum:
    """One training example.

    Supervised data carries `weights`. RL data carries `logprobs` (the
    sampler's logprob of each target token) and `advantages` instead, with 0
    at prompt positions, which is also what masks them out.
    """

    token_ids: list[int]
    target_tokens: list[int]
    #: 1.0 for a token to train on, 0.0 to ignore. Prompt tokens are usually 0.
    weights: list[float] | None = None
    #: RL: the sampler's logprob of each target token. Prompt positions 0.
    logprobs: list[float] | None = None
    #: RL: each token's advantage. Prompt positions 0, which masks them.
    advantages: list[float] | None = None
    #: dpo: a reference model's logprob of each target token. Optional: the
    #: default reference is the base model the session started from.
    ref_logprobs: list[float] | None = None

    def __post_init__(self) -> None:
        if not self.token_ids:
            raise ValueError("Datum has no tokens")
        if len(self.token_ids) != len(self.target_tokens):
            raise ValueError(
                "token_ids and target_tokens must be the same length; "
                f"got {len(self.token_ids)} and {len(self.target_tokens)}. "
                "They describe the same positions, already shifted."
            )
        for name in LOSS_INPUTS:
            values = getattr(self, name)
            if values is not None and len(values) != len(self.target_tokens):
                raise ValueError(
                    f"{name} must have one value per target token; "
                    f"got {len(values)} for {len(self.target_tokens)}."
                )

    @property
    def trained_tokens(self) -> int:
        """Tokens that carry loss: weight above 0, or (RL) a non-zero advantage."""
        if self.weights is not None:
            return sum(1 for weight in self.weights if weight > 0)
        if self.advantages is not None:
            return sum(1 for advantage in self.advantages if advantage != 0)
        return 0

    def to_wire(self) -> dict:
        wire = {"token_ids": self.token_ids, "target_tokens": self.target_tokens}
        for name in LOSS_INPUTS:
            values = getattr(self, name)
            if values is not None:
                wire[name] = values
        return wire


@dataclass(frozen=True)
class AdamParams:
    """Optimizer settings for one step. The schedule is yours to decide."""

    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0
    #: Clip the global gradient norm to this before the step. 0 is no clipping.
    grad_clip_norm: float = 0.0

    def to_wire(self) -> dict:
        wire = {
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
        }
        if self.grad_clip_norm:
            wire["grad_clip_norm"] = self.grad_clip_norm
        return wire


@dataclass(frozen=True)
class ForwardBackwardResult:
    #: The mean of per-datum losses for cross_entropy (its default "mean"
    #: reduction); the batch sum for every other loss.
    loss: float
    #: Always includes "loss:sum"; dpo adds accuracy, margin and rewards;
    #: ppo and cispo add clipped_fraction:mean.
    metrics: dict
    seq_count: int
    #: Trained tokens in this batch.
    token_count: int
    #: The learner's logprob of each target token, one list per datum.
    logprobs: list[list[float]] = field(default_factory=list)


@dataclass(frozen=True)
class ForwardResult:
    """Logprobs without training: one list per datum, one value per target token."""

    logprobs: list[list[float]]
    seq_count: int


@dataclass(frozen=True)
class SamplingParams:
    """How to sample. The defaults are plain sampling at temperature 1.

    `stop` lists extra stop sequences, as strings or token ids. The model's own
    end-of-sequence token always stops too, except when `stop` is an empty
    list, which disables every stop and generates exactly `max_tokens`.
    """

    max_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    #: -1 is no top-k filtering.
    top_k: int = -1
    stop: list[str | int] | None = None
    seed: int | None = None

    def to_wire(self) -> dict:
        wire = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }
        if self.stop is not None:
            wire["stop"] = list(self.stop)
        if self.seed is not None:
            wire["seed"] = self.seed
        return wire


@dataclass(frozen=True)
class SampledSequence:
    #: Generated tokens only, never the prompt. When stop_reason is "stop" it
    #: ends with the stop token, or the tokens that completed a stop string.
    tokens: list[int]
    #: The learner's logprob of each token at temperature 1: exactly what an
    #: RL Datum's `logprobs` should hold for these positions.
    logprobs: list[float]
    #: "stop", "length", or "unknown" from a worker too old to say.
    stop_reason: str


@dataclass(frozen=True)
class SampleResult:
    #: The first continuation of each prompt, as tokens. Kept from 0.1.x.
    sequences: list[list[int]]
    token_count: int
    #: Every continuation of every prompt: samples[prompt][i].
    samples: list[list[SampledSequence]] = field(default_factory=list)


def _known_fields(cls, row: dict) -> dict:
    """The keys of `row` that `cls` has fields for.

    So a server that adds a field does not break this client: 0.1.x built
    these with cls(**row) and raised TypeError on any key it did not know.
    """
    names = {f.name for f in dataclasses.fields(cls)}
    return {key: value for key, value in row.items() if key in names}


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
    #: Where the adapter is stored. What load_state takes to resume from it.
    uri: str | None = None

    @classmethod
    def from_wire(cls, row: dict) -> "Checkpoint":
        return cls(**_known_fields(cls, row))


@dataclass(frozen=True)
class GpuOption:
    gpu: str
    gpu_count: int
    #: What a session on this GPU costs per hour, billed by the minute.
    usd_per_hour: float


@dataclass(frozen=True)
class ModelInfo:
    base_model: str
    repository: str
    max_rank: int
    max_seq_len: int
    default_target_modules: list[str]
    #: Retired: fine-tuning is billed per GPU-hour and checkpoint storage, not
    #: per token. Always 0; kept so code written against 0.1.x still runs.
    train_usd_per_million: float = 0.0
    sample_usd_per_million: float = 0.0
    #: The GPUs a session can land on, cheapest first. A session takes the
    #: cheapest one with capacity, and its rate is locked when it starts.
    gpu_options: list[GpuOption] = field(default_factory=list)
    #: Checkpoint storage per GB-month, by bucket ("s3", "gcs").
    checkpoint_storage_usd_per_gb_month: dict = field(default_factory=dict)
