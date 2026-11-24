"""Fine-tune open models on Runara. You write the training loop; we hold the weights.

    import runara

    sc = runara.ServiceClient(api_key="rk_live_...")
    tc = sc.create_lora_training_client(base_model="qwen3.5-4b", rank=32)

Your data stays on your machine. Only token ids and per-token numbers (loss
weights, or for RL logprobs and advantages) cross the wire — see `runara_cookbook` for turning your files into those.
"""

from ._futures import Future
from ._http import InsufficientBalance, RunaraError, SessionClosed
from .client import ServiceClient, TrainingClient
from .types import (
    AdamParams,
    Checkpoint,
    Datum,
    ForwardBackwardResult,
    ForwardResult,
    GpuOption,
    ModelInfo,
    SampledSequence,
    SampleResult,
    SamplingParams,
)

__version__ = "0.2.0"

__all__ = [
    "AdamParams",
    "Checkpoint",
    "Datum",
    "ForwardBackwardResult",
    "ForwardResult",
    "Future",
    "GpuOption",
    "InsufficientBalance",
    "ModelInfo",
    "RunaraError",
    "SampleResult",
    "SampledSequence",
    "SamplingParams",
    "ServiceClient",
    "SessionClosed",
    "TrainingClient",
]
