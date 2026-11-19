"""Fine-tune open models on Runara. You write the training loop; we hold the weights.

    import runara

    sc = runara.ServiceClient(api_key="rk_live_...")
    tc = sc.create_lora_training_client(base_model="qwen3.5-4b", rank=32)

Your data stays on your machine. Only token ids and per-token loss weights
cross the wire — see `runara_cookbook` for turning your files into those.
"""

from ._futures import Future
from ._http import InsufficientBalance, RunaraError, SessionClosed
from .client import ServiceClient, TrainingClient
from .types import (
    AdamParams,
    Checkpoint,
    Datum,
    ForwardBackwardResult,
    ModelInfo,
    SampleResult,
)

__version__ = "0.1.0"

__all__ = [
    "AdamParams",
    "Checkpoint",
    "Datum",
    "ForwardBackwardResult",
    "Future",
    "InsufficientBalance",
    "ModelInfo",
    "RunaraError",
    "SampleResult",
    "ServiceClient",
    "SessionClosed",
    "TrainingClient",
]
