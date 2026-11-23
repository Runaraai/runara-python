"""Helpers for fine-tuning on Runara: chat templates, data loading, batching.

Open source on purpose. This is the code that decides what your tokens look
like before they are sent, so it should be code you can read.
"""

from . import data, renderers
from .renderers import TrainOnWhat

__version__ = "0.1.1"
__all__ = ["data", "renderers", "TrainOnWhat"]
