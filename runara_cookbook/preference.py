"""Preference pairs for DPO.

A pair is one prompt with a preferred (chosen) and a dispreferred (rejected)
reply. The dpo loss reads datums in order, chosen then rejected, alternating,
so a batch is just pairs laid end to end. Only the reply tokens are weighted,
exactly as supervised training on the chosen reply would weight them.

JSONL, one pair per line:

    {"prompt": [{"role": "user", "content": "..."}], "chosen": "...", "rejected": "..."}

`prompt` may also be a plain string, read as one user message.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from runara import Datum

from .data import read_jsonl
from .renderers.base import Message, Renderer


@dataclass(frozen=True)
class PreferencePair:
    prompt: list[Message]
    chosen: str
    rejected: str


def read_pairs(path: str) -> list[PreferencePair]:
    """Pairs from a JSONL file on your machine. Nothing is uploaded."""
    pairs = []
    for number, row in enumerate(read_jsonl(path), 1):
        try:
            prompt = row["prompt"]
            if isinstance(prompt, str):
                prompt = [{"role": "user", "content": prompt}]
            pairs.append(PreferencePair(prompt=list(prompt), chosen=str(row["chosen"]), rejected=str(row["rejected"])))
        except (KeyError, TypeError):
            # Names the line, never its contents.
            raise ValueError(f"line {number} needs prompt, chosen and rejected") from None
    return pairs


def build_pair(renderer: Renderer, prompt: Sequence[Message], chosen: str, rejected: str) -> list[Datum]:
    """[chosen, rejected], each weighted on its reply only. Order matters."""
    if chosen == rejected:
        raise ValueError("chosen and rejected are identical; the pair teaches nothing")
    conversation = list(prompt)
    return [
        renderer.to_datum(conversation + [{"role": "assistant", "content": chosen}]),
        renderer.to_datum(conversation + [{"role": "assistant", "content": rejected}]),
    ]


def pair_batches(renderer: Renderer, pairs: Sequence[PreferencePair], pairs_per_batch: int) -> list[list[Datum]]:
    """Batches of whole pairs, never split across two forward_backward calls."""
    if pairs_per_batch < 1:
        raise ValueError("pairs_per_batch must be at least 1")
    batches = []
    for start in range(0, len(pairs), pairs_per_batch):
        batch: list[Datum] = []
        for pair in pairs[start : start + pairs_per_batch]:
            batch.extend(build_pair(renderer, pair.prompt, pair.chosen, pair.rejected))
        batches.append(batch)
    return batches
