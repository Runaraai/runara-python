"""Reading your data, on your machine.

Nothing here talks to Runara. These helpers open files you already have and
hand the rows to a renderer; what leaves your machine is whatever the renderer
produced, which is token ids and floats.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator, Sequence

from runara.types import Datum

from .renderers.base import Message, Renderer, TrainOnWhat


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file from your disk. One JSON object per line."""
    rows: list[dict] = []
    with Path(path).open() as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                # Name the line rather than the content: an error message is a
                # place training data leaks into logs by accident.
                raise ValueError(f"{path} line {number} is not valid JSON") from error
    if not rows:
        raise ValueError(f"{path} has no rows")
    return rows


def conversations(rows: Sequence[dict], key: str = "messages") -> list[list[Message]]:
    """Pull the chat turns out of raw rows, checking the shape once up front."""
    out: list[list[Message]] = []
    for index, row in enumerate(rows):
        messages = row.get(key)
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"row {index} has no {key!r} list")
        for message in messages:
            if not isinstance(message, dict) or "role" not in message or "content" not in message:
                raise ValueError(
                    f"row {index}: every message needs a 'role' and 'content'"
                )
        out.append(messages)
    return out


def build_data(
    renderer: Renderer,
    conversation_list: Sequence[Sequence[Message]],
    train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    max_tokens: int | None = None,
) -> list[Datum]:
    """Render every conversation into a Datum, dropping ones that are too long.

    Dropped rather than truncated: a truncated conversation ends mid-sentence,
    and training the model to stop there teaches it to do the same.
    """
    data: list[Datum] = []
    for messages in conversation_list:
        rendered = renderer.build_supervised_example(messages, train_on_what)
        if max_tokens is not None and len(rendered.tokens) > max_tokens:
            continue
        data.append(renderer.to_datum(messages, train_on_what))
    return data


def batches(items: Sequence[Datum], size: int, shuffle: bool = False, seed: int = 0) -> Iterator[list[Datum]]:
    """Yield fixed-size batches. The last one may be smaller."""
    if size < 1:
        raise ValueError("batch size must be at least 1")
    order = list(range(len(items)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for start in range(0, len(order), size):
        yield [items[index] for index in order[start : start + size]]


def token_count(data: Sequence[Datum]) -> int:
    """Trained tokens across a dataset — what one epoch will cost."""
    return sum(datum.trained_tokens for datum in data)
