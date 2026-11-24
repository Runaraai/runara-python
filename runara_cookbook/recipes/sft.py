"""Supervised fine-tuning on conversations: the Get started loop, as a recipe.

Data, one conversation per line; the last assistant reply is what is learned:

    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import runara

from .. import data as cookbook_data
from ..renderers.base import Message, Renderer
from . import _cli


def train(
    tc: runara.TrainingClient,
    renderer: Renderer,
    conversations: Sequence[Sequence[Message]],
    *,
    epochs: int = 3,
    batch_size: int = 8,
    adam: runara.AdamParams = runara.AdamParams(learning_rate=1e-4),
    max_tokens: int | None = 4096,
    log: Callable[[dict], None] = _cli.log,
) -> list[dict]:
    dataset = cookbook_data.build_data(renderer, conversations, max_tokens=max_tokens)
    if not dataset:
        raise ValueError("no conversation fits in max_tokens")
    history = []
    for epoch in range(epochs):
        for batch in cookbook_data.batches(dataset, batch_size, shuffle=True, seed=epoch):
            # Submitted together; the loop blocks only on the result.
            fb = tc.forward_backward(batch)
            tc.optim_step(adam)
            record = {"epoch": epoch, "loss": fb.result().loss, "tokens": cookbook_data.token_count(batch)}
            log(record)
            history.append(record)
    return history


def main() -> None:
    p = _cli.parser(__doc__.splitlines()[0])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    renderer = _cli.renderer_for(args)
    conversations = cookbook_data.conversations(cookbook_data.read_jsonl(args.data))
    with _cli.training_client(args) as tc:
        train(tc, renderer, conversations, epochs=args.epochs, batch_size=args.batch_size,
              adam=runara.AdamParams(learning_rate=args.learning_rate))
        if args.save_as:
            print(tc.save_state(name=args.save_as).result())


if __name__ == "__main__":
    main()
