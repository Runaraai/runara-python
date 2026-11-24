"""DPO on preference pairs: make the chosen reply likelier than the rejected one.

Against the base model the session started from, which is the reference by
default and costs nothing extra. Data, one pair per line (see
runara_cookbook.preference):

    {"prompt": [{"role": "user", "content": "..."}], "chosen": "...", "rejected": "..."}

Watch `accuracy` (the share of pairs ranked the right way round) and `margin`
rise; if accuracy stays near 0.5, raise the learning rate or beta.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import runara

from ..preference import PreferencePair, pair_batches, read_pairs
from ..renderers.base import Renderer
from . import _cli


def train(
    tc: runara.TrainingClient,
    renderer: Renderer,
    pairs: Sequence[PreferencePair],
    *,
    epochs: int = 1,
    pairs_per_batch: int = 4,
    beta: float = 0.1,
    adam: runara.AdamParams = runara.AdamParams(learning_rate=1e-5),
    log: Callable[[dict], None] = _cli.log,
) -> list[dict]:
    if not pairs:
        raise ValueError("no pairs to train on")
    batches = pair_batches(renderer, pairs, pairs_per_batch)
    history = []
    for epoch in range(epochs):
        for batch in batches:
            fb = tc.forward_backward(batch, "dpo", {"beta": beta})
            tc.optim_step(adam)
            metrics = fb.result().metrics
            record = {
                "epoch": epoch,
                "loss": float(metrics.get("dpo_loss", 0.0)),
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "margin": float(metrics.get("margin", 0.0)),
            }
            log(record)
            history.append(record)
    return history


def main() -> None:
    p = _cli.parser(__doc__.splitlines()[0])
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--pairs-per-batch", type=int, default=4)
    p.add_argument("--beta", type=float, default=0.1)
    p.set_defaults(learning_rate=1e-5)
    args = p.parse_args()
    renderer = _cli.renderer_for(args)
    with _cli.training_client(args) as tc:
        train(tc, renderer, read_pairs(args.data), epochs=args.epochs, pairs_per_batch=args.pairs_per_batch,
              beta=args.beta, adam=runara.AdamParams(learning_rate=args.learning_rate))
        if args.save_as:
            print(tc.save_state(name=args.save_as).result())


if __name__ == "__main__":
    main()
