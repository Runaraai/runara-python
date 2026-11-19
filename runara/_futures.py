"""Futures that let a training loop stay ahead of the network.

A step is two calls — ``forward_backward`` then ``optim_step`` — and a loop
that awaits each one spends most of its wall-clock time waiting for a round
trip while the GPU sits idle. The worker already guarantees that calls for one
session are applied in the order they arrive, so the client can submit both and
collect later.

That ordering guarantee is what makes this safe, and it is the worker's
per-session lock that provides it. Without it, two in-flight calls would race
on the same adapter and the optimizer could apply a step before the gradients
it was meant to apply.
"""

from __future__ import annotations

from concurrent.futures import Future as _Future
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Future(Generic[T]):
    """A submitted request. Call ``result()`` when you actually need the value.

    Deliberately thin: it is one outstanding HTTP request, not a task in a
    scheduler. There is no polling, no result store and no reconnection
    protocol, because there is nothing to reconnect to — the request either
    returns or raises.
    """

    def __init__(self, inner: _Future) -> None:
        self._inner = inner

    def result(self, timeout: float | None = None) -> T:
        """Block until the worker answers, re-raising whatever it raised."""
        return self._inner.result(timeout)

    def done(self) -> bool:
        return self._inner.done()

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        state = "done" if self.done() else "pending"
        return f"<runara.Future {state}>"


class Submitter:
    """Runs requests for one session, one at a time, in submission order.

    A single worker thread rather than a pool: the ordering the protocol
    promises is per session, and a pool would let two requests for the same
    session overlap. Pipelining comes from the client not blocking, not from
    the requests running concurrently.
    """

    def __init__(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="runara")

    def submit(self, fn: Callable[..., T], *args, **kwargs) -> Future[T]:
        return Future(self._pool.submit(fn, *args, **kwargs))

    def close(self) -> None:
        self._pool.shutdown(wait=True)
