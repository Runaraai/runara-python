"""HTTP, with the retry policy spelled out.

Two different endpoints with different failure characters:

* The **control plane** creates sessions and lists checkpoints. Slow, rare,
  and safe to retry only because session creation is idempotent on a request
  id the client generates.
* The **worker** runs the training calls. Fast, frequent, and **not** safe to
  retry: a ``forward_backward`` that timed out may well have applied its
  gradients, and sending it again would train on the same batch twice. So the
  worker path retries only on errors that prove the request never arrived.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

#: Only these are retried on the worker path: the request did not land.
_CONNECTION_ERRORS = (urllib.error.URLError, TimeoutError, ConnectionError)


class RunaraError(Exception):
    """An error the service reported, with its code where there is one."""

    def __init__(self, message: str, *, status: int | None = None, code: str | None = None):
        super().__init__(message)
        self.status = status
        self.code = code


class InsufficientBalance(RunaraError):
    """The organization is out of credit. Training stops until it is topped up."""


class SessionClosed(RunaraError):
    """The session has ended, or is ending because the balance ran out."""


def _raise_for(status: int, body: Any, url: str) -> None:
    error = body.get("error") if isinstance(body, dict) else None
    code = error.get("code") if isinstance(error, dict) else (error if isinstance(error, str) else None)
    message = error.get("message") if isinstance(error, dict) else str(error or body)

    if status == 402 or code == "INSUFFICIENT_BALANCE":
        raise InsufficientBalance(message, status=status, code=code)
    if code in {"session_closed", "session_closing", "session_not_resident"}:
        raise SessionClosed(message, status=status, code=code)
    raise RunaraError(f"{message} ({url.rsplit('/', 1)[-1]})", status=status, code=code)


def request(
    method: str,
    url: str,
    token: str,
    body: dict | None = None,
    *,
    timeout: float = 600.0,
    retries: int = 0,
) -> Any:
    """One request, retried only when it provably did not arrive.

    `retries` defaults to zero. The caller decides, because only the caller
    knows whether repeating the call is safe — and for a training step it is
    not: a timed-out forward_backward may have already applied its gradients.
    """
    payload = json.dumps(body).encode() if body is not None else None
    attempt = 0
    while True:
        http_request = urllib.request.Request(
            url,
            data=payload,
            method=method,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "User-Agent": "runara-python",
            },
        )
        try:
            with urllib.request.urlopen(http_request, timeout=timeout) as response:
                raw = response.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as error:
            try:
                parsed = json.loads(error.read())
            except Exception:  # noqa: BLE001 - a non-JSON error body is still an error
                parsed = {"error": {"message": error.reason}}
            # A 503 means the worker is still loading its weights; that is a
            # wait, not a failure, and the request never reached the model.
            if error.code == 503 and attempt < retries:
                attempt += 1
                time.sleep(min(2**attempt, 10))
                continue
            _raise_for(error.code, parsed, url)
        except _CONNECTION_ERRORS as error:
            if attempt < retries:
                attempt += 1
                time.sleep(min(2**attempt, 10))
                continue
            raise RunaraError(f"Could not reach {url.rsplit('/', 2)[0]}: {error}") from error
