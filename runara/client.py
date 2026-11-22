"""The client. Four primitives against base weights you never have to host.

    import runara

    sc = runara.ServiceClient(api_key="rk_live_...")
    tc = sc.create_lora_training_client(base_model="qwen3.5-4b", rank=32)

    for batch in batches:
        fb = tc.forward_backward(batch)          # submitted, not awaited
        tc.optim_step(runara.AdamParams(1e-4))   # submitted straight after
        print(fb.result().loss)                  # the loop blocks here

    tc.save_state(name="support-v1")

Your data never leaves your machine as text. The cookbook turns whatever you
have into `Datum`s locally, and only token ids and loss weights are sent.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass

from . import _http
from ._futures import Future, Submitter
from .types import (
    AdamParams,
    Checkpoint,
    Datum,
    ForwardBackwardResult,
    ModelInfo,
    SampleResult,
)

DEFAULT_BASE_URL = "https://runara.ai/api/v1"
#: Refresh this far before expiry, so a long call cannot start on a token that
#: dies mid-flight.
TOKEN_REFRESH_MARGIN_SECONDS = 120
#: Nothing is kept warm, so a new session usually waits for a GPU to be rented
#: for it. Ask the control plane how that wait is going at this interval.
SESSION_POLL_INTERVAL_SECONDS = 5.0
#: Stop waiting for a worker after this long. The control plane fails a session
#: that never got capacity after fifteen minutes, so this only bounds a control
#: plane that stops answering.
SESSION_START_TIMEOUT_SECONDS = 20 * 60


class ServiceClient:
    """Your account. Opens training sessions and reads what you have saved."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        key = api_key or os.environ.get("RUNARA_API_KEY")
        if not key:
            raise ValueError(
                "No API key. Pass api_key=... or set RUNARA_API_KEY. "
                "Create one in the Runara console under Settings."
            )
        self.api_key = key
        self.base_url = (base_url or os.environ.get("RUNARA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")

    # -- catalog -----------------------------------------------------------

    def list_models(self) -> list[ModelInfo]:
        """Base models open for training, with their per-token prices.

        A fixed catalog rather than any HuggingFace id: the weights have to be
        already resident on a worker for sessions to share them, which is what
        makes per-token pricing possible at all.
        """
        body = self._control("GET", "/finetune/models")
        return [
            ModelInfo(
                base_model=row["base_model"],
                repository=row["repository"],
                max_rank=row["max_rank"],
                max_seq_len=row["max_seq_len"],
                default_target_modules=row["default_target_modules"],
                train_usd_per_million=row["train_usd_per_million"],
                sample_usd_per_million=row["sample_usd_per_million"],
            )
            for row in body["data"]
        ]

    def list_checkpoints(self, base_model: str | None = None) -> list[Checkpoint]:
        path = "/finetune/checkpoints"
        if base_model:
            path += f"?base_model={base_model}"
        body = self._control("GET", path)
        return [Checkpoint(**row) for row in body["data"]]

    # -- sessions ----------------------------------------------------------

    def create_lora_training_client(
        self,
        base_model: str,
        rank: int = 32,
        *,
        alpha: int | None = None,
        target_modules: list[str] | None = None,
        method: str = "lora",
        request_id: str | None = None,
    ) -> "TrainingClient":
        """Open a session and get a client for it.

        `request_id` makes this idempotent. One is generated if you do not
        supply it, so a retry after a timeout returns the session you already
        have rather than opening a second one you would also be billed for.

        Nothing is kept warm, so a session usually exists before its worker
        does. This waits for the worker and returns a client that can talk to
        it, and raises once the session fails or the wait runs out.
        """
        payload = {
            "request_id": request_id or f"sdk-{uuid.uuid4().hex}",
            "base_model": base_model,
            "rank": rank,
            "method": method,
        }
        if alpha is not None:
            payload["alpha"] = alpha
        if target_modules is not None:
            payload["target_modules"] = target_modules

        body = self._control("POST", "/finetune/sessions", payload)
        if body.get("status") == "pending_capacity":
            body = self._wait_for_worker(body["session_id"])
        return TrainingClient(
            service=self,
            session_id=body["session_id"],
            worker_url=body["worker_url"].rstrip("/"),
            token=body["session_token"],
            token_expires_at=body["expires_at"],
            # The worker is rented for minutes at an address nobody owns, so
            # there is no public certificate for it. The control plane, which
            # we authenticate normally, tells us exactly which certificate it
            # will present; we trust that one and refuse anything else.
            worker_cert_pem=body["worker_cert_pem"],
        )

    # -- plumbing ----------------------------------------------------------

    def _wait_for_worker(self, session_id: str) -> dict:
        """Wait for the session to be placed on a worker, then get a token.

        The create call answers `pending_capacity` when no GPU was warm for it,
        with no address and no token. Polling the session is what the control
        plane asks for; a session it cannot place is failed by its watchdog,
        which this reports instead of waiting on a session that will never run.
        """
        deadline = time.monotonic() + SESSION_START_TIMEOUT_SECONDS
        while True:
            status = self._control("GET", f"/finetune/sessions/{session_id}")["status"]
            if status == "pending_capacity":
                if time.monotonic() >= deadline:
                    raise _http.RunaraError(
                        f"No worker was started for session {session_id} within "
                        f"{SESSION_START_TIMEOUT_SECONDS // 60} minutes."
                    )
                time.sleep(SESSION_POLL_INTERVAL_SECONDS)
                continue
            if status in ("closed", "closing"):
                raise _http.SessionClosed(
                    f"Session {session_id} ended before a worker was ready for it."
                )
            if status == "failed":
                raise _http.RunaraError(
                    f"Session {session_id} failed before a worker was ready for it."
                )
            issued = self._control("POST", f"/finetune/sessions/{session_id}/token")
            return {"session_id": session_id, **issued}

    def _control(self, method: str, path: str, body: dict | None = None):
        # The control plane is retried: these calls are rare, and the one that
        # mutates anything is idempotent on request_id.
        return _http.request(
            method, self.base_url + path, self.api_key, body, timeout=60.0, retries=2
        )


@dataclass
class _SessionToken:
    value: str
    expires_at: int

    def stale(self, now_ms: float) -> bool:
        return self.expires_at - now_ms < TOKEN_REFRESH_MARGIN_SECONDS * 1000


class TrainingClient:
    """One training session. The four primitives, and the loop is yours."""

    def __init__(
        self,
        service: ServiceClient,
        session_id: str,
        worker_url: str,
        token: str,
        token_expires_at: int,
        worker_cert_pem: str,
    ) -> None:
        self.service = service
        self.session_id = session_id
        if not worker_url.startswith("https://"):
            # Never negotiable. This connection carries the customer's
            # training tokens, and a plaintext worker_url would hand them to
            # anyone on the path while everything still appeared to work.
            raise ValueError(f"worker_url must be https, got {worker_url!r}")
        self.worker_url = worker_url
        self._token = _SessionToken(token, token_expires_at)
        self._pinned = _http._PinnedWorker(worker_cert_pem)
        self._submitter = Submitter()

    # -- the primitives ----------------------------------------------------

    def forward_backward(
        self, data: list[Datum], loss_fn: str = "cross_entropy"
    ) -> Future[ForwardBackwardResult]:
        """Accumulate gradients for one batch. Returns immediately.

        Submit this and `optim_step` back to back without awaiting: the worker
        applies calls for a session in the order they arrive, so the network
        round trip overlaps your next batch of tokenization instead of
        stalling the GPU.
        """
        payload = {"data": [datum.to_wire() for datum in data], "loss_fn": loss_fn}

        def run() -> ForwardBackwardResult:
            body = self._call("forward_backward", payload)
            return ForwardBackwardResult(
                loss=body["loss"],
                metrics=body.get("metrics", {}),
                seq_count=body.get("seq_count", len(data)),
                token_count=body.get("token_count", 0),
            )

        return self._submitter.submit(run)

    def optim_step(self, params: AdamParams) -> Future[int]:
        """Apply the accumulated gradients. Returns the new step number."""

        def run() -> int:
            return self._call("optim_step", {"adam": params.to_wire()})["step"]

        return self._submitter.submit(run)

    def sample(
        self,
        prompts: list[list[int]],
        max_tokens: int = 64,
        temperature: float = 1.0,
    ) -> Future[SampleResult]:
        """Generate from the adapter as it stands, without ending the session."""

        def run() -> SampleResult:
            body = self._call(
                "sample",
                {"prompts": prompts, "max_tokens": max_tokens, "temperature": temperature},
            )
            return SampleResult(
                sequences=body["sequences"], token_count=body.get("token_count", 0)
            )

        return self._submitter.submit(run)

    def save_state(self, name: str) -> Future[dict]:
        """Save the adapter. Blocks on everything you submitted before it.

        Ordering matters here more than anywhere: saving before your last
        optim_step has been applied would write a checkpoint one step behind
        what you think you trained. The worker's per-session ordering gives
        that for free, which is why this goes through the same queue.
        """

        def run() -> dict:
            return self._call("save_state", {"name": name})

        return self._submitter.submit(run)

    def load_state(self, uri: str) -> Future[dict]:
        """Resume from a checkpoint saved earlier. Blocks on prior work.

        The counterpart of save_state, and what makes a lost worker
        recoverable: a session is placed on whichever GPU had capacity, so the
        adapter in VRAM goes when that machine does. The weights in object
        storage do not.

        `uri` is the storage location from a checkpoint — list_checkpoints()
        returns it. It must belong to your organization; the worker compares
        it against the session token the control plane signed, and answers
        "not found" for anything else rather than confirming it exists.

        Through the same queue as everything else: loading while a
        forward_backward is still in flight would train one batch against the
        old weights and the next against the new, and the loss curve would
        look merely noisy.
        """

        def run() -> dict:
            return self._call("load_state", {"uri": uri})

        return self._submitter.submit(run)

    # -- session -----------------------------------------------------------

    def status(self) -> dict:
        return self.service._control("GET", f"/finetune/sessions/{self.session_id}")

    def close(self) -> None:
        """End the session and release its slot on the worker.

        Worth doing explicitly: an abandoned session holds an adapter slot
        until the watchdog reclaims it half an hour later.
        """
        try:
            self.service._control("DELETE", f"/finetune/sessions/{self.session_id}")
        finally:
            self._submitter.close()
            # The pinned certificate lives in a temp file; remove it.
            self._pinned.close()

    def __enter__(self) -> "TrainingClient":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # -- plumbing ----------------------------------------------------------

    def _call(self, action: str, payload: dict) -> dict:
        token = self._fresh_token()
        # retries=0: a forward_backward that timed out may already have applied
        # its gradients, and sending it again would train on the batch twice.
        return _http.request(
            "POST",
            f"{self.worker_url}/v1/sessions/{self.session_id}/{action}",
            token,
            payload,
            retries=0,
            ssl_context=self._pinned.context,
        )

    def _fresh_token(self) -> str:
        if not self._token.stale(time.time() * 1000):
            return self._token.value
        body = self.service._control(
            "POST", f"/finetune/sessions/{self.session_id}/token"
        )
        self._token = _SessionToken(body["session_token"], body["expires_at"])
        # The worker can move between refreshes in principle; follow it.
        self.worker_url = body.get("worker_url", self.worker_url).rstrip("/")
        return self._token.value
