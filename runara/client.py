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
import urllib.parse
import uuid
from dataclasses import dataclass

from . import _http
from ._futures import Future, Submitter
from ._http import RunaraError
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

# The app, not the marketing site: runara.ai is a Framer page and 404s here.
DEFAULT_BASE_URL = "https://app.runara.ai/api/v1"
#: Refresh this far before expiry, so a long call cannot start on a token that
#: dies mid-flight.
TOKEN_REFRESH_MARGIN_SECONDS = 120
#: How often to ask again while a GPU is being started for a new session.
POLL_SECONDS = 10


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
        """Base models open for training, with what they cost.

        Billed per GPU-hour on the GPU a session lands on (gpu_options,
        cheapest first) and per GB-month for checkpoints you keep. A fixed
        catalog rather than any HuggingFace id: the weights have to be
        resident on a worker before a session can train against them.
        """
        body = self._control("GET", "/finetune/models")
        return [
            ModelInfo(
                base_model=row["base_model"],
                repository=row["repository"],
                max_rank=row["max_rank"],
                max_seq_len=row["max_seq_len"],
                default_target_modules=row["default_target_modules"],
                train_usd_per_million=row.get("train_usd_per_million", 0.0),
                sample_usd_per_million=row.get("sample_usd_per_million", 0.0),
                gpu_options=[
                    GpuOption(gpu=g["gpu"], gpu_count=g["gpu_count"], usd_per_hour=g["usd_per_hour"])
                    for g in row.get("gpu_options", [])
                ],
                checkpoint_storage_usd_per_gb_month=row.get("checkpoint_storage_usd_per_gb_month", {}),
            )
            for row in body["data"]
        ]

    def list_checkpoints(self, base_model: str | None = None) -> list[Checkpoint]:
        """Your saved checkpoints, newest first, each with the `uri` load_state takes."""
        # include=uri asks for the storage location. Opt-in, because 0.1.x
        # clients built Checkpoint(**row) and would break on a field they did
        # not know; this client ignores unknown fields.
        query = {"include": "uri"}
        if base_model:
            query["base_model"] = base_model
        body = self._control("GET", "/finetune/checkpoints?" + urllib.parse.urlencode(query))
        return [Checkpoint.from_wire(row) for row in body["data"]]

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
        wait_timeout: float = 25 * 60,
    ) -> "TrainingClient":
        """Open a session and get a client for it.

        Blocks until the session has a GPU. Nothing is kept warm, so the first
        session on a model usually waits a few minutes while one boots;
        `wait_timeout` (seconds) bounds that wait.

        `request_id` makes this idempotent. One is generated if you do not
        supply it, so a retry after a timeout returns the session you already
        have rather than opening a second one you would also be billed for.
        The wait relies on exactly that: it replays the same request.
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

        deadline = time.monotonic() + wait_timeout
        announced = False
        while True:
            body = self._control("POST", "/finetune/sessions", payload)
            if body.get("worker_url"):
                break
            if time.monotonic() >= deadline:
                raise RunaraError(
                    f"No GPU was ready for session {body.get('session_id')} after "
                    f"{wait_timeout:.0f}s. Retry with request_id={payload['request_id']!r} "
                    "to keep waiting for the same session."
                )
            if not announced:
                print(
                    f"runara: starting a GPU for session {body.get('session_id')}; "
                    "this usually takes a few minutes...",
                    flush=True,
                )
                announced = True
            time.sleep(POLL_SECONDS)

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
        #: The worker's capabilities, fetched on first use of a newer feature.
        self._capabilities: dict | None = None
        #: Set when a forward_backward fails, so the optim_step queued behind
        #: it is refused rather than applying a partial or stale gradient.
        self._failed_forward_backward: BaseException | None = None
        #: Metrics from the most recent optim_step, e.g. {"grad_norm": 0.41}.
        self.last_optim_metrics: dict = {}

    # -- the primitives ----------------------------------------------------

    def forward_backward(
        self,
        data: list[Datum],
        loss_fn: str = "cross_entropy",
        loss_fn_config: dict | None = None,
    ) -> Future[ForwardBackwardResult]:
        """Accumulate gradients for one batch under `loss_fn`. Returns immediately.

        Losses: "cross_entropy" (supervised; weights), "importance_sampling",
        "ppo", "cispo", "dro" (RL; logprobs and advantages) and "dpo"
        (preference pairs, chosen then rejected, alternating; weights).
        `loss_fn_config` sets each loss's knobs, e.g. ppo's
        clip_low_threshold / clip_high_threshold, dro's and dpo's beta, or
        cross_entropy's reduction ("mean", the default, or "sum").

        Submit this and `optim_step` back to back without awaiting: the worker
        applies calls for a session in the order they arrive, so the network
        round trip overlaps your next batch of tokenization instead of
        stalling the GPU.
        """
        if loss_fn != "cross_entropy" or loss_fn_config:
            self._require(f"loss_fn={loss_fn!r}", lambda caps: loss_fn in caps.get("losses", ()))
        payload: dict = {"data": [datum.to_wire() for datum in data], "loss_fn": loss_fn}
        if loss_fn_config:
            payload["loss_fn_config"] = dict(loss_fn_config)

        def run() -> ForwardBackwardResult:
            try:
                body = self._call("forward_backward", payload)
            except BaseException as error:
                self._failed_forward_backward = error
                raise
            return ForwardBackwardResult(
                loss=body["loss"],
                metrics=body.get("metrics", {}),
                seq_count=body.get("seq_count", len(data)),
                token_count=body.get("token_count", 0),
                logprobs=[o.get("logprobs", []) for o in body.get("loss_fn_outputs", [])],
            )

        return self._submitter.submit(run)

    def forward(self, data: list[Datum], *, reference: bool = False) -> Future[ForwardResult]:
        """The logprob of each target token, without training.

        `reference=True` scores against the base model the session started
        from (the adapter disabled): a free reference policy for KL penalties
        and DPO, with no second session.
        """
        self._require("forward()", lambda caps: caps.get("forward") is True)
        payload = {"data": [datum.to_wire() for datum in data], "reference": reference}

        def run() -> ForwardResult:
            body = self._call("forward", payload)
            return ForwardResult(
                logprobs=[o["logprobs"] for o in body["loss_fn_outputs"]],
                seq_count=body.get("seq_count", len(data)),
            )

        return self._submitter.submit(run)

    def optim_step(self, params: AdamParams) -> Future[int]:
        """Apply the accumulated gradients. Returns the new step number.

        Refused, with the reason, if the forward_backward queued before it
        failed: stepping then would apply a stale or partial gradient. The
        failure is cleared once reported, so the next batch proceeds.
        The step's metrics (grad_norm) land in `last_optim_metrics`.
        """
        if params.grad_clip_norm:
            self._require("grad_clip_norm", lambda caps: caps.get("optim", {}).get("grad_clip_norm") is True)

        def run() -> int:
            failed, self._failed_forward_backward = self._failed_forward_backward, None
            if failed is not None:
                raise RunaraError(
                    "optim_step skipped: the forward_backward before it failed "
                    f"({failed}). No update was applied."
                )
            body = self._call("optim_step", {"adam": params.to_wire()})
            self.last_optim_metrics = body.get("metrics", {})
            return body["step"]

        return self._submitter.submit(run)

    def sample(
        self,
        prompts: list[list[int]],
        max_tokens: int = 64,
        temperature: float = 1.0,
        *,
        num_samples: int = 1,
        sampling_params: SamplingParams | None = None,
    ) -> Future[SampleResult]:
        """Generate from the adapter as it stands, without ending the session.

        `num_samples` continuations of each prompt, each with the learner's
        logprob of every token and a stop reason: the raw material of an RL
        step. `sampling_params` adds top_p, top_k, stop and seed; when given,
        it replaces `max_tokens` and `temperature`.
        """
        params = sampling_params or SamplingParams(max_tokens=max_tokens, temperature=temperature)
        legacy = num_samples == 1 and sampling_params is None
        if not legacy:
            self._require("num_samples / sampling_params", lambda caps: "sample" in caps)
            payload = {"prompts": prompts, "num_samples": num_samples, "sampling_params": params.to_wire()}
        else:
            # The shape every worker understands; newer ones answer with
            # samples, older ones with sequences only.
            payload = {"prompts": prompts, "max_tokens": max_tokens, "temperature": temperature}

        def run() -> SampleResult:
            body = self._call("sample", payload)
            if "samples" in body:
                samples = [
                    [SampledSequence(s["tokens"], s.get("logprobs", []), s.get("stop_reason", "unknown")) for s in group]
                    for group in body["samples"]
                ]
            else:
                samples = [[SampledSequence(tokens, [], "unknown")] for tokens in body["sequences"]]
            return SampleResult(
                sequences=body.get("sequences", [group[0].tokens for group in samples]),
                token_count=body.get("token_count", 0),
                samples=samples,
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
            # Drain first: a save_state still queued when the session closes
            # would be refused, and the adapter it was meant to keep is lost.
            self._submitter.close()
            self.service._control("DELETE", f"/finetune/sessions/{self.session_id}")
        finally:
            # The pinned certificate lives in a temp file; remove it.
            self._pinned.close()

    def __enter__(self) -> "TrainingClient":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # -- plumbing ----------------------------------------------------------

    def capabilities(self) -> dict:
        """What this session's worker can do. Fetched once, then cached.

        A worker started before RL support answers 404, which reads as the
        original protocol: supervised training and plain sampling only.
        """
        if self._capabilities is None:
            try:
                self._capabilities = _http.request(
                    "GET",
                    f"{self.worker_url}/v1/capabilities",
                    self._fresh_token(),
                    retries=1,
                    timeout=30.0,
                    ssl_context=self._pinned.context,
                )
            except RunaraError as error:
                if error.status != 404:
                    raise
                self._capabilities = {"protocol": 1, "losses": ["cross_entropy"]}
        return self._capabilities

    def _require(self, feature: str, supported) -> None:
        """Refuse a feature this session's worker lacks, before sending anything.

        Otherwise an older worker answers with an opaque 400, or worse, an
        older one that ignored loss_fn would quietly train cross-entropy.
        """
        if not supported(self.capabilities()):
            raise RunaraError(
                f"This session's worker does not support {feature}: it started before "
                "that feature existed. Close this session and start a new one; new "
                "sessions run on current workers."
            )

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
