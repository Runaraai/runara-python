"""The client against a real HTTP server, not a mock.

The decisions worth testing are about what happens when things go wrong: a
timed-out training call that may already have applied its gradients, a token
that expires mid-run, a balance that runs out. All of those are invisible if
the transport is stubbed.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import ssl
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import runara
from runara import AdamParams, Datum, InsufficientBalance, SessionClosed


class Recorder:
    """A stand-in for both the control plane and a worker."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []
        self.token_expiry_ms = int((time.time() + 3600) * 1000)
        self.fail_next: tuple[int, dict] | None = None
        self.step = 0
        #: How many session-create calls answer "GPU still starting" first.
        self.pending_creates = 0
        #: What GET /v1/capabilities answers. None plays a worker started
        #: before capabilities existed, which answers 404.
        self.capabilities: dict | None = {
            "protocol": 2,
            "losses": ["cross_entropy", "importance_sampling", "ppo", "cispo", "dro", "dpo"],
            "forward": True,
            "sample": {"num_samples": 64, "logprobs": True},
            "optim": {"grad_clip_norm": True},
        }

    def handle(self, method: str, path: str, body: dict) -> tuple[int, dict]:
        self.calls.append((method, path, body))
        if self.fail_next is not None:
            status, payload = self.fail_next
            self.fail_next = None
            return status, payload

        if path == "/api/v1/finetune/sessions" and method == "POST" and self.pending_creates:
            self.pending_creates -= 1
            # The real server's shape while a worker boots: same session, no worker.
            return 200, {
                "session_id": "sess_1", "status": "pending_capacity",
                "worker_url": None, "session_token": None,
                "expires_at": None, "worker_cert_pem": None,
            }
        if path == "/api/v1/finetune/sessions" and method == "POST":
            return 200, {
                "session_id": "sess_1",
                "worker_url": self.worker_base,
                "worker_cert_pem": self.worker_cert,
                "session_token": "tok_1",
                "expires_at": self.token_expiry_ms,
            }
        if path.endswith("/token"):
            return 200, {
                "session_token": "tok_refreshed",
                "worker_url": self.worker_base,
                "expires_at": int((time.time() + 3600) * 1000),
            }
        if path == "/v1/capabilities" and method == "GET":
            if self.capabilities is None:
                return 404, {"error": "not_found"}
            return 200, self.capabilities
        if path.endswith("/forward_backward"):
            return 200, {
                "loss": 2.5, "metrics": {"loss": 2.5, "loss:sum": 2.5}, "seq_count": 1, "token_count": 7,
                "loss_fn_outputs": [{"logprobs": [-0.5] * len(d["target_tokens"])} for d in body.get("data", [])],
            }
        if path.endswith("/forward"):
            return 200, {
                "loss_fn_outputs": [{"logprobs": [-0.25] * len(d["target_tokens"])} for d in body.get("data", [])],
                "seq_count": len(body.get("data", [])),
            }
        if path.endswith("/optim_step"):
            self.step += 1
            return 200, {"ok": True, "step": self.step, "metrics": {"grad_norm": 0.5}}
        if path.endswith("/sample") and "sampling_params" in body:
            n = body.get("num_samples", 1)
            return 200, {
                "samples": [
                    [{"tokens": [5, 6], "logprobs": [-0.1, -0.2], "stop_reason": "stop"}] * n
                    for _ in body["prompts"]
                ],
                "sequences": [[5, 6] for _ in body["prompts"]],
                "token_count": 2 * n * len(body["prompts"]),
            }
        if path.endswith("/sample"):
            return 200, {"sequences": [[1, 2, 3]], "token_count": 3}
        if path.startswith("/api/v1/finetune/checkpoints"):
            return 200, {"data": [{
                "checkpoint_id": "ck_1", "name": "v1", "kind": "sampler_weights", "step": 3,
                "base_model": "qwen3.5-4b", "base_model_repository": "Qwen/Qwen3.5-4B",
                "rank": 32, "size_bytes": 1024, "files": [], "created_at": 1, "expires_at": None,
                "uri": "s3://bucket/finetune/org/sess/sampler_weights/3",
                # A field from a newer server this client has never heard of.
                "storage_usd_per_month": 0.01,
            }]}
        if path.endswith("/load_state"):
            return 200, {"ok": True, "step": self.step}
        if path.endswith("/save_state"):
            return 200, {"checkpointKey": "finetune/o/s/sampler_weights/1"}
        if path == "/api/v1/finetune/models":
            return 200, {"data": [{
                "base_model": "qwen3.5-4b", "repository": "Qwen/Qwen3.5-4B",
                "max_rank": 64, "max_seq_len": 4096, "default_target_modules": ["q_proj"],
                "train_usd_per_million": 0.74, "sample_usd_per_million": 1.0,
            }]}
        if method == "DELETE":
            return 200, {"status": "closed"}
        return 404, {"error": {"code": "NOT_FOUND", "message": "no route"}}


@pytest.fixture
def service():
    recorder = Recorder()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            return

        def _run(self, method: str):
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            body = json.loads(raw) if raw else {}
            status, payload = recorder.handle(method, self.path, body)
            encoded = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            self._run("GET")

        def do_POST(self):
            self._run("POST")

        def do_DELETE(self):
            self._run("DELETE")

    # Two servers, because production has two very different connections.
    #
    # The control plane is an ordinary public service. The worker is a box
    # rented for minutes at an address nobody owns, serving a certificate no
    # public CA will ever vouch for — so the client is handed that exact
    # certificate and must trust only it. Serving real TLS here means the
    # tests exercise that pinning rather than describing it: if the client
    # ever regressed to skipping verification, or to plain http, these fail.
    control = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    control_base = f"http://127.0.0.1:{control.server_address[1]}"
    threading.Thread(target=control.serve_forever, daemon=True).start()

    certdir = tempfile.mkdtemp(prefix="runara-test-tls-")
    certfile = os.path.join(certdir, "worker.crt")
    keyfile = os.path.join(certdir, "worker.key")
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
            "-keyout", keyfile, "-out", certfile,
            "-subj", "/CN=127.0.0.1", "-addext", "subjectAltName=IP:127.0.0.1",
        ],
        check=True, capture_output=True,
    )

    worker = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(certfile=certfile, keyfile=keyfile)
    worker.socket = ssl_context.wrap_socket(worker.socket, server_side=True)
    threading.Thread(target=worker.serve_forever, daemon=True).start()

    recorder.base = control_base
    recorder.worker_base = f"https://127.0.0.1:{worker.server_address[1]}"
    recorder.worker_cert = pathlib.Path(certfile).read_text()

    client = runara.ServiceClient(api_key="rk_test", base_url=control_base + "/api/v1")
    try:
        yield client, recorder
    finally:
        control.shutdown()
        control.server_close()
        worker.shutdown()
        worker.server_close()
        shutil.rmtree(certdir, ignore_errors=True)


def batch() -> list[Datum]:
    return [Datum(token_ids=[1, 2, 3], target_tokens=[2, 3, 4], weights=[0.0, 1.0, 1.0])]


class TestSessionLifecycle:
    def test_opens_a_session_and_gets_a_worker_to_talk_to(self, service):
        client, recorder = service
        tc = client.create_lora_training_client("qwen3.5-4b", rank=16)

        assert tc.session_id == "sess_1"
        _, _, body = recorder.calls[0]
        assert body["base_model"] == "qwen3.5-4b"
        assert body["rank"] == 16
        # Generated for the caller: a retry after a timeout must return the
        # session they already have, not open a second billable one.
        assert body["request_id"].startswith("sdk-")

    def test_an_explicit_request_id_is_used_verbatim(self, service):
        client, recorder = service
        client.create_lora_training_client("qwen3.5-4b", request_id="my-own-id-0001")
        assert recorder.calls[0][2]["request_id"] == "my-own-id-0001"

    def test_closing_releases_the_slot(self, service):
        client, recorder = service
        tc = client.create_lora_training_client("qwen3.5-4b")
        tc.close()
        assert any(method == "DELETE" for method, _, _ in recorder.calls)

    def test_context_manager_closes_even_when_the_loop_raises(self, service):
        client, recorder = service
        with pytest.raises(RuntimeError):
            with client.create_lora_training_client("qwen3.5-4b"):
                raise RuntimeError("training blew up")
        # Otherwise a crashed script leaves a GPU slot held until the watchdog
        # reclaims it half an hour later.
        assert any(method == "DELETE" for method, _, _ in recorder.calls)

    def test_waits_for_a_gpu_that_is_still_starting(self, service, monkeypatch):
        # Nothing is kept warm, so this is the first response almost every
        # customer sees. It must be waited out, not crashed on.
        monkeypatch.setattr(runara.client, "POLL_SECONDS", 0)
        client, recorder = service
        recorder.pending_creates = 2
        tc = client.create_lora_training_client("qwen3.5-4b")

        assert tc.worker_url == recorder.worker_base
        creates = [body for method, path, body in recorder.calls if path.endswith("/finetune/sessions")]
        assert len(creates) == 3
        # Replayed, not reopened: a new request_id would be a second billable session.
        assert len({body["request_id"] for body in creates}) == 1

    def test_gives_up_waiting_with_the_request_id_to_resume(self, service, monkeypatch):
        monkeypatch.setattr(runara.client, "POLL_SECONDS", 0)
        client, recorder = service
        recorder.pending_creates = 10_000
        with pytest.raises(runara.RunaraError, match="my-own-id-0002"):
            client.create_lora_training_client(
                "qwen3.5-4b", request_id="my-own-id-0002", wait_timeout=0
            )

    def test_closing_waits_for_a_queued_save(self, service):
        client, recorder = service
        with client.create_lora_training_client("qwen3.5-4b") as tc:
            tc.save_state(name="last")  # deliberately not awaited
        paths = [path for _, path, _ in recorder.calls]
        assert any(p.endswith("/save_state") for p in paths)
        save = next(i for i, p in enumerate(paths) if p.endswith("/save_state"))
        close = next(i for i, (m, _, _) in enumerate(recorder.calls) if m == "DELETE")
        # Closing first would refuse the save and lose the adapter.
        assert save < close


class TestPipelining:
    def test_both_calls_are_submitted_before_either_is_awaited(self, service):
        client, _ = service
        tc = client.create_lora_training_client("qwen3.5-4b")

        fb = tc.forward_backward(batch())
        step = tc.optim_step(AdamParams(learning_rate=1e-4))
        # Neither has been awaited, and the loop could already be tokenizing
        # the next batch. That overlap is the whole point of the futures.
        assert fb.result().loss == 2.5
        assert step.result() == 1
        tc.close()

    def test_calls_reach_the_worker_in_submission_order(self, service):
        client, recorder = service
        tc = client.create_lora_training_client("qwen3.5-4b")

        futures = []
        for _ in range(3):
            futures.append(tc.forward_backward(batch()))
            futures.append(tc.optim_step(AdamParams(learning_rate=1e-4)))
        for future in futures:
            future.result()

        actions = [path.rsplit("/", 1)[-1] for _, path, _ in recorder.calls if "sessions/sess_1" in path]
        # An optim_step that overtook its forward_backward would apply the
        # previous step's gradients, which no error would reveal.
        assert actions == ["forward_backward", "optim_step"] * 3
        tc.close()

    def test_a_failure_surfaces_when_the_future_is_awaited(self, service):
        client, recorder = service
        tc = client.create_lora_training_client("qwen3.5-4b")
        recorder.fail_next = (409, {"error": {"code": "session_closed", "message": "ended"}})

        future = tc.forward_backward(batch())
        # Submitting did not raise; collecting does. A loop that never calls
        # result() would otherwise train into the void.
        with pytest.raises(SessionClosed):
            future.result()
        tc.close()


class TestTokens:
    def test_refreshes_a_token_that_is_about_to_expire(self, service):
        client, recorder = service
        tc = client.create_lora_training_client("qwen3.5-4b")
        # Expires in a minute: inside the refresh margin, so the next call
        # must not start on a token that dies mid-flight.
        tc._token.expires_at = int((time.time() + 60) * 1000)

        tc.forward_backward(batch()).result()

        assert any(path.endswith("/token") for _, path, _ in recorder.calls)
        assert tc._token.value == "tok_refreshed"
        tc.close()

    def test_does_not_refresh_a_token_with_life_left(self, service):
        client, recorder = service
        tc = client.create_lora_training_client("qwen3.5-4b")
        tc.forward_backward(batch()).result()
        assert not any(path.endswith("/token") for _, path, _ in recorder.calls)
        tc.close()


class TestErrors:
    def test_an_empty_balance_raises_something_a_loop_can_catch(self, service):
        client, recorder = service
        recorder.fail_next = (
            402,
            {"error": {"code": "INSUFFICIENT_BALANCE", "message": "Add credits"}},
        )
        with pytest.raises(InsufficientBalance):
            client.create_lora_training_client("qwen3.5-4b")

    def test_a_missing_api_key_says_where_to_get_one(self, monkeypatch):
        monkeypatch.delenv("RUNARA_API_KEY", raising=False)
        with pytest.raises(ValueError, match="console"):
            runara.ServiceClient()


class TestDatum:
    def test_refuses_arrays_that_describe_different_positions(self):
        # The commonest hand-rolled mistake, and one that otherwise trains
        # happily against misaligned targets.
        with pytest.raises(ValueError, match="same length"):
            Datum(token_ids=[1, 2, 3], target_tokens=[2, 3], weights=[1.0, 1.0])

    def test_counts_only_tokens_that_carry_loss(self):
        datum = Datum(
            token_ids=[1, 2, 3, 4], target_tokens=[2, 3, 4, 5], weights=[0.0, 0.0, 1.0, 1.0]
        )
        # What you are billed for: the prompt is masked and trains nothing.
        assert datum.trained_tokens == 2


def test_lists_the_catalog_with_prices(service):
    client, _ = service
    models = client.list_models()
    assert models[0].base_model == "qwen3.5-4b"
    assert models[0].train_usd_per_million == 0.74


class TestWorkerIsPinned:
    """The data plane's security rests entirely on this, so prove it.

    The worker holds customer training tokens on a connection to an address
    we do not own, certified by nobody. What makes that safe is that the
    control plane names the certificate and the client accepts no other. A
    regression to skipping verification would leave every test above passing.
    """

    def test_a_different_certificate_is_refused(self, service):
        client, recorder = service
        # A valid certificate, correctly signed, for the right address — just
        # not the one the control plane named. This is the attack: someone
        # between the client and the GPU presenting their own certificate.
        certdir = tempfile.mkdtemp(prefix="runara-impostor-")
        try:
            other_cert = os.path.join(certdir, "other.crt")
            other_key = os.path.join(certdir, "other.key")
            subprocess.run(
                [
                    "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                    "-days", "1", "-keyout", other_key, "-out", other_cert,
                    "-subj", "/CN=127.0.0.1",
                    "-addext", "subjectAltName=IP:127.0.0.1",
                ],
                check=True, capture_output=True,
            )
            recorder.worker_cert = pathlib.Path(other_cert).read_text()

            tc = client.create_lora_training_client(base_model="m", rank=8)
            with pytest.raises(Exception) as caught:
                tc.forward_backward(
                    [Datum(token_ids=[1, 2], target_tokens=[2, 3], weights=[0.0, 1.0])]
                ).result()
            # Specifically a TLS failure, not a timeout or a 500 that happened
            # to look like one.
            assert "certificate" in str(caught.value).lower() or "ssl" in str(
                caught.value
            ).lower()
        finally:
            shutil.rmtree(certdir, ignore_errors=True)

    def test_a_plaintext_worker_url_is_refused(self, service):
        client, recorder = service
        # A downgrade must fail loudly here rather than quietly putting the
        # customer's tokens on the wire in the clear.
        recorder.worker_base = recorder.worker_base.replace("https://", "http://")
        with pytest.raises(ValueError, match="https"):
            client.create_lora_training_client(base_model="m", rank=8)


class TestLoadState:
    def test_resumes_through_the_same_queue(self, service):
        # Ordering is the point. Loading while a forward_backward is still in
        # flight would train one batch against the old weights and the next
        # against the new, and the curve would look merely noisy.
        client, recorder = service
        tc = client.create_lora_training_client(base_model="m", rank=8)
        tc.forward_backward([Datum(token_ids=[1, 2], target_tokens=[2, 3], weights=[1.0, 1.0])])
        result = tc.load_state("gs://bucket/finetune/org/sess/sampler_weights/10").result()
        assert result["ok"] is True

        actions = [p.rsplit("/", 1)[-1] for _, p, _ in recorder.calls if "/sessions/" in p]
        assert actions.index("forward_backward") < actions.index("load_state")

    def test_sends_the_uri_the_worker_checks(self, service):
        client, recorder = service
        tc = client.create_lora_training_client(base_model="m", rank=8)
        uri = "gs://bucket/finetune/org/sess/sampler_weights/10"
        tc.load_state(uri).result()
        body = next(b for _, p, b in recorder.calls if p.endswith("/load_state"))
        # Verbatim: the worker compares it against the org in the signed token.
        assert body["uri"] == uri


def test_defaults_to_the_app_not_the_marketing_site(monkeypatch):
    # runara.ai is the Framer site and answers /api/v1 with a 404 page.
    monkeypatch.delenv("RUNARA_BASE_URL", raising=False)
    assert runara.ServiceClient(api_key="rk_test").base_url == "https://app.runara.ai/api/v1"


def rl_batch() -> list[Datum]:
    return [Datum(token_ids=[1, 2, 3], target_tokens=[2, 3, 4], logprobs=[0.0, -0.4, -0.6], advantages=[0.0, 1.0, 1.0])]


def worker_calls(recorder, action):
    return [body for method, path, body in recorder.calls if path.endswith(f"/{action}")]


class TestLosses:
    def test_an_rl_loss_and_its_config_reach_the_worker(self, service):
        client, recorder = service
        tc = client.create_lora_training_client("qwen3.5-4b")
        result = tc.forward_backward(rl_batch(), "ppo", {"clip_high_threshold": 1.3}).result()
        sent = worker_calls(recorder, "forward_backward")[-1]
        assert sent["loss_fn"] == "ppo"
        assert sent["loss_fn_config"] == {"clip_high_threshold": 1.3}
        assert sent["data"][0] == {
            "token_ids": [1, 2, 3], "target_tokens": [2, 3, 4],
            "logprobs": [0.0, -0.4, -0.6], "advantages": [0.0, 1.0, 1.0],
        }
        assert result.logprobs == [[-0.5, -0.5, -0.5]]
        assert result.metrics["loss:sum"] == 2.5

    def test_supervised_data_still_sends_exactly_what_0_1_sent(self, service):
        client, recorder = service
        tc = client.create_lora_training_client("qwen3.5-4b")
        tc.forward_backward(batch()).result()
        sent = worker_calls(recorder, "forward_backward")[-1]
        assert sent == {"data": [{"token_ids": [1, 2, 3], "target_tokens": [2, 3, 4], "weights": [0.0, 1.0, 1.0]}], "loss_fn": "cross_entropy"}

    def test_an_old_worker_is_refused_an_rl_loss_before_anything_is_sent(self, service):
        # Old workers ignored loss_fn: sending RL data to one would have
        # trained it as cross-entropy, silently.
        client, recorder = service
        recorder.capabilities = None
        tc = client.create_lora_training_client("qwen3.5-4b")
        with pytest.raises(runara.RunaraError, match="start a new one"):
            tc.forward_backward(rl_batch(), "importance_sampling")
        assert worker_calls(recorder, "forward_backward") == []

    def test_an_old_worker_still_trains_supervised(self, service):
        client, recorder = service
        recorder.capabilities = None
        tc = client.create_lora_training_client("qwen3.5-4b")
        assert tc.forward_backward(batch()).result().loss == 2.5


class TestForward:
    def test_returns_logprobs_and_can_ask_for_the_reference(self, service):
        client, recorder = service
        tc = client.create_lora_training_client("qwen3.5-4b")
        result = tc.forward(batch(), reference=True).result()
        assert result.logprobs == [[-0.25, -0.25, -0.25]]
        assert worker_calls(recorder, "forward")[-1]["reference"] is True

    def test_refused_on_an_old_worker(self, service):
        client, recorder = service
        recorder.capabilities = None
        tc = client.create_lora_training_client("qwen3.5-4b")
        with pytest.raises(runara.RunaraError, match="forward"):
            tc.forward(batch())


class TestSamplingV2:
    def test_num_samples_and_params_come_back_with_logprobs(self, service):
        client, recorder = service
        tc = client.create_lora_training_client("qwen3.5-4b")
        params = runara.SamplingParams(max_tokens=8, temperature=0.8, top_p=0.95, stop=["\n"], seed=1)
        result = tc.sample([[1, 2], [3]], num_samples=3, sampling_params=params).result()
        sent = worker_calls(recorder, "sample")[-1]
        assert sent == {
            "prompts": [[1, 2], [3]], "num_samples": 3,
            "sampling_params": {"max_tokens": 8, "temperature": 0.8, "top_p": 0.95, "top_k": -1, "stop": ["\n"], "seed": 1},
        }
        assert [len(group) for group in result.samples] == [3, 3]
        first = result.samples[0][0]
        assert (first.tokens, first.logprobs, first.stop_reason) == ([5, 6], [-0.1, -0.2], "stop")

    def test_a_0_1_style_call_sends_the_old_shape_and_still_fills_samples(self, service):
        client, recorder = service
        recorder.capabilities = None  # works against any worker
        tc = client.create_lora_training_client("qwen3.5-4b")
        result = tc.sample([[1]], max_tokens=3, temperature=0.0).result()
        assert worker_calls(recorder, "sample")[-1] == {"prompts": [[1]], "max_tokens": 3, "temperature": 0.0}
        assert result.sequences == [[1, 2, 3]]
        assert result.samples[0][0].stop_reason == "unknown"

    def test_multiple_samples_are_refused_on_an_old_worker(self, service):
        client, recorder = service
        recorder.capabilities = None
        tc = client.create_lora_training_client("qwen3.5-4b")
        with pytest.raises(runara.RunaraError):
            tc.sample([[1]], num_samples=4)


class TestAFailedStepIsNotApplied:
    def test_the_optim_step_behind_a_failed_forward_backward_is_refused(self, service):
        client, recorder = service
        tc = client.create_lora_training_client("qwen3.5-4b")
        recorder.fail_next = (500, {"error": "engine_error"})
        fb = tc.forward_backward(batch())
        step = tc.optim_step(AdamParams(1e-4))
        with pytest.raises(runara.RunaraError):
            fb.result()
        with pytest.raises(runara.RunaraError, match="optim_step skipped"):
            step.result()
        assert worker_calls(recorder, "optim_step") == []

        # Reported once, then cleared: the next batch trains normally.
        tc.forward_backward(batch())
        assert tc.optim_step(AdamParams(1e-4)).result() == 1

    def test_optim_metrics_are_kept_and_clipping_is_sent(self, service):
        client, recorder = service
        tc = client.create_lora_training_client("qwen3.5-4b")
        tc.forward_backward(batch())
        tc.optim_step(AdamParams(1e-4, grad_clip_norm=1.0)).result()
        assert worker_calls(recorder, "optim_step")[-1]["adam"]["grad_clip_norm"] == 1.0
        assert tc.last_optim_metrics == {"grad_norm": 0.5}


class TestCatalogAndCheckpoints:
    def test_models_read_with_or_without_the_retired_token_prices(self, service):
        client, recorder = service
        model = client.list_models()[0]
        assert model.train_usd_per_million == 0.74 and model.gpu_options == []

    def test_checkpoints_carry_a_uri_and_tolerate_fields_this_client_does_not_know(self, service):
        client, recorder = service
        checkpoints = client.list_checkpoints(base_model="qwen 3.5")
        assert checkpoints[0].uri == "s3://bucket/finetune/org/sess/sampler_weights/3"
        path = [p for m, p, _ in recorder.calls if "checkpoints" in p][-1]
        assert "include=uri" in path and "base_model=qwen+3.5" in path


class TestDatumForRL:
    def test_rl_inputs_must_line_up_with_the_targets(self):
        with pytest.raises(ValueError, match="advantages"):
            Datum(token_ids=[1, 2], target_tokens=[2, 3], logprobs=[0.0, -1.0], advantages=[1.0])

    def test_trained_tokens_counts_advantaged_positions(self):
        assert rl_batch()[0].trained_tokens == 2
