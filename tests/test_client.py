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

    def handle(self, method: str, path: str, body: dict) -> tuple[int, dict]:
        self.calls.append((method, path, body))
        if self.fail_next is not None:
            status, payload = self.fail_next
            self.fail_next = None
            return status, payload

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
        if path.endswith("/forward_backward"):
            return 200, {"loss": 2.5, "metrics": {"loss": 2.5}, "seq_count": 1, "token_count": 7}
        if path.endswith("/optim_step"):
            self.step += 1
            return 200, {"ok": True, "step": self.step}
        if path.endswith("/sample"):
            return 200, {"sequences": [[1, 2, 3]], "token_count": 3}
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
