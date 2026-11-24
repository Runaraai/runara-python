"""The RL and preference helpers, and the recipe loops, without a network.

The helpers are where an alignment mistake would hide: an RL datum with its
advantages one position off trains the wrong tokens with no error. The
recipes are checked for what they send, against a client that records calls.
"""

from __future__ import annotations

import json
from concurrent.futures import Future as _Future

import pytest

import runara
from runara import Datum, SampledSequence
from runara._futures import Future
from runara_cookbook import renderers
from runara_cookbook.preference import build_pair, pair_batches, read_pairs, PreferencePair
from runara_cookbook.recipes import dpo, rl_math, sft
from runara_cookbook.rl import build_rl_datum, group_advantages, rl_data


class Tok:
    """One token per character, with a chat template that marks roles."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, tokens, **_kwargs):
        return "".join(chr(t) for t in tokens)

    def apply_chat_template(self, conversation, tokenize=False, **kwargs):
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in conversation)
        return text + "<assistant>" if kwargs.get("add_generation_prompt") else text


@pytest.fixture
def renderer():
    return renderers.get("generic", Tok())


def seq(text: str, lp: float = -0.5) -> SampledSequence:
    tokens = [ord(c) for c in text]
    return SampledSequence(tokens=tokens, logprobs=[lp] * len(tokens), stop_reason="stop")


class TestAdvantages:
    def test_reward_minus_group_mean(self):
        assert group_advantages([1.0, 0.0, 0.0, 1.0]) == [0.5, -0.5, -0.5, 0.5]

    def test_a_uniform_group_has_no_signal(self):
        assert group_advantages([1.0, 1.0]) == [0.0, 0.0]


class TestRLDatum:
    def test_layout_masks_the_prompt_and_aligns_the_reply(self):
        d = build_rl_datum([10, 11, 12], SampledSequence([20, 21], [-0.1, -0.2], "stop"), advantage=0.5)
        # full = 10 11 12 20 21; inputs drop the last, targets the first.
        assert d.token_ids == [10, 11, 12, 20]
        assert d.target_tokens == [11, 12, 20, 21]
        # Position 2 predicts the reply's first token: that is where the
        # sampler's logprobs and the advantage begin.
        assert d.logprobs == [0.0, 0.0, -0.1, -0.2]
        assert d.advantages == [0.0, 0.0, 0.5, 0.5]
        assert d.weights is None

    def test_a_reply_without_logprobs_is_refused_with_the_reason(self):
        with pytest.raises(ValueError, match="logprobs"):
            build_rl_datum([1, 2], SampledSequence([3], [], "unknown"), 1.0)

    def test_rl_data_drops_average_replies_and_uniform_groups(self):
        group = [seq("ab"), seq("cd"), seq("ef")]
        data = rl_data([1, 2], group, [1.0, 0.0, 0.5])
        assert len(data) == 2  # the reply that scored exactly the mean is dropped
        assert rl_data([1, 2], group, [1.0, 1.0, 1.0]) == []

    def test_the_generation_prompt_is_the_training_prompt(self, renderer):
        # A sampled reply appended to the generation prompt must line up with
        # what supervised training on the same conversation would see.
        messages = [{"role": "user", "content": "hi"}]
        prompt = renderer.generation_prompt(messages)
        sft_datum = renderer.to_datum(messages + [{"role": "assistant", "content": "yo"}])
        assert sft_datum.token_ids[: len(prompt) - 1] == prompt[:-1]
        assert renderer.decode(prompt).endswith("<assistant>")


class TestPreferencePairs:
    def test_chosen_then_rejected_each_weighted_on_its_reply(self, renderer):
        chosen, rejected = build_pair(renderer, [{"role": "user", "content": "q"}], "good", "bad")
        assert sum(chosen.weights) == len("good</assistant>")
        assert sum(rejected.weights) == len("bad</assistant>")

    def test_an_identical_pair_is_refused(self, renderer):
        with pytest.raises(ValueError, match="identical"):
            build_pair(renderer, [{"role": "user", "content": "q"}], "same", "same")

    def test_batches_never_split_a_pair(self, renderer):
        pairs = [PreferencePair([{"role": "user", "content": str(i)}], "a", "b") for i in range(5)]
        batches = pair_batches(renderer, pairs, pairs_per_batch=2)
        assert [len(b) for b in batches] == [4, 4, 2]

    def test_reads_pairs_and_names_a_bad_line_without_its_contents(self, tmp_path):
        path = tmp_path / "pairs.jsonl"
        path.write_text(
            json.dumps({"prompt": "plain string prompt", "chosen": "a", "rejected": "b"}) + "\n"
            + json.dumps({"prompt": "secret prompt text", "chosen": "a"}) + "\n"
        )
        with pytest.raises(ValueError) as error:
            read_pairs(str(path))
        assert "line 2" in str(error.value) and "secret" not in str(error.value)
        path.write_text(json.dumps({"prompt": "p", "chosen": "a", "rejected": "b"}) + "\n")
        assert read_pairs(str(path))[0].prompt == [{"role": "user", "content": "p"}]


class TestMathReward:
    @pytest.mark.parametrize(
        "reply, answer, reward",
        [
            ("so the total is 72", "72", 1.0),
            ("The answer is 1,234.", "Natalia ... #### 1234", 1.0),
            ("I think 71", "72", 0.0),
            ("no idea", "72", 0.0),
            ("it is 5.0", "5", 1.0),
        ],
    )
    def test_compares_final_numbers(self, reply, answer, reward):
        assert rl_math.exact_match_reward(reply, answer) == reward


class FakeTrainingClient:
    """Records calls; samples are scripted; every future is already done."""

    def __init__(self, replies: list[str]):
        self.replies = replies
        self.calls: list[tuple] = []
        self.last_optim_metrics = {"grad_norm": 0.25}

    @staticmethod
    def _done(value) -> Future:
        inner: _Future = _Future()
        inner.set_result(value)
        return Future(inner)

    def sample(self, prompts, num_samples=1, sampling_params=None, **_):
        self.calls.append(("sample", len(prompts), num_samples))
        group = [seq(self.replies[i % len(self.replies)]) for i in range(num_samples)]
        return self._done(runara.SampleResult(sequences=[], token_count=0, samples=[group for _ in prompts]))

    def forward_backward(self, data, loss_fn="cross_entropy", loss_fn_config=None):
        self.calls.append(("forward_backward", len(data), loss_fn, loss_fn_config))
        metrics = {"dpo_loss": 0.6, "accuracy": 0.75, "margin": 0.1} if loss_fn == "dpo" else {}
        return self._done(runara.ForwardBackwardResult(loss=0.5, metrics=metrics, seq_count=len(data), token_count=1))

    def optim_step(self, params):
        self.calls.append(("optim_step", params.learning_rate))
        return self._done(1)


class TestRecipes:
    def test_rl_math_rewards_right_answers_and_trains_on_the_difference(self, renderer):
        tc = FakeTrainingClient(replies=["the answer is 4", "maybe 5"])
        problems = [(rl_math.problem_messages("2+2?"), "4")]
        history = rl_math.train(tc, renderer, problems, steps=2, prompts_per_step=1, group_size=4, log=lambda r: None)
        assert history[0]["reward"] == 0.5
        kinds = [c[0] for c in tc.calls]
        assert kinds == ["sample", "forward_backward", "optim_step"] * 2
        _, n_datums, loss_fn, _ = tc.calls[1]
        assert (n_datums, loss_fn) == (4, "importance_sampling")
        assert history[0]["grad_norm"] == 0.25

    def test_rl_math_skips_the_update_when_no_reply_differs(self, renderer):
        tc = FakeTrainingClient(replies=["4"])
        rl_math.train(tc, renderer, [(rl_math.problem_messages("2+2?"), "4")], steps=1, prompts_per_step=1,
                      group_size=3, log=lambda r: None)
        assert [c[0] for c in tc.calls] == ["sample"]

    def test_dpo_sends_whole_pairs_with_beta(self, renderer):
        tc = FakeTrainingClient(replies=[])
        pairs = [PreferencePair([{"role": "user", "content": "q"}], "good", "bad")] * 3
        history = dpo.train(tc, renderer, pairs, pairs_per_batch=2, beta=0.2, log=lambda r: None)
        fbs = [c for c in tc.calls if c[0] == "forward_backward"]
        assert [(n, loss, cfg) for _, n, loss, cfg in fbs] == [(4, "dpo", {"beta": 0.2}), (2, "dpo", {"beta": 0.2})]
        assert history[0]["accuracy"] == 0.75

    def test_sft_trains_every_batch(self, renderer):
        tc = FakeTrainingClient(replies=[])
        conversations = [[{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": "a"}] for i in range(5)]
        sft.train(tc, renderer, conversations, epochs=2, batch_size=2, log=lambda r: None)
        fbs = [c for c in tc.calls if c[0] == "forward_backward"]
        assert len(fbs) == 6 and all(c[2] == "cross_entropy" for c in fbs)
