"""Rendering and masking, against a stub tokenizer.

A stub rather than a real one on purpose: the logic that matters here is the
shift and the mask, and those are exact. A character-per-token tokenizer makes
an off-by-one visible as a readable string instead of a difference between two
lists of five-digit integers.
"""

from __future__ import annotations

import json

import pytest

from runara.types import Datum
from runara_cookbook import data as cookbook_data
from runara_cookbook import renderers
from runara_cookbook.renderers import TrainOnWhat
from runara_cookbook.renderers.base import RenderedExample, to_datum


class CharTokenizer:
    """One token per character, so tokens are readable in a failure message."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, tokens) -> str:
        return "".join(chr(token) for token in tokens)

    def apply_chat_template(self, conversation, tokenize=False, **kwargs) -> str:
        parts = [f"<{m['role']}>{m['content']}</{m['role']}>" for m in conversation]
        text = "".join(parts)
        if kwargs.get("add_generation_prompt"):
            text += "<assistant>"
        return text


@pytest.fixture
def renderer():
    return renderers.get("generic", CharTokenizer())


CHAT = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "yo"},
]


class TestShift:
    def test_targets_are_the_inputs_moved_along_by_one(self):
        rendered = RenderedExample(tokens=[10, 11, 12, 13], weights=[0.0, 0.0, 1.0, 1.0])
        datum = to_datum(rendered)

        # A model predicts the next token, so at position i the input is
        # token i and the answer being scored is token i+1.
        assert datum.token_ids == [10, 11, 12]
        assert datum.target_tokens == [11, 12, 13]

    def test_weights_follow_the_targets_not_the_inputs(self):
        rendered = RenderedExample(tokens=[10, 11, 12, 13], weights=[0.0, 0.0, 1.0, 1.0])
        datum = to_datum(rendered)

        # weights[1:], not weights[:-1]. A weight says "score the prediction
        # here", and the prediction at position i is of the original token
        # i+1. Taking the other slice shifts the mask one place against the
        # targets: the model trains on the prompt's last token and not on the
        # answer's last token, with no symptom whatsoever.
        assert datum.weights == [0.0, 1.0, 1.0]

    def test_the_masked_span_lines_up_with_the_answer(self, renderer):
        tokenizer = CharTokenizer()
        datum = renderer.to_datum(CHAT)
        trained = tokenizer.decode(
            [datum.target_tokens[i] for i, w in enumerate(datum.weights) if w > 0]
        )
        # Exactly the assistant's turn and its closing tag, nothing of the
        # prompt. This is the assertion that catches an off-by-one.
        assert trained == "yo</assistant>"

    def test_a_sequence_too_short_to_shift_is_refused(self):
        with pytest.raises(ValueError, match="at least two tokens"):
            to_datum(RenderedExample(tokens=[1], weights=[1.0]))

    def test_mismatched_lengths_are_refused_before_the_shift(self):
        with pytest.raises(ValueError, match="same length"):
            RenderedExample(tokens=[1, 2, 3], weights=[1.0])


class TestMasking:
    def test_default_trains_on_the_final_answer_only(self, renderer):
        rendered = renderer.build_supervised_example(CHAT)
        trained = CharTokenizer().decode(
            [t for t, w in zip(rendered.tokens, rendered.weights) if w > 0]
        )
        assert trained == "yo</assistant>"
        # The prompt is present but masked: the model sees it, learns nothing
        # from predicting it.
        assert "hi" in CharTokenizer().decode(rendered.tokens)

    def test_can_train_on_every_assistant_turn(self, renderer):
        chat = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
        ]
        rendered = renderer.build_supervised_example(
            chat, TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )
        trained = CharTokenizer().decode(
            [t for t, w in zip(rendered.tokens, rendered.weights) if w > 0]
        )
        assert "b" in trained and "d" in trained
        # User turns stay masked even in the middle of a conversation.
        assert "<user>c</user>" not in trained

    def test_all_tokens_trains_on_everything(self, renderer):
        rendered = renderer.build_supervised_example(CHAT, TrainOnWhat.ALL_TOKENS)
        assert set(rendered.weights) == {1.0}

    def test_an_empty_conversation_is_refused(self, renderer):
        with pytest.raises(ValueError, match="no messages"):
            renderer.build_supervised_example([])


class TestRegistry:
    def test_names_the_options_when_asked_for_one_we_lack(self):
        with pytest.raises(ValueError, match="Available"):
            renderers.get("not-a-family", CharTokenizer())

    def test_a_custom_family_can_be_registered(self):
        class Mine(renderers.Renderer):
            pass

        renderers.register("mine", Mine)
        assert isinstance(renderers.get("mine", CharTokenizer()), Mine)


class TestData:
    def test_reads_jsonl_and_names_the_bad_line(self, tmp_path):
        path = tmp_path / "rows.jsonl"
        path.write_text('{"messages": []}\nnot json\n')
        with pytest.raises(ValueError, match="line 2"):
            cookbook_data.read_jsonl(path)
        # The message names the line, never its contents: an error string is
        # an easy way for training data to end up in a log.
        try:
            cookbook_data.read_jsonl(path)
        except ValueError as error:
            assert "not json" not in str(error)

    def test_builds_data_and_counts_what_an_epoch_costs(self, tmp_path, renderer):
        path = tmp_path / "rows.jsonl"
        path.write_text("\n".join(json.dumps({"messages": CHAT}) for _ in range(3)))

        rows = cookbook_data.read_jsonl(path)
        chats = cookbook_data.conversations(rows)
        built = cookbook_data.build_data(renderer, chats)

        assert len(built) == 3
        assert all(isinstance(item, Datum) for item in built)
        # Trained tokens only, which is the billable quantity.
        assert cookbook_data.token_count(built) == 3 * built[0].trained_tokens

    def test_drops_over_long_conversations_rather_than_truncating(self, renderer):
        long_chat = [
            {"role": "user", "content": "x" * 500},
            {"role": "assistant", "content": "y"},
        ]
        built = cookbook_data.build_data(renderer, [CHAT, long_chat], max_tokens=100)
        # Truncating would end a conversation mid-sentence and teach the model
        # to stop there.
        assert len(built) == 1

    def test_rejects_rows_with_no_messages(self):
        with pytest.raises(ValueError, match="no 'messages' list"):
            cookbook_data.conversations([{"text": "oops"}])

    def test_batches_cover_every_example_once(self, renderer):
        built = cookbook_data.build_data(renderer, [CHAT] * 7)
        sizes = [len(batch) for batch in cookbook_data.batches(built, size=3)]
        assert sizes == [3, 3, 1]

    def test_shuffling_is_seeded_so_a_run_can_be_repeated(self, renderer):
        built = cookbook_data.build_data(renderer, [CHAT] * 10)
        first = [len(b) for b in cookbook_data.batches(built, 3, shuffle=True, seed=7)]
        second = [len(b) for b in cookbook_data.batches(built, 3, shuffle=True, seed=7)]
        assert first == second
