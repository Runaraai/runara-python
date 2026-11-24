"""Turning a conversation into tokens and a loss mask.

This is the part of fine-tuning that is easy to get wrong and never tells you.
Two mistakes in particular:

**The shift.** A language model predicts the *next* token, so the targets are
the inputs moved along by one. `to_datum` does that once, correctly. Doing it
twice — or not at all — trains the model toward the wrong position, and the
loss still falls, just toward something useless.

**The template.** Every model family wraps a conversation in its own control
tokens, and training on a format the model does not see at inference quietly
costs you quality. Which is why the renderer builds the same token sequence
the model will actually be prompted with, rather than something close to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence

from runara.types import Datum

Message = dict  # {"role": "user" | "assistant" | "system", "content": str}


class TrainOnWhat(str, Enum):
    """Which tokens carry loss.

    The default trains on the last assistant message only, which is what you
    want for most supervised fine-tuning: you are teaching the model to answer,
    not to produce the questions it was asked.
    """

    LAST_ASSISTANT_MESSAGE = "last_assistant_message"
    ALL_ASSISTANT_MESSAGES = "all_assistant_messages"
    ALL_TOKENS = "all_tokens"


@dataclass(frozen=True)
class RenderedExample:
    """Tokens and their per-token weights, before the shift."""

    tokens: list[int]
    weights: list[float]

    def __post_init__(self) -> None:
        if len(self.tokens) != len(self.weights):
            raise ValueError(
                f"tokens and weights must be the same length; "
                f"got {len(self.tokens)} and {len(self.weights)}"
            )


def to_datum(rendered: RenderedExample) -> Datum:
    """Shift once, and only once.

    `token_ids[i]` is the input at position *i*; `target_tokens[i]` is what the
    model should produce there. Because a model predicts the next token, that
    target is the *following* token of the original sequence — so inputs drop
    the last token and targets drop the first.

    The weights follow the targets, not the inputs: a weight says "score the
    prediction at this position", and the prediction being scored at position
    *i* is of the original token *i+1*. Using `weights[:-1]` instead would
    shift the mask one place against the targets, so the model would be
    trained on the prompt's final token and not on the answer's final token —
    an error of exactly one position, with no symptom at all.
    """
    if len(rendered.tokens) < 2:
        raise ValueError("need at least two tokens to form an input and a target")
    return Datum(
        token_ids=rendered.tokens[:-1],
        target_tokens=rendered.tokens[1:],
        weights=rendered.weights[1:],
    )


class Tokenizer(Protocol):
    """The slice of a HuggingFace tokenizer a renderer needs."""

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...

    def decode(self, tokens: Sequence[int], **kwargs) -> str: ...

    def apply_chat_template(
        self, conversation: Sequence[Message], tokenize: bool = ..., **kwargs
    ) -> str: ...


class Renderer:
    """Base class. A family-specific renderer overrides the template calls."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

    # -- what subclasses provide ------------------------------------------

    def render_prefix(self, messages: Sequence[Message]) -> str:
        """The prompt text the model would be given for these messages."""
        return self.tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )

    def render_full(self, messages: Sequence[Message]) -> str:
        """The whole conversation, including the final answer."""
        return self.tokenizer.apply_chat_template(list(messages), tokenize=False)

    # -- the usable surface ------------------------------------------------

    def build_supervised_example(
        self,
        messages: Sequence[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> RenderedExample:
        """Tokens plus a 0/1 mask, in the format this model is served in."""
        if not messages:
            raise ValueError("no messages to render")

        if train_on_what is TrainOnWhat.ALL_TOKENS:
            tokens = self.tokenizer.encode(
                self.render_full(messages), add_special_tokens=False
            )
            return RenderedExample(tokens, [1.0] * len(tokens))

        if train_on_what is TrainOnWhat.ALL_ASSISTANT_MESSAGES:
            return self._mask_each_assistant_turn(messages)

        return self._mask_final_answer(messages)

    def to_datum(
        self,
        messages: Sequence[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> Datum:
        """Render and shift in one call. What you send to `forward_backward`."""
        return to_datum(self.build_supervised_example(messages, train_on_what))

    def generation_prompt(self, messages: Sequence[Message]) -> list[int]:
        """The tokens to sample a reply to `messages` from: what `sample()` takes.

        Tokenized exactly as the prompt part of a training example is, so an
        RL datum built from this prompt plus a sampled reply lines up with
        what supervised training on the same conversation would have seen.
        """
        if not messages:
            raise ValueError("no messages to render")
        return self.tokenizer.encode(self.render_prefix(messages), add_special_tokens=False)

    def decode(self, tokens: Sequence[int]) -> str:
        """A sampled reply as text, without special tokens such as end-of-turn."""
        try:
            return self.tokenizer.decode(list(tokens), skip_special_tokens=True)
        except TypeError:
            # A minimal tokenizer that takes no keyword arguments.
            return self.tokenizer.decode(list(tokens))

    # -- masking -----------------------------------------------------------

    def _mask_final_answer(self, messages: Sequence[Message]) -> RenderedExample:
        """Train on the last message only, with everything before it masked.

        The split is done on *text* and then tokenized separately, rather than
        tokenizing once and guessing a boundary. Guessing is where off-by-one
        mask errors come from, and they are invisible: the model simply learns
        a little of the prompt along with the answer.
        """
        prefix_text = self.render_prefix(messages[:-1])
        full_text = self.render_full(messages)
        if not full_text.startswith(prefix_text):
            raise ValueError(
                "this model's chat template does not extend its own prompt; "
                "the renderer for this family needs to override _mask_final_answer"
            )

        prompt = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        # Trailing whitespace is trimmed because chat templates append a
        # newline after the stop token, and generation ends *at* the stop
        # token. Training on what follows it spends signal teaching the model
        # to emit something it will never be asked to produce. Verified
        # against tinker-cookbook's qwen3 renderer, which trims the same token.
        completion_text = full_text[len(prefix_text) :].rstrip()
        completion = self.tokenizer.encode(completion_text, add_special_tokens=False)
        if not completion:
            raise ValueError("the final message rendered to no tokens")

        return RenderedExample(
            tokens=prompt + completion,
            weights=[0.0] * len(prompt) + [1.0] * len(completion),
        )

    def _mask_each_assistant_turn(self, messages: Sequence[Message]) -> RenderedExample:
        """Train on every assistant turn, masking user and system turns.

        Boundaries come from rendering the conversation one message longer
        each time and taking the difference. Using the *generation prompt* as
        the running prefix instead looks natural and is wrong: after an
        assistant turn it ends with a dangling header that the next render
        does not extend, so the split silently stops lining up.

        The assistant's own header falls inside the trained span here, unlike
        the single-answer case which masks it. That is deliberate — across a
        multi-turn conversation the model has to learn where its turns begin.
        """
        tokens: list[int] = []
        weights: list[float] = []
        previous = ""
        for index, message in enumerate(messages):
            current = self.render_full(messages[: index + 1])
            if not current.startswith(previous):
                raise ValueError(
                    "this chat template does not extend its own output; "
                    "this family needs its own renderer"
                )
            chunk = self.tokenizer.encode(
                current[len(previous) :], add_special_tokens=False
            )
            tokens.extend(chunk)
            weights.extend(
                [1.0 if message.get("role") == "assistant" else 0.0] * len(chunk)
            )
            previous = current
        return RenderedExample(tokens, weights)
