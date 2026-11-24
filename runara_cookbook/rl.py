"""Turning sampled replies and their rewards into RL training data.

The loop is GRPO-shaped, as in Tinker's "first RL" tutorial: sample a group of
replies per prompt, score each, and train on how much better or worse each did
than its group's average. Replies that did exactly average carry no signal and
are dropped, and so is any group where every reply scored the same.

The data layout is the part that is easy to get wrong without an error:

    full    = prompt + reply
    input   = full[:-1]          target = full[1:]
    logprobs   = [0] * (len(prompt) - 1) + the sampler's logprob of each reply token
    advantages = [0] * (len(prompt) - 1) + [advantage] * len(reply)

The zero advantages on the prompt are also what masks it out of the loss.
"""

from __future__ import annotations

from collections.abc import Sequence

from runara import Datum, SampledSequence


def group_advantages(rewards: Sequence[float]) -> list[float]:
    """Each reward minus its group's mean. No std scaling, as in Tinker."""
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    return [float(reward - mean) for reward in rewards]


def build_rl_datum(prompt_tokens: Sequence[int], sequence: SampledSequence, advantage: float) -> Datum:
    """One reply, aligned for an RL loss (importance_sampling, ppo, cispo, dro)."""
    prompt = list(prompt_tokens)
    reply = list(sequence.tokens)
    if not prompt:
        raise ValueError("the prompt has no tokens")
    if not reply:
        raise ValueError("the reply has no tokens; skip empty replies")
    if len(sequence.logprobs) != len(reply):
        raise ValueError(
            "this reply has no per-token logprobs, which RL needs. Sample with "
            "num_samples or sampling_params on a current worker; a session on an "
            "older worker returns tokens only."
        )
    full = prompt + reply
    pad = len(prompt) - 1
    return Datum(
        token_ids=full[:-1],
        target_tokens=full[1:],
        logprobs=[0.0] * pad + [float(lp) for lp in sequence.logprobs],
        advantages=[0.0] * pad + [float(advantage)] * len(reply),
    )


def rl_data(
    prompt_tokens: Sequence[int],
    group: Sequence[SampledSequence],
    rewards: Sequence[float],
) -> list[Datum]:
    """A group's training data: one datum per reply that beat or trailed the mean."""
    if len(group) != len(rewards):
        raise ValueError(f"{len(group)} replies but {len(rewards)} rewards")
    advantages = group_advantages(rewards)
    return [
        build_rl_datum(prompt_tokens, sequence, advantage)
        for sequence, advantage in zip(group, advantages)
        if advantage != 0 and sequence.tokens
    ]
