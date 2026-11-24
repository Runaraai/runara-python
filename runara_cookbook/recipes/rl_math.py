"""RL on problems with a checkable answer: sample a group, reward the right ones, repeat.

The shape of Tinker's "first RL" tutorial (GRPO on GSM8K): for each problem,
sample `group_size` replies; reward 1 for the correct final number, 0
otherwise; train with importance_sampling on each reply's advantage over its
group. No reward model, no labels beyond the answers.

Data, one problem per line. `answer` may be GSM8K's full solution ending in
"#### 42"; only the final number is compared:

    {"question": "Natalia sold clips to 48 of her friends ...", "answer": "72"}
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import runara

from .. import data as cookbook_data
from ..renderers.base import Message, Renderer
from ..rl import rl_data
from . import _cli

#: Asks for the answer in a form exact_match_reward can find.
INSTRUCTION = "Solve the problem. End your reply with the final answer as a number on its own line."

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def final_number(text: str) -> str | None:
    """The last number in `text`, commas dropped, or None. GSM8K's #### wins."""
    if "####" in text:
        text = text.rsplit("####", 1)[1]
    found = _NUMBER.findall(text)
    if not found:
        return None
    value = found[-1].replace(",", "")
    return value[:-2] if value.endswith(".0") else value


def exact_match_reward(reply: str, answer: str) -> float:
    """1.0 if the reply's final number is the answer's, else 0.0."""
    got, want = final_number(reply), final_number(answer)
    return 1.0 if got is not None and got == want else 0.0


def problem_messages(question: str) -> list[Message]:
    return [{"role": "user", "content": f"{INSTRUCTION}\n\n{question}"}]


def train(
    tc: runara.TrainingClient,
    renderer: Renderer,
    problems: Sequence[tuple[Sequence[Message], str]],
    *,
    steps: int = 20,
    prompts_per_step: int = 8,
    group_size: int = 8,
    sampling_params: runara.SamplingParams = runara.SamplingParams(max_tokens=256, temperature=1.0),
    adam: runara.AdamParams = runara.AdamParams(learning_rate=4e-5),
    loss_fn: str = "importance_sampling",
    loss_fn_config: dict | None = None,
    reward_fn: Callable[[str, str], float] = exact_match_reward,
    log: Callable[[dict], None] = _cli.log,
) -> list[dict]:
    """`steps` rounds of sample, score, train. Returns one record per step."""
    if not problems:
        raise ValueError("no problems to train on")
    history = []
    for step in range(steps):
        start = (step * prompts_per_step) % len(problems)
        batch = [problems[(start + i) % len(problems)] for i in range(prompts_per_step)]
        prompts = [renderer.generation_prompt(messages) for messages, _ in batch]
        # One call samples every prompt's whole group, from the adapter as it
        # stands now: on-policy, with no snapshot to take first.
        result = tc.sample(prompts, num_samples=group_size, sampling_params=sampling_params).result()

        data: list[runara.Datum] = []
        rewards: list[float] = []
        for (_, answer), prompt, group in zip(batch, prompts, result.samples):
            group_rewards = [reward_fn(renderer.decode(s.tokens), answer) for s in group]
            rewards.extend(group_rewards)
            data.extend(rl_data(prompt, group, group_rewards))

        record: dict = {"step": step, "reward": sum(rewards) / len(rewards), "datums": len(data)}
        if data:
            fb = tc.forward_backward(data, loss_fn, loss_fn_config)
            tc.optim_step(adam)
            record["loss"] = fb.result().loss
            if tc.last_optim_metrics.get("grad_norm") is not None:
                record["grad_norm"] = float(tc.last_optim_metrics["grad_norm"])
        # else: every group scored all-right or all-wrong; nothing to learn from.
        log(record)
        history.append(record)
    return history


def main() -> None:
    p = _cli.parser(__doc__.splitlines()[0])
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--prompts-per-step", type=int, default=8)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=256)
    p.set_defaults(learning_rate=4e-5)
    args = p.parse_args()
    renderer = _cli.renderer_for(args)
    problems = [(problem_messages(row["question"]), str(row["answer"])) for row in cookbook_data.read_jsonl(args.data)]
    with _cli.training_client(args) as tc:
        train(tc, renderer, problems, steps=args.steps, prompts_per_step=args.prompts_per_step,
              group_size=args.group_size,
              sampling_params=runara.SamplingParams(max_tokens=args.max_tokens, temperature=1.0),
              adam=runara.AdamParams(learning_rate=args.learning_rate))
        if args.save_as:
            print(tc.save_state(name=args.save_as).result())


if __name__ == "__main__":
    main()
