# runara

Fine-tune open models without hosting them. You write the training loop; we
hold the base weights and run the forward and backward passes.

```bash
pip install "runara[cookbook]"
export RUNARA_API_KEY=rk_live_...
```

```python
import runara
from runara_cookbook import data, renderers
from transformers import AutoTokenizer

# Tokenized on your machine; only token ids and a loss mask are sent.
renderer = renderers.get("generic", AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct"))
dataset = data.build_data(renderer, data.conversations(data.read_jsonl("my_tickets.jsonl")))

sc = runara.ServiceClient()                      # reads RUNARA_API_KEY
with sc.create_lora_training_client(base_model="qwen-2.5-7b", rank=32) as tc:  # waits for a GPU
    for epoch in range(3):
        for batch in data.batches(dataset, 8, shuffle=True, seed=epoch):
            fb = tc.forward_backward(batch)              # submitted, not awaited
            tc.optim_step(runara.AdamParams(1e-4))       # submitted straight after
            print(fb.result().loss)                      # the loop blocks here
    tc.save_state(name="support-v1").result()
```

`base_model` is a slug from the console's Models tab, and the tokenizer is
that model's Repository. `my_tickets.jsonl` holds one
`{"messages": [{"role": ..., "content": ...}, ...]}` per line.

## Your data does not leave your machine

`open(...)` runs on your computer. The cookbook tokenizes locally and builds a
per-token loss mask, so what crosses the wire is a list of integers and a list
of floats — no text, no filenames, no schema. A two-turn conversation is about
900 bytes of that.

You can check this yourself before sending anything:

```python
datum = renderer.to_datum(messages)
print(datum.token_ids)       # integers
print(datum.weights)         # 0.0 for the prompt, 1.0 for what you train on
```

What we do keep is the adapter you write with `save_state`, plus token counts
and loss values for billing and the charts in the console.

## Why the loop is yours

There is no job to submit and no config to fill in. You decide the batching,
the schedule, when to stop, and what the loss means. That is more work than a
managed API and it is the point: it is the difference between tuning a model
and filling in a form.

`forward_backward` and `optim_step` both return immediately. Submit them back
to back and collect later — the worker applies calls for a session in the
order they arrive, so the round trip overlaps your next batch of tokenization
instead of stalling the GPU.

## Reinforcement learning

Sample a group of replies per prompt, score them however you like, and train
on how much better or worse each did than its group's average:

```python
from runara_cookbook.rl import rl_data

prompt = renderer.generation_prompt([{"role": "user", "content": question}])
group = tc.sample([prompt], num_samples=8,
                  sampling_params=runara.SamplingParams(max_tokens=256)).result().samples[0]
rewards = [my_reward(renderer.decode(s.tokens)) for s in group]   # your code, your machine

data = rl_data(prompt, group, rewards)        # advantages = reward - group mean
if data:
    tc.forward_backward(data, "importance_sampling")
    tc.optim_step(runara.AdamParams(learning_rate=4e-5))
```

Each sampled sequence carries the policy's logprob of every token. These are
scored exactly as training scores them, so on-policy the importance ratio is
1. `python -m runara_cookbook.recipes.rl_math` is the whole loop, on problems
with checkable numeric answers.

## Preferences (DPO)

```python
from runara_cookbook.preference import build_pair

data = build_pair(renderer, prompt_messages, chosen="...", rejected="...")   # chosen first
out = tc.forward_backward(data, "dpo", {"beta": 0.1}).result()
print(out.metrics["accuracy"], out.metrics["margin"])
```

The reference policy is the base model your session started from, at no
extra cost. To compare against something else, send `ref_logprobs` on each
datum. The whole loop is in `python -m runara_cookbook.recipes.dpo`.

## Losses

| `loss_fn` | Data | `loss_fn_config` |
|---|---|---|
| `cross_entropy` | `weights` | `reduction`: `"mean"` (default) or `"sum"` |
| `importance_sampling` | `logprobs`, `advantages` | none |
| `ppo` | `logprobs`, `advantages` | `clip_low_threshold` 0.8, `clip_high_threshold` 1.2 |
| `cispo` | `logprobs`, `advantages` | `clip_low_threshold` 0.0, `clip_high_threshold` 4.0 |
| `dro` | `logprobs`, `advantages` | `beta` (required) |
| `dpo` | `weights`, chosen/rejected alternating, optional `ref_logprobs` | `beta` 0.1 |

The formulas are Tinker's, so its recipes and hyperparameters carry over. The
one difference: `cross_entropy` defaults to a per-example mean, so a long
masked prompt doesn't dilute the loss. Pass `{"reduction": "sum"}` for
Tinker's sum. The RL losses and `dpo` sum over tokens, as Tinker's do.

`tc.forward(data)` returns each target token's logprob without training, and
`tc.forward(data, reference=True)` returns it under the base model.
`AdamParams(grad_clip_norm=1.0)` clips gradients, and each step's gradient norm
is in `tc.last_optim_metrics`.

A session on a worker started before these features existed refuses them up
front, naming the feature. Start a new session to get a current worker.

## What you get out

```python
for checkpoint in sc.list_checkpoints():
    print(checkpoint.name, checkpoint.size_bytes, checkpoint.base_model)
```

Each checkpoint's `uri` is what `tc.load_state(uri)` takes to resume from it.
Optimizer state isn't saved yet, so a resumed run restarts Adam's moments.

Checkpoints are standard PEFT adapters — `adapter_config.json` and
`adapter_model.safetensors` — so they load in vLLM, SGLang or transformers
without conversion. Download them from the console, or serve them on Runara
alongside the base model.

## Errors worth catching

| | |
|---|---|
| `runara.InsufficientBalance` | The organization is out of credit. |
| `runara.SessionClosed` | The session ended, or is ending because the balance ran out. |
| `runara.RunaraError` | Everything else, with `.status` and `.code`. |

A training call is never retried automatically. One that timed out may already
have applied its gradients, and sending it again would train on the same batch
twice.

## Checked against the reference

The renderers are verified to produce byte-identical tokens, targets and loss
masks to `tinker-cookbook` for the same conversation and tokenizer. Chat
templating is the part of fine-tuning that goes wrong silently, so it is
checked against an independent implementation rather than only against itself.

## Installing

The client has no dependencies at all — it is stdlib, so installing it cannot
conflict with whatever `torch`, `transformers` or `numpy` versions your
training code already pins. The cookbook's dependencies (`transformers` for tokenizers) come with the `runara[cookbook]` extra.
