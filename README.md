# runara

Fine-tune open models without hosting them. You write the training loop; we
hold the base weights and run the forward and backward passes.

```bash
pip install runara runara-cookbook
```

```python
import runara
from runara_cookbook import renderers

sc = runara.ServiceClient(api_key="rk_live_...")
tc = sc.create_lora_training_client(base_model="qwen3.5-4b", rank=32)

renderer = renderers.get("qwen3_5", tokenizer=tc.get_tokenizer())
rows = [json.loads(line) for line in open("my_tickets.jsonl")]   # your disk

for epoch in range(3):
    for chunk in batches(rows, 8):
        data = [renderer.to_datum(row["messages"]) for row in chunk]
        fb = tc.forward_backward(data)              # submitted, not awaited
        tc.optim_step(runara.AdamParams(1e-4))      # submitted straight after
        print(fb.result().loss)                     # the loop blocks here

tc.save_state(name="support-v1")
```

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

## What you get out

```python
for checkpoint in sc.list_checkpoints():
    print(checkpoint.name, checkpoint.size_bytes, checkpoint.base_model)
```

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

## Installing

The client has no dependencies at all — it is stdlib, so installing it cannot
conflict with whatever `torch`, `transformers` or `numpy` versions your
training code already pins. The heavy dependencies live in `runara-cookbook`.
