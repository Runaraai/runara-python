# Changelog

## 0.2.0 (unreleased)

RL and preference training. Code written against 0.1.x keeps working.

- **Losses.** `forward_backward(data, loss_fn, loss_fn_config)` takes
  `importance_sampling`, `ppo`, `cispo` and `dro` (RL), and `dpo` (preference
  pairs), as well as `cross_entropy`. The formulas are Tinker's, so its recipes
  and hyperparameters transfer.
  - RL losses sum over tokens, as Tinker does. `cross_entropy` keeps its
    per-datum mean unless you pass `{"reduction": "sum"}`.
- **RL data.** `Datum` takes `logprobs` and `advantages`; `weights` is now
  optional. For `dpo` it can also take `ref_logprobs`. By default the reference
  is the base model your session started from.
- **`forward(data, reference=False)`** returns each target token's logprob
  without training. With `reference=True` it scores against the base model.
- **Sampling.** `sample(prompts, num_samples=..., sampling_params=SamplingParams(...))`
  returns `samples[prompt][i]`, each with tokens, per-token logprobs and a stop
  reason. `SamplingParams` adds `top_p`, `top_k`, `stop` (strings or token ids)
  and `seed`. The logprobs are the learner's, so on-policy importance ratios
  are exactly 1.
- **Optimizer.** `AdamParams(grad_clip_norm=...)` clips gradients. Each step's
  gradient norm lands in `TrainingClient.last_optim_metrics`.
- **Safety.** An `optim_step` queued behind a failed `forward_backward` is
  refused, so it cannot apply a stale or partial gradient. The failure is
  reported once, then cleared.
- **Older workers.** A session on a worker that predates these features
  raises a clear error naming the feature, instead of failing opaquely. One
  that ignored `loss_fn` would have trained cross-entropy without saying so.
- **Checkpoints.** Each one carries the `uri` that `load_state` takes, and
  `Checkpoint` ignores fields it doesn't know.
- **Prices.** `ModelInfo` gains `gpu_options` and
  `checkpoint_storage_usd_per_gb_month`. Fine-tuning is billed per GPU-hour and
  for checkpoint storage; the per-token price fields are always 0.

## 0.1.1

Made the quickstart work against a real backend.

## 0.1.0

First release: supervised LoRA fine-tuning.
