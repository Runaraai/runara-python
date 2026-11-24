"""What every recipe's main() shares: arguments, the tokenizer, the session."""

from __future__ import annotations

import argparse

import runara

from .. import renderers


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--base-model", required=True, help="base_model from list_models(), e.g. qwen-2.5-7b")
    p.add_argument("--repository", required=True, help="its HuggingFace repository, for the tokenizer")
    p.add_argument("--data", required=True, help="a JSONL file on this machine")
    p.add_argument("--renderer", default="generic", help=f"one of {', '.join(renderers.available())}")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--save-as", default=None, help="save the adapter under this name at the end")
    return p


def renderer_for(args) -> renderers.Renderer:
    # Imported here so the cookbook's helpers work without transformers.
    from transformers import AutoTokenizer

    return renderers.get(args.renderer, AutoTokenizer.from_pretrained(args.repository))


def training_client(args) -> runara.TrainingClient:
    return runara.ServiceClient().create_lora_training_client(base_model=args.base_model, rank=args.rank)


def log(record: dict) -> None:
    print("  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in record.items()), flush=True)
