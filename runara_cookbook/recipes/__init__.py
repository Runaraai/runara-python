"""Worked training loops: supervised (sft), RL on verifiable answers (rl_math), and DPO (dpo).

Each module has a `train(...)` function that takes a TrainingClient and a
renderer, so it can be called from your own code, and a `main()` you can run:

    python -m runara_cookbook.recipes.sft --base-model qwen-2.5-7b --repository Qwen/Qwen2.5-7B-Instruct --data my_data.jsonl
    python -m runara_cookbook.recipes.rl_math --base-model qwen-2.5-7b --repository Qwen/Qwen2.5-7B-Instruct --data problems.jsonl
    python -m runara_cookbook.recipes.dpo --base-model qwen-2.5-7b --repository Qwen/Qwen2.5-7B-Instruct --data pairs.jsonl

They are meant to be read and copied: the loop is yours.
"""
