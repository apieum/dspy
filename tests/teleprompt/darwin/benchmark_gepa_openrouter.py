"""Opt-in apples-to-apples benchmark for Darwin and official DSPy GEPA.

This file is intentionally not named ``test_*.py``: it never runs as part of
the normal test suite and makes real OpenRouter requests only when invoked
directly.

Example::

    OPENROUTER_API_KEY=... \
    OPENROUTER_MODEL=google/gemini-2.0-flash-001 \
    .venv/bin/python tests/teleprompt/darwin/benchmark_gepa_openrouter.py

Use ``--budget`` to cap each optimizer's optimization budget. The script runs
mutation-only first so the comparison isolates the GEPA implementations; use
``--merge`` for a second comparison including official merge behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from copy import deepcopy
from typing import Any

import dspy

from dspy.teleprompt.darwin import GEPAMute


def make_dataset() -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Return a small deterministic arithmetic train/validation split."""
    train = [
        ("What is 7 + 5?", "12"),
        ("What is 9 + 6?", "15"),
        ("What is 8 + 4?", "12"),
        ("What is 13 - 5?", "8"),
        ("What is 18 - 9?", "9"),
        ("What is 6 * 3?", "18"),
    ]
    validation = [
        ("What is 11 + 7?", "18"),
        ("What is 14 - 6?", "8"),
        ("What is 5 * 4?", "20"),
        ("What is 16 + 3?", "19"),
    ]

    def examples(items):
        return [
            dspy.Example(question=question, answer=answer).with_inputs("question")
            for question, answer in items
        ]

    return examples(train), examples(validation)


def score(example, prediction) -> float:
    answer = str(getattr(prediction, "answer", "")).strip()
    return float(answer == str(example.answer).strip())


def official_metric(gold, prediction, trace=None, pred_name=None, pred_trace=None) -> float:
    return score(gold, prediction)


def darwin_metric(example, prediction, trace=None):
    value = score(example, prediction)
    feedback = "The answer is correct." if value else "The answer is incorrect; solve the arithmetic carefully."
    return value, feedback


def make_lm(model: str, api_key: str) -> dspy.LM:
    if not model.startswith("openrouter/"):
        model = f"openrouter/{model}"
    return dspy.LM(
        model,
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
        temperature=0.0,
        max_tokens=512,
        cache=False,
    )


def usage(lm: dspy.LM) -> dict[str, int]:
    totals = {"requests": len(getattr(lm, "history", [])), "prompt_tokens": 0, "completion_tokens": 0}
    for record in getattr(lm, "history", []):
        record_usage = record.get("usage", {}) if isinstance(record, dict) else {}
        totals["prompt_tokens"] += int(record_usage.get("prompt_tokens", 0) or 0)
        totals["completion_tokens"] += int(record_usage.get("completion_tokens", 0) or 0)
    totals["total_tokens"] = totals["prompt_tokens"] + totals["completion_tokens"]
    return totals


def run_darwin(trainset, valset, lm, budget: int) -> dict[str, Any]:
    student = dspy.Predict("question -> answer")
    optimizer = GEPAMute(
        metric=darwin_metric,
        max_calls=budget,
        minibatch_size=3,
        patience=2,
        verbose=False,
    )
    started = time.perf_counter()
    with dspy.context(lm=lm):
        compiled = optimizer.compile(student, trainset=trainset, valset=valset)
    elapsed = time.perf_counter() - started
    result = optimizer.get_last_result()
    history = getattr(result, "history", []) if result is not None else []
    return {
        "optimizer": "darwin",
        "final_score": result.candidates[0].average_score() if result and result.candidates else None,
        "seed_score": history[0].get("best_score") if history else None,
        "generations": len(history),
        "elapsed_seconds": round(elapsed, 2),
        "lm_usage": usage(lm),
        "final_instruction": getattr(compiled.predictors()[0].signature, "instructions", ""),
    }


def run_official(trainset, valset, task_lm, reflection_lm, budget: int, use_merge: bool) -> dict[str, Any]:
    student = dspy.Predict("question -> answer")
    optimizer = dspy.GEPA(
        metric=official_metric,
        reflection_lm=reflection_lm,
        max_metric_calls=budget,
        reflection_minibatch_size=3,
        use_merge=use_merge,
        max_merge_invocations=2,
        seed=1,
        track_stats=True,
    )
    started = time.perf_counter()
    with dspy.context(lm=task_lm):
        compiled = optimizer.compile(student, trainset=trainset, valset=valset)
    elapsed = time.perf_counter() - started
    details = getattr(compiled, "detailed_results", None)
    scores = list(getattr(details, "val_aggregate_scores", []) or [])
    return {
        "optimizer": "official_dspy_gepa",
        "final_score": max(scores) if scores else None,
        "seed_score": scores[0] if scores else None,
        "metric_calls": getattr(details, "total_metric_calls", None),
        "elapsed_seconds": round(elapsed, 2),
        "task_lm_usage": usage(task_lm),
        "reflection_lm_usage": usage(reflection_lm),
        "final_instruction": getattr(compiled.predictors()[0].signature, "instructions", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=24, help="Optimization budget per optimizer")
    parser.add_argument("--merge", action="store_true", help="Enable official GEPA merge proposals")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    model = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required; this benchmark makes paid requests.")
    if args.budget <= 0:
        raise SystemExit("--budget must be positive")

    trainset, valset = make_dataset()
    darwin_lm = make_lm(model, api_key)
    official_task_lm = make_lm(model, api_key)
    official_reflection_lm = make_lm(model, api_key)

    results = [
        run_darwin(trainset, valset, darwin_lm, args.budget),
        run_official(trainset, valset, official_task_lm, official_reflection_lm, args.budget, args.merge),
    ]
    print(json.dumps({"model": model, "budget_per_optimizer": args.budget, "merge": args.merge, "results": results}, indent=2))


if __name__ == "__main__":
    main()
