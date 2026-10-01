"""Assemble a replay: `uv run python -m replay e-001 --n 100`.

Instructor-provided. Wires :func:`replay.harness.replay_case` and
:func:`replay.harness.summarize_rollouts` to the real agent and the real
world reset, exactly as the docstrings describe:

  - the reset is :func:`replay.rollout.world_reset` (re-seed the same
    directory before every rollout, so no write leaks between rollouts),
  - the runner plays the case with :func:`replay.rollout.run_case`, scores
    it with :func:`replay.rollout.apply_checks` (plus the case's pinned
    judges when API keys are present), and maps transport errors to
    :class:`replay.harness.ReplayInfraError` so only infrastructure gets
    retried, never a verdict.

Needs an API key for the agent's model and the judge's, if the case names
judges). Writes one JSON record per rollout to replay/results/<case>.jsonl
and prints the summary.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from agents.exceptions import MaxTurnsExceeded, ModelBehaviorError

from replay.harness import ReplayInfraError, replay_case, summarize_rollouts
from replay.rollout import (
    apply_checks,
    judge_reply_with_text,
    load_cases,
    load_frozen_judge,
    retrieved_docs_text,
    run_case,
    world_reset,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "replay" / "results"

# Transport-shaped exceptions that count as infrastructure, not verdicts.
_INFRA_ERRNOS = (
    "timeout",
    "timed out",
    "rate limit",
    "429",
    "connection",
    "disconnected",
    "500",
    "502",
    "503",
    "504",
    "529",
    "internal server error",
    "service unavailable",
    "overloaded",
)


def judge_provider_key(model: str) -> str | None:
    """Name the API key a judge's model needs, from its provider prefix."""
    from agent.agent import LITELLM_COURSE_MODELS

    model = LITELLM_COURSE_MODELS.get(model, model)
    if model.startswith(("gpt", "o1", "o3", "o4", "openai/")):
        return "OPENAI_API_KEY"
    if model.startswith(("claude", "anthropic/")):
        return "ANTHROPIC_API_KEY"
    if model.startswith(("together_ai/", "zai-org/")):
        return "TOGETHER_API_KEY"
    if model.startswith(("gemini", "gemini/")):
        return "GEMINI_API_KEY"
    return None


def _is_infra_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _INFRA_ERRNOS)


def make_runner(
    case: dict[str, Any],
    world_root: Path,
    model: str | None,
    prompt_template: str | None = None,
) -> Any:
    """One rollout: run the case on the (already reset) world, apply the
    code checks and judges, and return the harness record."""
    db_path = world_root / "cartwheel.db"
    judges = {mode: load_frozen_judge(mode) for mode in case["expected"].get("judges", {})}
    for mode, judge in judges.items():
        key = judge_provider_key(judge["model"])
        if key and not os.environ.get(key):
            raise ValueError(f"the {mode} judge uses {judge['model']} and needs {key}")

    def runner() -> dict[str, Any]:
        try:
            started = time.perf_counter()
            transcript = run_case(case, model=model, prompt_template=prompt_template)
            agent_latency_seconds = time.perf_counter() - started
        except (ModelBehaviorError, MaxTurnsExceeded) as exc:
            # The model misbehaved, e.g., it called a tool that does not
            # exist. That is a failing verdict for this run, not an error.
            return {
                "passed": False,
                "failure_modes": [f"agent_error:{type(exc).__name__}"],
                "error": str(exc)[:500],
                "steps": 0,
                "tool_calls": [],
                "usage": {"requests": 0, "input_tokens": 0, "output_tokens": 0},
                "final_reply": "",
                "agent_latency_seconds": time.perf_counter() - started,
            }
        except Exception as exc:  # noqa: BLE001 - classified right below
            if _is_infra_error(exc):
                raise ReplayInfraError(str(exc)) from exc
            raise
        outcome = apply_checks(case, transcript, db_path)
        failure_modes = list(outcome["failed"])
        docs = retrieved_docs_text(transcript)
        judge_reasons = {}
        for mode, judge in judges.items():
            expected = case["expected"]["judges"][mode]
            try:
                verdict, judge_reasons[mode] = judge_reply_with_text(
                    judge, transcript["final_reply"], docs
                )
                judge_reasons[mode] = judge_reasons[mode][:1000]
            except Exception as exc:  # noqa: BLE001
                if _is_infra_error(exc):
                    raise ReplayInfraError(str(exc)) from exc
                raise
            if verdict != expected:
                failure_modes.append(f"judge:{mode}")
        return {
            "passed": not failure_modes,
            "failure_modes": failure_modes,
            "steps": transcript["steps"],
            "tool_calls": [
                c["name"] for t in transcript["turns"] for c in t["tool_calls"]
            ],
            "usage": transcript["usage"],
            "final_reply": transcript["final_reply"],
            "judge_reasons": judge_reasons,
            "agent_latency_seconds": agent_latency_seconds,
        }

    return runner


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay one evaluation case n times.")
    parser.add_argument("case_id", help="evaluation case id, e.g. e-001")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    cases = {c["id"]: c for c in load_cases()}
    if args.case_id not in cases:
        raise SystemExit(f"no evaluation case {args.case_id!r} in eval_cases/cases.jsonl")
    case = cases[args.case_id]

    world_root = Path(tempfile.mkdtemp(prefix=f"replay-{case['id']}-"))
    reset = world_reset(world_root)
    runner = make_runner(case, world_root, args.model)

    records = replay_case(runner, reset, n=args.n)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{case['id']}.jsonl"
    with open(out_path, "w") as out:
        for record in records:
            out.write(json.dumps(record) + "\n")

    summary = summarize_rollouts(records)
    print(json.dumps(summary, indent=2))
    print(f"\n{summary['failures']}/{summary['n']} rollouts failed "
          f"(rate {summary['failure_rate']:.3f}, "
          f"CI {summary['ci_low']:.3f}-{summary['ci_high']:.3f}). "
          f"Records: {out_path}")
    print("Use the measured rate to decide what to do. A high, consistent rate is a "
          "regression to fix and block in CI. A rare failure should become an evaluation case.")


if __name__ == "__main__":
    main()
