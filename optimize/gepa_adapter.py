"""Use GEPA to improve the Cartwheel system prompt on development cases."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from agent.agent import LITELLM_COURSE_MODELS, SYSTEM_PROMPT_TEMPLATE
from observability.instrument import load_env
from replay.__main__ import make_runner
from replay.harness import replay_case
from replay.rollout import load_cases, world_reset

from optimize.runner import model_provider_key, required_judge_keys, runs_for_case
from optimize.workflow import (
    BUDGET_PATH,
    CASES_PATH,
    RESULTS_DIR,
    SPLIT_PATH,
    now_utc,
    read_json,
    reserve_search_calls,
    validate_split,
    validate_model_selection,
    write_json,
)

REQUIRED_FIELDS = ("{role}", "{user_id}", "{store_id}")
SMOKE_METRIC_CALLS = 4


def litellm_model_id(model: str) -> str:
    """Translate a course model name, e.g. glm-5.2, into the LiteLLM identifier."""
    return LITELLM_COURSE_MODELS.get(model, model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run one development case for a few evaluations without charging the budget",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        help="stop after about this many evaluated case runs (default: the remaining budget)",
    )
    args = parser.parse_args()
    try:
        from gepa.optimize_anything import (
            EngineConfig,
            GEPAConfig,
            ReflectionConfig,
            optimize_anything,
        )
    except ImportError as exc:
        raise SystemExit("run `uv sync --extra optimization` before the GEPA path") from exc

    load_env()
    config = read_json(Path(__file__).with_name("config.json"))
    validate_model_selection(config)
    task_model = config["models"]["development_and_search"]
    reflection_model = litellm_model_id(config["models"]["gepa_reflection"])
    split = read_json(SPLIT_PATH)
    validate_split(split)
    cases_by_id = {case["id"]: case for case in load_cases(CASES_PATH)}
    cases = [cases_by_id[case_id] for case_id in split["development_case_ids"]]
    missing = sorted(key for key in required_judge_keys(cases) if not os.getenv(key))
    for model in (task_model, reflection_model):
        key = model_provider_key(model)
        if key and not os.getenv(key):
            missing.append(key)
    if missing:
        raise SystemExit("GEPA needs " + ", ".join(sorted(set(missing))))

    units = [
        {"case_id": case["id"], "sample": sample + 1}
        for case in cases
        for sample in range(runs_for_case(case))
    ]
    if args.smoke:
        units = units[:1]
    budget = read_json(BUDGET_PATH)
    remaining = budget["maximum_calls"] - budget["used_calls"]
    limit = min(args.max_runs or remaining, remaining)
    # GEPA checks its limit between steps, so one step can run past the limit by
    # up to one full scoring of the development cases plus one reflection batch.
    # Leave that much room so the search never hits the budget check mid-step.
    gepa_limit = SMOKE_METRIC_CALLS if args.smoke else limit - (len(units) + 3)
    if not args.smoke and gepa_limit < len(units):
        raise SystemExit(
            f"GEPA needs at least {2 * len(units) + 3} evaluated case runs, "
            f"but only {limit} are allowed"
        )
    run_dir = RESULTS_DIR / ("gepa-smoke" if args.smoke else "gepa-workspace")

    # GEPA passes each dataset item to the evaluator as the keyword `example`.
    def evaluate(candidate: str, example: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        unit = example
        if not args.smoke:
            reserve_search_calls(1, f"gepa:{unit['case_id']}:{unit['sample']}")
        missing_fields = [field for field in REQUIRED_FIELDS if field not in candidate]
        if missing_fields:
            return 0.0, {
                "failed_checks": ["prompt keeps the injected session fields"],
                "detail": "candidate is missing " + ", ".join(missing_fields),
            }
        case = cases_by_id[unit["case_id"]]
        with tempfile.TemporaryDirectory(prefix="hw8-gepa-") as temp:
            root = Path(temp)
            record = replay_case(
                make_runner(case, root, task_model, candidate), world_reset(root), n=1
            )[0]
        score = 1.0 if record["passed"] else 0.0
        return score, {
            "case_id": case["id"],
            "failed_checks": record.get("failure_modes", []),
            "expected_behavior": case["expected"].get("assertions", []),
        }

    try:
        result = optimize_anything(
            SYSTEM_PROMPT_TEMPLATE,
            evaluator=evaluate,
            dataset=units,
            objective=(
                "Improve the Cartwheel system prompt so the agent passes the development "
                "case checks and the saved Homework 5 judges. Keep the three session fields, "
                "and do not weaken access control, confirmation, or refund safety rules."
            ),
            background=(
                f"The task model is {task_model}. Failed checks and expected behavior are "
                "returned after each evaluated case run. Higher scores are better."
            ),
            config=GEPAConfig(
                engine=EngineConfig(
                    run_dir=str(run_dir),
                    seed=0,
                    max_metric_calls=gepa_limit,
                    parallel=False,
                    max_workers=1,
                    raise_on_exception=True,
                ),
                reflection=ReflectionConfig(reflection_lm=reflection_model),
            ),
        )
    except Exception as exc:
        # Record what happened. GEPA keeps its progress in run_dir and resumes
        # from it when the same command runs again.
        if not args.smoke:
            write_json(
                RESULTS_DIR / "gepa-result.json",
                {
                    "stopped_early": True,
                    "error": str(exc)[:500],
                    "workspace": str(run_dir),
                    "search_budget": read_json(BUDGET_PATH),
                },
            )
        raise SystemExit(
            f"GEPA stopped early: {str(exc)[:300]}\n"
            f"Its progress is saved in {run_dir}. Run the same command again to resume."
        ) from exc
    if args.smoke:
        print(
            json.dumps(
                {
                    "smoke": "passed",
                    "task_model": task_model,
                    "reflection_model": reflection_model,
                    "evaluations": result.total_metric_calls,
                    "candidates": len(result.candidates),
                },
                indent=2,
            )
        )
        return
    prompt_path = RESULTS_DIR / "gepa-best-prompt.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(result.best_candidate.rstrip() + "\n")
    summary = {
        "created_at": now_utc(),
        "task_model": task_model,
        "reflection_model": reflection_model,
        "best_score": result.val_aggregate_scores[result.best_idx],
        "total_evals": result.total_metric_calls,
        "best_prompt": str(prompt_path),
    }
    write_json(
        RESULTS_DIR / "gepa-result.json",
        {
            "candidates": result.candidates,
            "parents": result.parents,
            "development_scores": result.val_aggregate_scores,
            "discovery_eval_counts": result.discovery_eval_counts,
            "best_candidate_index": result.best_idx,
            "total_evaluated_case_runs": result.total_metric_calls,
        },
    )
    write_json(RESULTS_DIR / "gepa-summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
