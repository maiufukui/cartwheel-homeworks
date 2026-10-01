"""Check that every selected model can finish one case before the split is made.

Run this before ``optimize.prepare``. After preparation the model selection is
locked, so a model that fails here should be replaced in optimize/config.json
first. The check does not use the search budget.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from observability.instrument import load_env
from replay.__main__ import make_runner
from replay.harness import ReplayInfraError, replay_case
from replay.rollout import load_cases, world_reset

from optimize.runner import read_prices
from optimize.workflow import CASES_PATH, read_json, validate_model_selection

CONFIG_PATH = Path(__file__).with_name("config.json")


def main() -> None:
    load_env()
    config = read_json(CONFIG_PATH)
    validate_model_selection(config)
    cases = load_cases(CASES_PATH)
    if not cases:
        raise SystemExit("eval_cases/cases.jsonl has no cases")
    case = cases[0]
    results = {}
    for model in config["models"]["comparison"]:
        try:
            read_prices(CONFIG_PATH, model)
            with tempfile.TemporaryDirectory(prefix="hw8-check-") as temp:
                root = Path(temp)
                record = replay_case(
                    make_runner(case, root, model), world_reset(root), n=1
                )[0]
        except (ReplayInfraError, ValueError) as exc:
            results[model] = {"ok": False, "problem": str(exc)[:300]}
            continue
        agent_errors = [m for m in record["failure_modes"] if m.startswith("agent_error")]
        if agent_errors:
            results[model] = {"ok": False, "problem": record.get("error", agent_errors[0])}
        else:
            results[model] = {"ok": True, "case": case["id"], "passed": record["passed"]}
    print(json.dumps(results, indent=2))
    failed = [model for model, result in results.items() if not result["ok"]]
    if failed:
        raise SystemExit(
            "Replace these models in optimize/config.json before running "
            "optimize.prepare: " + ", ".join(failed)
        )
    print("Every selected model finished one case.")


if __name__ == "__main__":
    main()
