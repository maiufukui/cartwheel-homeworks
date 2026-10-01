"""Create the fixed Homework 8 case split and search budget."""

from __future__ import annotations

import json

from replay.rollout import load_cases

from optimize.workflow import (
    BUDGET_PATH,
    CASES_PATH,
    REPO_ROOT,
    SPLIT_PATH,
    grader_hash,
    make_split,
    now_utc,
    sha256_file,
    split_membership_hash,
    validate_model_selection,
    write_json,
)


def main() -> None:
    if SPLIT_PATH.exists() or BUDGET_PATH.exists():
        raise SystemExit(
            "Homework 8 state already exists. Keep the existing split and budget; "
            "do not reset them."
        )
    validate_model_selection(
        json.loads((REPO_ROOT / "optimize" / "config.json").read_text())
    )
    cases = load_cases(CASES_PATH)
    kinds = {
        kind: sum(case["kind"] == kind for case in cases)
        for kind in ("regression", "capability")
    }
    if len(cases) < 6 or any(kinds[kind] < 2 for kind in kinds):
        raise SystemExit(
            "Homework 8 needs at least six evaluation cases, including at least "
            "two regression cases and two capability cases. Add cases before "
            "creating the development and test split."
        )
    split = make_split(cases)
    split.update(
        {
            "created_at": now_utc(),
            "cases_sha256": sha256_file(CASES_PATH),
            "grader_sha256": grader_hash(),
        }
    )
    split["membership_sha256"] = split_membership_hash(split)
    write_json(SPLIT_PATH, split)
    write_json(
        BUDGET_PATH,
        {"maximum_calls": 150, "used_calls": 0, "reservations": []},
    )
    print(
        json.dumps(
            {
                "development_cases": len(split["development_case_ids"]),
                "test_cases": len(split["test_case_ids"]),
                "search_budget": 150,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
