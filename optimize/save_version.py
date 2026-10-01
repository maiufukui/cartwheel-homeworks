"""Record the starting version or the selected final version for Homework 8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from optimize.workflow import (
    BUDGET_PATH,
    FINAL_VERSION_PATH,
    RESULTS_DIR,
    SPLIT_PATH,
    STARTING_VERSION_PATH,
    canonical_hash,
    current_commit,
    grader_hash,
    now_utc,
    read_json,
    validate_split,
    validate_model_selection,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("starting", "final"))
    parser.add_argument(
        "--development-result",
        type=Path,
        default=RESULTS_DIR / "latest-development.json",
    )
    parser.add_argument(
        "--layer", choices=("prompt", "tool", "harness"), action="append"
    )
    parser.add_argument("--method", choices=("manual", "gepa", "improve-loop"), action="append")
    args = parser.parse_args()

    split = read_json(SPLIT_PATH)
    validate_split(split)
    result = read_json(args.development_result)
    config = read_json(Path(__file__).with_name("config.json"))
    validate_model_selection(config)
    development_model = config["models"]["development_and_search"]
    model_selection_sha256 = canonical_hash(config["models"])
    commit = current_commit()
    destination = STARTING_VERSION_PATH if args.kind == "starting" else FINAL_VERSION_PATH
    if destination.exists():
        raise SystemExit(f"{destination} already exists and must not be replaced")
    if args.kind == "final" and not STARTING_VERSION_PATH.exists():
        raise SystemExit("record the starting version before saving the final version")
    if args.kind == "final":
        starting_version = read_json(STARTING_VERSION_PATH)
        if starting_version["model_selection_sha256"] != model_selection_sha256:
            raise SystemExit("the selected models changed after the starting version was saved")
    if result["split"] != "development":
        raise SystemExit("the saved version needs a development result")
    if result["git_commit"] != commit:
        raise SystemExit("rerun the development command on the current commit")
    if split["grader_sha256"] != grader_hash():
        raise SystemExit("the cases, evaluation tests, or saved judges changed after the split")
    if result["model"] != development_model:
        raise SystemExit(
            f"the development result must use {development_model}, as selected in "
            "optimize/config.json"
        )

    record = {
        "kind": args.kind,
        "created_at": now_utc(),
        "git_commit": commit,
        "development_result": str(args.development_result),
        "development_model": development_model,
        "comparison_models": config["models"]["comparison"],
        "model_selection_sha256": model_selection_sha256,
        "development_score": result["score"],
        "write_pass_5": result["write_pass_5"],
        "prompt_sha256_12": result["prompt_sha256_12"],
        "grader_sha256": split["grader_sha256"],
        "split_membership_sha256": split["membership_sha256"],
    }
    if args.kind == "final":
        if not args.layer or not args.method:
            raise SystemExit("the final version needs --layer and --method")
        record["changed_layers"] = args.layer
        record["method"] = args.method
        record["search_budget"] = read_json(BUDGET_PATH)
    write_json(destination, record)
    print(json.dumps(record, indent=2))
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
