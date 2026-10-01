"""Plan and run the one Homework 8 test batch."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from observability.instrument import load_env
from optimize.workflow import (
    CASES_PATH,
    FINAL_VERSION_PATH,
    REPO_ROOT,
    RESULTS_DIR,
    SPLIT_PATH,
    STARTING_VERSION_PATH,
    TEST_PLAN_PATH,
    TEST_RUN_PATH,
    canonical_hash,
    grader_hash,
    mark_dominated,
    now_utc,
    read_json,
    validate_split,
    validate_model_selection,
    write_json,
)

def make_plan() -> dict[str, Any]:
    validate_split(read_json(SPLIT_PATH))
    starting_version = read_json(STARTING_VERSION_PATH)
    final_version = read_json(FINAL_VERSION_PATH)
    config = read_json(REPO_ROOT / "optimize" / "config.json")
    validate_model_selection(config)
    models = config["models"]["comparison"]
    current_grader_hash = grader_hash()
    if (
        starting_version["grader_sha256"] != current_grader_hash
        or final_version["grader_sha256"] != current_grader_hash
    ):
        raise ValueError("the score definition changed after a version was saved")
    if starting_version["split_membership_sha256"] != final_version["split_membership_sha256"]:
        raise ValueError("the development and test membership differs between saved versions")
    model_selection_sha256 = canonical_hash(config["models"])
    if (
        starting_version["model_selection_sha256"] != model_selection_sha256
        or final_version["model_selection_sha256"] != model_selection_sha256
    ):
        raise ValueError("the selected models changed after a version was saved")
    plan = {
        "created_at": now_utc(),
        "configurations": [
            {
                "configuration": 1,
                "name": "starting-version-model-1",
                "parent": "",
                "git_commit": starting_version["git_commit"],
                "model": models[0],
            },
            {
                "configuration": 2,
                "name": "starting-version-model-2",
                "parent": "",
                "git_commit": starting_version["git_commit"],
                "model": models[1],
            },
            {
                "configuration": 3,
                "name": "starting-version-model-3",
                "parent": "",
                "git_commit": starting_version["git_commit"],
                "model": models[2],
            },
            {
                "configuration": 4,
                "name": "final-version-model-3",
                "parent": "starting-version-model-3",
                "git_commit": final_version["git_commit"],
                "model": models[2],
            },
        ],
    }
    plan["plan_sha256"] = canonical_hash(plan)
    return plan


def plan_command() -> None:
    if TEST_RUN_PATH.exists():
        raise SystemExit("the test batch has already started")
    if TEST_PLAN_PATH.exists():
        raise SystemExit("the test plan already exists and must not be replaced")
    plan = make_plan()
    write_json(TEST_PLAN_PATH, plan)
    print(json.dumps(plan, indent=2))


def run_one_configuration(
    configuration: dict[str, Any], plan: dict[str, Any], output: Path
) -> dict[str, Any]:
    git_root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    number = configuration["configuration"]
    with tempfile.TemporaryDirectory(prefix=f"hw8-configuration-{number}-") as temp:
        worktree = Path(temp) / "repo"
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(worktree), configuration["git_commit"]],
            cwd=git_root,
            check=True,
            capture_output=True,
            text=True,
        )
        try:
            worktree_cartwheel = worktree / REPO_ROOT.resolve().relative_to(git_root.resolve())
            env = os.environ.copy()
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["PYTHONPATH"] = os.pathsep.join(
                [str(worktree_cartwheel), str(REPO_ROOT), env.get("PYTHONPATH", "")]
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "optimize.runner",
                    "--split",
                    "test",
                    "--candidate",
                    configuration["name"],
                    "--model",
                    configuration["model"],
                    "--cases-file",
                    str(CASES_PATH),
                    "--split-file",
                    str(SPLIT_PATH),
                    "--config",
                    str(REPO_ROOT / "optimize" / "config.json"),
                    "--output",
                    str(output),
                    "--test-plan-hash",
                    plan["plan_sha256"],
                ],
                cwd=worktree_cartwheel,
                env=env,
                check=True,
            )
        finally:
            subprocess.run(
                ["git", "worktree", "remove", str(worktree)],
                cwd=git_root,
                check=True,
                capture_output=True,
                text=True,
            )
    return read_json(output)


def run_command() -> None:
    load_env()  # the worktrees have no .env, so pass the keys through os.environ
    plan = read_json(TEST_PLAN_PATH)
    plan_without_hash = {
        key: value for key, value in plan.items() if key != "plan_sha256"
    }
    expected_hash = canonical_hash(plan_without_hash)
    if plan["plan_sha256"] != expected_hash:
        raise SystemExit("the saved test plan changed after it was created")
    state = read_json(TEST_RUN_PATH) if TEST_RUN_PATH.exists() else {
        "plan_sha256": plan["plan_sha256"],
        "started_at": now_utc(),
        "completed_configurations": [],
    }
    if state["plan_sha256"] != plan["plan_sha256"]:
        raise SystemExit("the existing test run belongs to a different plan")
    write_json(TEST_RUN_PATH, state)

    configurations: list[dict[str, Any]] = []
    for configuration in plan["configurations"]:
        number = configuration["configuration"]
        output = RESULTS_DIR / f"test-configuration-{number}.json"
        if number not in state["completed_configurations"]:
            result = run_one_configuration(configuration, plan, output)
            state["completed_configurations"].append(number)
            write_json(TEST_RUN_PATH, state)
        else:
            result = read_json(output)
        configurations.append(
            {
                "configuration": number,
                "name": configuration["name"],
                "parent": configuration["parent"],
                "git_commit": configuration["git_commit"],
                "model": configuration["model"],
                "score": result["score"],
                "write_pass_5": result["write_pass_5"],
                "cost_per_100_conversations_usd": result["cost_per_100_conversations_usd"],
                "median_latency_seconds": result["median_latency_seconds"],
            }
        )

    marked = mark_dominated(configurations)
    csv_path = RESULTS_DIR / "frontier.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(marked[0]))
        writer.writeheader()
        writer.writerows(marked)
    state["completed_at"] = now_utc()
    state["frontier_csv"] = str(csv_path)
    write_json(TEST_RUN_PATH, state)
    print(json.dumps(marked, indent=2))
    print(f"Saved {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "run"))
    args = parser.parse_args()
    plan_command() if args.command == "plan" else run_command()


if __name__ == "__main__":
    main()
