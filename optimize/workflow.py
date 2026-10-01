"""Small, testable rules shared by the Homework 8 commands."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = REPO_ROOT / "optimize" / "state"
RESULTS_DIR = REPO_ROOT / "optimize" / "results"
CASES_PATH = REPO_ROOT / "eval_cases" / "cases.jsonl"
SPLIT_PATH = STATE_DIR / "split.json"
BUDGET_PATH = STATE_DIR / "search_budget.json"
STARTING_VERSION_PATH = STATE_DIR / "starting_version.json"
FINAL_VERSION_PATH = STATE_DIR / "final_version.json"
TEST_PLAN_PATH = STATE_DIR / "test_plan.json"
TEST_RUN_PATH = STATE_DIR / "test_run.json"


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def git_value(*args: str, cwd: Path = REPO_ROOT) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def current_commit(cwd: Path = REPO_ROOT) -> str:
    return git_value("rev-parse", "HEAD", cwd=cwd)


def grader_hash(root: Path = REPO_ROOT) -> str:
    """Hash every file that defines the Homework 8 scores."""
    paths = [root / "eval_cases" / "cases.jsonl"]
    for folder in (root / "tests" / "eval", root / "analysis" / "state" / "judges"):
        if folder.exists():
            paths.extend(path for path in folder.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def is_expected_write(case: dict[str, Any]) -> bool:
    """Whether a case expects the agent to change data, e.g., issue a refund."""
    from replay.rollout import WRITE_TOOLS

    return any(
        check["check"] == "refund_status"
        or (check["check"] == "tool_called" and check.get("name") in WRITE_TOOLS)
        for check in case["expected"].get("checks", [])
    )


def make_split(cases: list[dict[str, Any]], seed: str = "cartwheel-hw8-v1") -> dict[str, Any]:
    """Place about one third of each (kind, writes data) group in the test set."""
    if len(cases) < 6:
        raise ValueError("Homework 8 needs at least six evaluation cases")
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation case identifiers must be unique")
    groups: dict[tuple[str, bool], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        groups[(case["kind"], is_expected_write(case))].append(case)
    test_ids: list[str] = []
    for (kind, writes), group in sorted(groups.items()):
        ordered = sorted(
            group,
            key=lambda case: sha256_bytes(f"{seed}:{kind}:{case['id']}".encode()),
        )
        # A group of one stays in development; larger groups keep at least one
        # case on each side.
        test_count = 0 if len(group) < 2 else min(len(group) - 1, max(1, round(len(group) / 3)))
        test_ids.extend(case["id"] for case in ordered[:test_count])
    all_ids = {case["id"] for case in cases}
    test_set = set(test_ids)
    return {
        "seed": seed,
        "development_case_ids": sorted(all_ids - test_set),
        "test_case_ids": sorted(test_set),
    }


def split_membership_hash(split: dict[str, Any]) -> str:
    return canonical_hash(
        {
            "seed": split["seed"],
            "development_case_ids": split["development_case_ids"],
            "test_case_ids": split["test_case_ids"],
        }
    )


def validate_split(split: dict[str, Any], cases_path: Path = CASES_PATH) -> None:
    current_hash = sha256_file(cases_path)
    if split.get("cases_sha256") != current_hash:
        raise ValueError("eval_cases/cases.jsonl changed after the Homework 8 split was made")
    development = set(split["development_case_ids"])
    test = set(split["test_case_ids"])
    if development & test:
        raise ValueError("the development and test case lists overlap")
    if split.get("membership_sha256") != split_membership_hash(split):
        raise ValueError("the Homework 8 development or test membership changed")


def validate_model_selection(config: dict[str, Any]) -> None:
    models = config.get("models", {})
    comparison = models.get("comparison", [])
    development = models.get("development_and_search")
    if len(comparison) != 3 or len(set(comparison)) != 3:
        raise ValueError("optimize/config.json must list three different comparison models")
    if comparison[2] != development:
        raise ValueError(
            "the third comparison model must match the development and search model"
        )
    if not models.get("gepa_reflection"):
        raise ValueError("optimize/config.json must name a GEPA reflection model")


def reserve_search_calls(required: int, label: str, path: Path = BUDGET_PATH) -> dict[str, Any]:
    """Reserve calls before a search run, so a failed run cannot hide spend."""
    budget = read_json(path)
    used = int(budget["used_calls"])
    limit = int(budget["maximum_calls"])
    if required < 1:
        raise ValueError("required calls must be positive")
    if used + required > limit:
        raise ValueError(
            f"search needs {required} calls, but only {limit - used} of {limit} remain"
        )
    budget["used_calls"] = used + required
    budget.setdefault("reservations", []).append(
        {"label": label, "calls": required, "reserved_at": now_utc()}
    )
    write_json(path, budget)
    return budget


def release_search_calls(count: int, label: str, path: Path = BUDGET_PATH) -> dict[str, Any]:
    """Return runs that were reserved but never executed, e.g., after a crash."""
    budget = read_json(path)
    budget["used_calls"] = max(0, int(budget["used_calls"]) - count)
    budget.setdefault("reservations", []).append(
        {"label": label, "calls": -count, "reserved_at": now_utc(), "note": "released after a failed run"}
    )
    write_json(path, budget)
    return budget


def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Return whether left is at least as good and no more costly than right."""
    cost_field = "cost_per_100_conversations_usd"
    no_worse = left["score"] >= right["score"] and left[cost_field] <= right[cost_field]
    strictly_better = left["score"] > right["score"] or left[cost_field] < right[cost_field]
    return no_worse and strictly_better


def mark_dominated(configurations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    marked = []
    for configuration in configurations:
        row = dict(configuration)
        row["dominated"] = any(
            other["configuration"] != configuration["configuration"]
            and dominates(other, configuration)
            for other in configurations
        )
        marked.append(row)
    return marked
