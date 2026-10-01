"""Offline contract tests for the supplied Homework 8 workflow."""

from __future__ import annotations

import json

import pytest

from agent.agent import render_system_prompt
from agent.auth import AuthContext
from optimize.runner import runs_for_case
from optimize.workflow import (
    make_split,
    mark_dominated,
    reserve_search_calls,
    split_membership_hash,
    validate_model_selection,
)


def _case(number: int, kind: str) -> dict:
    return {
        "id": f"e-{number:03d}",
        "kind": kind,
        "expected": {"checks": []},
    }


def test_hw8_split_is_stable_disjoint_and_twenty_ten() -> None:
    cases = [*(_case(i, "regression") for i in range(1, 21))]
    cases.extend(_case(i, "capability") for i in range(21, 31))
    first = make_split(cases)
    second = make_split(list(reversed(cases)))

    assert first == second
    assert len(first["development_case_ids"]) == 20
    assert len(first["test_case_ids"]) == 10
    assert not set(first["development_case_ids"]) & set(first["test_case_ids"])
    first["membership_sha256"] = split_membership_hash(first)
    assert len(first["membership_sha256"]) == 64


def test_hw8_budget_refuses_an_overrun(tmp_path) -> None:
    path = tmp_path / "budget.json"
    path.write_text(json.dumps({"maximum_calls": 3, "used_calls": 0, "reservations": []}))

    reserve_search_calls(2, "first", path)
    with pytest.raises(ValueError, match="only 1 of 3 remain"):
        reserve_search_calls(2, "second", path)

    assert json.loads(path.read_text())["used_calls"] == 2


def test_hw8_write_case_runs_five_times() -> None:
    write_case = {
        "expected": {"checks": [{"check": "refund_status", "order_id": 1}]}
    }
    read_case = {"expected": {"checks": [{"check": "tool_called", "name": "get_order"}]}}

    assert runs_for_case(write_case) == 5
    assert runs_for_case(read_case) == 1


def test_hw8_frontier_marks_only_strictly_worse_points() -> None:
    configurations = [
        {
            "configuration": 1,
            "score": 0.8,
            "cost_per_100_conversations_usd": 4.0,
        },
        {
            "configuration": 2,
            "score": 0.8,
            "cost_per_100_conversations_usd": 5.0,
        },
        {
            "configuration": 3,
            "score": 0.9,
            "cost_per_100_conversations_usd": 6.0,
        },
        {
            "configuration": 4,
            "score": 0.7,
            "cost_per_100_conversations_usd": 1.0,
        },
    ]

    marked = {
        row["configuration"]: row["dominated"]
        for row in mark_dominated(configurations)
    }

    assert marked == {1: False, 2: True, 3: False, 4: False}


def test_hw8_prompt_candidate_keeps_injected_session_fields() -> None:
    context = AuthContext(user_id=7, role="merchant", store_id=2)
    rendered = render_system_prompt(
        context, "Role {role}; user {user_id}; store {store_id}."
    )

    assert rendered == "Role merchant; user 7; store 2."


def test_hw8_model_selection_keeps_the_development_model_in_the_comparison() -> None:
    config = {
        "models": {
            "development_and_search": "model-c",
            "comparison": ["model-a", "model-b", "model-c"],
            "gepa_reflection": "model-r",
        }
    }
    validate_model_selection(config)

    config["models"]["comparison"][2] = "model-d"
    with pytest.raises(ValueError, match="third comparison model"):
        validate_model_selection(config)


def test_hw10_trace_normalization_keeps_provider_cost_fields() -> None:
    from analysis.helpers.normalization import normalize_trace

    trace = normalize_trace(
        {
            "id": "cost-trace-1",
            "input": "question",
            "output": "answer",
            "observations": [
                {
                    "id": "generation-1",
                    "type": "GENERATION",
                    "name": "support-agent",
                    "model": "model-a",
                    "output": "answer",
                    "usageDetails": {
                        "input": 80,
                        "input_cached_tokens": 20,
                        "output": 10,
                    },
                    "costDetails": {"input": 0.01, "output": 0.02},
                    "totalCost": 0.03,
                    "latency": 0.8,
                    "timeToFirstToken": 0.2,
                }
            ],
        }
    )

    observation = trace["observations"][0]
    assert observation["usage_details"]["input_cached_tokens"] == 20
    assert observation["cost_details"] == {"input": 0.01, "output": 0.02}
    assert observation["total_cost"] == 0.03
    assert observation["latency_seconds"] == 0.8
    assert observation["time_to_first_token_seconds"] == 0.2
