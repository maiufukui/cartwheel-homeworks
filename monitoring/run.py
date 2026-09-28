"""Fetch Langfuse traces, sample, judge, correct, and write monitoring scores."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from analysis.helpers.normalization import _flatten
from analysis.review_app.grouping import cluster_by_conversation
from monitoring.correct import corrected_mode_prevalence
from monitoring.run_judges import judge_sample, judge_test_data
from monitoring.sample import DEFAULT_RISK_GROUPS, select_traces
from monitoring.write_scores import build_score_records, post_scores
from scenarios.validate import load_jsonl

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "monitoring" / "config.json"
SCENARIOS_PATH = REPO_ROOT / "scenarios" / "monitoring_scenarios.jsonl"
ARTIFACTS_DIR = REPO_ROOT / "monitoring" / "artifacts"
HISTORY_PATH = REPO_ROOT / "monitoring" / "history.jsonl"


def _parse_instant(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _in_window(timestamp: str | None, start: datetime, end: datetime) -> bool:
    if not timestamp:
        return False
    instant = _parse_instant(str(timestamp))
    return start <= instant <= end


def _tool_names(messages: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for message in messages:
        if message.get("role") != "tool_call":
            continue
        name = message.get("name")
        if not name:
            continue
        cleaned = str(name).split(".")[-1]
        names.add(cleaned)
    return sorted(names)


def _user_turn_count(messages: list[dict[str, Any]]) -> int:
    return sum(1 for message in messages if message.get("role") == "user")


def _merge_trace_group(group: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge ordered traces into one conversation; id is the final trace."""
    group = sorted(group, key=lambda trace: trace.get("timestamp") or "")
    if len(group) == 1:
        conversation = dict(group[0])
    else:
        conversation = dict(group[0])
        for later in group[1:]:
            conversation["trace"] = conversation["trace"] + later["trace"]
            conversation["observations"] = conversation["observations"] + later[
                "observations"
            ]
            conversation["models"] = list(
                dict.fromkeys(conversation["models"] + later["models"])
            )
        conversation["text"] = _flatten(conversation["trace"])
        conversation["features"] = dict(conversation.get("features") or {})
        conversation["features"]["turn_count"] = sum(
            message.get("role") in {"user", "assistant"}
            for message in conversation["trace"]
        )
        conversation["features"]["tool_call_count"] = sum(
            message.get("role") == "tool_call" for message in conversation["trace"]
        )
    final = group[-1]
    conversation["id"] = final["id"]
    conversation["trace_id"] = final["trace_id"]
    conversation["text"] = conversation.get("text") or _flatten(conversation["trace"])
    conversation["tools"] = _tool_names(conversation["trace"])
    conversation["turn_count"] = _user_turn_count(conversation["trace"])
    return conversation


def _models_on_trace(trace: dict[str, Any]) -> list[str]:
    models = list(trace.get("models") or [])
    stats = trace.get("stats") or {}
    if stats.get("model"):
        models.append(str(stats["model"]))
    return models


def _model_matches(actual: str, expected: str) -> bool:
    if actual == expected:
        return True
    prefix = expected.rstrip("/") + "-"
    if actual.startswith(prefix):
        return True
    if actual.startswith(f"{expected}/"):
        return True
    return False


def _validate_models(traces: list[dict[str, Any]], expected_model: str) -> None:
    for trace in traces:
        for model in _models_on_trace(trace):
            if model and not _model_matches(model, expected_model):
                raise ValueError(
                    f"trace {trace.get('id')} used model {model!r}, expected {expected_model!r}"
                )


def _fetch_normalized_traces(limit: int = 10000) -> list[dict[str, Any]]:
    from analysis.helpers import langfuse_io

    if not langfuse_io.is_configured():
        raise langfuse_io.LangfuseNotConfigured(
            "set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST "
            "to fetch traces for monitoring"
        )
    return langfuse_io.fetch_traces(limit=limit)


def build_scenario_conversations(
    traces: list[dict[str, Any]],
    *,
    start: datetime,
    end: datetime,
    expected_ids: set[str],
    expected_model: str,
) -> list[dict[str, Any]]:
    """One merged conversation per scenario in the time window."""
    in_window = [
        trace
        for trace in traces
        if _in_window(trace.get("timestamp"), start, end)
    ]
    _validate_models(in_window, expected_model)

    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in in_window:
        scenario = (trace.get("meta") or {}).get("scenario_id")
        if scenario:
            by_scenario[str(scenario)].append(trace)

    missing = sorted(expected_ids - set(by_scenario))
    if missing:
        raise ValueError(
            f"period is missing {len(missing)} scenario conversations: {missing[:5]}"
        )

    conversations = [
        _merge_trace_group(by_scenario[scenario_id])
        for scenario_id in sorted(expected_ids)
    ]
    if len(conversations) != len(expected_ids):
        raise ValueError("period must produce exactly one conversation per scenario")
    return conversations


def build_session_conversations(
    traces: list[dict[str, Any]],
    *,
    start: datetime,
    end: datetime,
    expected_model: str,
) -> list[dict[str, Any]]:
    """Group traces in the window by session into conversations."""
    in_window = [
        trace
        for trace in traces
        if _in_window(trace.get("timestamp"), start, end)
    ]
    _validate_models(in_window, expected_model)
    groups = cluster_by_conversation(in_window)
    return [_merge_trace_group(group) for group in groups if group]


def _load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _period_config(config: dict[str, Any], label: str) -> dict[str, Any]:
    for period in config.get("periods") or []:
        if period.get("label") == label:
            return period
    raise ValueError(f"unknown period label {label!r}")


def _configured_risk_groups(config: dict[str, Any]) -> dict[str, Any]:
    names = config.get("risk_groups") or []
    missing = [name for name in names if name not in DEFAULT_RISK_GROUPS]
    if missing:
        raise ValueError(f"unknown risk groups in config: {missing}")
    return {name: DEFAULT_RISK_GROUPS[name] for name in names}


def _split_verdicts(
    plan: dict[str, Any], verdicts: dict[str, int]
) -> tuple[dict[str, int], dict[str, int]]:
    random_ids = [str(trace["id"]) for trace in plan["random"]]
    random_verdicts = {trace_id: verdicts[trace_id] for trace_id in random_ids}

    risk_ids: list[str] = []
    seen: set[str] = set()
    for name in plan["risk_groups"]:
        for trace in plan["risk_groups"][name]:
            trace_id = str(trace["id"])
            if trace_id not in seen:
                seen.add(trace_id)
                risk_ids.append(trace_id)
    risk_verdicts = {trace_id: verdicts[trace_id] for trace_id in risk_ids}
    return random_verdicts, risk_verdicts


def _save_verdicts(
    batch_label: str, random_verdicts: dict[str, int], risk_verdicts: dict[str, int]
) -> None:
    directory = ARTIFACTS_DIR / batch_label
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "random_verdicts.json").write_text(
        json.dumps(random_verdicts, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (directory / "risk_verdicts.json").write_text(
        json.dumps(risk_verdicts, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_history(record: dict[str, Any]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if HISTORY_PATH.exists():
        for line in HISTORY_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    label = record.get("period")
    rows = [row for row in rows if row.get("period") != label]
    rows.append(record)
    HISTORY_PATH.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _update_chart(config: dict[str, Any]) -> None:
    if not HISTORY_PATH.exists():
        return
    points = []
    for line in HISTORY_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        points.append(
            {
                "label": row["period"],
                "corrected": row["corrected"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
            }
        )
    from monitoring.chart import prevalence_chart

    svg = prevalence_chart(
        points,
        threshold=float(config.get("threshold", 0.15)),
        mode=str(config.get("judge_mode", "failure_mode")),
    )
    (REPO_ROOT / "monitoring" / "prevalence.svg").write_text(svg, encoding="utf-8")


def _print_plan_summary(
    config: dict[str, Any],
    *,
    batch_label: str,
    conversation_count: int,
    trace_count: int,
    plan: dict[str, Any],
) -> None:
    random_count = len(plan["random"])
    risk_count = sum(len(traces) for traces in plan["risk_groups"].values())
    judge_calls = len(plan["to_judge"])
    print(f"batch: {batch_label}")
    print(f"cartwheel model: {config.get('model')}")
    print(f"judge: {config.get('judge_id')} ({config.get('judge_mode')})")
    print(f"agent conversations in window: {conversation_count}")
    print(f"langfuse traces in window: {trace_count}")
    print(f"random sample size: {random_count}")
    print(f"risk-group trace selections (with duplicates): {risk_count}")
    print(f"judge calls (union): {judge_calls}")


def run_period(
    config: dict[str, Any],
    period: dict[str, Any],
    *,
    judge: bool,
    write_scores: bool,
) -> None:
    expected_ids = {record["id"] for record in load_jsonl(SCENARIOS_PATH)}
    start = _parse_instant(period["from"])
    end = _parse_instant(period["to"])
    batch_label = str(period["label"])
    expected_model = str(config["model"])

    raw_traces = _fetch_normalized_traces()
    trace_count = sum(
        1 for trace in raw_traces if _in_window(trace.get("timestamp"), start, end)
    )
    conversations = build_scenario_conversations(
        raw_traces,
        start=start,
        end=end,
        expected_ids=expected_ids,
        expected_model=expected_model,
    )

    risk_groups = _configured_risk_groups(config)
    plan = select_traces(
        conversations,
        random_rate=float(config["random_rate"]),
        risk_groups=risk_groups,
        seed=7,
    )
    _print_plan_summary(
        config,
        batch_label=batch_label,
        conversation_count=len(conversations),
        trace_count=trace_count,
        plan=plan,
    )
    if not judge:
        print("Skipping judge calls (pass --judge after you approve).")
        return

    verdicts = judge_sample(str(config["judge_id"]), plan["to_judge"])
    random_verdicts, risk_verdicts = _split_verdicts(plan, verdicts)
    _save_verdicts(batch_label, random_verdicts, risk_verdicts)

    test_labels, test_preds = judge_test_data(str(config["judge_id"]))
    estimate = corrected_mode_prevalence(
        [random_verdicts[str(trace["id"])] for trace in plan["random"]],
        test_labels,
        test_preds,
    )
    mode = str(config["judge_mode"])
    records = build_score_records(
        mode, random_verdicts, risk_verdicts, estimate, batch_label
    )
    if write_scores:
        post_scores(records)

    history = {
        "period": batch_label,
        "judge_id": config["judge_id"],
        "model": expected_model,
        "trace_count": trace_count,
        "conversation_count": len(conversations),
        "random_sample_count": len(plan["random"]),
        "risk_sample_count": len(risk_verdicts),
        "raw": estimate["raw"],
        "corrected": estimate["corrected"],
        "ci_low": estimate["ci_low"],
        "ci_high": estimate["ci_high"],
        "failure_sensitivity": estimate["failure_sensitivity"],
        "pass_specificity": estimate["pass_specificity"],
    }
    _append_history(history)
    _update_chart(config)
    print(
        f"corrected prevalence: {estimate['corrected']} "
        f"[{estimate['ci_low']}, {estimate['ci_high']}] (raw {estimate['raw']})"
    )


def run_last_hours(
    config: dict[str, Any],
    hours: int,
    *,
    judge: bool,
    write_scores: bool,
) -> None:
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    batch_label = end.strftime("%Y-%m-%dT%H:%MZ")
    expected_model = str(config["model"])

    raw_traces = _fetch_normalized_traces()
    trace_count = sum(
        1 for trace in raw_traces if _in_window(trace.get("timestamp"), start, end)
    )
    conversations = build_session_conversations(
        raw_traces,
        start=start,
        end=end,
        expected_model=expected_model,
    )

    if not conversations:
        print(f"batch: {batch_label}")
        print(f"cartwheel model: {expected_model}")
        print("agent conversations in window: 0")
        print("langfuse traces in window:", trace_count)
        print("No eligible conversations; skipping judge.")
        history = {
            "period": batch_label,
            "judge_id": config["judge_id"],
            "model": expected_model,
            "trace_count": trace_count,
            "conversation_count": 0,
            "random_sample_count": 0,
            "risk_sample_count": 0,
            "raw": 0.0,
            "corrected": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "failure_sensitivity": None,
            "pass_specificity": None,
        }
        _append_history(history)
        _update_chart(config)
        return

    risk_groups = _configured_risk_groups(config)
    plan = select_traces(
        conversations,
        random_rate=float(config["random_rate"]),
        risk_groups=risk_groups,
        seed=7,
    )
    _print_plan_summary(
        config,
        batch_label=batch_label,
        conversation_count=len(conversations),
        trace_count=trace_count,
        plan=plan,
    )
    if not judge:
        print("Skipping judge calls (pass --judge after you approve).")
        return

    verdicts = judge_sample(str(config["judge_id"]), plan["to_judge"])
    random_verdicts, risk_verdicts = _split_verdicts(plan, verdicts)
    _save_verdicts(batch_label, random_verdicts, risk_verdicts)

    test_labels, test_preds = judge_test_data(str(config["judge_id"]))
    estimate = corrected_mode_prevalence(
        [random_verdicts[str(trace["id"])] for trace in plan["random"]],
        test_labels,
        test_preds,
    )
    mode = str(config["judge_mode"])
    records = build_score_records(
        mode, random_verdicts, risk_verdicts, estimate, batch_label
    )
    if write_scores:
        post_scores(records)

    history = {
        "period": batch_label,
        "judge_id": config["judge_id"],
        "model": expected_model,
        "trace_count": trace_count,
        "conversation_count": len(conversations),
        "random_sample_count": len(plan["random"]),
        "risk_sample_count": len(risk_verdicts),
        "raw": estimate["raw"],
        "corrected": estimate["corrected"],
        "ci_low": estimate["ci_low"],
        "ci_high": estimate["ci_high"],
        "failure_sensitivity": estimate["failure_sensitivity"],
        "pass_specificity": estimate["pass_specificity"],
    }
    _append_history(history)
    _update_chart(config)
    print(
        f"corrected prevalence: {estimate['corrected']} "
        f"[{estimate['ci_low']}, {estimate['ci_high']}] (raw {estimate['raw']})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Cartwheel monitoring job.")
    parser.add_argument(
        "--period",
        choices=["before", "after"],
        help="Run one configured comparison period (50 scenarios).",
    )
    parser.add_argument(
        "--last-hours",
        type=int,
        metavar="N",
        help="Run over the previous N hours, grouping by session id.",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Call the frozen judge for --period runs (paid).",
    )
    parser.add_argument(
        "--write-scores",
        action="store_true",
        help="Post score records to Langfuse after judging (--period runs).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="For --last-hours, print counts only (no judge or Langfuse scores).",
    )
    args = parser.parse_args()
    if bool(args.period) == bool(args.last_hours):
        parser.error("specify exactly one of --period or --last-hours")

    config = _load_config()
    if args.period:
        period = _period_config(config, args.period)
        run_period(config, period, judge=args.judge, write_scores=args.write_scores)
        return
    run_last_hours(
        config,
        int(args.last_hours),
        judge=not args.dry_run,
        write_scores=not args.dry_run,
    )


if __name__ == "__main__":
    main()
