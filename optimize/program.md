# Cartwheel improve loop program

The setup for `.agents/skills/improve-loop/SKILL.md` in Homework 8.

## Goal

Fix the failure mode named in `optimize/results/target.json` on the development cases.

## Files you may edit

The agent files listed in `optimize/allowlist.txt`. Change one layer per attempt: the prompt (`SYSTEM_PROMPT_TEMPLATE` in `agent/agent.py`), a tool (its description in `agent/agent.py` or its code in `agent/tools.py`), or the harness (the other allowed files). Do not weaken the permission checks in `agent/auth.py`.

## Files you must never edit

`eval_cases/`, `tests/`, `analysis/state/judges/`, and `optimize/state/`. Never run `optimize.runner --split test` or `optimize.frontier`.

## Evaluation

```bash
uv run python -m optimize.runner --split development --candidate <short-name> --search
```

The command prints `score` and `write_pass_5` and saves the result to `optimize/results/latest-development.json`. `run_details` in the result shows each run's reply, failed checks, and judge reasons.

## Keep rule

Keep a change when `score` improves and `write_pass_5` does not drop.

## Budget

150 evaluated case runs in total, tracked in `optimize/state/search_budget.json`. The `--search` flag charges each run and refuses a run that would exceed the budget.

## Log

Append one JSON object per attempt to `optimize/results/improve-loop.jsonl` with `candidate`, `changed_layer`, `changed_files`, `rationale`, `development_score`, `write_pass_5`, `decision`, `git_commit`, and `result_file`.
