---
name: improve-loop
description: Run an autoresearch loop that hill climbs on a project's evaluation score. Edit only allowed files, one change at a time, rerun the evaluation, keep a change only when the score improves, and log every attempt until the budget is spent. Use when a project provides a program.md that names the editable files, the evaluation command, and the budget.
---

# Improve loop

An autoresearch loop for any project. The human writes `program.md`; you run the loop it describes.

## Setup

1. Find the project's `program.md` (e.g., `optimize/program.md`) and read it fully. It defines:
   - the files you may edit,
   - the files you must never edit (the evaluator, the tests, the held-out cases, and the budget state),
   - the evaluation command and the metric it reports,
   - the keep rule, the budget, and the log file.
2. If any of the above is missing, stop and ask the human. Never guess an evaluation command or a metric.
3. Run the evaluation once on the current code to record a baseline score, and log it as the current best.

## Loop

1. Read the latest evaluation result and pick one failure to fix.
2. Make one small change in the allowed files only. Prefer deleting or simplifying code over adding more.
3. Commit the change, then run the evaluation command.
4. Keep the commit only if the keep rule in `program.md` is met. Otherwise revert the edited files to the last kept commit.
5. Append one line to the log: the change, the files changed, a one-line rationale, the score, the decision, the commit, and the result file.
6. Stop when the budget is spent, when two changes in a row do not improve the score, or when the score cannot improve further.

## Rules

- Never edit the evaluator, the tests, the held-out cases, or the budget, and never run the held-out cases.
- Never change a setting to make the evaluation easier, e.g., a timeout, a seed, or a sample size.
- Never weaken access controls or other safety checks to raise the score.
- Treat a small score gain with suspicion. When the evaluation set is small, rerun a promising change before keeping it.

Finish with a report: the changes kept, the score before and after, the budget used, and the failures that remain.
