# Homework 7 monitoring — insufficient_order_context

**Judge:** `insufficient_order_context-v1` (frozen HW5)  
**Cartwheel model:** `gpt-5.5`  
**Scenarios:** 50 in `scenarios/monitoring_scenarios.jsonl` (HW3 set)  
**Sampling:** 20% random (`random_rate` 0.2 → 10 traces per period) + `policy_lookup` risk group  
**Threshold:** corrected prevalence **0.15** (`monitoring/config.json`)

Period comparison uses Langfuse windows in `config.json` (**before** = HW3 run, **after** = Sep 2026 re-run). Scheduled CI uses `--last-hours 24` on a self-hosted runner against local Langfuse.

---

## 1. Did the corrected failure estimate move between the two periods?

**No meaningful movement.** Both periods gave the same point estimates:

| Period | Raw (random 10) | Corrected | 95% CI |
|--------|-----------------|-----------|--------|
| before | 0.8 | 0.5704 | 0.0 – 1.0 |
| after  | 0.8 | 0.5704 | 0.0 – 1.0 |

So the judge flagged the same share of the random sample (8/10) and Rogan–Gladen correction landed on the same corrected rate for both deployment windows.

---

## 2. Do the intervals support a conclusion, or is the result uncertain?

**Uncertain.** With only **n = 10** random traces per period, the bootstrap interval spans **0.0–1.0** for both runs. That width reflects (a) small sample size and (b) judge calibration on the held-out split (failure sensitivity ≈ 0.91, pass specificity ≈ 0.34 for this judge), which adds extra uncertainty in the correction.

We cannot claim a statistically precise drop or rise between before and after; the data are compatible with many true prevalence values.

---

## 3. What did the risk groups reveal that the random estimate did not?

The **`policy_lookup`** group pulls every conversation where the agent called `get_policy` or `search_help_center` (12 traces per period here), regardless of the random draw. Those traces are **not** used in prevalence math; they are for **deeper inspection** of policy-heavy threads where `insufficient_order_context` may be more likely.

The random sample alone answers “how often does this mode fail in the population?” The risk group answers “show me more policy-tool conversations to read and possibly send to error analysis,” including cases that never appeared in the random 10.

---

## 4. What action should happen if the estimate crosses the threshold?

Configured threshold: **0.15** corrected prevalence. Both periods estimate **~0.57**, above the line, so under our policy we would **start error analysis** on traces flagged by the monitor (random and/or risk verdict scores in Langfuse), confirm failures with human review, and **mint new Homework 6 evaluation cases** for confirmed patterns—not change production behavior from a single noisy point estimate alone.

If a future run crossed 0.15 with a **narrower** interval, the same pipeline applies: pull flagged traces → label → add regression/capability cases → re-run HW6 CI.

---

## Artifacts

- `history.jsonl` — period-level counts and prevalence  
- `prevalence.svg` — before/after vs threshold  
- `artifacts/{before,after}/` — random and risk verdict JSON  
- `.github/workflows/monitor.yml` — daily + manual monitor (self-hosted runner + local Langfuse)
