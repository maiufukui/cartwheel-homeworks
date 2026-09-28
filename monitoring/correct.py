"""Bias-corrected failure prevalence for a monitoring period."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _rogan_failure_point(
    sample_preds: np.ndarray,
    test_labels: np.ndarray,
    test_preds: np.ndarray,
) -> float:
    """Rogan-Gladen failure prevalence (1 = failure present)."""
    labels_pass = 1 - test_labels.astype(float)
    preds_pass = 1 - test_preds.astype(float)
    unlabeled_pass = 1 - sample_preds.astype(float)

    positive = labels_pass == 1
    negative = labels_pass == 0
    if not positive.any() or not negative.any() or not unlabeled_pass.size:
        return float("nan")
    tpr = preds_pass[positive].mean()
    tnr = (1 - preds_pass[negative]).mean()
    denominator = tpr + tnr - 1
    if denominator == 0:
        return float("nan")
    observed = unlabeled_pass.mean()
    pass_rate = (observed + tnr - 1) / denominator
    pass_rate = float(min(max(pass_rate, 0.0), 1.0))
    return 1 - pass_rate


def corrected_mode_prevalence(
    sample_preds: Sequence[int],
    test_labels: Sequence[int],
    test_preds: Sequence[int],
    confidence: float = 0.95,
    bootstrap_iterations: int = 20000,
    seed: int | None = 7,
) -> dict[str, Any]:
    """Bias-corrected live prevalence for one mode from sampled verdicts.

    The contract, precisely:

      1. ``raw`` is the uncorrected flag rate: ``mean(sample_preds)``.
      2. Compute the frozen judge's failure sensitivity and pass specificity
         from ``test_labels`` and ``test_preds``. Both use the monitoring
         convention that 1 means a failure is present. Failure sensitivity is
         the flagged fraction of human-labeled failures. Pass specificity is
         the unflagged fraction of human-labeled passes.
      3. Compute the Rogan-Gladen point estimate, then resample the held-out
         records and sampled predictions to obtain a percentile-bootstrap
         interval. Use a seeded NumPy generator so the committed result is
         reproducible.
      4. Resample the monitoring predictions and the paired held-out records
         independently with replacement. Keep their original sample sizes.
         Discard a draw if the correction cannot be computed. Clamp each
         retained estimate to [0, 1], then take the percentile interval.
         Raise ``ValueError`` if no replicate is valid.

    Args:
        sample_preds: the judge's 0/1 verdicts over the UNIFORM BASE sample
            only (never the risk strata; they are biased toward failure by
            design).
        test_labels: human labels for the frozen Homework 5 judge's test
            split.
        test_preds: the frozen judge's predictions on that test split.
        confidence: interval confidence level.
        bootstrap_iterations: number of percentile-bootstrap replicates.
        seed: numpy seed for a reproducible interval; None leaves the RNG
            untouched.

    Returns:
        {"raw", "corrected", "ci_low", "ci_high", "confidence",
         "failure_sensitivity", "pass_specificity", "n_sample"}
        with "corrected" clamped to [0, 1] and rates rounded to 4 places.

    Raises:
        ValueError: if an input is empty, the held-out inputs have different
            lengths, a value is not 0 or 1, a class is absent, the judge is
            missing a usable correction, or no bootstrap replicate is valid.
    """
    sample = np.asarray(list(sample_preds), dtype=int)
    labels = np.asarray(list(test_labels), dtype=int)
    preds = np.asarray(list(test_preds), dtype=int)

    if sample.size == 0 or labels.size == 0 or preds.size == 0:
        raise ValueError("sample and held-out inputs must be non-empty")
    if labels.size != preds.size:
        raise ValueError("held-out labels and predictions must match")
    for name, values in (
        ("sample_preds", sample),
        ("test_labels", labels),
        ("test_preds", preds),
    ):
        if not np.isin(values, [0, 1]).all():
            raise ValueError(f"{name} must contain only 0 or 1")

    failure_mask = labels == 1
    pass_mask = labels == 0
    if not failure_mask.any() or not pass_mask.any():
        raise ValueError("held-out split must contain both failures and passes")

    failure_sensitivity = float(preds[failure_mask].mean())
    pass_specificity = float((1 - preds[pass_mask]).mean())

    raw = float(sample.mean())
    corrected = _rogan_failure_point(sample, labels, preds)
    if np.isnan(corrected):
        raise ValueError("judge correction is not usable for these inputs")
    corrected = float(min(max(corrected, 0.0), 1.0))

    rng = np.random.default_rng(seed)
    bootstrap = np.empty(bootstrap_iterations, dtype=float)
    for index in range(bootstrap_iterations):
        test_indices = rng.integers(0, labels.size, labels.size)
        sample_indices = rng.integers(0, sample.size, sample.size)
        bootstrap[index] = _rogan_failure_point(
            sample[sample_indices],
            labels[test_indices],
            preds[test_indices],
        )
    bootstrap = bootstrap[~np.isnan(bootstrap)]
    if bootstrap.size == 0:
        raise ValueError("no valid bootstrap replicates")
    bootstrap = np.clip(bootstrap, 0.0, 1.0)
    alpha = (1 - confidence) / 2
    ci_low = float(np.quantile(bootstrap, alpha))
    ci_high = float(np.quantile(bootstrap, 1 - alpha))

    return {
        "raw": round(raw, 4),
        "corrected": round(corrected, 4),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
        "confidence": confidence,
        "failure_sensitivity": round(failure_sensitivity, 4),
        "pass_specificity": round(pass_specificity, 4),
        "n_sample": int(sample.size),
    }
