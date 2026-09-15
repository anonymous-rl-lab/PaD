"""Test statistics and permutation summaries (protocol sections 5 and 6)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def auroc(y, s):
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else 0.5


def average_precision(y, s):
    return float(average_precision_score(y, s)) if np.any(y) else 0.0


def correct(y, direction):
    return int(np.sum((direction >= 0) == (y > 0)))


def jonikas_statistics(y, out, reference=None):
    """Primary AUROC gain plus the secondary statistics of Section 5."""
    pad, add = out["PaD"], out["A_match"]
    row = dict(
        delta=auroc(y, pad["score"]) - auroc(y, add["score"]),
        auroc_pad=auroc(y, pad["score"]),
        auroc_amatch=auroc(y, add["score"]),
        delta_ap=average_precision(y, pad["score"]) - average_precision(y, add["score"]),
    )
    pos = y > 0
    repaired = int(np.sum(pos & (pad["direction"] >= 0) & (add["direction"] < 0)))
    harmed = int(np.sum(pos & (pad["direction"] < 0) & (add["direction"] >= 0)))
    row["direction_repairs_minus_harms"] = repaired - harmed
    if "A_rbf" in out:
        row["auroc_arbf"] = auroc(y, out["A_rbf"]["score"])
        row["delta_vs_arbf"] = row["auroc_pad"] - row["auroc_arbf"]
    if reference is not None:
        row["delta_vs_p_only"] = row["auroc_pad"] - auroc(y, reference)
    return row


def costanzo_statistics(y, out, reference=None):
    pad, add = out["PaD"], out["A_match"]
    row = dict(
        delta=float(correct(y, pad["direction"]) - correct(y, add["direction"])),
        correct_pad=correct(y, pad["direction"]),
        correct_amatch=correct(y, add["direction"]),
        auroc_pad=auroc(y, pad["score"]),
        auroc_amatch=auroc(y, add["score"]),
    )
    if "A_rbf" in out:
        row["correct_arbf"] = correct(y, out["A_rbf"]["direction"])
        row["delta_vs_arbf"] = float(row["correct_pad"] - row["correct_arbf"])
    if reference is not None:
        row["delta_vs_p_only"] = float(row["correct_pad"] - correct(y, reference))
    return row


def selection_columns(out, prefix_map):
    row = {}
    for family, prefix in prefix_map.items():
        if family not in out:
            continue
        for i, c in enumerate(out[family]["weight_counts"]):
            row["%s_w%d" % (prefix, i)] = int(c)
        for i, c in enumerate(out[family]["lambda_counts"]):
            row["%s_l%d" % (prefix, i)] = int(c)
    return row


def summarize(observed, deltas, two_sided_floor=True):
    """One-sided right-tail permutation p-value and the flexibility offset.

    ``p = (1 + #{ Delta_r >= Delta_obs }) / (1 + R)`` is exact in finite samples
    and needs no large-sample argument.  ``delta_flex`` is the mean of the null,
    that is what the same learner buys from flexibility alone on data with no
    correspondence to exploit; it is reported as a scientific quantity, not a
    diagnostic.
    """
    deltas = np.asarray(deltas, float)
    R = len(deltas)
    p_one = (1 + int(np.sum(deltas >= observed))) / (1 + R)
    p_left = (1 + int(np.sum(deltas <= observed))) / (1 + R)
    p_two = min(1.0, 2 * min(p_one, p_left)) if two_sided_floor else 2 * min(p_one, p_left)
    sd = float(deltas.std(ddof=1)) if R > 1 else float("nan")
    flex = float(deltas.mean())
    return dict(
        replicates=R,
        delta_observed=float(observed),
        delta_flex=flex,
        null=dict(
            mean=flex,
            sd=sd,
            q025=float(np.percentile(deltas, 2.5)),
            q500=float(np.percentile(deltas, 50)),
            q975=float(np.percentile(deltas, 97.5)),
            minimum=float(deltas.min()),
            maximum=float(deltas.max()),
        ),
        standardized_effect=float((observed - flex) / sd) if sd and np.isfinite(sd) and sd > 0 else None,
        p_one_sided=p_one,
        p_two_sided=p_two,
        resolution_floor=1.0 / (1 + R),
    )


def holm(pvalues):
    """Holm step-down adjusted p-values, returned in input order."""
    items = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(items):
        value = min(1.0, (len(pvalues) - rank) * pvalues[i])
        running = max(running, value)
        adjusted[i] = running
    return adjusted
