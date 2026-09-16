"""Run the same-rule single-channel controls.

Order matters and follows the specification: the degeneracy pre-check of
section 2 is computed and written before any headline number is reported. Its
one gate is section 2.2 -- a candidate that is constant on every outer fold
stops the run. The other three items are records, not gates.

    python -m scc.run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .families import ROOT, SINGLE, SCCPanel, build
import kernel as k  # noqa: E402

SPEC = ROOT / "protocols/SCC_SINGLE_CHANNEL_SPEC.md"
RESULTS = ROOT / "scc_results"
CONST_TOL = 1e-12

# family, panel, D block
CELLS = [
    ("P_only", "costanzo", "D"),
    ("D_only", "costanzo", "D"),
    ("P_only", "jonikas", "D"),
    ("D_only", "jonikas", "D"),
    ("Dplus_only", "jonikas", "D+"),
]

# already in the paper; used only to place the pre-registered bands
REFERENCE = {
    "jonikas": dict(PaD=0.881114, A_match=0.827664, A_rbf=0.810982, frozen_P=0.536443),
    "costanzo": dict(PaD=611, A_match=613, A_rbf=606, frozen_P=597),
}


def spec_sha256():
    return hashlib.sha256(SPEC.read_bytes()).hexdigest()


def fit_candidates(fp, family, st):
    """All 35 candidates, outer predictions, no selection yet."""
    d = fp.data[family]
    inner, outer, reverse = fp._blank()
    for i, theta in enumerate(d.weights):
        fp._pass(d, theta, i * len(k.LAMBDAS), [(inner, outer, reverse)], st)
    return inner, outer, reverse


def statistic(panel, y, score, direction):
    if panel == "costanzo":
        return int(np.sum((direction >= 0) == (y > 0)))
    return float(roc_auc_score(y, score)) if len(set(y)) == 2 else 0.5


def pair_bootstrap(panel, y, score, direction, pair_of_record, draws=4000, seed=20260915):
    """The descriptive pair bootstrap the paper already uses.

    Resamples unordered pairs, so reciprocal records move as a unit.
    """
    rng = np.random.default_rng(seed)
    n_pairs = int(pair_of_record.max()) + 1
    index = [np.flatnonzero(pair_of_record == p) for p in range(n_pairs)]
    vals = []
    for _ in range(draws):
        take = np.concatenate([index[p] for p in rng.integers(0, n_pairs, n_pairs)])
        yy = y[take]
        if panel == "costanzo":
            vals.append(int(np.sum((direction[take] >= 0) == (yy > 0))) * len(y) / len(take))
        else:
            if len(set(yy)) < 2:
                continue
            vals.append(roc_auc_score(yy, score[take]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)], len(vals)


def precheck_cell(fp, family, panel, st, outer, reverse):
    """Specification section 2, for one cell."""
    d = fp.data[family]
    full = np.arange(fp.n)

    # 2.1 kernel rank and conditioning of the two base Gram matrices
    base = d.components(full)[0]
    kernels = []
    for name, M in zip(("linear", "rbf"), base):
        sym = (M + M.T) / 2
        ev = np.linalg.eigvalsh(sym)
        nz = np.abs(ev)
        kernels.append(
            dict(
                kernel=name,
                shape=list(M.shape),
                numerical_rank=int(np.linalg.matrix_rank(M)),
                condition_number=float(np.linalg.cond(M)),
                min_eigenvalue=float(ev.min()),
                max_eigenvalue=float(ev.max()),
                near_zero_eigenvalues=int((nz < 1e-10 * max(nz.max(), 1e-300)).sum()),
            )
        )

    # 2.2 non-constant scores  (THE GATE)
    const = np.zeros((outer.shape[0], fp.M), bool)
    for c in range(outer.shape[0]):
        for m in range(fp.M):
            v = outer[c, st.te[m]]
            const[c, m] = float(np.std(v)) <= CONST_TOL
    anywhere = int(const.any(1).sum())
    everywhere = int(const.all(1).sum())

    # 2.3 support of the output statistic across outer folds
    per_fold = []
    for m in range(fp.M):
        te = st.te[m]
        per_fold.append(statistic(panel, fp.y[te], outer[0, te], (outer[0, te] - reverse[0, te]) / 2))
    return (
        dict(
            base_kernels=kernels,
            candidates=int(outer.shape[0]),
            outer_groups=int(fp.M),
            candidates_constant_on_some_fold=anywhere,
            candidates_constant_on_every_fold=everywhere,
            gate_passed=bool(everywhere == 0),
            statistic_distinct_values_across_outer_folds=int(len(set(per_fold))),
            statistic_per_outer_fold_sample=per_fold[:8],
        ),
        everywhere == 0,
    )


def run(out_dir=None, draws=4000):
    from threadpoolctl import threadpool_limits

    digest = spec_sha256()
    out = Path(out_dir or RESULTS)
    (out / "scc_predictions").mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    precheck, rows, preds = {}, [], {}

    with threadpool_limits(limits=1):
        for family, panel, block in CELLS:
            label = "%s_%s" % (family, panel) + ("" if block == "D" else "_Dplus")
            t0 = time.monotonic()
            fp, dplus_report = build(panel, (family,), block)
            d = fp.data[family]
            # the family factory is a hook; assert it actually fired, and that
            # the candidate mixes over exactly the two base kernels the
            # specification names. A family returning three components with
            # two-element weights would silently drop the third under zip().
            assert getattr(d, "channel", None) == SINGLE[family], (
                "family factory did not fire for %s: got %s" % (family, type(d).__name__)
            )
            assert len(d.weights[0]) == 2 and len(d.components(np.arange(fp.n))[0]) == 2, (
                "%s must mix exactly two base kernels" % family
            )
            st = fp.structure(None)
            assert all(
                np.array_equal(st.tr[m], fp.archived_allowed[m]) for m in range(fp.M)
            ), "exclusion sets differ from the archived ones"

            inner, outer, reverse = fit_candidates(fp, family, st)
            pre, ok = precheck_cell(fp, family, panel, st, outer, reverse)
            pre["label"] = label
            pre["block"] = block
            if dplus_report is not None:
                pre["d_plus"] = {q: dplus_report[q] for q in dplus_report if q != "restored"}
            precheck[label] = pre
            if not ok:
                precheck[label]["stopped"] = (
                    "a candidate is constant on every outer fold; section 6 says report and stop"
                )
                _write_precheck(out, digest, precheck, stopped=label)
                raise SystemExit("PRE-CHECK FAILED for %s -- see scc_precheck.json" % label)

            score, rev, wc, lc = fp._assemble(fp.data[family], inner, outer, reverse, st)
            direction = (score - rev) / 2
            y = fp.y
            # 2.4 selector reachability
            precheck[label]["weights_ever_selected"] = int((wc > 0).sum())
            precheck[label]["weight_selection_counts"] = [int(x) for x in wc]
            precheck[label]["effectively_single_model"] = bool((wc > 0).sum() == 1)

            interval, used = pair_bootstrap(panel, y, score, direction, fp.pair_of_record, draws)
            table = fp.frame[["a", "b", "pair"]].copy()
            table["target"] = y
            table["score"] = score
            table["reverse_score"] = rev
            table["direction"] = direction
            table["support"] = (score + rev) / 2
            path = out / "scc_predictions" / ("%s.csv" % label)
            table.to_csv(path, index=False)
            preds[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

            rows.append(
                dict(
                    spec_sha256=digest,
                    label=label,
                    family=family,
                    panel=panel,
                    block=block,
                    n_records=int(fp.n),
                    n_pairs=int(fp.n_pairs),
                    auroc=float(roc_auc_score(y, score)) if len(set(y)) == 2 else 0.5,
                    average_precision=float(average_precision_score(y, score)) if np.any(y) else 0.0,
                    correct=int(np.sum((direction >= 0) == (y > 0))),
                    statistic=statistic(panel, y, score, direction),
                    ci95_low=interval[0],
                    ci95_high=interval[1],
                    bootstrap_draws=used,
                    **{"w%d" % i: int(c) for i, c in enumerate(wc)},
                    **{"lam%d" % i: int(c) for i, c in enumerate(lc)},
                    predictions_sha256=preds[path.name],
                    seconds=time.monotonic() - t0,
                )
            )
            print(
                "  %-22s %-9s %s = %s   95%% [%.4f, %.4f]   weights used %d/7   %.1fs"
                % (label, panel, "correct" if panel == "costanzo" else "AUROC",
                   rows[-1]["statistic"], interval[0], interval[1],
                   precheck[label]["weights_ever_selected"], rows[-1]["seconds"]),
                flush=True,
            )

    _write_precheck(out, digest, precheck)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "scc_results.csv", index=False)
    (out / "scc_predictions" / "MANIFEST.json").write_text(
        json.dumps(dict(spec_sha256=digest, files=preds), indent=2) + "\n"
    )
    readings = _readings(frame, digest)
    (out / "scc_readings.json").write_text(json.dumps(readings, indent=2) + "\n")
    print("\n  total %.1f s" % (time.monotonic() - start))
    print(json.dumps(readings["outcomes"], indent=2, ensure_ascii=False))
    return frame, readings


def _write_precheck(out, digest, precheck, stopped=None):
    (out / "scc_precheck.json").write_text(
        json.dumps(
            dict(
                spec_sha256=digest,
                section="2",
                gate="2.2 -- a candidate constant on every outer fold stops the run",
                note="2.1 and 2.2 precede any reported result; 2.3 and 2.4 are records, "
                     "and 2.4 is by construction a property of the completed selection",
                stopped_at=stopped,
                cells=precheck,
            ),
            indent=2,
        )
        + "\n"
    )


def _readings(frame, digest):
    """Specification section 3, applied verbatim."""
    g = {r["label"]: r for r in frame.to_dict("records")}
    out = {}
    d_j = g.get("D_only_jonikas", {}).get("statistic")
    if d_j is not None:
        out["J"] = dict(
            D_only_auroc=d_j,
            outcome=("J1" if d_j >= 0.870 else "J2" if d_j >= 0.800 else "J3" if d_j >= 0.650 else "J4"),
        )
    dp = g.get("Dplus_only_jonikas_Dplus", {}).get("statistic")
    if dp is not None and d_j is not None:
        diff = dp - d_j
        out["X"] = dict(
            Dplus_only_auroc=dp, difference=diff,
            outcome=("X1" if diff > 0.02 else "X3" if diff < -0.02 else "X2"),
        )
    d_c = g.get("D_only_costanzo", {}).get("statistic")
    if d_c is not None:
        out["C"] = dict(
            D_only_correct=d_c,
            outcome=("C1" if d_c >= 611 else "C2" if d_c >= 597 else "C3"),
        )
    p_j = g.get("P_only_jonikas", {}).get("statistic")
    p_c = g.get("P_only_costanzo", {}).get("statistic")
    out["P"] = dict(
        P_only_jonikas_auroc=p_j, frozen_P_jonikas=REFERENCE["jonikas"]["frozen_P"],
        P_only_costanzo_correct=p_c, frozen_P_costanzo=REFERENCE["costanzo"]["frozen_P"],
        jonikas_exceeds=None if p_j is None else bool(p_j > REFERENCE["jonikas"]["frozen_P"]),
        costanzo_exceeds=None if p_c is None else bool(p_c > REFERENCE["costanzo"]["frozen_P"]),
        consequence="if P_only materially exceeds the frozen score, every comparison "
                    "against the frozen P row must be restated against P_only",
    )
    return dict(spec_sha256=digest, reference=REFERENCE, outcomes=out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="scc.run", description=__doc__)
    ap.add_argument("--out", default=None)
    ap.add_argument("--draws", type=int, default=4000)
    a = ap.parse_args(argv)
    run(out_dir=a.out, draws=a.draws)
    return 0


if __name__ == "__main__":
    sys.exit(main())
