"""Panel drivers: observed statistic, placebo reference, permutation replicates.

Protocol v1.2.  Two things differ from a textbook permutation test and are
named in every output record.

The exclusion is provenance adaptive (section 4.6).  Permuting D moves where a
record's measurements came from without moving its name, so the archived
name-based endpoint exclusion no longer bounds what the training set knows
about the test record's D provenance.  Every replicate therefore rebuilds the
nested structure from the union of the name-sharing and provenance-sharing
graphs.

That denser exclusion shrinks training pools under the null but not for the
untouched observed statistic, which would bias the comparison.  The reference
the p-value is measured against is therefore placebo matched: ``K`` draws that
keep the real D blocks in place but apply the exclusion graph of a permutation.
The untouched observed value is reported alongside for continuity with the
paper, and is not the reference.

The resulting p-value is approximate, not exact.  Records sharing genes are not
exchangeable, and the exclusion graphs of the null and placebo sides are equal
in law but not identical.
"""
from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import dplus, operators, stats
from .frozen import ROOT, FrozenPanel

RESULTS = ROOT / "ccct_results"
PREFIX = {"PaD": "pad", "A_match": "am", "A_rbf": "arbf"}
PRIMARY_VARIANT = {"jonikas": "primary", "costanzo": "V1_fold"}

_STATE = {}


def reference_p_only(panel):
    path = ROOT / "reference" / panel / "pair_predictions.csv"
    return pd.read_csv(path, float_precision="round_trip").P_frozen.to_numpy()


def statistics(panel, y, out, reference):
    fn = stats.jonikas_statistics if panel == "jonikas" else stats.costanzo_statistics
    row = fn(y, out, reference)
    row.update(stats.selection_columns(out, PREFIX))
    row["min_train_pairs"] = int(min(out["_pool_sizes"]))
    row["mean_train_pairs"] = float(np.mean(out["_pool_sizes"]))
    return row


def make_panel(panel, families, block="D"):
    fp = FrozenPanel(panel, families)
    report = None
    if block == "D+":
        if panel != "jonikas":
            raise ValueError("D+ is defined for the Jonikas panel only")
        xd, xdr, odd, report = dplus.build(fp.frame)
        fp.install_base(xd, xdr, odd)
    else:
        fp.restore()
    return fp, report


def _init(panel, families, variant, block, base_seed, replicates, placebo):
    from threadpoolctl import threadpool_limits

    _STATE["limits"] = threadpool_limits(limits=1)
    fp, _ = make_panel(panel, families, block)
    op = operators.build(panel, fp.frame, fp.original_block(), variant=variant)
    _STATE.update(
        panel=panel,
        fp=fp,
        op=op,
        base=fp.original_block(),
        reference=reference_p_only(panel),
        children=np.random.SeedSequence(base_seed).spawn(replicates),
        placebo_children=np.random.SeedSequence(base_seed + 1).spawn(max(placebo, 1)),
    )


def _one(kind_r):
    kind, r = kind_r
    fp, op = _STATE["fp"], _STATE["op"]
    if kind == "null":
        rng = np.random.Generator(np.random.PCG64(_STATE["children"][r]))
        block, order = op(rng)
        fp.set_D(*block)
    else:  # placebo: real D, permuted exclusion graph
        rng = np.random.Generator(np.random.PCG64(_STATE["placebo_children"][r]))
        order = op.draw(rng)
        fp.set_D(*_STATE["base"])
    out = fp.run(order=order)
    row = statistics(_STATE["panel"], fp.y, out, _STATE["reference"])
    row["r"] = r
    row["kind"] = kind
    return row


def _map(jobs, workers, label, total, start):
    rows = []
    if workers > 1:
        import multiprocessing as mp

        with mp.get_context("fork").Pool(workers) as pool:
            for row in pool.imap_unordered(_one, jobs, chunksize=1):
                rows.append(row)
                if len(rows) % 20 == 0:
                    print(
                        "  %s %d/%d  %.1f min" % (label, len(rows), total, (time.monotonic() - start) / 60),
                        flush=True,
                    )
    else:
        for job in jobs:
            rows.append(_one(job))
    return rows


def exclusion_bites(fp, op, draws=8, seed=101):
    """Does the provenance rule remove anything beyond the archived exclusion?"""
    rng = np.random.Generator(np.random.PCG64(seed))
    for _ in range(draws):
        st = fp.structure(op.draw(rng))
        for m in range(fp.M):
            if not np.array_equal(st.tr[m], fp.archived_allowed[m]):
                return True
    return False


def costanzo_equivalence(y, out, delta_flex, region=13.0, draws=10000, seed=20260915):
    """Paired instance bootstrap for ``Delta_C - delta_flex_C`` (section 5)."""
    pad = (out["PaD"]["direction"] >= 0) == (y > 0)
    add = (out["A_match"]["direction"] >= 0) == (y > 0)
    d = pad.astype(float) - add.astype(float)
    n = len(d)
    rng = np.random.default_rng(seed)
    boot = np.array([d[rng.integers(0, n, n)].sum() for _ in range(draws)])
    lo, hi = np.percentile(boot - delta_flex, [2.5, 97.5])
    return dict(
        statistic=float(d.sum() - delta_flex),
        interval=[float(lo), float(hi)],
        region=[-region, region],
        equivalence_holds=bool(lo >= -region and hi <= region),
        bootstrap_draws=draws,
        reading=(
            "additive suffices on Costanzo"
            if (lo >= -region and hi <= region)
            else "no evidence of a correspondence gain"
        ),
    )


def run_panel(
    panel,
    replicates,
    base_seed,
    variant=None,
    block="D",
    families=("PaD", "A_match"),
    workers=1,
    out_dir=None,
    protocol_sha256=None,
    label=None,
    placebo=100,
):
    variant = variant or PRIMARY_VARIANT[panel]
    start = time.monotonic()
    fp, block_report = make_panel(panel, families, block)
    op = operators.build(panel, fp.frame, fp.original_block(), variant=variant)
    reference = reference_p_only(panel)

    # ---- replicate zero -----------------------------------------------------
    st0 = fp.structure(None)
    archived_match = all(
        np.array_equal(st0.tr[m], fp.archived_allowed[m]) for m in range(fp.M)
    )
    if not archived_match:
        raise AssertionError("identity provenance graph differs from the archived exclusion")
    observed_out = fp.run(structure=st0)
    observed = statistics(panel, fp.y, observed_out, reference)
    block_ident, order_ident = op.identity()
    fp.set_D(*block_ident)
    identity = statistics(panel, fp.y, fp.run(order=order_ident), reference)
    if any(
        not np.array_equal(np.asarray(observed[key]), np.asarray(identity[key]))
        for key in observed
    ):
        raise AssertionError("identity permutation did not reproduce the unpermuted run")
    fp.set_D(*fp.original_block())

    bites = exclusion_bites(fp, op)
    n_placebo = placebo if bites else 0

    _init(panel, families, variant, block, base_seed, replicates, n_placebo)
    jobs = [("placebo", r) for r in range(n_placebo)] + [("null", r) for r in range(replicates)]
    rows = _map(jobs, workers, "%s/%s" % (panel, variant), len(jobs), start)
    table = pd.DataFrame(rows)
    nulls = table[table.kind == "null"].sort_values("r").reset_index(drop=True)
    placebos = table[table.kind == "placebo"].sort_values("r").reset_index(drop=True)

    if n_placebo:
        ref_delta = float(placebos.delta.mean())
        placebo_record = dict(
            draws=n_placebo,
            mean=ref_delta,
            sd=float(placebos.delta.std(ddof=1)),
            q025=float(np.percentile(placebos.delta, 2.5)),
            q975=float(np.percentile(placebos.delta, 97.5)),
            mean_train_pairs=float(placebos.mean_train_pairs.mean()),
        )
    else:
        ref_delta = float(observed["delta"])
        placebo_record = dict(
            draws=0,
            mean=ref_delta,
            sd=0.0,
            note="provenance rule is a no-op for this variant; untouched observed value is the reference",
        )

    summary = stats.summarize(ref_delta, nulls.delta.to_numpy())
    summary["p_kind"] = "approximate"
    summary["approximation_sources"] = [
        "records sharing genes are not exchangeable",
        "null and placebo exclusion graphs are equal in law, not identical",
    ]
    record = dict(
        protocol_sha256=protocol_sha256,
        protocol_version="1.2",
        panel=panel,
        variant=variant,
        block=block,
        primary_variant=(variant == PRIMARY_VARIANT[panel] and block == "D"),
        statistic="auroc_pad_minus_amatch" if panel == "jonikas" else "correct_pad_minus_amatch",
        label=label or ("%s_%s%s" % (panel, variant, "" if block == "D" else "_Dplus")),
        n_records=int(fp.n),
        n_pairs=int(fp.n_pairs),
        n_positive=int((fp.y > 0).sum()),
        strata=op.strata,
        outer_groups=int(fp.M),
        families=list(families),
        seed_stream=dict(base_seed=base_seed, algorithm="PCG64", spawn="per-replicate child"),
        provenance_exclusion=dict(
            rule="union of name-sharing and provenance-sharing graphs (section 4.6)",
            identity_matches_archived=True,
            active=bool(bites),
            observed_mean_train_pairs=observed["mean_train_pairs"],
            null_mean_train_pairs=float(nulls.mean_train_pairs.mean()),
        ),
        untouched_observed=observed,
        placebo_reference=placebo_record,
        identity_replicate_bitwise_equal=True,
        **summary,
    )
    if panel == "costanzo":
        record["equivalence"] = costanzo_equivalence(fp.y, observed_out, summary["delta_flex"])
    if block_report is not None:
        record["d_plus"] = block_report
    record["environment"] = environment()
    record["runtime_seconds"] = time.monotonic() - start
    record["failed_replicates"] = int(replicates - len(nulls))

    out_dir = Path(out_dir or RESULTS)
    (out_dir / "panels").mkdir(parents=True, exist_ok=True)
    (out_dir / "replicates").mkdir(parents=True, exist_ok=True)
    stem = record["label"]
    table.to_csv(out_dir / "replicates" / ("%s.csv" % stem), index=False)
    (out_dir / "panels" / ("ccct_%s.json" % stem)).write_text(
        json.dumps(record, indent=2, default=_json_default) + "\n"
    )
    print(
        "PANEL %s variant=%s block=%s R=%d  ref=%.10g (untouched %.10g)  flex=%.10g  p=%.5g  %.1f min"
        % (panel, variant, block, replicates, record["delta_observed"], observed["delta"],
           record["delta_flex"], record["p_one_sided"], record["runtime_seconds"] / 60),
        flush=True,
    )
    return record


def _json_default(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.bool_):
        return bool(x)
    raise TypeError(type(x).__name__)


def environment():
    import scipy
    import sklearn

    return dict(
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        pandas=pd.__version__,
        scikit_learn=sklearn.__version__,
        release=(ROOT / "VERSION").read_text().strip(),
        cpu_count=os.cpu_count(),
    )
