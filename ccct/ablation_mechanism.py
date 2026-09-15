"""Section 15.4: B1 with D+, the demonstration that the mechanism's difficulty
is a property of the exchange-invariant interface.

Prediction recorded in the protocol: ``A_match(D+)`` attains near-zero error,
because the restored ``c23`` carries the sign of ``-Y`` deterministically in the
noiseless summary.  Reporting it is the point.  The paper is better off stating
that the mechanism's D channel is a direction oracle in raw form than having a
reviewer derive it from Supplementary S4.2 -- and the contrast with the Jonikas
panel, where the same restoration is at chance, is the empirical justification
for the interface.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import stats
from .mechanism import CONTROL_FAMILIES, MechanismBlock, SETTINGS
from .panels import RESULTS, _json_default, environment


def run(setting="B1", seeds=(17, 29, 43), ntrain=512, ntest=4096, out_dir=None, protocol_sha256=None):
    from threadpoolctl import threadpool_limits

    start = time.monotonic()
    rows = []
    with threadpool_limits(limits=1):
        for seed in seeds:
            entry = dict(seed=seed)
            for block in ("D", "D+"):
                mb = MechanismBlock(SETTINGS[setting]["scenario"], seed, ntrain, ntest, block=block)
                out = mb.run()
                y = mb.zte["y"]
                for family in CONTROL_FAMILIES:
                    errors = int(np.sum((out[family]["direction"] >= 0) != (y > 0)))
                    entry["%s(%s)_errors" % (family, block)] = errors
                    entry["%s(%s)_error_rate" % (family, block)] = errors / ntest
                    entry["%s(%s)_auroc" % (family, block)] = stats.auroc(y, out[family]["score"])
                if block == "D+":
                    entry["dplus_scales"] = mb.dplus_scales
                entry["theory"] = mb.theory
            rows.append(entry)
            print(
                "  seed %d  A_match(D) %.4f -> A_match(D+) %.4f   joint(D) %.4f -> joint(D+) %.4f"
                % (seed, entry["A_match(D)_error_rate"], entry["A_match(D+)_error_rate"],
                   entry["joint(D)_error_rate"], entry["joint(D+)_error_rate"]),
                flush=True,
            )
    mean = lambda key: float(np.mean([r[key] for r in rows]))
    record = dict(
        protocol_sha256=protocol_sha256,
        section="15.4",
        setting=setting,
        seeds=list(seeds),
        n_train=ntrain,
        n_test=ntest,
        per_seed=rows,
        mean_error_rate={
            key: mean(key)
            for key in (
                "joint(D)_error_rate", "A_match(D)_error_rate",
                "joint(D+)_error_rate", "A_match(D+)_error_rate",
            )
        },
        additive_error_lower=rows[0]["theory"]["additive_error_lower"],
        prediction="A_match(D+) attains near-zero error",
        prediction_held=bool(mean("A_match(D+)_error_rate") < 0.01),
        reading=(
            "The mechanism's raw ordered profile is a direction oracle; restoring it "
            "lets the additive control reach near-zero error, so the mechanism's "
            "difficulty is a property of the exchange-invariant retained "
            "representation, not of the assay."
        ),
        environment=environment(),
        runtime_seconds=time.monotonic() - start,
    )
    out = Path(out_dir or RESULTS)
    (out / "controls").mkdir(parents=True, exist_ok=True)
    (out / "controls" / "ccct_ablation_mechanism.json").write_text(
        json.dumps(record, indent=2, default=_json_default) + "\n"
    )
    print("  mean error rates:", json.dumps(record["mean_error_rate"], indent=None), flush=True)
    print("  prediction (A_match(D+) near zero) held:", record["prediction_held"], flush=True)
    return record
