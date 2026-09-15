#!/usr/bin/env python3
"""Independent verification of the Cross-Channel Correspondence Test outputs.

Recomputes every reported p-value, flexibility offset and decision from the
stored replicate tables, re-checks the protocol hash carried by each output
record, and re-runs replicate zero.  It does not trust any number written by
the run that produced the results.

    python verify_ccct.py                 # recompute from ccct_results/
    python verify_ccct.py --identity      # also re-run the replicate-zero gate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ccct import stats  # noqa: E402

RESULTS = ROOT / "ccct_results"
PROTOCOL = ROOT / "protocols/CCCT_FROZEN_PROTOCOL.md"
HASH_FILE = ROOT / "protocols/CCCT_FROZEN_PROTOCOL.sha256"


def close(a, b, tol=1e-12, what=""):
    assert abs(float(a) - float(b)) <= tol, (what, a, b)


def protocol_hash():
    digest = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    if HASH_FILE.exists():
        recorded = HASH_FILE.read_text().split()[0]
        assert digest == recorded, ("protocol hash drifted", digest, recorded)
    return digest


def verify_record(record, table, digest):
    """Recompute a panel or control record from its replicate table."""
    out = dict(label=record.get("label") or record.get("setting"))
    assert record.get("protocol_sha256") == digest, ("stale protocol hash", out["label"])
    if "kind" in table.columns:
        nulls = table[table.kind == "null"]
        placebos = table[table.kind == "placebo"]
    else:
        nulls, placebos = table, table.iloc[:0]

    assert len(nulls) == record["replicates"], (out["label"], len(nulls), record["replicates"])
    assert record["failed_replicates"] == 0, out["label"]
    assert nulls.delta.notna().all(), out["label"]
    assert sorted(nulls.r.tolist()) == list(range(len(nulls))), "replicate index gap"

    if len(placebos):
        close(record["placebo_reference"]["mean"], placebos.delta.mean(), 1e-12, "placebo mean")
        close(record["delta_observed"], placebos.delta.mean(), 1e-12, "reference is placebo mean")
    recomputed = stats.summarize(record["delta_observed"], nulls.delta.to_numpy())
    for key in ("delta_flex", "p_one_sided", "p_two_sided"):
        close(recomputed[key], record[key], 1e-12, key)
    for key in ("mean", "sd", "q025", "q500", "q975"):
        close(recomputed["null"][key], record["null"][key], 1e-12, "null." + key)
    out["replicates"] = int(len(nulls))
    out["placebo_draws"] = int(len(placebos))
    out["delta_observed"] = record["delta_observed"]
    out["delta_flex"] = recomputed["delta_flex"]
    out["p_one_sided"] = recomputed["p_one_sided"]
    out["exceedances"] = int(np.sum(nulls.delta.to_numpy() >= record["delta_observed"]))
    out["recomputed"] = True
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--identity", action="store_true", help="re-run the replicate-zero gate")
    args = ap.parse_args(argv)
    results = Path(args.results)
    digest = protocol_hash()
    report = {"protocol_sha256": digest, "panels": [], "controls": []}

    for kind, folder in (("panels", "panels"), ("controls", "controls")):
        for path in sorted((results / folder).glob("ccct_*.json")):
            record = json.loads(path.read_text())
            stem = record.get("label") or record.get("setting")
            csv = results / "replicates" / (
                "%s.csv" % (stem if kind == "panels" else "control_%s" % stem)
            )
            table = pd.read_csv(csv, float_precision="round_trip")
            report[kind].append(verify_record(record, table, digest))

    primary = [p for p in report["panels"] if p["label"] in ("jonikas_primary", "costanzo_V1_fold")]
    if len(primary) == 2:
        primary.sort(key=lambda p: p["label"])
        adjusted = stats.holm([p["p_one_sided"] for p in primary])
        report["holm"] = {p["label"]: a for p, a in zip(primary, adjusted)}

    controls = {c["label"]: c for c in report["controls"]}
    if "B1" in controls and "B4" in controls:
        b4 = json.loads((results / "controls/ccct_B4.json").read_text())
        size = b4.get("empirical_size")
        report["B1_rejects"] = controls["B1"]["p_one_sided"] <= 0.05
        report["B4_size"] = size
        report["B4_size_in_interval"] = None if size is None else bool(0.031 <= size <= 0.073)

    if args.identity:
        from ccct.run import identity

        class _A:
            out = str(results)

        report["identity_exit_code"] = identity(_A())

    (results / "ccct_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("VERIFY CCCT: all stored p-values, offsets and null summaries recomputed from replicate tables.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
