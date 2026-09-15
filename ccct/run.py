"""Command line entry point for the Cross-Channel Correspondence Test."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from .frozen import ROOT, FrozenPanel
from . import dplus, operators, panels, stats
from .panels import RESULTS, _json_default, environment

PROTOCOL = ROOT / "protocols/CCCT_FROZEN_PROTOCOL.md"
HASH_FILE = ROOT / "protocols/CCCT_FROZEN_PROTOCOL.sha256"


def protocol_sha256():
    if not PROTOCOL.exists():
        return None
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


# ---------------------------------------------------------------- identity ---
def identity(args):
    """Replicate zero: the frozen procedure, the fast runner and the identity
    permutation must agree bit for bit, and the archived reference must agree to
    the declared tolerance."""
    from threadpoolctl import threadpool_limits
    import pandas as pd
    import kernel as k
    from nested import train as release_train

    report = {"protocol_sha256": protocol_sha256(), "checks": []}
    with threadpool_limits(limits=1):
        for panel in ("costanzo", "jonikas"):
            dataset = {"costanzo": "Costanzo", "jonikas": "KEGG"}[panel]
            release = {}
            for family in ("PaD", "A_match"):
                release_train(dataset, family)
                out = Path(__import__("os").environ.get("PAD_OUTPUT_ROOT", str(ROOT / "runs")))
                df = pd.read_csv(out / panel / ("%s_predictions.csv" % family), float_precision="round_trip")
                release[family] = (df.score.to_numpy(), df.reverse_score.to_numpy())

            fp = FrozenPanel(panel, ("PaD", "A_match"))
            fp.restore()
            st0 = fp.structure(None)
            archived = all(
                np.array_equal(st0.tr[m], fp.archived_allowed[m]) for m in range(fp.M)
            )
            plain = fp.run(share=False, structure=st0)
            shared = fp.run(share=True, structure=st0)
            op = operators.build(
                panel, fp.frame, fp.original_block(), variant=panels.PRIMARY_VARIANT[panel]
            )
            block_ident, order_ident = op.identity()
            fp.set_D(*block_ident)
            ident = fp.run(order=order_ident)

            entry = dict(panel=panel, variant=panels.PRIMARY_VARIANT[panel])
            entry["identity_provenance_graph_matches_archived"] = bool(archived)
            entry["runner_matches_release_bitwise"] = all(
                np.array_equal(plain[f]["score"], release[f][0])
                and np.array_equal(plain[f]["reverse"], release[f][1])
                for f in ("PaD", "A_match")
            )
            entry["shared_kernel_pass_bitwise"] = all(
                np.array_equal(plain[f]["score"], shared[f]["score"])
                and np.array_equal(plain[f]["reverse"], shared[f]["reverse"])
                for f in ("PaD", "A_match")
            )
            entry["identity_permutation_bitwise"] = all(
                np.array_equal(shared[f]["score"], ident[f]["score"])
                and np.array_equal(shared[f]["reverse"], ident[f]["reverse"])
                for f in ("PaD", "A_match")
            )
            ref = pd.read_csv(ROOT / "reference" / panel / "pair_predictions.csv", float_precision="round_trip")
            archived = ref.PD_frozen.to_numpy() if panel == "jonikas" else None
            if panel == "costanzo":
                ic = pd.read_csv(ROOT / "reference/costanzo/internal_controls.csv", float_precision="round_trip")
                archived = ic.V4_score.to_numpy()
                archived_add = ic.A_match_score.to_numpy()
            else:
                archived_add = ref.A_match.to_numpy()
            entry["archived_max_abs_deviation"] = dict(
                PaD=float(np.abs(plain["PaD"]["score"] - archived).max()),
                A_match=float(np.abs(plain["A_match"]["score"] - archived_add).max()),
            )
            y = fp.y
            if panel == "costanzo":
                entry["decision_quantities"] = dict(
                    correct_pad=stats.correct(y, plain["PaD"]["direction"]),
                    correct_amatch=stats.correct(y, plain["A_match"]["direction"]),
                    expected=[611, 613],
                )
                entry["decision_quantities_match"] = [
                    entry["decision_quantities"]["correct_pad"],
                    entry["decision_quantities"]["correct_amatch"],
                ] == [611, 613]
            else:
                a_pad = stats.auroc(y, plain["PaD"]["score"])
                a_add = stats.auroc(y, plain["A_match"]["score"])
                entry["decision_quantities"] = dict(
                    auroc_pad=a_pad, auroc_amatch=a_add, delta=a_pad - a_add,
                    expected=[0.8811143505021056, 0.8276643990929706],
                )
                entry["decision_quantities_match"] = (
                    abs(a_pad - 0.8811143505021056) < 1e-12 and abs(a_add - 0.8276643990929706) < 1e-12
                )
            entry["passed"] = bool(
                entry["identity_provenance_graph_matches_archived"]
                and entry["shared_kernel_pass_bitwise"]
                and entry["identity_permutation_bitwise"]
                and entry["decision_quantities_match"]
                and max(entry["archived_max_abs_deviation"].values()) < 1e-12
            )
            report["checks"].append(entry)
            print(json.dumps(entry, indent=2, default=_json_default), flush=True)

    report["passed"] = all(c["passed"] for c in report["checks"])
    report["environment"] = environment()
    out_dir = Path(args.out or RESULTS)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ccct_identity.json").write_text(json.dumps(report, indent=2, default=_json_default) + "\n")
    print("IDENTITY CHECK:", "PASSED" if report["passed"] else "FAILED", flush=True)
    return 0 if report["passed"] else 1


# --------------------------------------------------------------- ablation ---
def ablation(args):
    """Section 15: five learners on Jonikas, three contrasts, four readings."""
    from threadpoolctl import threadpool_limits

    start = time.monotonic()
    with threadpool_limits(limits=1):
        deviations = dplus.verify_context()
        assert max(deviations.values()) == 0.0, deviations
        results = {}
        fp_d, _ = panels.make_panel("jonikas", ("PaD", "A_match", "A_rbf"), block="D")
        y = fp_d.y
        out = fp_d.run()
        for family in fp_d.families:
            results["%s(D)" % family] = _metrics(y, out[family])
        fp_p, report = panels.make_panel("jonikas", ("PaD", "A_match", "A_rbf"), block="D+")
        out_plus = fp_p.run()
        for family in fp_p.families:
            results["%s(D+)" % family] = _metrics(y, out_plus[family])

    d1 = results["PaD(D)"]["auroc"] - results["A_match(D+)"]["auroc"]
    d2 = results["PaD(D+)"]["auroc"] - results["A_match(D+)"]["auroc"]
    d3 = results["A_match(D+)"]["auroc"] - results["A_match(D)"]["auroc"]
    readings = dict(
        R1=bool(d1 <= 0.01),
        R2_delta_2=bool(d2 >= 0.03),
        R3=bool(abs(d3) <= 0.01),
    )
    record = dict(
        protocol_sha256=protocol_sha256(),
        section="15",
        panel="jonikas",
        d_plus=report,
        context_rebuild_deviations=deviations,
        learners=results,
        contrasts=dict(
            Delta_1_pad_D_minus_amatch_Dplus=d1,
            Delta_2_pad_Dplus_minus_amatch_Dplus=d2,
            Delta_3_amatch_Dplus_minus_amatch_D=d3,
        ),
        readings=readings,
        note=(
            "R2 additionally requires the CCCT on D+ to reject at Holm-adjusted "
            "p <= 0.05; that part is decided by the D+ panel run, not here."
        ),
        environment=environment(),
        runtime_seconds=time.monotonic() - start,
    )
    out_dir = Path(args.out or RESULTS)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ccct_ablation.json").write_text(json.dumps(record, indent=2, default=_json_default) + "\n")
    for name, m in results.items():
        print("  %-14s AUROC %.6f  AP %.6f" % (name, m["auroc"], m["average_precision"]), flush=True)
    print("  Delta_1 = %.6f   Delta_2 = %.6f   Delta_3 = %.6f" % (d1, d2, d3), flush=True)
    print("  readings:", readings, flush=True)
    return 0


def _metrics(y, out):
    return dict(
        auroc=stats.auroc(y, out["score"]),
        average_precision=stats.average_precision(y, out["score"]),
        weight_counts=[int(c) for c in out["weight_counts"]],
        lambda_counts=[int(c) for c in out["lambda_counts"]],
    )


# ------------------------------------------------------------------ panels ---
def panel(args):
    panels.run_panel(
        args.panel,
        replicates=args.replicates,
        base_seed=args.seed,
        variant=args.variant,
        block=args.block,
        families=tuple(args.families),
        workers=args.workers,
        out_dir=args.out,
        protocol_sha256=protocol_sha256(),
        label=args.label,
        placebo=args.placebo,
    )
    return 0


def control(args):
    from . import mechanism

    for setting in args.settings:
        mechanism.run_control(
            setting,
            replicates=args.replicates,
            base_seed=args.seed,
            seed=args.mechanism_seed,
            ntrain=args.ntrain,
            ntest=args.ntest,
            workers=args.workers,
            out_dir=args.out,
            protocol_sha256=protocol_sha256(),
        )
    return 0


def power(args):
    from . import powercurve

    powercurve.run(
        grid=args.grid,
        datasets=args.datasets,
        replicates=args.replicates,
        base_seed=args.seed,
        workers=args.workers,
        out_dir=args.out,
        protocol_sha256=protocol_sha256(),
    )
    return 0


def summarize(args):
    out_dir = Path(args.out or RESULTS)
    record = dict(protocol_sha256=protocol_sha256(), panels={}, controls={}, decision=None)
    for path in sorted((out_dir / "panels").glob("ccct_*.json")):
        record["panels"][path.stem[5:]] = json.loads(path.read_text())
    for path in sorted((out_dir / "controls").glob("ccct_*.json")):
        record["controls"][path.stem[5:]] = json.loads(path.read_text())

    primary = [k for k in record["panels"] if record["panels"][k]["variant"] == "primary" and record["panels"][k]["block"] == "D"]
    primary.sort()
    if primary:
        raw = [record["panels"][k]["p_one_sided"] for k in primary]
        adj = stats.holm(raw)
        for key, a in zip(primary, adj):
            record["panels"][key]["p_holm"] = a
        record["holm_family"] = dict(zip(primary, adj))

    controls = record["controls"]
    b1_ok = controls.get("B1", {}).get("rejected") is True
    size = controls.get("B4", {}).get("p_one_sided")
    b4_rate = controls.get("B4", {}).get("empirical_size")
    valid = b1_ok and (b4_rate is None or 0.031 <= b4_rate <= 0.073)
    jon = record["panels"].get("jonikas_primary")
    if jon is not None:
        p = jon.get("p_holm", jon["p_one_sided"])
        if not valid:
            record["decision"] = "CCCT_INVALID"
        else:
            record["decision"] = "CCCT_SUPPORTED" if p <= 0.05 else "CCCT_NOT_SUPPORTED"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ccct_summary.json").write_text(json.dumps(record, indent=2, default=_json_default) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "panels"}, indent=2, default=_json_default)[:2000])
    print("DECISION:", record["decision"], flush=True)
    return 0


def freeze(args):
    digest = protocol_sha256()
    if digest is None:
        raise SystemExit("protocols/CCCT_FROZEN_PROTOCOL.md not found")
    HASH_FILE.write_text(digest + "  CCCT_FROZEN_PROTOCOL.md\n")
    print(digest, flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ccct.run", description=__doc__)
    ap.add_argument("--out", default=None, help="output root (default ccct_results/)")
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("identity", help="replicate zero: bitwise gate on both panels")
    sub.add_parser("ablation", help="section 15 interface ablation on Jonikas")
    sub.add_parser("freeze", help="write the protocol SHA-256")
    sub.add_parser("summarize", help="collect results, apply Holm, state the decision")

    p = sub.add_parser("panel", help="run a panel's permutation replicates")
    p.add_argument("--panel", choices=["jonikas", "costanzo"], required=True)
    p.add_argument("--replicates", type=int, default=2000)
    p.add_argument("--placebo", type=int, default=100)
    p.add_argument("--seed", type=int, default=20260915)
    p.add_argument("--variant", default=None)
    p.add_argument("--block", default="D", choices=["D", "D+"])
    p.add_argument("--families", nargs="+", default=["PaD", "A_match"])
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--label", default=None)

    c = sub.add_parser("control", help="run mechanism control settings")
    c.add_argument("--settings", nargs="+", default=["B1", "B2", "B4", "B5"])
    c.add_argument("--replicates", type=int, default=500)
    c.add_argument("--seed", type=int, default=20260915)
    c.add_argument("--mechanism-seed", type=int, default=17)
    c.add_argument("--ntrain", type=int, default=512)
    c.add_argument("--ntest", type=int, default=4096)
    c.add_argument("--workers", type=int, default=1)

    w = sub.add_parser("power", help="power calibration at Jonikas dimensions")
    w.add_argument("--grid", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    w.add_argument("--datasets", type=int, default=200)
    w.add_argument("--replicates", type=int, default=500)
    w.add_argument("--seed", type=int, default=20260915)
    w.add_argument("--workers", type=int, default=1)

    args = ap.parse_args(argv)
    return {
        "identity": identity,
        "ablation": ablation,
        "panel": panel,
        "control": control,
        "power": power,
        "summarize": summarize,
        "freeze": freeze,
    }[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
