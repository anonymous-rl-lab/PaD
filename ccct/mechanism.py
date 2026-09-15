"""Mechanism controls and power calibration (protocol sections 7 and 15.4).

The four control settings are the ones the test must be shown to behave
correctly on before its panel verdict is believed:

    B1  (0.5, 2, 0.5)   strong coupled benefit                 must reject
    B5  (1, 0, 0.8)     D marginally independent of Y          must reject
    B4  (1, 0, 0)       D pure noise, H0 exactly true          must not reject
    B2  (1, 0.5, 0.5)   channels dependent but d = 0           must not reject

B4 is the Type I error check and B5 the informative positive control: there D
carries no marginal information about the label at all, so no marginal screen
would flag it, yet the joint learner gains.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import operators, stats
from .frozen import ROOT
from .panels import RESULTS, _json_default, environment

import engine as e  # noqa: E402  (path prepared by ccct.frozen)
import kernel as v4  # noqa: E402

SETTINGS = {
    "B1": dict(scenario="B1", expect="reject"),
    "B2": dict(scenario="B2", expect="not reject"),
    "B4": dict(scenario="B4", expect="not reject"),
    "B5": dict(scenario="B5", expect="reject"),
}
CONTROL_FAMILIES = ("joint", "A_match")


def _frame(ids):
    return pd.DataFrame(
        dict(a=[x + ":A" for x in ids], b=[x + ":B" for x in ids], pair=ids)
    )


def _fold(ids):
    import hashlib

    order = np.argsort([hashlib.sha256(x.encode()).hexdigest() for x in ids])
    fold = np.empty(len(ids), int)
    fold[order] = np.arange(len(ids)) % 5
    return fold


class MechanismBlock:
    """One train/test mechanism dataset under the frozen five-fold selection.

    Reproduces ``engine.train_four`` restricted to the two families the test
    compares, without the per-replicate file writes.
    """

    def __init__(self, scenario, seed, ntrain=512, ntest=4096, level="ideal", noise=(0.0, 0.0), phase="ccct"):
        self.scenario = scenario
        self.seed = seed
        self.ztr, self.mtr, _, self.audit_tr = e.dataset(phase, "train", scenario, seed, ntrain, level, noise)
        self.zte, self.mte, _, self.audit_te = e.dataset(phase, "test", scenario, seed, ntest, level, noise)
        assert self.audit_tr["stream"] != self.audit_te["stream"]
        self.gtr = _frame(self.mtr["instance_id"])
        self.fold = _fold(self.mtr["instance_id"])
        self.ntest = ntest
        self.base = {
            side: tuple(np.array(z[key], copy=True) for key in ("xd", "xdr", "odd"))
            for side, z in (("train", self.ztr), ("test", self.zte))
        }
        self.theory = e.theoretical(scenario)

    def install(self, train_block, test_block):
        for z, block in ((self.ztr, train_block), (self.zte, test_block)):
            z["xd"], z["xdr"], z["odd"] = block

    def run(self):
        out = {}
        for family in CONTROL_FAMILIES:
            d = e.Data(self.ztr, self.gtr, family)
            weights = d.weights
            configs = len(weights) * len(v4.LAMBDAS)
            cv = np.full((configs, len(self.ztr["y"])), np.nan)
            for f in range(5):
                tr = np.flatnonzero(self.fold != f)
                te = np.flatnonzero(self.fold == f)
                for i, theta in enumerate(weights):
                    fit = v4.fit(d, tr, theta)
                    for j, lam in enumerate(v4.LAMBDAS):
                        cv[i * 5 + j, te] = fit[lam][0][te]
            assert np.isfinite(cv).all()
            chosen = e.select(d, np.arange(len(self.ztr["y"])), cv)[0]
            i, lam = divmod(chosen, len(v4.LAMBDAS))
            model = e.fitted(d, np.arange(len(self.ztr["y"])), weights[i], v4.LAMBDAS[lam])
            s, sr = e.predict(model, self.zte)
            out[family] = dict(
                score=s, reverse=sr, direction=(s - sr) / 2, weight_index=i, lambda_index=lam
            )
        return out

    def statistic(self, out):
        y = self.zte["y"]
        joint, add = out["joint"], out["A_match"]
        errors = {f: int(np.sum((out[f]["direction"] >= 0) != (y > 0))) for f in CONTROL_FAMILIES}
        return dict(
            delta=float(errors["A_match"] - errors["joint"]),
            errors_joint=errors["joint"],
            errors_amatch=errors["A_match"],
            error_rate_joint=errors["joint"] / self.ntest,
            error_rate_amatch=errors["A_match"] / self.ntest,
            auroc_joint=stats.auroc(y, joint["score"]),
            auroc_amatch=stats.auroc(y, add["score"]),
            joint_weight_index=out["joint"]["weight_index"],
            joint_lambda_index=out["joint"]["lambda_index"],
            amatch_weight_index=out["A_match"]["weight_index"],
            amatch_lambda_index=out["A_match"]["lambda_index"],
        )


_STATE = {}


def _init(setting, seed, ntrain, ntest, base_seed, replicates):
    from threadpoolctl import threadpool_limits

    _STATE["limits"] = threadpool_limits(limits=1)
    block = MechanismBlock(SETTINGS[setting]["scenario"], seed, ntrain, ntest)
    _STATE.update(
        block=block,
        ops={
            side: operators.MechanismOperator(
                block.ztr["y"] if side == "train" else block.zte["y"], block.base[side]
            )
            for side in ("train", "test")
        },
        children=np.random.SeedSequence(base_seed).spawn(replicates),
    )


def _replicate(r):
    block, ops = _STATE["block"], _STATE["ops"]
    rng = np.random.Generator(np.random.PCG64(_STATE["children"][r]))
    block.install(ops["train"](rng), ops["test"](rng))
    row = block.statistic(block.run())
    row["r"] = r
    return row


def run_control(
    setting,
    replicates=500,
    base_seed=20260915,
    seed=17,
    ntrain=512,
    ntest=4096,
    workers=1,
    out_dir=None,
    protocol_sha256=None,
    alpha=0.05,
):
    start = time.monotonic()
    block = MechanismBlock(SETTINGS[setting]["scenario"], seed, ntrain, ntest)
    ops = {
        side: operators.MechanismOperator(
            block.ztr["y"] if side == "train" else block.zte["y"], block.base[side]
        )
        for side in ("train", "test")
    }
    observed = block.statistic(block.run())
    block.install(ops["train"].identity(), ops["test"].identity())
    identity = block.statistic(block.run())
    if any(observed[key] != identity[key] for key in observed):
        raise AssertionError("identity permutation changed the mechanism run: %r %r" % (observed, identity))

    rows = []
    _init(setting, seed, ntrain, ntest, base_seed, replicates)
    if workers > 1:
        import multiprocessing as mp

        ctx = mp.get_context("fork")
        with ctx.Pool(workers) as pool:
            for row in pool.imap_unordered(_replicate, range(replicates), chunksize=1):
                rows.append(row)
                if len(rows) % 25 == 0:
                    print("  %s %d/%d  %.1f min" % (setting, len(rows), replicates, (time.monotonic() - start) / 60), flush=True)
    else:
        for r in range(replicates):
            rows.append(_replicate(r))

    table = pd.DataFrame(rows).sort_values("r").reset_index(drop=True)
    summary = stats.summarize(observed["delta"], table.delta.to_numpy())
    record = dict(
        protocol_sha256=protocol_sha256,
        setting=setting,
        scenario=SETTINGS[setting]["scenario"],
        required_behavior=SETTINGS[setting]["expect"],
        seed=seed,
        n_train=ntrain,
        n_test=ntest,
        strata=ops["test"].strata,
        theory=block.theory,
        observed=observed,
        identity_replicate_bitwise_equal=True,
        seed_stream=dict(base_seed=base_seed, algorithm="PCG64", spawn="per-replicate child"),
        alpha=alpha,
        rejected=bool(summary["p_one_sided"] <= alpha),
        environment=environment(),
        failed_replicates=int(replicates - len(table)),
        **summary,
    )
    record["decision_matches_requirement"] = bool(
        record["rejected"] == (SETTINGS[setting]["expect"] == "reject")
    )
    out_dir = Path(out_dir or RESULTS)
    (out_dir / "controls").mkdir(parents=True, exist_ok=True)
    (out_dir / "replicates").mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "replicates" / ("control_%s.csv" % setting), index=False)
    (out_dir / "controls" / ("ccct_%s.json" % setting)).write_text(
        json.dumps(record, indent=2, default=_json_default) + "\n"
    )
    record["runtime_seconds"] = time.monotonic() - start
    print(
        "CONTROL %s R=%d delta_obs=%g delta_flex=%.4g p=%.5g rejected=%s (required %s) %.1f min"
        % (setting, replicates, record["delta_observed"], record["delta_flex"], record["p_one_sided"],
           record["rejected"], SETTINGS[setting]["expect"], record["runtime_seconds"] / 60),
        flush=True,
    )
    return record
