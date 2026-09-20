"""Interface ablation: the restored-direction D+ block (protocol section 15).

Supplementary Section S4.2 states that ordered, uncompressed double-perturbation
profiles reveal the upstream endpoint, and the retained Jonikas context
coordinates (18 to 22) are exchange invariant, so they discard that order.  This
module restores it: three odd context descriptors computed from exactly the
inputs that produce coordinates 20 to 22, appended to the frozen 23-coordinate
block.

Provenance.  The release ships the prepared P/D interface, not the historical
Jonikas D adapter, so the partner sets are rebuilt here from the archived
``080930a_DM_data.mat``.  The rebuild is not assumed: ``verify_context``
recomputes the frozen coordinates 18 to 22 from the raw arrays and requires
bitwise equality with ``data/prepared/KEGG.npz`` before any D+ block is
returned.  That check passing is what makes the three new coordinates
descriptors of the same measurements rather than a re-derivation of unknown
fidelity.

    M(A,B)   = { E not in {A,B} : L[A,E] and L[B,E] both observed }
    M_H(A,B) = { E not in {A,B} : r[A,E] and r[B,E] both observed },  r = L - H
    c23 = tanh( mean_{E in M_H} ( r[A,E] - r[B,E] ) / d23 )
    c24 = frac_{E in M_H}( r[A,E] > r[B,E] ) - frac_{E in M_H}( r[A,E] < r[B,E] )
    c25 = tanh( mean_{E in M}   ( L[A,E] - L[B,E] ) / d25 )

``c24`` is written as a signed rank fraction rather than the ``2 * frac(>) - 1``
of the protocol text.  The two agree exactly whenever no partner ties, which is
the case on this panel, but only the signed form is exactly antisymmetric under
endpoint exchange when ties do occur, and exact antisymmetry is a hard
requirement: the D odd block must flip sign under exchange for the permutation
operator of Section 4 to preserve the (D, Y) marginal.  The mechanism profiles
of Section 15.4 tie on six of seven coordinates by construction, so the
distinction is not academic there.

``d23`` and ``d25`` are the median nonzero absolute raw value over the unlabeled
panel, floored at 0.01, the rule the existing odd descriptors use.  They are
computed once, label-free, and frozen; being functions of the multiset of raw
values they are permutation invariant, so Lemma 4.1 covers them unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.io as sio

from frozen_panel import ROOT

MAT = ROOT / "data/jonikas/080930a_DM_data.mat"
PREPARED = ROOT / "data/prepared"
SCALE_FLOOR = 0.01
N_NEW = 3


def _corr(x, y):
    """The release's finite-input correlation readout (``readouts.corr``)."""
    if len(x) < 3 or min(np.std(x), np.std(y)) < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def load_raw():
    m = sio.loadmat(MAT)
    names = [str(x[0]) for x in m["qnames_out"].ravel()]
    L = m["DM_array_merged"]
    H = m["DM_Hill_merged"]
    return names, L, H, L - H


def partner_masks(frame, names, L, R):
    index = {n: i for i, n in enumerate(names)}
    out = []
    for a, b in zip(frame.a, frame.b):
        i, j = index[a], index[b]
        keep = np.ones(len(names), bool)
        keep[[i, j]] = False
        out.append(
            (
                i,
                j,
                keep & np.isfinite(L[i]) & np.isfinite(L[j]),
                keep & np.isfinite(R[i]) & np.isfinite(R[j]),
            )
        )
    return out


def verify_context(frame=None):
    """Recompute frozen coordinates 18-22 from the raw arrays.

    Returns the per-coordinate maximum absolute deviation; the caller requires
    all five to be exactly zero.
    """
    frame = pd.read_csv(PREPARED / "KEGG.csv") if frame is None else frame
    xd = dict(np.load(PREPARED / "KEGG.npz"))["xd"]
    names, L, H, R = load_raw()
    got = np.zeros((len(frame), 5))
    for k, (i, j, mM, mH) in enumerate(partner_masks(frame, names, L, R)):
        la, lb = L[i][mM], L[j][mM]
        ra, rb = R[i][mH], R[j][mH]
        got[k, 0] = _corr(la, lb)
        got[k, 1] = _corr(ra, rb)
        got[k, 2] = np.tanh(np.log1p(mH.sum()) / 4)
        got[k, 3] = np.tanh(np.mean(np.abs(ra - rb))) if mH.sum() else 0.0
        got[k, 4] = np.mean(np.sign(ra) == np.sign(rb)) if mH.sum() else 0.5
    return {
        "coordinate_%d" % (18 + c): float(np.abs(got[:, c] - xd[:, 18 + c]).max())
        for c in range(5)
    }


def raw_descriptors(frame=None):
    """The three unscaled odd context descriptors, plus their availability."""
    frame = pd.read_csv(PREPARED / "KEGG.csv") if frame is None else frame
    names, L, H, R = load_raw()
    n = len(frame)
    raw = np.zeros((n, N_NEW))
    have = np.zeros((n, N_NEW), bool)
    ties = 0
    for k, (i, j, mM, mH) in enumerate(partner_masks(frame, names, L, R)):
        if mH.sum():
            ra, rb = R[i][mH], R[j][mH]
            raw[k, 0] = np.mean(ra - rb)
            raw[k, 1] = np.mean(np.sign(ra - rb))
            ties += int(np.sum(ra == rb))
            have[k, 0] = have[k, 1] = True
        if mM.sum():
            raw[k, 2] = np.mean(L[i][mM] - L[j][mM])
            have[k, 2] = True
    return raw, have, dict(partner_ties=int(ties))


def scales(raw):
    """Frozen label-free scaling constants: median nonzero |raw|, floored."""
    out = []
    for c in (0, 2):
        v = np.abs(raw[:, c])
        v = v[v > 0]
        out.append(max(float(np.median(v)) if len(v) else 0.0, SCALE_FLOOR))
    return out


def coordinate_diagnostics(frame, y, new, pair_of_record, draws=4000, seed=20260915):
    """Direct directional content of each restored descriptor.

    Delta_3 of section 15.3 asks whether restoring direction changes what the
    additive control achieves, which mixes two questions: does the restored
    direction help, and do three extra coordinates hurt.  This measures the
    first one on its own -- each descriptor scored against the reference
    annotations, with a bootstrap over the 84 unordered pairs so the reciprocal
    structure is resampled as a unit.
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    n_pairs = int(pair_of_record.max()) + 1
    rows = []
    for j, name in enumerate(("c23_mean_residual_difference", "c24_signed_rank", "c25_mean_profile_difference")):
        v = new[:, j]
        point = float(roc_auc_score(y, v))
        vals = []
        for _ in range(draws):
            pick = rng.integers(0, n_pairs, n_pairs)
            take = np.concatenate([np.flatnonzero(pair_of_record == p) for p in pick])
            if len(set(y[take])) < 2:
                continue
            vals.append(roc_auc_score(y[take], v[take]))
        lo, hi = np.percentile(vals, [2.5, 97.5])
        rows.append(
            dict(
                coordinate=name,
                auroc=point,
                ci95=[float(lo), float(hi)],
                positive_sign_on_annotated_positives=int(np.sum(v[y > 0] > 0)),
                n_annotated_positives=int(np.sum(y > 0)),
                reads_as=("chance" if lo <= 0.5 <= hi else ("directional" if lo > 0.5 else "reversed")),
                bootstrap_draws=len(vals),
            )
        )
    return rows


def build(frame=None, strict=True):
    """Return ``(xd_plus, xdr_plus, odd_plus, report)`` for the Jonikas panel."""
    frame = pd.read_csv(PREPARED / "KEGG.csv") if frame is None else frame
    if strict:
        deviations = verify_context(frame)
        worst = max(deviations.values())
        if worst != 0.0:
            raise AssertionError(
                "raw context rebuild does not reproduce the frozen coordinates "
                "bitwise; worst deviation %r" % deviations
            )
    z = dict(np.load(PREPARED / "KEGG.npz"))
    raw, have, diag = raw_descriptors(frame)
    d23, d25 = scales(raw)
    new = np.zeros_like(raw)
    new[:, 0] = np.where(have[:, 0], np.tanh(raw[:, 0] / d23), 0.0)
    new[:, 1] = np.where(have[:, 1], raw[:, 1], 0.0)
    new[:, 2] = np.where(have[:, 2], np.tanh(raw[:, 2] / d25), 0.0)
    xd_plus = np.hstack([z["xd"], new])
    xdr_plus = np.hstack([z["xdr"], -new])
    odd_plus = np.hstack([z["odd"], new])
    report = dict(
        coordinates=int(xd_plus.shape[1]),
        restored=new,
        odd_coordinates=int(odd_plus.shape[1]),
        symmetric_coordinates=int(xd_plus.shape[1] - odd_plus.shape[1]),
        scale_c23=d23,
        scale_c25=d25,
        scale_floor=SCALE_FLOOR,
        available_c23=int(have[:, 0].sum()),
        available_c25=int(have[:, 2].sum()),
        context_missing_records=int((~have[:, 0]).sum()),
        **diag,
    )
    return xd_plus, xdr_plus, odd_plus, report
