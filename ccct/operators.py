"""Permutation operators (protocol section 4).

The permutation unit is the unordered gene pair.  A pair's whole D measurement
block moves as one object: the forward descriptor vector, the reverse
descriptor vector and the odd descriptor block.  P blocks, labels, folds, gene
identities and folds never move.

Orientation.  D odd descriptors are antisymmetric under endpoint exchange, so a
permutation that ignored orientation would destroy the (D, Y) marginal.  Every
operator below therefore moves blocks in a label-aligned orientation.  Because
the release stores both orientations of every record, an aligned move copies
stored arrays rather than recomputing them, and

    flip(xd, xdr, odd) = (xdr, xd, -odd)

is an exact involution: no rounding is introduced anywhere, and the identity
permutation returns every array bit for bit.
"""
from __future__ import annotations

import numpy as np


def flip(block):
    xd, xdr, odd = block
    return xdr, xd, -odd


def take(block, index):
    xd, xdr, odd = block
    return xd[index], xdr[index], odd[index]


def _where_flip(mask, block):
    """Return ``block`` with ``flip`` applied on every row where ``mask``."""
    xd, xdr, odd = block
    m = mask[:, None]
    return (
        np.where(m, xdr, xd),
        np.where(m, xd, xdr),
        np.where(m, -odd, odd),
    )


class CostanzoOperator:
    """Truth-aligned global permutation of 657 one-record pairs.

    ``target`` says whether the stored canonical endpoint order is the inherited
    true orientation.  Expressing every block along its true orientation turns
    the panel into a single exchangeable pool, so no stratification by label is
    needed; the assignment step re-expresses each block in the destination
    pair's stored orientation.
    """

    name = "costanzo"

    def __init__(self, frame, block, within_fold=False, within_target=False):
        self.reverse_mask = frame.target.to_numpy() != 1
        self.n = len(self.reverse_mask)
        self.truth = _where_flip(self.reverse_mask, block)
        self.within_fold = within_fold
        self.within_target = within_target
        fold = frame.fold.to_numpy()
        target = frame.target.to_numpy()
        if within_fold:
            keys = [("fold %d" % f, fold == f) for f in sorted(set(fold))]
        elif within_target:
            keys = [("target %d" % t, target == t) for t in sorted(set(target))]
        else:
            keys = [("panel", np.ones(self.n, bool))]
        self.blocks = [np.flatnonzero(m) for _, m in keys]
        self.strata = {name: int(m.sum()) for name, m in keys}

    def draw(self, rng):
        order = np.arange(self.n)
        for ix in self.blocks:
            order[ix] = ix[rng.permutation(len(ix))]
        return order

    def __call__(self, rng):
        order = self.draw(rng)
        return _where_flip(self.reverse_mask, take(self.truth, order)), order

    def identity(self):
        order = np.arange(self.n)
        return _where_flip(self.reverse_mask, take(self.truth, order)), order


class JonikasOperator:
    """Stratified permutation of 84 two-record pairs.

    Stratum O holds the 21 pairs carrying one annotated positive direction;
    stratum U holds the 63 pairs whose two records are both reference negatives.
    Reference negatives are unannotated records, not confirmed reverse
    relations, so stratum U is aligned canonically rather than by truth.

    Within a stratum a pair carries an ordered slot pair: slot 0 is the
    annotated positive record in stratum O and the canonically ordered record in
    stratum U, slot 1 is its reciprocal.  Both slots move together, which
    preserves exchange antisymmetry exactly and keeps the record-level (D, Y)
    empirical joint distribution unchanged.
    """

    name = "jonikas"

    def __init__(self, frame, block, by_context=False):
        pairs = list(dict.fromkeys(frame.pair))
        label = frame.label.to_numpy()
        a = frame.a.to_numpy()
        b = frame.b.to_numpy()
        context = frame.double_observed.to_numpy()
        slot0, slot1, tags = [], [], []
        for p in pairs:
            ix = np.flatnonzero((frame.pair == p).to_numpy())
            assert len(ix) == 2, (p, ix)
            pos = ix[label[ix] > 0]
            if len(pos) == 1:
                first = int(pos[0])
                tag = "O"
            elif len(pos) == 0:
                first = int(ix[0] if a[ix[0]] < b[ix[0]] else ix[1])
                tag = "U"
            else:
                raise AssertionError("pair with two annotated directions: %s" % p)
            second = int(ix[0] if ix[1] == first else ix[1])
            assert a[first] == b[second] and b[first] == a[second]
            if by_context:
                assert context[first] == context[second]
                tag = tag + ("+ctx" if context[first] else "-ctx")
            slot0.append(first)
            slot1.append(second)
            tags.append(tag)
        self.pairs = pairs
        self.slot0 = np.asarray(slot0)
        self.slot1 = np.asarray(slot1)
        self.tags = np.asarray(tags)
        self.n = len(frame)
        self.block = block
        self.groups = [np.flatnonzero(self.tags == t) for t in sorted(set(tags))]
        self.strata = {t: int((self.tags == t).sum()) for t in sorted(set(tags))}

    def _assign(self, order):
        xd, xdr, odd = self.block
        nxd = np.array(xd, copy=True)
        nxdr = np.array(xdr, copy=True)
        nodd = np.array(odd, copy=True)
        src0 = self.slot0[order]
        src1 = self.slot1[order]
        for dst, src in ((self.slot0, src0), (self.slot1, src1)):
            nxd[dst] = xd[src]
            nxdr[dst] = xdr[src]
            nodd[dst] = odd[src]
        return nxd, nxdr, nodd

    def draw(self, rng):
        order = np.arange(len(self.pairs))
        for ix in self.groups:
            order[ix] = ix[rng.permutation(len(ix))]
        return order

    def __call__(self, rng):
        order = self.draw(rng)
        return self._assign(order), order

    def identity(self):
        order = np.arange(len(self.pairs))
        return self._assign(order), order


class MechanismOperator:
    """Within-label permutation of mechanism records.

    Mechanism D descriptors are exchange symmetric (the reverse block equals the
    forward block and the local odd block is exactly zero), so orientation
    alignment is vacuous and the operator reduces to a label-stratified
    permutation.  Training and test records are permuted inside their own set,
    which keeps each set's ``(D, Y)`` marginal exactly fixed.
    """

    name = "mechanism"

    def __init__(self, y, block):
        self.n = len(y)
        self.block = block
        self.groups = [np.flatnonzero(y == v) for v in np.unique(y)]
        self.strata = {("label %g" % v): int((y == v).sum()) for v in np.unique(y)}

    def _assign(self, order):
        xd, xdr, odd = self.block
        return xd[order], xdr[order], odd[order]

    def draw(self, rng):
        order = np.arange(self.n)
        for ix in self.groups:
            order[ix] = ix[rng.permutation(len(ix))]
        return order

    def __call__(self, rng):
        order = self.draw(rng)
        return self._assign(order), order

    def identity(self):
        order = np.arange(self.n)
        return self._assign(order), order


def build(panel, frame, block, variant="primary", y=None):
    if panel == "costanzo":
        return CostanzoOperator(
            frame,
            block,
            within_fold=(variant == "V1_fold"),
            within_target=(variant == "V5_target"),
        )
    if panel == "jonikas":
        return JonikasOperator(frame, block, by_context=(variant == "V2_context"))
    if panel == "mechanism":
        return MechanismOperator(y, block)
    raise ValueError(panel)
