"""Frozen procedure runner with a provenance-adaptive exclusion structure.

The scientific content of this module is nothing: every number it produces is
produced by the release functions ``kernel.fit`` and
``selection.select_from_inner``, called with the same arguments and in the same
order as ``nested.train``.  What it adds is two things.

First, it builds the nested split structure from the *union provenance graph*
of protocol section 4.6 rather than from names alone.  A D permutation moves
where a record's measurements came from without moving its name, so a
name-based endpoint exclusion no longer bounds what the training set knows
about the test record's D provenance.  For outer test group ``g`` under
permutation ``sigma`` the training pool is

    T_sigma(g) = { t : genes(t) & genes(g) = 0  and  genes(sigma(t)) & genes(sigma(g)) = 0 }

and inner validation applies the same two-sided rule inside the pool.  With
``sigma = identity`` this is exactly the archived scheme, which
``ccct.run identity`` asserts.

Second, candidate kernels whose weight on the third mixing component is zero
are built identically by PaD and by the matched additive control, so their fits
are evaluated once and shared between the two families.  That is a pure
bookkeeping saving; ``ccct.run identity`` asserts it is bitwise neutral.

Lemma 4.1 of the protocol covers the panel-level adapter constants, which are
functions of the multiset of D values and therefore permutation invariant.  The
training-fold RMS constants are not invariant and are recomputed inside
``kernel.fit`` on every replicate exactly as in the release.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src/pad", ROOT / "experiments/mechanism", ROOT / "verification"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import kernel as k  # noqa: E402
from controls import Control  # noqa: E402
from selection import select_from_inner  # noqa: E402

DATASETS = {"jonikas": "KEGG", "costanzo": "Costanzo"}
NCONFIG = len(k.WEIGHTS) * len(k.LAMBDAS)

SHARED_THETAS = tuple(i for i, t in enumerate(k.WEIGHTS) if t[2] == 0.0)


class _Col:
    def __init__(self, arr):
        self._arr = arr

    def to_numpy(self):
        return self._arr


class _Frame:
    """Column shim; ``select_from_inner`` reads ``g.pair.to_numpy()`` 35 times
    per group and the release rebuilds it from pandas each time.  Same values."""

    def __init__(self, g):
        self.frame = g
        self.a = g.a
        self.b = g.b
        self.pair = _Col(g.pair.to_numpy())


class Structure:
    """Per-permutation nested split structure."""

    __slots__ = ("te", "tr", "trmask", "inner", "pool_sizes", "degenerate")

    def __init__(self, te, tr, trmask, inner, pool_sizes, degenerate):
        self.te = te
        self.tr = tr
        self.trmask = trmask
        self.inner = inner
        self.pool_sizes = pool_sizes
        self.degenerate = degenerate


class FrozenPanel:
    def __init__(self, panel, families=("PaD", "A_match")):
        if panel not in DATASETS:
            raise ValueError(panel)
        self.panel = panel
        self.dataset = DATASETS[panel]
        self.families = tuple(families)
        self.data = {
            f: (k.Data(self.dataset) if f == "PaD" else Control(self.dataset, f))
            for f in self.families
        }
        base = self.data[self.families[0]]
        self.frame = base.g
        self.n = base.n
        self.y = base.y
        self.directional = base.directional
        for d in self.data.values():
            d.g = _Frame(d.g)

        # ---- pair / group / gene incidence (all permutation invariant) ------
        frame = self.frame
        pairs = list(dict.fromkeys(frame.pair))
        pair_index = {p: i for i, p in enumerate(pairs)}
        self.pairs = pairs
        self.n_pairs = len(pairs)
        self.pair_of_record = np.array([pair_index[p] for p in frame.pair])
        genes = sorted(set(frame.a) | set(frame.b))
        gene_index = {g: i for i, g in enumerate(genes)}
        self.genes = genes
        self.pair_gene = np.zeros((self.n_pairs, len(genes)), bool)
        for a, b, p in zip(frame.a, frame.b, frame.pair):
            self.pair_gene[pair_index[p], gene_index[a]] = True
            self.pair_gene[pair_index[p], gene_index[b]] = True

        self.group_keys = list(base.groups)
        self.M = len(self.group_keys)
        self.group_records = [base.groups[key] for key in self.group_keys]
        self.group_pairs = [
            np.unique(self.pair_of_record[te]) for te in self.group_records
        ]
        self.group_record_mask = np.zeros((self.M, self.n), bool)
        for m, te in enumerate(self.group_records):
            self.group_record_mask[m, te] = True
        self.archived_allowed = [base.allowed[key] for key in self.group_keys]

        # ---- snapshot of the permuted channel -------------------------------
        self.original = {
            key: np.array(base.z[key], copy=True) for key in ("xd", "xdr", "odd")
        }
        self.invariant_P = {
            f: (k.rbf(d.z["xp"], d.z["xp"]), k.rbf(d.z["xpr"], d.z["xp"]))
            for f, d in self.data.items()
        }
        self.identity_order = np.arange(self.n_pairs)

    # ------------------------------------------------------------ channel ---
    def install_base(self, xd, xdr, odd):
        """Replace the panel's D channel (used by the section 15 ablation)."""
        self.original = {
            "xd": np.array(xd, copy=True),
            "xdr": np.array(xdr, copy=True),
            "odd": np.array(odd, copy=True),
        }
        self.restore()

    def original_block(self):
        return tuple(np.array(self.original[key], copy=True) for key in ("xd", "xdr", "odd"))

    def set_D(self, xd, xdr, odd):
        D = k.rbf(xd, xd)
        Dr = k.rbf(xdr, xd)
        for family, d in self.data.items():
            d.z["xd"] = xd
            d.z["xdr"] = xdr
            d.z["odd"] = odd
            P, Pr = self.invariant_P[family]
            d.J = P * D
            d.Jr = Pr * Dr
            if family != "PaD":
                d.P, d.Pr, d.D, d.Dr = P, Pr, D, Dr

    def restore(self):
        self.set_D(*self.original_block())

    # ---------------------------------------------------------- structure ---
    def group_gene(self, order):
        """Gene incidence of each outer group under provenance ``order``."""
        out = np.zeros((self.M, self.pair_gene.shape[1]), bool)
        for m, gp in enumerate(self.group_pairs):
            out[m] = self.pair_gene[order[gp]].any(0)
        return out

    def structure(self, order=None):
        """Nested split structure under the provenance permutation ``order``.

        ``order[p]`` is the pair whose D block pair ``p`` received.  The name
        rule and the provenance rule are applied together at the outer level and
        again inside the pool at the inner level.
        """
        order = self.identity_order if order is None else np.asarray(order)
        name_group = self.group_gene(self.identity_order)
        prov_group = name_group if order is self.identity_order else self.group_gene(order)
        pg = self.pair_gene
        pg_src = pg[order]
        # pair p is admissible for group m when it shares no gene by name and
        # none by provenance
        ok = (~(pg.astype(np.uint8) @ name_group.astype(np.uint8).T).astype(bool)) & (
            ~(pg_src.astype(np.uint8) @ prov_group.astype(np.uint8).T).astype(bool)
        )
        allowed_mask = ok[self.pair_of_record]  # (n_records, M)

        te, tr, trmask, inner, sizes = [], [], [], [], []
        degenerate = 0
        for m in range(self.M):
            mask = allowed_mask[:, m]
            idx = np.flatnonzero(mask)
            te.append(self.group_records[m])
            tr.append(idx)
            trmask.append(mask)
            sizes.append(int(len(np.unique(self.pair_of_record[idx]))))
            specs = []
            if len(idx) == 0:
                degenerate += 1
                inner.append(specs)
                continue
            for m2 in range(self.M):
                val = np.flatnonzero(mask & self.group_record_mask[m2])
                if not len(val):
                    continue
                imask = mask & allowed_mask[:, m2]
                itr = np.flatnonzero(imask)
                if not len(itr):
                    degenerate += 1
                    continue
                specs.append((val, itr, imask.tobytes()))
            inner.append(specs)
        return Structure(te, tr, trmask, inner, sizes, degenerate)

    # ------------------------------------------------------------- passes ---
    def _pass(self, d, theta, base_row, targets, st):
        cache = {}
        lambdas = k.LAMBDAS
        for m in range(self.M):
            te = st.te[m]
            for val, itr, key in st.inner[m]:
                fit = cache.get(key)
                if fit is None:
                    fit = cache[key] = k.fit(d, itr, theta)
                for j, lam in enumerate(lambdas):
                    row = fit[lam][0][val]
                    for inner, _o, _r in targets:
                        inner[m, base_row + j, val] = row
            key = st.trmask[m].tobytes()
            fit = cache.get(key)
            if fit is None:
                fit = cache[key] = k.fit(d, st.tr[m], theta)
            for j, lam in enumerate(lambdas):
                fwd = fit[lam][0][te]
                rev = fit[lam][1][te]
                for _i, outer, reverse in targets:
                    outer[base_row + j, te] = fwd
                    reverse[base_row + j, te] = rev

    def _blank(self):
        return (
            np.full((self.M, NCONFIG, self.n), np.nan),
            np.full((NCONFIG, self.n), np.nan),
            np.full((NCONFIG, self.n), np.nan),
        )

    def _assemble(self, d, inner, outer, reverse, st):
        score = np.full(self.n, np.nan)
        rev = np.full(self.n, np.nan)
        weight_counts = np.zeros(len(k.WEIGHTS), int)
        lambda_counts = np.zeros(len(k.LAMBDAS), int)
        for m in range(self.M):
            tr, te = st.tr[m], st.te[m]
            block = inner[m][:, tr]
            assert np.isfinite(block).all(), "inner predictions incomplete"
            chosen = select_from_inner(d, tr, block)[0]
            score[te] = outer[chosen, te]
            rev[te] = reverse[chosen, te]
            wi, li = divmod(chosen, len(k.LAMBDAS))
            weight_counts[wi] += 1
            lambda_counts[li] += 1
        assert np.isfinite(score).all()
        return score, rev, weight_counts, lambda_counts

    def run(self, order=None, share=True, structure=None):
        st = structure if structure is not None else self.structure(order)
        if st.degenerate:
            raise RuntimeError("degenerate exclusion pool (%d groups)" % st.degenerate)
        buffers = {f: self._blank() for f in self.families}
        joint = [f for f in ("PaD", "A_match") if f in self.families]
        shared = set(SHARED_THETAS) if (share and len(joint) == 2) else set()
        for family in self.families:
            d = self.data[family]
            weights = k.WEIGHTS if family in ("PaD", "A_match") else d.weights
            for i, theta in enumerate(weights):
                if family in joint and i in shared and family != joint[0]:
                    continue
                targets = (
                    [buffers[f] for f in joint]
                    if (family in joint and i in shared)
                    else [buffers[family]]
                )
                self._pass(d, theta, i * len(k.LAMBDAS), targets, st)
        out = {}
        for family in self.families:
            inner, outer, reverse = buffers[family]
            score, rev, wc, lc = self._assemble(self.data[family], inner, outer, reverse, st)
            out[family] = dict(
                score=score,
                reverse=rev,
                direction=(score - rev) / 2,
                weight_counts=wc,
                lambda_counts=lc,
            )
        out["_pool_sizes"] = st.pool_sizes
        return out
