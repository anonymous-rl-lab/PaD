"""Single-channel kernel families (specification section 1).

Each family keeps the frozen learning rule unchanged -- the class-balanced
square loss, the solve, the nested endpoint-excluded folds, the reversal
grouping, the paired one-standard-error selector and the five regularizers --
and changes only which kernels the candidate mixes over.

Capacity is matched to ``A_rbf``: seven mixing weights over two base kernels,
times five regularizers, is 35 candidates, the same grid the existing controls
search. ``w`` takes the same seven values ``A_rbf`` already uses.

    P_only      w k_P^lin  + (1-w) k_P^rbf
    D_only      w k_D^lin  + (1-w) k_D^rbf

``k_P^lin`` is the release's P-linear component, the outer product of the frozen
scalar P score normalized by its training-fold RMS. ``k_D^lin`` acts on the odd
D descriptors and ``k_D^rbf`` on the complete retained D vector, mirroring how
PaD uses them. Both linear components are rebuilt per training fold by the
release's own ``Data.components``, so the training-only normalization convention
is inherited rather than reimplemented.
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
from controls import RBF_W  # noqa: E402

from frozen_panel import FrozenPanel  # noqa: E402
import dplus  # noqa: E402

SINGLE = {"P_only": "P", "D_only": "D", "Dplus_only": "D"}
WEIGHTS = [(w, 1.0 - w) for w in RBF_W]


class SingleChannel(k.Data):
    """One channel, two kernels, the release's rule around them."""

    def __init__(self, name, channel):
        super().__init__(name)
        z = self.z
        self.channel = channel
        self.weights = WEIGHTS
        self.P = k.rbf(z["xp"], z["xp"])
        self.Pr = k.rbf(z["xpr"], z["xp"])
        self.D = k.rbf(z["xd"], z["xd"])
        self.Dr = k.rbf(z["xdr"], z["xd"])

    def components(self, tr):
        # super() rebuilds the P-linear and D-odd components against this
        # training fold; we keep one of them and pair it with its RBF twin.
        cc, rr = super().components(tr)
        if self.channel == "P":
            return [cc[0], self.P], [rr[0], self.Pr]
        return [cc[1], self.D], [rr[1], self.Dr]

    def base_kernels(self, tr):
        """The two base Gram matrices, for the degeneracy pre-check."""
        cc, _ = self.components(tr)
        return cc


class SCCPanel(FrozenPanel):
    """A panel whose families may be single-channel."""

    def make_family(self, family):
        if family in SINGLE:
            return SingleChannel(self.dataset, SINGLE[family])
        return super().make_family(family)


def build(panel, families, block="D"):
    """Construct a panel, installing D+ where the cell calls for it."""
    fp = SCCPanel(panel, tuple(families))
    report = None
    if block == "D+":
        if panel != "jonikas":
            raise ValueError("D+ is defined for the Jonikas panel only")
        # same bitwise gate as the ablation: the rebuild must reproduce the
        # frozen context coordinates exactly or no block is returned
        xd, xdr, odd, report = dplus.build(fp.frame)
        fp.install_base(xd, xdr, odd)
    else:
        fp.restore()
    return fp, report
