"""Inherited five Proposer features, with original thresholds and coefficients."""
import math
import numpy as np
from scipy.stats import skew, kurtosis
P_THRESHOLD=0.01
EFFECT_THRESHOLD=0.50
class FeatureBuilder:
    def __init__(self, data):
        self.data = data
        self.cache = {}

    def feature(self, u: str, v: str):
        key = (u, v)
        if key in self.cache:
            return self.cache[key].copy()
        gi, si = self.data["gene_index"], self.data["source_index"]
        mask = np.ones(len(self.data["sources"]), dtype=bool)
        for endpoint in (u, v):
            if endpoint in si:
                mask[si[endpoint]] = False
        xu = self.data["effect"][gi[u], mask]
        xv = self.data["effect"][gi[v], mask]
        pu = self.data["pvalue"][gi[u], mask]
        pv = self.data["pvalue"][gi[v], mask]

        def node_stats(x, p):
            count = np.sum((p < P_THRESHOLD) & (np.abs(x) >= EFFECT_THRESHOLD))
            return np.asarray([
                math.log(float(np.var(x, ddof=1)) + 1e-10),
                float(np.nan_to_num(skew(x, bias=False))),
                float(np.nan_to_num(kurtosis(x, fisher=True, bias=False))),
                math.log1p(float(count)),
            ])

        zu = (xu - xu.mean()) / (xu.std(ddof=1) + 1e-8)
        zv = (xv - xv.mean()) / (xv.std(ddof=1) + 1e-8)
        nonlinear = float(np.mean(zu * zu * zv - zv * zv * zu))
        x = np.r_[node_stats(xu, pu) - node_stats(xv, pv), nonlinear]
        self.cache[key] = x
        return x.copy()
