# Shared supervised input

`experiments/jonikas/external.py:data()` returns a finite 168 x 31 matrix from the frozen KEGG NPZ.
Column numbering below is zero-based. No reference labels or gene identifiers
are added as covariates.

| Columns | Exact source | Meaning |
|---|---|---|
| 0 | `z['p']` | Frozen P directional score; the joint RBF uses tanh(p/2), and the P linear component retains p |
| 1–7 | `z['xp'][:,1:]` | Expression correlation, cosine, overlap, shared/union response counts, variance product, count imbalance summaries |
| 8–15 | `z['odd']` | The eight D odd coordinates supplied to the original linear component |
| 16–30 | `z['xd'][:,8:]` | Fifteen D symmetric quality, phenotype, missingness and matched third-party context summaries |

For Jonikas, `z['odd']` is exactly `z['xd'][:,:8]`. These D values already
contain the historical nonlinear measurement transforms. The word “raw” in
the execution protocol distinguishes the values supplied to a component from
newly rescaled values; it does not describe untransformed biological assays.
No feature is recomputed to suit an external competitor.

SVM fits its StandardScaler inside each inner/final training subset. Boosted
trees use the supplied coordinates without an additional cross-record scaler.
P+D's component RMS remains training-only under its original solver; its
frozen RBF input adapters retain the historical unlabeled panel convention.

The original complete KEGG CSV also contains historical Réd and diagnostic
columns. Those columns are not selected into X. Feature extraction selects
only the NPZ arrays above. The supplementary `data/prepared/frozen_P.json`
records the inherited P training; this comparison does not refit P.
