# GitHub reproducibility release v5 — aligned to paper v31

This release aligns the repository with manuscript **v31** and integrates the
source-support experiments the paper reports. The frozen science is unchanged:
the shared learner, the seven-weight / five-regularizer selection rule, the
mechanism protocol, the prepared inputs and every archived reference prediction
are byte-identical to v4.

## What the repository is for

The manuscript archive carries the paper, its evidence ledgers and lightweight
verification. This repository carries the part that archive explicitly does not:
the machinery that **refits** the results — the 120 mechanism learners, the
external comparators, the Réd port, the yeast refits, the same-rule
single-channel controls, and the source-support experiments.

## Added in v5

| Path | Contents |
|---|---|
| `experiments/source/` | The two source-support experiments as integrated in paper v31: joint-control (pure RBF J) refits, the source ablation branches (S, SM, SH, SMH, SMHV, SMHB, FULL, SHVB), the frozen source-task hardening, and the shared statistics generators that rebuild every reported interval from frozen scores. |
| `scc/`, `scc_results/` | Same-rule single-channel controls (`P_only`, `D_only`, `Dplus_only`) fitted under the frozen rule at matched capacity, with the eleven per-record ledgers the appendix cites. |
| `src/pad/frozen_panel.py` | The panel runner these two use. Every number still comes from `kernel.fit` and `selection.select_from_inner`; the module only caches the archived split structure and shares candidate kernels that two families build identically. |
| `src/pad/dplus.py` | The restored-direction D+ block, gated on reproducing the frozen context coordinates bitwise from the archived MAT file. |

## Removed in v5

Nothing that produces a number in the paper was removed. What went:

| Removed | Why |
|---|---|
| Cross-channel correspondence test (package, results, protocol, verifier, docs) | Its own pre-registered decision rule returned `CCCT_INVALID` — the B4 size check failed its interval — so it reports no evidence and does not appear in v31. |
| An independent-replication feasibility proposal | A design for work that was never run. |
| Paper-edit drafts | Superseded by the v31 manuscript. |
| SCC process reports | Their data survives; v31 cites the ledgers, not the write-ups. |
| v4 release-process documents | Replaced by this file. |
| `runs/` | Generated output that should never have been tracked. |

## Verify

```bash
python -m pip install -r requirements-core.txt
python reproduce.py verify                              # archived evidence, encoders, integrity
python experiments/source/code/verify_stored_results.py # source and J statistics from frozen scores
python -m scc.run                                       # refits the five single-channel cells
```

`reproduce.py verify` recomputes every delivered metric and checks the manifest;
it does not retrain. The source verifier regenerates all reported source/J
descriptive statistics from frozen scores and asserts 97 numeric checks. `scc.run`
does refit, in about two and a half minutes.
