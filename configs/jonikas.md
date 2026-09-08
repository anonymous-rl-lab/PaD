# Jonikas external-method comparison v1 — frozen before new supervised fits

Date: 2026-09-08. Goal: compare frozen v4 P+D with external predictors on the
released KEGG ordering panel. This is a reused development panel, not a new
independent confirmatory cohort or an official benchmark leaderboard.

## Task and information

Exactly 168 ordered records, 84 unordered pairs, 21 positive reference records,
147 nonordered reference records and 30 endpoints. Primary: pooled AUROC;
secondary: average precision. Known-direction diagnostic: compare forward and
reverse scores only on the 21 positive records. A negative reference is not a
confirmed reverse direction. Full scores retain symmetric support. No forced
antisymmetrization, complementary-label augmentation or accuracy on all 168.

P+D is frozen at AUROC 0.8811143505021056. Use its exact KEGG.csv/npz and
label-endpoint exclusions. External supervised inputs retain all information
from P (8) and D (23): raw P score, seven P symmetric summaries, eight raw D
odd values, fifteen D symmetric values. No gene names, reference-network
features, Réd scores or hidden mechanism variables are supplied.

The input adapter's historical unlabeled panel transformations are preserved
for all learners. New SVM scaling is fitted only on each training subset.
These splits isolate supervision endpoints; they do not hide all test-gene
unlabeled phenotypes from the common input construction.

## Splits and selection

Outer: leave one unordered pair out and exclude every labeled record sharing
either endpoint. Inner: repeat pair endpoint exclusion within the outer
training pool. Reverse records remain grouped. There are 1,405 unique inner
training sets, each cached once per candidate/seed. All input identities and
class counts are audited before fitting.

All supervised methods use class-balanced training weights. Within a family,
choose greatest pooled inner AUROC, then AP, then lowest class-balanced
logistic loss, then candidate index. This is a predeclared external-method
selector, not the frozen P+D one-SE kernel selector. P+D is replayed under its
original selector. No outer result chooses a grid, seed, scoring sign or model.

Fixed external grids, selected before smoke/pilot:

- RBF SVM: C in {0.3,3,30}, gamma in {scale,0.1,1}; 9 candidates,
  training-only StandardScaler; deterministic, seed 17 identifier.
- XGBoost 3.0.5: max_depth in {2,4}, reg_lambda in {1,10}; 4 candidates,
  learning_rate=0.05, 300 trees, hist, subsample=colsample_bytree=1,
  one numerical thread; seed 17. Deterministic full-sampling algorithm, no
  redundant seed runs are planned.
- CatBoost 1.2.8: depth in {3,5}, l2_leaf_reg in {1,10}; 4 candidates,
  500 iterations, learning_rate=0.03, Logloss, one thread. Seeds 17,29,43;
  all reported, seed 17 is primary, no best-seed selection.
- Réd: original published settings rank=100, lambda_u=lambda_v=1e-4,
  alpha=beta=0.1, max_iter=200 and original early stopping. Seeds 42,43,44
  already exist; replay saved model scores, and rerun seed 42 to check source
  agreement. Original released single-direction probability is primary.
  No optimization or correction of the Réd scoring equation.
- APN: recover original author code and reference list; report published
  0.648 only as a literature value unless the original training is reproduced.
  Do not relabel a surrogate or a port without verification as APN.

## Compute stages and stopping

1. Smoke: frozen P+D replay, original-reference matching, all split checks,
   one fit of each supervised family; cap 10 minutes.
2. Pilot: one fixed outer pair (first in the frozen order) for all three
   families, seed 17, full fixed grids; cap one hour. Assess execution,
   candidate independence and cost, not a favorable partial score.
3. Formal if the interface and projected cost pass: all 84 outer pairs for
   SVM17, XGB17 and CatBoost17/29/43. 420 selected supervised models,
   35,125 unique candidate–inner-training-set fits before cache reuse, plus
   final fits. Four CPU workers, one thread/model. No neural epochs. Budget
   one hour wall clock after smoke; no GPU or paid compute job. Cache shared
   across pilot/formal. All seeds/candidates retained. No expansion of grids
   or P+D architecture based on results.

Per-fit ledger records actual calls and times. Verification replays all final
models and all 168 scores, checks source hashes, metrics, exact record identity
and endpoint separation. Reporting includes P+D, inherited P, two nonlinear
additive controls, raw biological probes, all new external models and all Réd
seeds. Published APN/Réd numbers are separate from local replay numbers.
Paired ranking changes and leave-one-gene sensitivity are descriptive; 3,087
positive–negative comparisons are not treated as independent observations.

No main-paper edit or silent replacement of the existing result is authorized
by this execution. A separate fully reproducible result package is delivered.
