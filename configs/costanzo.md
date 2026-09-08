# Costanzo 2016 external-method comparison v1

Frozen before opening new comparator results, 2026-09-08.

## Task and claim

Primary panel: all 657 previously covered response-orientation pairs, with the
same five endpoint-disjoint folds and inherited expression-response labels.
Costanzo supplies measured double-deletion growth fitness; this is not an
official Costanzo direct-edge benchmark. The frozen P+D result is 611/657.
Original P training ancestry and historical development remain inherited.

The literature audit has not identified a published leaderboard on these exact
labels, inputs and splits. Réd (Zitnik & Zupan, Bioinformatics 2014) is an external
published epistasis-order method. CatBoost, XGBoost and RBF-SVM are external
general classifiers adapted to precisely the retained P+D information; they
are strong comparator families, not claimed Costanzo direction SOTA records.
No opponent score is fabricated from an unrelated genetic-interaction-prediction
task. No paper or historical algorithm is overwritten.

## Supervised comparison

Inputs have 41 columns: raw frozen P score, the remaining seven retained P
coordinates, 12 raw odd D coordinates and 21 retained symmetric D coordinates.
These contain the same information used by the frozen Costanzo kernel. Gene
identities, fold numbers, truth, and candidate direct-expression cells are not
features. The inherited bounded symmetric measurement coordinates are kept;
scaling fitted by SVM uses each training subset only.

Train on forward/reverse augmentation, with reversed labels; evaluate the odd
part of the raw decision score. Reverse records never create new independent
test pairs. For each outer fold, inner validation uses the other four complete
endpoint folds. Choose highest inner correct count, then AUROC, then logistic
loss, then fixed candidate order. No outer-label selection. P+D retains its
original stability selector and 35-candidate grid; replay selection and scores.

CatBoost 1.2.8: depth {3,5,7} x l2_leaf_reg {1,5,20}, 500 iterations,
learning_rate=0.03, all other model settings recorded in code. Nine candidates.
XGBoost CPU 3.0.5: depth {2,4,6} x reg_lambda {1,10} x learning_rate
{0.03,0.1}, 300 trees, subsample=colsample_bytree=1. Twelve candidates.
RBF-SVM: C {0.03,0.3,3,30,300} x gamma {scale,0.01,0.1,1}; twenty candidates.
CatBoost and XGBoost seeds: 17,29,43, all retained. SVM and kernel replay are
deterministic. Seed 17 is the declared individual primary comparator; also
report seed mean/range, never the best seed as a selected method.

## Réd adaptation

Use author's released scoring code, with Python 3 syntax-only port already
present in the project. Validate against current author source. Extract DMA30
NxN rows for the 349 endpoints in the frozen 657-pair panel from the authentic
Costanzo raw archive. All within-panel double measurements, including third
partners, are available without labels. Aggregate replicate/layout double
fitness by median; single fitness by gene median across finite measurements;
H is the multiplicative single-fitness null. Missing G entries remain NaN.
No labels fit this transductive domain baseline. Gene names only index assays.
Input audit found 11 genes with no finite single fitness even in the official
single-mutant workbook. Their baseline S is 1 (wild type), with affected pairs
flagged; results on the common observed-single subset are descriptive secondary
checks for both methods, never replacements for the primary 657 denominator.
Defaults: rank100, lambda_u=lambda_v=1e-4, alpha=beta=0.1, max_iter=200;
seeds17,29,43. Retain the released asymmetric scoring implementation. Primary
direction score is p(A->B)-p(B->A); report released p(A->B)-0.5 separately and
exchange inconsistency. Do not repair the author's score or tune it on truth.
Missing/failed predictions are reported on the full denominator and coverage,
not silently dropped. This method receives D measurements only; it is not a
same-input attribution control. Supervised families above provide that control.

## Gates and budget

1. Input/code audit and a short smoke: endpoint isolation, finite features,
   reversal, original score replay, author-code compatibility.
2. Small validation: one nested outer fold of each new supervised family,
   Réd seed17 on the extracted panel, no parameter amendments from accuracy.
3. Formal: remaining folds and declared seeds if execution is finite/stable.

Upper first-round CPU budget: two hours; expected 10–30 minutes after source
download, calibrated by smoke timing. No GPU or paid compute service. Maximum
tree iterations and seeds are fixed above; no neural epochs. Shared inner
training sets may reuse fits. Record actual fits and durations, not just nominal
workflow count. Stop only for an interface/numerical/resource failure, retain
all successful and failed blocks, do not add seeds or search candidates.

## Reporting

Primary: correct/657 and accuracy; AUROC secondary. Report per-fold scores,
paired repairs/harms relative to P+D, predictions for every pair, and exact
McNemar as descriptive only (shared genes and historical development prevent
an iid confirmatory interpretation). Do not declare SOTA solely because this
bounded comparison wins. Keep previous 613/657 matched-additive result visible
as an internal comparator. No adaptive improvement of either P+D or opponents.
