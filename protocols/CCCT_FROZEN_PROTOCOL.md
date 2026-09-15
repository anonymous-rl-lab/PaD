# Cross-Channel Correspondence Test (CCCT): frozen protocol v1.2

Status: DRAFT, to be frozen before execution.
v1.1 added Section 15, the interface ablation.
v1.2 corrects four things after external review: (i) the test is approximate,
not exact, because shared genes break record exchangeability; (ii) a naive
D permutation opens a gene-identity leak through the endpoint exclusion, which
Section 4.6 closes with a provenance-aware exclusion graph and a
placebo-matched observed reference; (iii) Proposition CCCT-1 is one-directional
and the interpretation is rewritten accordingly; (iv) Costanzo non-rejection
is no longer read as "additive suffices"; an equivalence region replaces it.
Section 16 records why out-of-family certification is not part of this
protocol.
Scope: PaD paper, Evidence layer. Adds one prospectively specified test to the
existing Costanzo and Jonikas panels and to four mechanism control settings.
No new biological data, no change to the P encoder, the D adapters, the kernel
family, or the selection rule.

---

## 1. What this test decides

The paper currently reports a joint-minus-additive gain on Jonikas
(AUROC 0.881114 versus 0.827664, difference 0.053450) and a joint-minus-additive
loss on Costanzo (611/657 versus 613/657), and then states that the source of
the Jonikas gain "is not established" and that mixed-context robustness
"remains unresolved". This test replaces both statements with a prospectively
specified decision.

The question is exactly: **is the observed joint-minus-additive gain larger
than what the same joint learner would produce on data in which the two
channels carry no correspondence beyond what each carries about the label?**

The test does not require the mechanism constants `a`, `b`, `rho`, which
Supplementary Section S9 states cannot be estimated on the yeast records. It
conditions on the observed data and randomizes only the coupling between the
two channels.

---

## 2. Theoretical anchor

**Proposition CCCT-1 (additivity under conditional independence).**
Let `Y` be a binary answer label, and let `P` and `D` be the two channel
summaries. If `P` is conditionally independent of `D` given `Y`, then

```
log [ Pr(Y=+1 | P,D) / Pr(Y=-1 | P,D) ]
  = log(pi_+ / pi_-) + log [ p(P|Y=+1) / p(P|Y=-1) ]
                     + log [ p(D|Y=+1) / p(D|Y=-1) ]
  = const + a(P) + b(D).
```

Hence under conditional independence the Bayes-optimal score is additive, and
because any strictly increasing transform of the posterior is also
Bayes-optimal for bipartite ranking, the population joint-minus-additive gain
is exactly zero for both classification risk and AUROC.

This is the converse face of Main Theorem 1. Theorem 1 says that when the
background overlap is positive, every additive scorer has risk at least
`e/2`, a lower bound on absolute risk that an oracle attains; the loss relative
to the joint optimum is at least `e/2 - Phi(-r)`, not "exactly `e/2`".
Proposition CCCT-1 says that when the channels are conditionally independent
given the answer, additive scoring loses nothing.

**The implication runs one way only.** Conditional independence implies
additive sufficiency. Conditional dependence does not imply additive
insufficiency: two Gaussian classes with a common non-diagonal covariance have
`P` and `D` dependent given `Y`, yet the Bayes score is the linear
discriminant `alpha P + beta D`. Three population regimes must therefore be
kept apart: (i) conditionally independent, joint gain zero; (ii) dependent but
additive-sufficient, joint gain zero; (iii) dependent and additive-insufficient,
joint gain positive. The permutation null below is the distribution of the
statistic under (i). Because the statistic is the joint-minus-additive gain
itself, regime (ii) also predicts a population gain of zero, so an observed
excess over the (i)-null is not explained by either (i) or (ii); but the null
is only guaranteed to be the correct reference under (i). Rejection is
therefore read as "the joint learner's gain exceeds what correspondence-free
data produce", not as a proof that additive scoring is insufficient in
population.

Proof of the proposition: one line of the naive-Bayes factorization above,
plus Menon and Williamson (2014) for the ranking half. Include it in the
supplement as a numbered proposition, stated as a one-way implication.

**Null hypothesis.** `H0 : P independent of D given Y`.

**Consequence used by the test.** Under `H0`, and if records were exchangeable
given their labels, reassigning D blocks among same-label records would leave
the joint distribution of the data unchanged and the test would be exact.
Records on both panels share genes and are not exchangeable, and the endpoint
exclusion that the evaluation uses to handle that dependence interacts with
the permutation (Section 4.6). The test is therefore **approximate**: the
permutation removes cross-channel correspondence while the provenance-aware
exclusion of Section 4.6 preserves the evaluation's information boundary, and
the residual asymmetry is declared, quantified by the placebo reference, and
not assumed away.

---

## 3. Frozen inputs

Nothing in this list is refitted or re-derived by the test.

| Object | Source | Status |
|---|---|---|
| P encoder: five-statistic score, coefficients, scales | `verification/frozen_P.json`, Table S1 | frozen |
| P nonlinear summaries (8 coordinates) | S2.1, Eq. (S3) | frozen |
| D adapters: Jonikas 23 coordinates (8 odd, 15 symmetric), Costanzo 33 coordinates (12 odd, 21 symmetric) | S2.2, S2.3, `evidence/Costanzo_feature_schema.json` | frozen |
| Panel-level label-free scaling constants (median nonzero absolute descriptor, unlabeled panel RMS) | S2.2, S2.3 | frozen, and see Lemma 4.1 |
| Kernel family, `h=0.5`, 7 simplex weights, 5 regularizers, 35 candidates | Eq. (2)-(3), S3 | frozen |
| Paired one-standard-error selector, Eq. (S7) | S3 | frozen |
| Outer and inner split structure | S3, S11.2 | frozen |
| Labels, folds, gene identities, P blocks | evidence CSVs | frozen, never permuted |

Everything downstream of the data, that is the training-fold RMS
normalization, the 35 candidate fits, the selector, and the final predictions,
is recomputed from scratch inside every replicate. The test statistic is the
output of the complete frozen procedure applied to a dataset, not a rescoring
of stored predictions.

---

## 4. The permutation operator

### 4.1 Unit and invariance

**Permutation unit: the unordered gene pair.** The entire D measurement block
of a pair moves as one object, including its odd descriptors, symmetric
descriptors, availability flags, missingness masks, measurement counts, and
third-partner context coordinates. Nothing inside a D block is permuted
separately. P blocks, labels, folds and gene identities never move.

**Lemma 4.1 (frozen adapter constants are permutation invariant).**
The panel-level scaling constants are functions of the multiset of D
descriptor values over the whole panel. A permutation is a bijection on that
multiset, so every such constant is unchanged. The frozen adapter therefore
requires no recomputation and introduces no label leakage under permutation.
Training-fold RMS constants are not invariant and are recomputed per replicate
by the existing code path.

### 4.2 Orientation alignment

D odd descriptors are antisymmetric under endpoint exchange. A permutation
that ignored orientation would destroy the `(D, Y)` marginal and invalidate the
test. Both panels therefore move D blocks in a label-aligned orientation.

Write a D block as `D = (D_odd, D_sym)` and let `flip(D) = (-D_odd, D_sym)`.

### 4.3 Costanzo panel

Structure: 657 unordered pairs, one record each, stored in canonical
(lexicographic) endpoint order, `target in {0,1}` indicating whether the
canonical order is the inherited true orientation. Counts: 344 with
`target=1`, 313 with `target=0`. Five endpoint-disjoint folds of sizes
149, 105, 138, 159, 106 over 349 genes.

**Truth alignment.** For pair `g`, define

```
D_truth(g) = D(g)            if target(g) == 1
D_truth(g) = flip(D(g))      if target(g) == 0
```

so that every `D_truth` is expressed along the true orientation. This makes
the whole panel one exchangeable pool and removes the need to stratify by
label.

**Permutation (primary: within fold).** For each fold `f`, draw a uniform
permutation `sigma_f` of the pairs in that fold. Assign

```
D_perm(g) = D_truth(sigma_f(g))         if target(g) == 1
D_perm(g) = flip(D_truth(sigma_f(g)))   if target(g) == 0
```

Each pair keeps its own P, target, genes and fold; only the D block is
reassigned, and it is reassigned from a pair in the same fold. The multiset of
truth-aligned D blocks within each fold and the joint distribution of
`(D, target)` are preserved exactly. Because the five outer folds are
endpoint-disjoint and inner selection uses the other four folds, every D block
in a test fold has provenance inside that fold, so the fold-level endpoint
exclusion continues to hold for D provenance without any further rule. This is
why within-fold permutation is primary on Costanzo; a global permutation
(sensitivity variant `V1_global`) would move D provenance across folds and
reopen the leak described in Section 4.6.

### 4.4 Jonikas panel

Structure: 84 unordered pairs, 2 ordered records each, 168 records, 21
positive and 147 reference-negative annotations, 30 genes. Exactly 21 pairs
carry one positive direction (stratum **O**, oriented) and 63 pairs carry two
negative records (stratum **U**, unoriented). Context availability:
122 records with an observed double phenotype, 46 without.

Reference negatives are unannotated records, not confirmed reverse relations.
The alignment below respects that: it never treats a negative record as
evidence of the reverse direction.

**Stratum O, 21 pairs.** Let `pos(g)` be the endpoint order of the annotated
positive record. Define `D_align(g)` as the D block expressed in `pos(g)`
orientation. Draw a uniform permutation `sigma_O` of the 21 pairs and assign
`D_align(g) <- D_align(sigma_O(g))`, then write back both records of `g`:
the positive record receives `D_align`, its reciprocal receives
`flip(D_align)`.

**Stratum U, 63 pairs.** No annotated direction exists. Express D in canonical
endpoint order, draw a uniform permutation `sigma_U` of the 63 pairs, assign,
and write back both records with the antisymmetric convention as above.

Permuting the two strata separately preserves the empirical joint distribution
of `(D, Y)` at the record level, including the fact that the 147 negatives are
a mixture of 21 oriented-reverse and 126 unoriented D values. Reciprocal
records always move together, so exchange antisymmetry is preserved exactly.

### 4.5 Sensitivity variants (secondary, same statistic)

| Variant | Definition | Purpose |
|---|---|---|
| `V1_global` | Costanzo: truth-aligned permutation across folds, with the Section 4.6 provenance exclusion applied | quantifies what fold restriction alone changes |
| `V2_context` | Jonikas: add context availability (`double_observed`) to the stratification | removes availability heterogeneity |
| `V3_two_sided` | primary strata, two-sided p-value | pre-registered sensitivity on tail choice |
| `V4_rbf` | statistic computed against `A_rbf` instead of `A_match` | second additive control |

Variants are reported alongside the primary and do not alter the primary
decision.

### 4.6 Provenance and the endpoint exclusion

The Jonikas evaluation is leave-pair-out: for test pair `g`, every training
record sharing an endpoint with `g` is removed, and inner validation repeats
the rule. The purpose is to block gene-identity shortcuts for the test genes.
A D permutation changes where each record's D data comes from without
changing the record's name, so a name-based exclusion no longer bounds the
information in the training set. Two channels have to be examined.

**Channel 1 (not a leak).** A training record `t` receives the D block of a
pair sharing a gene with `g`. That block is paired with `t`'s label, which is
unrelated to `g`'s genes, and the test record's own D is no longer `g`'s. No
label information about the test genes reaches the test prediction through
this route.

**Channel 2 (a real leak).** The test record receives the D block of
`h = sigma(g) = (X, Y')`. Training records whose D blocks come from pairs
sharing `X` or `Y'` carry `X`'s single phenotype and partner context, aligned
by label in stratum O. A learner can associate "`X`'s phenotype in the
label-aligned first slot" with the positive label and then match it on the
test record, which holds `D_h` in exactly that alignment. This shortcut is
blocked on the real data by the endpoint exclusion and is open under a naive
permutation. It is available to both learners, so its effect on the
difference statistic has no guaranteed sign, and it makes the null
distribution incomparable to the observed statistic.

**Fix: exclusion on the union provenance graph.** Let `genes(t)` be a pair's
named endpoints and `genes(sigma(t))` the endpoints of the pair whose D block
`t` received. Under permutation `sigma`, for outer test pair `g`, the training
pool is

```
T_sigma(g) = { t : genes(t) ∩ genes(g) = ∅  AND  genes(sigma(t)) ∩ genes(sigma(g)) = ∅ }
```

and inner validation applies the same two-sided rule within the pool. Under
the identity permutation this is the original scheme. Under a random
permutation it removes, on average, about twice as many pairs (roughly 73 to
62 training pairs on Jonikas), and it closes Channel 2 by construction: no
training D block shares a gene with the test record's D provenance, and no
training label shares a gene with the test record's P. A single global
permutation compatible with all 84 fixed exclusion sets is not required; the
exclusion adapts to each test pair.

**Placebo-matched observed reference.** The denser exclusion shrinks
training sets under the null but not for the observed statistic, which would
bias the comparison. The observed reference is therefore computed under the
same regime: draw `K = 100` placebo permutations `sigma'_k`, keep the real D
blocks in place, but apply the exclusion graph `T_{sigma'_k}`. Define

```
Delta_obs^placebo = mean_k Delta( real D, exclusion T_{sigma'_k} )
```

and report its distribution (its spread measures how much the exclusion
regime alone moves the statistic). The p-value in Section 6 compares the null
replicates, each computed with permuted D under its own `T_{sigma_r}`, against
`Delta_obs^placebo`. The untouched observed value (0.0534 on Jonikas) is
reported alongside for continuity with the paper but is not the reference.

**What remains approximate.** Even with matched exclusion, records sharing
genes are not exchangeable, and the exclusion graphs of the null and placebo
sides are equal in law but not identical. The reported p-value is an
approximate permutation p-value with these two sources of approximation
named in the paper. Costanzo needs neither the union rule nor the placebo,
because within-fold permutation keeps provenance inside the test fold.

---

## 5. Test statistics

Let `Pi` denote the frozen complete procedure (encode, normalize per training
fold, fit 35 candidates, select, predict) and let `Delta(dataset)` be computed
after `Pi` has run on that dataset.

**Jonikas primary.**
```
Delta_J = AUROC(PaD) - AUROC(A_match)          over all 168 records
```
Ties receive half credit, matching S7.2. Observed value: `0.0534499514`.
Equivalently, net concordance over the 21 x 147 = 3087 comparisons divided by
3087; the two are identical and the paper already reports 165/3087.

**Costanzo primary.**
```
Delta_C = correct(PaD) - correct(A_match)      over all 657 pairs
```
A zero score is assigned to the positive class, matching S11.2. Observed
value: `611 - 613 = -2`.

**Costanzo equivalence reading.** Non-rejection is not evidence that additive
scoring suffices. To say that, use the same device the paper already uses for
B2: a pre-declared equivalence region. Declare `[-0.02, +0.02]` in accuracy,
that is `[-13, +13]` pairs, and read "additive suffices on Costanzo" only if
the paired instance-bootstrap interval for `Delta_C - delta_flex_C` lies
inside that region. Otherwise report "no evidence of a correspondence gain",
which is the weaker and correct statement.

**Secondary statistics** (reported, not decision-bearing):
`Delta_AP` on Jonikas; direction repairs minus harms on the 21 positive
annotations; `Delta` against `A_rbf` on both panels; the PaD-minus-P-only
difference on both panels.

---

## 6. Null distribution, p-values, and the flexibility offset

Run `R = 2000` independent permutation replicates per panel. Replicate `r`
produces `Delta_r`, computed with permuted D under its own provenance
exclusion `T_{sigma_r}` (Jonikas) or within fold (Costanzo). The one-sided
right-tail p-value is

```
p = (1 + #{ r : Delta_r >= Delta_ref }) / (1 + R)
```

with `Delta_ref = Delta_obs^placebo` on Jonikas (Section 4.6) and
`Delta_ref = Delta_obs` on Costanzo. The resolution floor is
`1/2001 = 0.0005`. This is an approximate permutation p-value; the two
sources of approximation are stated in Section 4.6 and repeated in the paper
wherever the number appears.

Report, for each panel and each variant:

- `Delta_obs`
- `delta_flex = mean_r(Delta_r)` and `sd_r(Delta_r)`
- standardized effect `(Delta_obs - delta_flex) / sd_r(Delta_r)`
- `p_one_sided`, `p_two_sided`
- the 2.5th, 50th and 97.5th percentiles of the null

`delta_flex` is a reported scientific quantity, not a diagnostic. It is the
AUROC that the product kernel buys from flexibility alone on data with no
correspondence to exploit. Reporting it converts the headline number into a
decomposition:

```
Delta_obs = delta_flex + (correspondence-attributable gain) + noise
```

If `delta_flex` is materially positive, say above 0.01 AUROC, that is itself a
finding worth stating: part of the 0.0535 was never attributable to the
mechanism, and the paper should say so.

**Monte Carlo error.** With `R = 2000` and a true p near 0.05, the standard
error of `p` is about 0.005. If the primary p lands in `[0.02, 0.10]`, extend
that panel to `R = 10000` under the same seed stream and report the extended
value as primary. Declare this rule before execution; it is a precision rule,
not an outcome-dependent stopping rule, because the decision threshold is
fixed and only the estimate of `p` is refined.

**Multiplicity.** Two primary tests, one per panel. Holm correction at
family `alpha = 0.05`. Secondary statistics and variants are descriptive and
carry no correction.

---

## 7. Controls: the test must be shown to work before it is believed

Run the identical test on four mechanism settings from Table S9, using the
existing generator and the complete encoders, at `n = 512` training and
`n = 4096` test instances per replicate, `R = 500` per setting.

| Setting | `(a,b,rho)` | Truth about correspondence | Required behavior |
|---|---|---|---|
| B1 | `(0.5, 2, 0.5)` | strong coupled benefit, ideal joint error 0.0187 | **reject**, `p <= 0.002` |
| B5 | `(1, 0, 0.8)` | D marginally independent of Y, helps by shared-noise removal | **reject** |
| B4 | `(1, 0, 0)` | D is pure noise, `P` and `D` genuinely independent given `Y` | **not reject**, empirical size near `alpha` |
| B2 | `(1, 0.5, 0.5)` | channels dependent but `d = 0`, population gain exactly zero | **not reject** |

B4 is the Type I error check: it is the one setting where `H0 : P ⊥ D | Y`
holds exactly by construction (`b = 0`, `rho = 0`, so D is noise independent
of everything), so the empirical rejection rate over 500 replicates estimates
the test size and must fall inside a binomial 95 percent interval around 0.05,
that is `[0.031, 0.073]`. Note that B2's redundancy condition is a different
statement, `Y ⊥ D | P`: the channels there are dependent given `Y`, so B2 is
not a null case for this test and is not used to estimate size.

B5 is the most informative positive control. There, D carries no marginal
information about the label at all, so no marginal screening or correlation
analysis would flag it, yet the joint learner gains. A test that detects B5 is
demonstrably detecting correspondence rather than marginal signal.

B2 is the conservative-behavior check: the channels are dependent given the
label, but the additive class already attains the joint optimum, so there is
nothing for the joint term to buy and the test should not fire. Together B2
and B4 show that the test tracks usable gain, not dependence per se.

**Power calibration at Jonikas dimensions.** Generate mechanism instances at
`n = 168` records with 21 positives and 147 negatives, sweeping the
conditional separation `d` over `{0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0}`, 200
datasets per grid point, `R = 500` permutations each. Report the power curve.
This states, before seeing the Jonikas result, what effect size the panel can
detect. If the Jonikas test does not reject, this curve is what distinguishes
"no correspondence" from "panel too small", and it is the single most useful
object to have in hand during the rebuttal period.

---

## 8. Pre-registered decision rule

Freeze before execution.

```
CCCT_SUPPORTED      Jonikas Holm-adjusted p <= 0.05 AND B1 rejects AND B4 size in [0.031,0.073]
CCCT_NOT_SUPPORTED  Jonikas Holm-adjusted p > 0.05, with controls passing
CCCT_INVALID        B1 fails to reject or B4 size falls outside its interval
```

Costanzo has no rejection requirement in either direction. It is reported
under the equivalence reading of Section 5: "additive suffices" only if the
paired interval lies inside the pre-declared region, otherwise "no evidence of
a correspondence gain". Non-rejection by itself is not read as the instrument
saying no.

If `CCCT_INVALID`, the test is not reported as evidence and the reason is
recorded. No re-specification of strata, statistics or thresholds after seeing
any panel result.

---

## 9. Pre-registered predictions

Record these, with the SHA-256 of this file, before running anything. They
are predictions, recorded so that the theory is seen to take a risk; they are
not decision rules. The decision rules are Section 8 and Section 15.5 only.

1. Jonikas primary `p <= 0.05`, one-sided, against the placebo reference.
2. Costanzo: the paired interval for `Delta_C - delta_flex_C` lies inside
   `[-13, +13]` pairs (the equivalence reading holds).
3. B1 and B5 reject; B2 and B4 do not.
4. `delta_flex` on Jonikas is positive and below 0.02 AUROC.

Prediction 4 is the one most likely to be wrong and is the most useful to
have recorded. If `delta_flex` turns out to be large, the honest reading is
that the product kernel's flexibility explains a substantial share of the
reported gain, and the paper must say so in the abstract.

---

## 10. Output schema

Canonical machine-readable output is JSON. One file per panel and per control
setting, plus one summary. CSV mirrors of the replicate tables are secondary.

```
ccct_results/
  ccct_summary.json
  panels/ccct_jonikas.json
  panels/ccct_costanzo.json
  controls/ccct_B1.json  ccct_B2.json  ccct_B4.json  ccct_B5.json
  power/ccct_power_jonikas_dims.json
  replicates/jonikas_replicates.csv      # r, seed, Delta, auroc_pad, auroc_amatch, selected_weights, selected_lambda
  replicates/costanzo_replicates.csv
  MANIFEST_SHA256.json
```

Per-panel JSON:

```json
{
  "protocol_sha256": "<hash of this file>",
  "panel": "jonikas",
  "variant": "primary",
  "statistic": "auroc_pad_minus_amatch",
  "n_records": 168, "n_pairs": 84, "n_positive": 21,
  "strata": {"oriented_pairs": 21, "unoriented_pairs": 63},
  "replicates": 2000,
  "seed_stream": {"base_seed": 20260915, "algorithm": "PCG64", "spawn": "per-replicate child"},
  "delta_observed": 0.0534499514,
  "null": {"mean": 0.0, "sd": 0.0, "q025": 0.0, "q500": 0.0, "q975": 0.0},
  "delta_flex": 0.0,
  "standardized_effect": 0.0,
  "p_one_sided": 0.0, "p_two_sided": 0.0, "p_holm": 0.0,
  "secondary": {"delta_ap": 0.0, "direction_repairs_minus_harms": 0, "vs_a_rbf": 0.0},
  "environment": {"python": "", "numpy": "", "scipy": "", "pad_causal": "0.1.1"},
  "runtime_seconds": 0.0,
  "failed_replicates": 0
}
```

`failed_replicates` must be zero. A replicate that raises, produces a
degenerate Gram matrix, or fails the selector is a protocol failure, not a
data point to drop. If any occur, record them, fix the cause, and rerun the
whole panel.

---

## 11. Reference implementation

Runs inside the `v0.1.1` source release, where the frozen encoders, adapters,
splits and selector already exist. Entry point:
`python -m ccct.run --panel jonikas --replicates 2000 --seed 20260915`.

```python
import hashlib, json, numpy as np

# ---------- permutation operators ----------

def flip_odd(D, odd_idx):
    """Endpoint exchange acting on one D block."""
    Dp = D.copy()
    Dp[odd_idx] = -Dp[odd_idx]
    return Dp

def permute_costanzo(D, target, odd_idx, rng):
    """D: (657, 33). target: (657,) in {0,1}. Truth-aligned global permutation."""
    n = D.shape[0]
    D_truth = np.stack([D[g] if target[g] == 1 else flip_odd(D[g], odd_idx)
                        for g in range(n)])
    sigma = rng.permutation(n)
    D_new = np.empty_like(D)
    for g in range(n):
        blk = D_truth[sigma[g]]
        D_new[g] = blk if target[g] == 1 else flip_odd(blk, odd_idx)
    return D_new

def permute_jonikas(D_pair, pos_order, stratum, odd_idx, rng):
    """D_pair: dict pair_id -> D block in pos_order (stratum O) or canonical
    order (stratum U). stratum: dict pair_id -> 'O' | 'U'.
    Returns a new dict; caller writes both ordered records back, the reciprocal
    receiving flip_odd of the assigned block."""
    out = {}
    for s in ('O', 'U'):
        ids = [p for p in D_pair if stratum[p] == s]
        sigma = rng.permutation(len(ids))
        for k, p in enumerate(ids):
            out[p] = D_pair[ids[sigma[k]]].copy()
    return out

# ---------- one replicate ----------

def replicate(panel, D_perm, frozen, statistic):
    """Runs the COMPLETE frozen procedure on permuted D.
    Must call the same code path as the reported results: training-fold RMS
    normalization, 35 candidates, paired one-standard-error selector, nested
    endpoint-excluded splits. Panel-level frozen scaling constants are
    permutation invariant (Lemma 4.1) and are NOT recomputed."""
    pad     = frozen.fit_predict(kind="pad",      D=D_perm)
    amatch  = frozen.fit_predict(kind="a_match",  D=D_perm)
    return statistic(pad, amatch)

# ---------- driver ----------

def run(panel, R, base_seed, frozen, statistic, permute, obs):
    ss = np.random.SeedSequence(base_seed)
    children = ss.spawn(R)                      # reproducible, order independent
    deltas = np.empty(R)
    for r, child in enumerate(children):
        rng = np.random.Generator(np.random.PCG64(child))
        deltas[r] = replicate(panel, permute(rng), frozen, statistic)
    p_one = (1 + int((deltas >= obs).sum())) / (1 + R)
    return {
        "delta_observed": float(obs),
        "delta_flex": float(deltas.mean()),
        "null": {"mean": float(deltas.mean()), "sd": float(deltas.std(ddof=1)),
                 "q025": float(np.percentile(deltas, 2.5)),
                 "q500": float(np.percentile(deltas, 50)),
                 "q975": float(np.percentile(deltas, 97.5))},
        "standardized_effect": float((obs - deltas.mean()) / deltas.std(ddof=1)),
        "p_one_sided": p_one,
        "p_two_sided": float(2 * min(p_one, 1 - p_one + 1 / (1 + R))),
    }
```

Implementation requirements:

1. `frozen.fit_predict` must be the same function object that produced the
   reported 611/657 and 0.881114. Verify this before any permutation by
   running it with the identity permutation and asserting bitwise equality
   with the stored predictions in `evidence/evidence_Costanzo.csv` and
   `evidence/external/jonikas/results/pair_predictions.csv`. This identity
   check is replicate zero and must pass, otherwise the whole run is void.
2. Seeding uses `SeedSequence.spawn`, so replicates are reproducible in any
   order and parallelize without stream collision. Record the base seed.
3. Parallelize over replicates with one numerical thread per worker, matching
   the determinism convention already used in S8.
4. Persist every replicate's `Delta`, selected simplex weight and selected
   regularizer. The distribution of selected weights under the null is a free
   and informative by-product: if the selector picks the pure product kernel
   about as often under permutation as on the real data, that is worth
   reporting.
5. Build the exclusion graph from provenance, not from names alone. Under
   permutation `sigma`, `excluded(g) = {t : shares_gene(t, g) or
   shares_gene(sigma(t), sigma(g))}` at the outer level, and the same rule
   inside the pool at the inner level. Assert that with `sigma = identity` the
   graph equals the stored one. Placebo runs use `sigma'_k` for the graph and
   the identity for D.

```python
def exclusion_pool(g, sigma, genes):
    Sg, Shg = genes[g], genes[sigma[g]]
    return [t for t in genes
            if not (genes[t] & Sg) and not (genes[sigma[t]] & Shg) and t != g]
```

**Budget.** Do not estimate from the cache-replay timings. Supplementary
S10.4 records that four full nested Jonikas evaluations without caches used
779,240 candidate-inner-fold evaluations, about 30 seconds each on one thread,
and a full Costanzo nested evaluation is of the same order. One replicate is
two learners on one panel, about one minute; the Jonikas placebo side adds
`K = 100` further evaluations. Two panels at `R = 2000` is therefore roughly
70 CPU-hours, about 4.5 hours on 16 cores, plus controls and the power sweep
overnight. **Run a 10-replicate pilot first and time it; freeze `R` only
after the pilot.** The pilot's replicates are discarded and not reused.

---

## 12. Freeze procedure

1. Fill in the four pre-registered predictions in Section 9.
2. Compute `sha256` of this file and write it to
   `protocols/CCCT_FROZEN_PROTOCOL.sha256`.
3. Commit before executing anything. The commit must contain no results.
4. Every output JSON carries `protocol_sha256` and is checked against that
   file by the verifier.
5. Add a `verify_ccct.py` that recomputes `p` from the stored replicate CSVs
   and re-checks the identity replicate, matching the existing
   `verify_v11.py` convention.

---

## 13. What the test establishes, and what it does not

**Establishes, on rejection.** The two channels are not conditionally
independent given the answer, and the dependence is of a kind that the joint
kernel converts into ranking gain beyond the flexibility offset. Combined with
Proposition CCCT-1, this attributes a nonzero part of the observed gain to
cross-channel correspondence, which is the mechanism Main Theorem 1 describes.

**Does not establish.** That additive scoring is insufficient in population:
dependent-but-additive-sufficient regimes exist (Section 2), and the null is
exact only under conditional independence. Nor exactness: shared genes break
exchangeability and the provenance-matched exclusion restores the information
boundary, not exchangeability. Nor that the correspondence is the shared
latent background `C` of Section 5.1. The mechanism's specific generative story
remains a model, and Supplementary Section S7.4 already reports that
P-to-background-proxy prediction on Jonikas is near chance. The wording in the
paper must stay at "cross-channel correspondence", never "the shared
background is identified". It also does not establish transfer to any other
panel, and it does not convert the development panels into held-out data.

**On non-rejection.** With the power curve of Section 7 in hand, a
non-rejection reads either as "no detectable correspondence at this panel
size" or as "the panel cannot detect effects below `d = x`". Both are
publishable statements, and both are better than the current "not
established". If Jonikas does not reject, the honest abstract sentence is:
the joint gain on Jonikas is not distinguishable from the joint learner's
flexibility on correspondence-free data at this sample size, and the certified
result stands on the mechanism panels alone.

---

## 14. How the result enters the paper

Both outcomes have a prepared home. Write both paragraphs before running.

**If supported.** Section 6.4 becomes "Cross-channel correspondence is
detectable on Jonikas and absent on Costanzo", with one table giving
`Delta_obs`, `delta_flex`, `p`, and the four control outcomes. The abstract
gains one clause: on one of two yeast panels the joint gain exceeds a
correspondence-free permutation null at `p = ...`, while on the other the
matched additive control is not exceeded. The Discussion's "not established"
sentence is deleted. This is the paper's only inferential statement on real
data and it should be stated as such.

**If not supported.** Section 6.4 becomes "A prospective test for
cross-channel correspondence, and what the yeast panels can detect", with the
power curve as the main figure. The contribution list changes (iii) to a
protocol contribution: a prospectively specified, provenance-matched test for
whether joint interpretation is warranted, validated on four mechanism
settings and applied to two panels. A test that returns "not warranted" on
Costanzo and "undetectable at this size" on Jonikas is a working instrument,
and it is a considerably stronger position than the current text.

Either way, this test is the thing that converts PaD from a method with a
theorem and two descriptive panels into a method that makes a falsifiable
claim about real data and reports the answer.

---

## 15. Companion experiment: interface ablation (restored-direction additive control)

### 15.1 Why this is required

Supplementary Section S4.2 states that ordered, uncompressed double-perturbation
profiles reveal the upstream endpoint. In the mechanism this is exact: with
partner W, the ordered profile pair is `(0, B e_W)` when A is upstream and
`(B e_W, 0)` when A is downstream, and `B = exp(nu C + sigma_D Z_D) > 0`
always. The raw D channel is therefore a noise-free direction oracle. Only the
exchange-invariant context coordinates (18 to 22: correlations, absolute
residual difference, sign agreement) discard the order and make D
"background-only". The abstract's claim that an assay carries no directional
information on its own is true of the retained representation, not of the
assay.

CCCT (Sections 1 to 14) tests whether the two retained channels are
conditionally dependent given the answer in a way the joint learner exploits.
It cannot tell whether that dependence is a property of the evidence or a
consequence of the interface having deleted direction. This section answers
that question directly: restore the discarded direction to the additive
control and see whether the gap closes.

This experiment applies to Jonikas only. Costanzo retains no third-partner
context, so there is nothing to restore.

### 15.2 Construction of D+

Let `M` and `M_H` be the partner sets already defined in S2.2, and
`r_{g,E} = L_{g,E} - H_{g,E}`. Add three odd context descriptors, computed from
exactly the inputs that produce coordinates 20 to 22:

```
c23 = tanh( mean_{E in M_H} (r_{A,E} - r_{B,E}) / d_23 )
c24 = 2 * frac_{E in M_H} ( r_{A,E} > r_{B,E} ) - 1          (0 when M_H empty)
c25 = tanh( mean_{E in M}   (L_{A,E} - L_{B,E}) / d_25 )
```

Each flips sign under endpoint exchange. `d_23` and `d_25` are the median
nonzero absolute value of the raw descriptor over the unlabeled panel, floored
at 0.01, the same rule the existing odd descriptors use. They are computed
once, label-free, before any fitting, and are then frozen. Missing-context
masks set all three to zero. `D+ = D concatenated with (c23, c24, c25)`,
giving 26 coordinates with 11 odd and 15 symmetric. The linear D kernel acts
on the 11 odd descriptors, the RBF D kernel on all 26, with `d_D = 26` in the
mean squared coordinate distance.

Nothing else changes: same P block, same 35 candidates, same selector, same
outer groups and inner endpoint exclusion.

### 15.3 Learners and contrasts

Fit, on the unpermuted Jonikas panel:

| Learner | Input | Status |
|---|---|---|
| `PaD(D)` | 8 P + 23 D | existing result, 0.881114 |
| `A_match(D)` | 8 P + 23 D | existing result, 0.827664 |
| `A_match(D+)` | 8 P + 26 D+ | new |
| `A_rbf(D+)` | 8 P + 26 D+ | new |
| `PaD(D+)` | 8 P + 26 D+ | new |

Report AUROC and AP for each, and three contrasts:

```
Delta_1 = AUROC PaD(D)     - AUROC A_match(D+)   does restoring direction close the gap additively
Delta_2 = AUROC PaD(D+)    - AUROC A_match(D+)   does synergy persist once direction is restored
Delta_3 = AUROC A_match(D+) - AUROC A_match(D)    how much usable direction the discarded context carried
```

Run CCCT (Section 4.4 strata, `R = 2000`) on `D+` as well, with statistic
`PaD(D+) - A_match(D+)`, so that the correspondence test and the interface
ablation are reported together. The D+ block, including the three new
coordinates and their masks, moves as one unit under permutation; Lemma 4.1
applies unchanged.

### 15.4 Mechanism control

Run B1 with D+ (same generator, same 512/4096 design, seeds 17, 29, 43).
Prediction: `A_match(D+)` attains near-zero error, because `c23` has sign
`-Y` deterministically in the noiseless summary and remains almost
deterministic at the 0.3% and 1% noise levels. Record and report this. It is
the demonstration that the mechanism's difficulty is a property of the
exchange-invariant interface, and the paper is better off stating it than
having a reviewer derive it from S4.2.

### 15.5 Pre-registered readings

Freeze before execution, alongside Section 9.

```
R1  Delta_1 <= 0.01
    The Jonikas gain is attributable to the interface's deletion of direction.
    Withdraw the real-data conditional-interpretation claim; restate Theorem 1
    as a result about exchange-invariant retained representations; keep the
    mechanism certification and the predictive comparison.

R2  Delta_2 >= 0.03 AND CCCT on D+ rejects at Holm-adjusted p <= 0.05
    Synergy persists beyond the interface. This is the strongest available
    real-data statement and goes in the abstract.

R3  |Delta_3| <= 0.01
    The discarded context carried no usable direction on this panel, unlike
    the mechanism. This is the empirical justification for the
    exchange-invariant interface and must be stated as such.

R4  anything else
    Report all three contrasts descriptively; the abstract makes no
    attribution claim on real data.
```

R1 and R2 are mutually exclusive by construction. R3 can hold together with
either. If R1 holds, the CCCT rejection on D (Section 8) is reinterpreted: the
correspondence the joint learner exploited was the direction the interface
removed, which is real conditional dependence but not the mechanism's
background coupling.

### 15.6 How it enters the paper

One additional row block in the main real-data table, five learners on
Jonikas, plus one sentence in Section 3 giving the reason for the
exchange-invariant context interface, which after this experiment will be an
empirical statement rather than an unexplained design choice. The abstract
sentence changes from "an assay can carry no directional information on its
own" to "a retained phenotype representation can carry no directional
information on its own", in every outcome.

### 15.7 Cost

Five learners on Jonikas: about two minutes. B1 with D+: about one minute per
seed. CCCT on D+ at `R = 2000`: about 14 CPU-hours, or `R = 1000` if the
budget is tight, with the same precision rule as Section 6. Total well under
the CCCT budget already allocated.

---

## 16. Out-of-family certification is not part of this protocol

An earlier suggestion proposed adding out-of-family generators (non-Gaussian
background variability, background-dependent elasticity, nonzero nulls) and
re-running the independent-test certification with the same additive lower
bound. That would be an error. The Hoeffding correction controls the
deviation of a test error from the model's risk on the test distribution; it
does not supply the additive lower bound on that distribution. The bound must
be re-derived for each generator or the certificate is void, and a void
certificate that "passes" would damage the paper more than any failure.

The dichotomy, recorded so that nobody runs it the wrong way:

- If the new generator satisfies the general overlap theorem's conditions
  (`T` balanced and independent of `(C, X, E)`, `P = psi(T, X)`,
  exchange-invariant `D = chi(C, E)`, `E ⊥ C`, `X ⊥ E | C`), recompute the
  overlap `e` for that generator by quadrature and certify against the
  recomputed bound. Non-Gaussian background variability is in this case.
  Background-dependent elasticity is in this case only after `P = psi(T, X, K)`
  is re-profiled and the overlap of `(X, K) | C` is computed; it is outside
  Corollary 1's closed form but not outside Theorem 1.
- If it does not, as with nonzero nulls, which can give the retained D
  directional information and break exchange invariance, report out-of-family
  predictive performance against the finite controls and do not call it a
  function-class certificate.

Neither branch is required for this submission. The fitted
non-antisymmetrized additive learner in the mechanism experiments (a
finite-control addition that needs no new bound) is the higher-priority
control.

---

## 17. Execution addendum (frozen with this file)

This section is part of the frozen document and is covered by its SHA-256. It
records the execution machine, the pilot measurement that Section 11 requires
before `R` is frozen, the values of `R` frozen on the basis of that pilot, and
every point at which execution departs from Sections 1 to 16. Nothing here
changes a decision rule, a statistic, a stratification or a threshold.

### 17.1 Execution environment

Four CPU cores, not the sixteen assumed in the Section 11 budget. Python
3.12.3, NumPy 2.3.5, SciPy 1.17.0, pandas 2.2.3, scikit-learn 1.8.0, release
`VERSION` 3. One numerical thread per worker throughout.

### 17.2 Pilot timing (Section 11 requirement)

Ten null replicates and six placebo replicates of the Jonikas primary panel,
under the Section 4.6 provenance exclusion, ran in 5.0 minutes of wall time on
four workers: about **50 CPU-seconds per replicate**, where one replicate is
the complete frozen procedure for PaD and for `A_match`. Costanzo under
within-fold permutation is about 10.6 CPU-seconds per replicate. One mechanism
control replicate is about 2.2 CPU-seconds. The pilot replicates are discarded
and are not reused.

### 17.3 Frozen replicate counts

| Run | `R` | Placebo `K` | Estimated wall time on four cores |
|---|---|---|---|
| Jonikas primary, block D | 1000 | 100 | 3.8 h |
| Costanzo primary, `V1_fold` | 2000 | 0 (rule is a no-op) | 1.5 h |
| Mechanism controls B1, B2, B4, B5 | 500 each | n/a | 0.3 h total |
| Jonikas, block D+ | 500 | 100 | 1.9 h |
| Sensitivity variants | 500 | 100 where active | as budget allows |

`R = 1000` on Jonikas rather than the 2000 of Section 6 is a budget decision
forced by the four-core machine, taken before any panel result was seen. It
costs resolution, not validity: the floor is `1/1001 = 0.000999` against a
Holm threshold of `0.025` for the smaller of two p-values, a factor of
twenty-five of headroom. The Section 6 precision rule is restated
correspondingly: if the Jonikas primary p lands in `[0.02, 0.10]` the run is
extended to `R = 3000` under the same seed stream and the extended value is
reported as primary. The decision threshold is unchanged, so this refines the
estimate of `p` and is not an outcome-dependent stopping rule.

### 17.4 Declared departures

1. **`c24` is written as a signed rank fraction**, `frac(r_A > r_B) - frac(r_A < r_B)`,
   rather than `2 * frac(r_A > r_B) - 1`. The two agree exactly when no partner
   ties; the Jonikas panel has zero ties, so on the panel of Section 15 they are
   the same number. Only the signed form is exactly antisymmetric under endpoint
   exchange when ties do occur, and exact antisymmetry is required for the
   Section 4 operator to preserve the `(D, Y)` marginal. The mechanism profiles
   of Section 15.4 tie on six of seven coordinates by construction.

2. **The identity check of Section 11 requirement 1 is formulated in process.**
   Bitwise equality with the archived CSVs is not attainable and not this code's
   fault: the release is bitwise deterministic across identical fresh processes
   but shifts by about `4e-15` under a different module import order. Replicate
   zero therefore asserts, in one process, (a) that the fast runner equals an
   unmodified `nested.train` bitwise, (b) that the shared-kernel pass is bitwise
   neutral, (c) that the identity permutation reproduces the unpermuted run
   bitwise, (d) that the identity provenance graph equals the archived exclusion
   set for every outer group, and (e) that the archived predictions agree to
   better than `1e-12` and that `611`, `613`, `0.8811143505021056` and
   `0.8276643990929706` are reproduced exactly. All five held on both panels.

3. **`V5_target`**, a Costanzo permutation stratified by the orientation label,
   is added to the Section 4.5 sensitivity list. Under it the multiset of D
   blocks within each label is preserved exactly, which is the literal reading of
   Proposition CCCT-1's null; the primary `V1_fold` and the global variant pool
   more widely. It is descriptive and carries no correction.

4. **The provenance rule of Section 4.6 is applied to every panel and variant**,
   not only to Jonikas. It is a verified no-op for Costanzo `V1_fold`
   (525.6 training pairs before and after), which is the argument of Section 4.6
   made mechanically rather than by hand, and it is what makes the global
   Costanzo variant a sensitivity only: there it collapses the pool to 55.7
   pairs. The placebo side is run only where the rule is active.

5. **The power calibration of Section 7 uses the mechanism's own five-fold
   selection at `n = 168` with 21 positives and 147 negatives**, not the Jonikas
   leave-pair-out procedure. Running the leave-pair-out procedure at every grid
   point would cost on the order of a thousand CPU-hours. Datasets are drawn by
   generating with `engine.raw` at the scenario's parameters, encoding with the
   complete encoders, and then sampling 21 positive and 147 negative records; the
   grid, the number of datasets per point and `R` per dataset are reduced from
   `(7, 200, 500)` to what the four-core budget allows, and the reduction is
   recorded in the output file. The curve is therefore a power statement about a
   procedure matched to the panel's dimensions and class balance, not about the
   leave-pair-out pipeline itself, and the paper must say so.

6. **Mechanism controls draw their own random stream** (`phase = "ccct"`), so no
   control replicate reuses a formal test instance. Their observed statistics are
   consequently not the Table S9 numbers and are not presented as such.

### 17.5 Observations made before this file was frozen

Recorded so that the pre-registration is auditable rather than overstated.
During implementation, before this commit:

- The Section 15 ablation was executed in full. Its readings `R1` to `R4` were
  fixed by the delivered protocol v1.1 before any execution and are unchanged.
- A three-replicate Jonikas smoke test returned deltas in `[-0.140, -0.041]`,
  and the ten-replicate pilot of Section 17.2 returned `delta_flex = -0.0421`.
  Pre-registered prediction 4 says `delta_flex` on Jonikas is positive and below
  0.02; it is left exactly as written, and it now looks likely to be refuted in
  sign. That is the outcome the prediction was recorded to expose.
- No panel p-value, no control decision and no power curve had been computed.
