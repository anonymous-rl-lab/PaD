# Same-rule single-channel controls (SCC)

Pre-registration. Written before any fit is run. Freeze this file, record its SHA-256
in every output record, and do not edit it after seeing any result.

**Compute budget: under one minute of fitting.** The hour is specification, the
degeneracy pre-check, and tabulation.

---

## 0. Why this experiment and not another

Three independent ICLR reviews of v14 and v14.2 each named the same missing control,
independently and without prompting:

> There is no D-only learner fitted under the same rule and the same selector. The
> frozen P score and the untrained phenotype probe are not substitutes, and the table
> caption says so.

This is the gap that caps the score. It is also the only named gap that a single fit
can close.

**This experiment deliberately produces no p-value.** There is no null distribution, no
permutation, no exchangeability assumption and therefore no calibration that can fail.
The output is point estimates from the paper's existing frozen learning rule applied to
restricted inputs, reported with the descriptive pair bootstrap the paper already uses.
That is the entire reason it cannot fail the way the correspondence test failed.

---

## 1. What is fitted

Five learner-panel cells. Each uses the **frozen learning rule unchanged**: the same
class-balanced square loss, the same solve, the same nested endpoint-excluded folds,
the same reversal grouping, the same paired one-standard-error selector, the same five
regularizers.

Only the kernel family changes, and each family is **capacity-matched to the existing
controls at 35 candidates** (7 weights x 5 regularizers), exactly as `A_rbf` is:

| Family | Kernel | Candidates | Panels |
|---|---|---|---|
| `P_only` | `w k_P^lin + (1-w) k_P^rbf` | 7 x 5 | Costanzo, Jonikas |
| `D_only` | `w k_D^lin + (1-w) k_D^rbf` | 7 x 5 | Costanzo, Jonikas |
| `Dplus_only` | `w k_{D+}^lin + (1-w) k_{D+}^rbf` | 7 x 5 | Jonikas |

`w` takes the same seven values already used by `A_rbf`. `k_D^lin` acts on the odd D
descriptors, `k_D^rbf` on the complete retained D vector, mirroring how PaD uses them.
`D+` is the 26-coordinate block of S13.1, accepted only under the same bitwise gate.

`P_only` is included because the current "frozen P score alone" row is an inherited
scalar, not a fitted learner. Without it, a reviewer can ask whether P was simply
undertrained, and the paper has no answer.

Reuse the archived fold assignments and exclusion sets byte-for-byte. Do not re-derive
them.

---

## 2. Degeneracy pre-check, to be run and reported BEFORE the main fits

This is the step whose omission invalidated the correspondence test. It is cheap.

1. **Kernel rank.** For each family and panel, report the numerical rank of each of the
   two base Gram matrices on the full panel, and the condition number. Record them.
2. **Non-constant scores.** For each of the 35 candidates in each family, confirm the
   fitted score is non-constant on at least one outer fold. Report the count of
   candidates that are constant anywhere. **If any candidate is constant on every fold,
   stop and report that before proceeding.**
3. **Statistic support.** Report the number of distinct values the output statistic takes
   across outer folds: correct-count on Costanzo, AUROC on Jonikas. This is a record,
   not a gate, since no permutation null is formed; it exists so that the support of
   every reported statistic is on file from now on.
4. **Selector reachability.** Report how many of the 7 weights are ever selected across
   outer groups. A family that always selects the same `w` is effectively a single
   model and must be reported as such.

Write these to `scc_precheck.json` before any main fit runs.

---

## 3. Pre-registered readings

Declared now, before running. The Jonikas bands are set against the values already in
the paper: PaD 0.881114, `A_match` 0.827664, `A_rbf` 0.810982, frozen P 0.536443.

### Jonikas, `D_only` AUROC

| Outcome | Band | What the paper must then say |
|---|---|---|
| **J1** | >= 0.870 | PaD's Jonikas result is D alone. **Withdraw the joint reading on that panel.** The real-data contribution becomes the Costanzo equivalence; the joint claim rests on the simulation certificate only. |
| **J2** | 0.800 to 0.870 | D carries most of it. Report PaD's margin over `D_only` as the joint increment, with the same descriptive pair bootstrap, and state that it is an increment over a strong single channel. |
| **J3** | 0.650 to 0.800 | Both channels contribute and the joint term does visible work. Report all three single-channel rows and the two joint rows together. |
| **J4** | < 0.650 | Neither channel alone succeeds. This is the strongest available real-data reading and the one the paper currently assumes without evidence. |

### Jonikas, `Dplus_only` against `D_only`

| Outcome | Condition | Reading |
|---|---|---|
| **X1** | `Dplus_only` exceeds `D_only` by more than 0.02 | The retained interface discards something usable on measured evidence. This sharpens the open question of why the fixed interface is worth studying, and it must be reported even though it weakens the interface's motivation. |
| **X2** | within 0.02 either way | The restored descriptors are inert for D alone on this panel, consistent with the per-coordinate diagnostics of S13.3. |
| **X3** | `Dplus_only` below `D_only` by more than 0.02 | The extension degrades a single-channel learner too, so the degradation in `A_match(D+)` is not specific to additive fusion. This weakens a contrast the paper currently draws. |

### Costanzo, `D_only` correct count out of 657

Reference points already in the paper: frozen P 597, `A_rbf` 606, PaD 611, `A_match` 613.

| Outcome | Band | Reading |
|---|---|---|
| **C1** | >= 611 | D alone matches or beats the joint learner. The Costanzo equivalence reading stands but its interpretation changes: the panel is D-driven, not additive-sufficient. |
| **C2** | 597 to 610 | Both channels contribute on Costanzo and neither dominates. |
| **C3** | < 597 | D alone is weaker than the inherited P score; Costanzo is P-driven. |

### Both panels, `P_only` against the frozen P score

If `P_only` materially exceeds the frozen score (597 on Costanzo, 0.536443 on Jonikas),
then the inherited encoder was undertrained and **every comparison against the frozen P
row in the paper must be restated against `P_only` instead.** Declare this now so the
result cannot be set aside later.

---

## 4. What this does not resolve

It says nothing about why a fixed retained interface is an object worth studying. That
is a positioning argument, not a measurement, and no fit will supply it.

It adds no function-class result. The certificate, Theorem 1 and Theorem 2 are untouched.

---

## 5. Outputs

- `scc_precheck.json`: section 2, written before any main fit.
- `scc_results.csv`: one row per learner-panel cell, with AUROC, AP, correct count,
  selected weight counts, selected regularizer counts, and the per-record predictions
  hashed.
- `scc_predictions/`: per-record scores, one file per cell, in the same format as the
  existing ledgers so `verify` can recompute the headline numbers from them.
- Every file carries this specification's SHA-256.

## 6. Stopping rule

There is none to violate: the experiment is five fits, all of which are run. No result
licenses a rerun with different settings. If a pre-check in section 2 fails, report the
failure and stop; do not adjust the family and retry.
