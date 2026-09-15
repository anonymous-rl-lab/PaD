# CCCT execution scope

Protocol `protocols/CCCT_FROZEN_PROTOCOL.md`, frozen at
`b76a942bae0d1acab9be8a1f81a8effdc7c083c4bfddada02e7e872f18dcfa8b`.

The frozen protocol is not edited to match what was executed. This file records
what was run and what was not, so the gap between the two is visible rather
than inferred. `ccct_results/ccct_summary.json` carries the same information
mechanically, in its `completeness` block.

## Executed

| Run | Protocol section | Status |
|---|---|---|
| Replicate zero, both panels | 11, requirement 1 | passed |
| Interface ablation, five learners on Jonikas | 15.3 | complete |
| Interface ablation, B1 with D+, seeds 17/29/43 | 15.4 | complete |
| Costanzo primary panel, `V1_fold`, R = 2000 | 4.3, 5, 6 | complete, 0 failed replicates |
| Mechanism controls B1, B5, B2, B4, R = 500 each | 7 | complete, all four as pre-registered |

### Results as executed

Costanzo primary, `V1_fold`, R = 2000. The provenance rule is a verified no-op
here (525.6 training pairs observed and under permutation alike), so no placebo
side is needed and the untouched observed value is the reference.

| | |
|---|---|
| `Delta_C` | −2 (611 versus 613 correct) |
| `delta_flex` | −0.5095 |
| null | median 0, sd 2.447, 2.5th/97.5th percentile −6 / +4 |
| standardized effect | −0.609 |
| approximate one-sided p | 0.8216 (1643 of 2000 replicates at or above the reference) |
| equivalence reading | `Delta_C - delta_flex_C` = −1.49, paired instance-bootstrap interval [−5.49, +2.51] pairs, inside the pre-declared [−13, +13] |

Mechanism controls, R = 500 each, own random stream (`phase = "ccct"`), so the
observed statistics are not the Table S9 numbers.

| Setting | `(a,b,rho)` | `Delta_obs` | `delta_flex` | p | decision | required |
|---|---|---|---|---|---|---|
| B1 | (0.5, 2, 0.5) | 977 | −6.14 | 0.001996 | reject | reject, `p <= 0.002` |
| B5 | (1, 0, 0.8) | 334 | 0.00 | 0.001996 | reject | reject |
| B2 | (1, 0.5, 0.5) | −1 | 0.18 | 0.9940 | not reject | not reject |
| B4 | (1, 0, 0) | −1 | −1.082 | 0.9341 | not reject | not reject |

Pre-registered prediction 2 (Costanzo equivalence) and prediction 3 (B1 and B5
reject, B2 and B4 do not) both hold. Predictions 1 and 4 concern the Jonikas
panel and are undecided.

B5 is the informative one: D there carries no marginal information about the
label at all, so no marginal screen would flag it, and the joint learner still
wins 334 instances out of 4,096 while the permutation null sits at exactly
zero. That is Proposition CCCT-1 visible in data -- destroy the cross-channel
correspondence and the joint-minus-additive gain goes to zero.

## Not executed

| Run | Protocol section | Reason |
|---|---|---|
| **Jonikas primary panel, R = 1000 + K = 100** | 6 | stopped by the author before it started |
| B4 size across independent datasets | 7 | not run; the Section 8 validity condition cannot be evaluated. The single-dataset B4 run above gives one p-value, not a rejection rate. |
| Power calibration at panel dimensions | 7 | not run |
| CCCT on D+ | 15.3 | not run; R2's second clause is undecided |
| Sensitivity variants V1/V2/V3/V4/V5 | 4.5 | not run |

## What this means for the claims

**The Section 8 decision rule is not evaluated.** It requires the Jonikas
Holm-adjusted p-value, a B1 rejection, and a B4 size inside `[0.031, 0.073]`.
Two of the three inputs are absent. No `CCCT_SUPPORTED`, `CCCT_NOT_SUPPORTED`
or `CCCT_INVALID` verdict exists, and none may be inferred from the runs that
did complete.

**The Holm family is not applied.** Section 6 defines it over the two primary
panels. Adjusting over one panel would understate the correction, so the
Costanzo p-value of 0.8216 is reported unadjusted and labelled as such. It is
far from any threshold, so the distinction does not change its reading.

**One secondary statistic is a point estimate only.** Section 5 lists `Delta`
against `A_rbf` on both panels. The Costanzo run fitted PaD and `A_match` only,
so the observed value is available (611 versus 606, a difference of 5) but it
has no null distribution. It is descriptive and carries no correction.

**Costanzo alone does not carry the correspondence claim.** Its pre-registered
role is the discriminant check, and its equivalence reading speaks to whether
additive scoring suffices on that panel. It says nothing about whether the
Jonikas gain is attributable to cross-channel correspondence. The paper's
"not established" sentence therefore stands unless and until the Jonikas panel
is run.

**The interface ablation is self-contained and does stand.** Sections 15.3 and
15.4 are unpermuted refits with a pre-registered set of readings; they do not
depend on any permutation run. Their conclusion -- that the mechanism's raw
ordered profile is a direction oracle while the same restoration on the real
panel is at chance -- is complete as reported.

## Pilot observations, recorded but not evidence

A ten-replicate pilot of the Jonikas primary panel was run before `R` was
frozen, as Section 11 requires, and its replicates were discarded. It is
recorded in addendum 17.2 and 17.5 and is reported here only so that the
decision to stop is not mistaken for the absence of any observation:
the placebo-matched reference was 0.0587 against an untouched observed value of
0.0534, `delta_flex` was −0.0421, the standardized effect was 2.52, and none of
the ten null replicates reached the reference. Ten replicates give a smallest
attainable p-value of 1/11; this is not a test result and is not reported as
one.

## Resuming

The runs above are independent. To complete the protocol later:

```bash
python -m ccct.run panel --panel jonikas --replicates 1000 --placebo 100 --workers 4
python -m ccct.run control --settings B4 --datasets 200 --replicates 39 --workers 4
python -m ccct.run power --datasets 40 --replicates 99 --workers 4
python -m ccct.run panel --panel jonikas --block D+ --replicates 500 --placebo 100 --workers 4 --label jonikas_primary_Dplus
python -m ccct.run summarize
python verify_ccct.py --identity
```

Nothing needs to be re-run first: the protocol hash, the frozen inputs and the
seed streams are unchanged, and `summarize` recomputes the family and the
decision from whatever is present.
