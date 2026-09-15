# Section 6.4 replacement text

The paper currently says the source of the Jonikas gain "is not established"
and that mixed-context robustness "remains unresolved". Both sentences are
replaced by a prospectively specified decision. This file carries both prepared
outcomes; the numbers are filled from `ccct_results/ccct_summary.json`.

Protocol `b76a942bae0d1acab9be8a1f81a8effdc7c083c4bfddada02e7e872f18dcfa8b`.

## What the test is, in the paper's own terms (shared by both outcomes)

Fix both marginals and randomize only the correspondence between the channels:
reassign each pair's whole D measurement block among pairs that share the
pair's answer stratum, in a label-aligned orientation, and refit the complete
frozen procedure -- training-fold normalization, 35 candidates, the paired
one-standard-error selector, endpoint-excluded nested splits -- from scratch on
every replicate. Proposition CCCT-1 makes the null exactly the statement that
the joint term has nothing to buy: if `P` is conditionally independent of `D`
given the answer, the Bayes-optimal score is additive and the population
joint-minus-additive gain is zero for classification and for ranking alike.
The implication runs one way only, so a rejection says the channels are
conditionally dependent in a way the joint kernel converts into ranking gain,
not that the shared background of Section 5.1 has been identified.

Two properties of the test are stated rather than buried. First, permuting D
moves where a record's measurements came from without moving its name, so the
name-based endpoint exclusion no longer bounds what the training set knows
about the test record's D provenance; every replicate therefore rebuilds the
nested structure from the union of the name-sharing and provenance-sharing
graphs, which reduces to the archived exclusion under the identity permutation.
Second, because that denser rule shrinks training pools under the null but not
for the untouched observed statistic, the reference is placebo matched: draws
that keep the real D blocks and apply a permutation's exclusion graph. The
resulting p-value is approximate. Records sharing genes are not exchangeable,
and the null and placebo exclusion graphs are equal in law but not identical.

## Outcome A (supported): "Cross-channel correspondence is detectable on Jonikas and absent on Costanzo"

> On Jonikas the joint-minus-additive gain exceeds a correspondence-free
> permutation null at approximate `p = <P_JONIKAS>` (`R = 1000`, Holm-adjusted
> over the two panels), against a placebo-matched reference of
> `<REF_JONIKAS>` and a flexibility offset of `<FLEX_JONIKAS>`. On Costanzo the
> matched additive control is not exceeded and the paired interval for the
> gain net of the flexibility offset is `<CI_COSTANZO>` pairs, inside the
> pre-declared equivalence region of `[-13, +13]`; we read that as additive
> scoring sufficing on that panel, not merely as a non-rejection.
>
> The instrument is calibrated on four mechanism settings before it is
> believed: it rejects where a coupled benefit exists (B1) and where D is
> marginally independent of the answer yet helps by shared-noise removal (B5),
> and does not reject where D is pure noise (B4) or where the channels are
> conditionally dependent but the additive class already attains the joint
> optimum (B2). B4 is the one setting where the null holds exactly by
> construction; the rejection rate across independent datasets is
> `<SIZE_B4>`.

Abstract gains one clause: on one of two yeast panels the joint gain exceeds a
correspondence-free permutation null, while on the other the matched additive
control is not exceeded. The Discussion's "not established" sentence is
deleted. This is the paper's only inferential statement on real data and it is
labelled as approximate.

## Outcome B (not supported): "A prospective test for cross-channel correspondence, and what the yeast panels can detect"

> The joint gain on Jonikas is not distinguishable from the joint learner's
> flexibility on correspondence-free data at this sample size (approximate
> `p = <P_JONIKAS>`). The power calibration of Figure <N>, at the panel's
> dimensions and class balance, reaches 80 percent power at a conditional
> separation of about `<D80>`, so the panel cannot detect effects below that;
> the result is "undetectable at this size", not "no correspondence".

Contribution (iii) becomes a protocol contribution: a prospectively specified,
placebo-matched test for whether joint interpretation is warranted, validated
on four mechanism settings and applied to two panels. A test that returns "not
warranted" on Costanzo and "undetectable at this size" on Jonikas is a working
instrument, and it is a stronger position than "unresolved".

## Reporting obligations under either outcome

- `delta_flex` is reported as a scientific quantity, not a diagnostic, and the
  headline number is presented as a decomposition into the flexibility offset,
  the correspondence-attributable part and noise. Pre-registered prediction 4
  said this offset would be positive and below 0.02 AUROC; if the executed
  value is negative, the paper states that the product kernel loses rather than
  gains on correspondence-free data and that the observed gain is therefore
  further from the null than the raw difference suggests.
- The word "exact" is not used of the permutation p-value anywhere.
- Costanzo non-rejection is reported under the equivalence reading, never as
  the instrument saying additivity is enough by itself.
